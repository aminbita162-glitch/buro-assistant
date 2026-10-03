"""
app/ingest/providers/imap_provider.py – IMAP mailbox adapter (row 5).

Credentials are read exclusively from environment variables:

    IMAP_HOST      – hostname, e.g. imap.gmail.com
    IMAP_PORT      – port number, default 993
    IMAP_USER      – login username / email
    IMAP_PASSWORD  – login password or app-password
    IMAP_MAILBOX   – mailbox to poll, default INBOX
    IMAP_USE_SSL   – "true" (default) | "false"

No credential is accepted from any other source.  The adapter is
instantiated lazily so the process starts even when IMAP_HOST is absent.
"""
from __future__ import annotations

import email
import email.header
import imaplib
import json
import os
import re
import uuid
from email.policy import default as email_default_policy
from typing import Iterator

from app.ingest.normalize import Attachment, NormalizedMessage
from app.ingest.providers.base import MailProvider

# Attachment content-types that are always allowed through (row 12).
ATTACHMENT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "text/plain",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)


def _decode_header_value(raw: str) -> str:
    """Decode RFC 2047 encoded header value to plain text."""
    parts = email.header.decode_header(raw or "")
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def _classify_attachments(msg: email.message.Message) -> tuple[list[Attachment], str]:
    """
    Walk the MIME tree, collect attachments, and return
    (attachment_list, attachment_state).

    attachment_state:
      "none"        – no attachments present
      "clean"       – all attachments are on the allowlist
      "quarantine"  – at least one attachment has a disallowed content-type
    """
    attachments: list[Attachment] = []
    has_quarantine = False

    for part in msg.walk():
        disposition = part.get_content_disposition()
        if disposition not in ("attachment", "inline"):
            continue
        filename = part.get_filename() or "attachment"
        content_type = part.get_content_type().lower()
        payload = part.get_payload(decode=True) or b""
        att = Attachment(
            filename=filename,
            content_type=content_type,
            size_bytes=len(payload),
        )
        attachments.append(att)
        if content_type not in ATTACHMENT_ALLOWLIST:
            has_quarantine = True

    if not attachments:
        return [], "none"
    return attachments, ("quarantine" if has_quarantine else "clean")


class IMAPProvider(MailProvider):
    """
    IMAP mailbox adapter.  Credentials come exclusively from environment
    variables (row 5).  The adapter fetches UNSEEN messages from the
    configured mailbox and yields NormalizedMessage objects.
    """

    @property
    def provider_name(self) -> str:
        return "imap"

    def _connect(self) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        host = os.environ["IMAP_HOST"]   # raises KeyError if absent – by design
        port = int(os.environ.get("IMAP_PORT", "993"))
        use_ssl = os.environ.get("IMAP_USE_SSL", "true").lower() != "false"
        user = os.environ["IMAP_USER"]
        password = os.environ["IMAP_PASSWORD"]

        if use_ssl:
            conn = imaplib.IMAP4_SSL(host, port)
        else:
            conn = imaplib.IMAP4(host, port)

        conn.login(user, password)
        return conn

    def fetch_new(self, tenant_id: int) -> Iterator[NormalizedMessage]:
        mailbox = os.environ.get("IMAP_MAILBOX", "INBOX")
        conn = self._connect()
        try:
            conn.select(mailbox, readonly=False)
            _status, data = conn.search(None, "UNSEEN")
            uids = data[0].split() if data[0] else []
            for uid in uids:
                _fetch_status, msg_data = conn.fetch(uid, "(RFC822)")
                if not msg_data or not msg_data[0]:
                    continue
                raw_bytes: bytes = msg_data[0][1]  # type: ignore[index]
                parsed = email.message_from_bytes(
                    raw_bytes, policy=email_default_policy
                )
                yield self._to_normalized(parsed, raw_bytes, tenant_id)
        finally:
            try:
                conn.logout()
            except Exception:  # noqa: BLE001
                pass

    def _to_normalized(
        self,
        parsed: email.message.Message,
        raw_bytes: bytes,
        tenant_id: int,
    ) -> NormalizedMessage:
        subject = _decode_header_value(str(parsed.get("Subject", "")))
        sender = _decode_header_value(str(parsed.get("From", "")))
        recipients_raw = str(parsed.get("To", ""))
        recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
        msg_id_header = str(parsed.get("Message-Id", "")).strip() or None

        # Provider message id: prefer Message-Id, fall back to a UUID so the
        # idempotency constraint is always satisfiable.
        provider_message_id = msg_id_header or str(uuid.uuid4())

        # Extract plain-text body.
        body_text = ""
        if parsed.is_multipart():
            for part in parsed.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_text = payload.decode(charset, errors="replace")
                        break
        else:
            payload = parsed.get_payload(decode=True)
            if payload:
                charset = parsed.get_content_charset() or "utf-8"
                body_text = payload.decode(charset, errors="replace")

        attachments, _attachment_state = _classify_attachments(parsed)

        raw: dict = {
            "headers": dict(parsed.items()),
            "size_bytes": len(raw_bytes),
        }

        return NormalizedMessage(
            provider=self.provider_name,
            provider_message_id=provider_message_id,
            tenant_id=tenant_id,
            message_id_header=msg_id_header,
            subject=subject,
            subject_normalized=NormalizedMessage.normalize_subject(subject),
            sender=sender,
            recipients=recipients,
            body_text=body_text,
            attachments=attachments,
            raw=raw,
        )
