"""
app/ingest/sender_auth.py – Phase 1: sender authentication.

Checks SPF, DKIM, and DMARC for an inbound message and returns a
SenderAuthResult carrying one of three values per check:

  "pass"    – the DNS record was found and the check succeeded
  "fail"    – the DNS record was found but the check failed
  "not_run" – live DNS is not configured; the check was skipped

Rules
-----
- A fail does NOT auto-send and does NOT delete the message.
  The pipeline layer enforces the send-block; this module only reads.
- The actual DNS lookups require the ``dnspython`` library (dnspython>=2).
  If the library is absent, every check returns "not_run".
- When SENDER_AUTH_ENABLED is not set to "1" in the environment, every
  check returns "not_run" so the app still runs without live DNS.

Only the sender domain is checked; no message bytes leave the process.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

AUTH_PASS = "pass"
AUTH_FAIL = "fail"
AUTH_NOT_RUN = "not_run"

VALID_VALUES = frozenset({AUTH_PASS, AUTH_FAIL, AUTH_NOT_RUN})


@dataclass
class SenderAuthResult:
    """Holds one auth result per check."""
    spf: str = AUTH_NOT_RUN
    dkim: str = AUTH_NOT_RUN
    dmarc: str = AUTH_NOT_RUN

    def __post_init__(self) -> None:
        for field_name, value in (
            ("spf", self.spf),
            ("dkim", self.dkim),
            ("dmarc", self.dmarc),
        ):
            if value not in VALID_VALUES:
                raise ValueError(
                    f"SenderAuthResult.{field_name} must be one of {VALID_VALUES!r}; got {value!r}"
                )

    @property
    def any_fail(self) -> bool:
        """True when at least one check has result 'fail'."""
        return AUTH_FAIL in (self.spf, self.dkim, self.dmarc)


# ---------------------------------------------------------------------------
# Sentinel result used when checks are skipped
# ---------------------------------------------------------------------------

NOT_RUN_RESULT = SenderAuthResult(
    spf=AUTH_NOT_RUN, dkim=AUTH_NOT_RUN, dmarc=AUTH_NOT_RUN
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_sender_auth(
    sender: str,
    raw_headers: Optional[dict] = None,
) -> SenderAuthResult:
    """
    Run SPF, DKIM, and DMARC checks for *sender*.

    Parameters
    ----------
    sender:
        The envelope sender address (e.g. ``"user@example.com"``).
    raw_headers:
        Optional dict of raw message headers.  Some DKIM validators read
        the ``DKIM-Signature`` header directly from here.  May be None.

    Returns
    -------
    SenderAuthResult
        Each field is "pass", "fail", or "not_run".
    """
    # Guard: skip if not explicitly enabled.
    if os.environ.get("SENDER_AUTH_ENABLED", "0") != "1":
        return NOT_RUN_RESULT

    # Guard: sender must contain a domain.
    if not sender or "@" not in sender:
        return NOT_RUN_RESULT

    domain = sender.split("@")[-1].lower().strip()
    if not domain:
        return NOT_RUN_RESULT

    spf = _check_spf(domain)
    dkim = _check_dkim(domain, raw_headers or {})
    dmarc = _check_dmarc(domain)

    return SenderAuthResult(spf=spf, dkim=dkim, dmarc=dmarc)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_spf(domain: str) -> str:
    """
    Return "pass" if a valid SPF TXT record exists for *domain*, "fail" if a
    record exists but is a hard-fail (``-all``), "not_run" if dnspython is
    absent or the lookup raises.
    """
    try:
        import dns.resolver  # type: ignore[import]
    except ImportError:
        return AUTH_NOT_RUN

    try:
        answers = dns.resolver.resolve(domain, "TXT", lifetime=5)
        for rdata in answers:
            txt = "".join(s.decode("utf-8", errors="replace") for s in rdata.strings)
            if txt.startswith("v=spf1"):
                # Hard-fail (``-all``) → fail; soft-fail (``~all``) → pass
                if "-all" in txt:
                    return AUTH_FAIL
                return AUTH_PASS
        # No SPF record found
        return AUTH_FAIL
    except Exception:  # noqa: BLE001
        return AUTH_NOT_RUN


def _check_dkim(domain: str, raw_headers: dict) -> str:
    """
    Return "pass" if a DKIM-Signature header is present and the selector key
    resolves in DNS.  Returns "fail" if the header is missing or the selector
    key does not resolve.  Returns "not_run" if dnspython is absent.
    """
    try:
        import dns.resolver  # type: ignore[import]
    except ImportError:
        return AUTH_NOT_RUN

    # Look for a DKIM-Signature header (case-insensitive key lookup).
    dkim_sig: Optional[str] = None
    for k, v in raw_headers.items():
        if k.lower() == "dkim-signature":
            dkim_sig = v
            break

    if not dkim_sig:
        return AUTH_FAIL

    # Extract the selector (``s=`` tag) and domain (``d=`` tag).
    selector = _dkim_tag(dkim_sig, "s")
    sig_domain = _dkim_tag(dkim_sig, "d")
    if not selector or not sig_domain:
        return AUTH_FAIL

    # Verify the public key record exists.
    key_record = f"{selector}._domainkey.{sig_domain}"
    try:
        dns.resolver.resolve(key_record, "TXT", lifetime=5)
        return AUTH_PASS
    except Exception:  # noqa: BLE001
        return AUTH_FAIL


def _check_dmarc(domain: str) -> str:
    """
    Return "pass" if a DMARC TXT record exists at ``_dmarc.<domain>``.
    Returns "fail" if no record is found.
    Returns "not_run" if dnspython is absent or the lookup raises unexpectedly.
    """
    try:
        import dns.resolver  # type: ignore[import]
    except ImportError:
        return AUTH_NOT_RUN

    dmarc_domain = f"_dmarc.{domain}"
    try:
        answers = dns.resolver.resolve(dmarc_domain, "TXT", lifetime=5)
        for rdata in answers:
            txt = "".join(s.decode("utf-8", errors="replace") for s in rdata.strings)
            if txt.startswith("v=DMARC1"):
                return AUTH_PASS
        return AUTH_FAIL
    except Exception:  # noqa: BLE001
        return AUTH_NOT_RUN


# ---------------------------------------------------------------------------
# DKIM tag parser helper
# ---------------------------------------------------------------------------

def _dkim_tag(header_value: str, tag: str) -> Optional[str]:
    """Extract the value of a DKIM header tag (``tag=value``)."""
    for part in header_value.split(";"):
        part = part.strip()
        if part.startswith(f"{tag}="):
            return part[len(tag) + 1:].strip()
    return None
