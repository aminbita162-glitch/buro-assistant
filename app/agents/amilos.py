"""
app/agents/amilos.py – Amilos, the reply agent (rows 14, 15).

Amilos accepts a TriageDecision dict and an approved template string.
It returns a dict valid against schemas/reply_draft.json.

Constraints (from DIRECTIVE.txt §4):
- Cannot invent a price, a legal promise, or a date not in the template variables.
- Output must pass schema validation before being returned (row 15).
- schema_version and prompt_version are stored on every draft (row 14).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from app.agents.decision_hash import compute_decision_hash

SCHEMA_VERSION = "1"
PROMPT_VERSION = "amilos-v1"

# Patterns that must never appear in a generated reply body.
_FORBIDDEN_PATTERNS = [
    re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?"),          # price: $1,234.56
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),       # date: 01/01/2024
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),              # ISO date: 2024-01-01
    re.compile(r"\bguarant(?:ee|y)\b", re.IGNORECASE), # legal promise
    re.compile(r"\bwarrant(?:y|ee)\b", re.IGNORECASE),
    re.compile(r"\bliab(?:le|ility)\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Schema validation (row 15)
# ---------------------------------------------------------------------------

def _load_schema() -> Dict[str, Any]:
    schema_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "schemas", "reply_draft.json"
    )
    with open(os.path.normpath(schema_path), encoding="utf-8") as f:
        return json.load(f)


_SCHEMA: Optional[Dict[str, Any]] = None


def _validate(data: Dict[str, Any]) -> None:
    required = {"schema_version", "prompt_version", "template_id",
                 "subject", "body", "decision_hash"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"ReplyDraft missing required fields: {missing}")
    try:
        import jsonschema  # type: ignore[import]
        global _SCHEMA
        if _SCHEMA is None:
            _SCHEMA = _load_schema()
        jsonschema.validate(instance=data, schema=_SCHEMA)
    except ImportError:
        pass


def _check_forbidden(body: str) -> None:
    """Raise ValueError if forbidden content is detected in the reply body."""
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(body):
            raise ValueError(
                f"Reply body contains forbidden content matching {pattern.pattern!r}"
            )

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class ReplyError(Exception):
    """Raised when Amilos cannot produce a valid draft."""


def draft_reply(
    triage_decision: Dict[str, Any],
    template: str,
    template_id: str,
    model: Optional[Any] = None,
    variables: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Produce a reply draft from *template* and optional *variables*.

    When *model* is ``None``, the template is used as-is after variable
    substitution (deterministic path, no model call).

    Parameters
    ----------
    triage_decision:
        The dict returned by :func:`app.agents.amin.triage`.
    template:
        Approved template string.  Use ``{variable_name}`` placeholders.
    template_id:
        ID that identifies the template in the registry.
    model:
        Optional model client with ``.call(prompt) -> dict``.
    variables:
        Template variable substitutions supplied by the policy layer.
    """
    variables = variables or {}
    try:
        body = template.format(**variables)
    except KeyError as exc:
        raise ReplyError(f"Template variable missing: {exc}") from exc

    subject = variables.get("subject", triage_decision.get("department", "Re: your message"))

    if model is not None:
        prompt = (
            f"Department: {triage_decision.get('department')}\n"
            f"Language: {triage_decision.get('language', 'en')}\n"
            f"Template: {template}\n"
            f"Variables: {json.dumps(variables)}\n"
            "Return JSON: {\"subject\": \"...\", \"body\": \"...\"}"
        )
        try:
            out = model.call(prompt)
            body = out.get("body", body)
            subject = out.get("subject", subject)
        except Exception as exc:
            raise ReplyError(f"Model call failed: {exc}") from exc

    _check_forbidden(body)

    decision_hash = compute_decision_hash(
        subject_normalized=str(subject).lower(),
        rule_hit=triage_decision.get("rule_hit"),
        action=triage_decision.get("action", "draft_reply"),
        rule_pack=None,
    )

    draft = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "template_id": template_id,
        "subject": subject,
        "body": body,
        "language": triage_decision.get("language", "en"),
        "decision_hash": decision_hash,
    }
    _validate(draft)
    return draft
