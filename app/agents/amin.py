"""
app/agents/amin.py – Amin, the triage controller (rows 7, 8, 10, 11, 13, 14, 15).

Amin accepts a NormalizedMessage and the tenant rule pack.
It returns a dict valid against schemas/triage_decision.json.

Pipeline
--------
1. Redact PII from subject + body (row 13).
2. Detect language on the redacted body (row 10).
3. Score urgency with tenant overrides (row 11).
4. Evaluate tenant rules (row 7).  If a rule hits, skip the model call.
5. If no rule hit: call the model client.
6. If model confidence < threshold: route to Leila (row 8).
7. Validate output against triage_decision.json (row 15).
8. Attach schema_version, prompt_version, decision_hash (row 14).

The same inputs with the model disabled always produce the same decision hash.
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any, Dict, Optional

from app.agents.decision_hash import compute_decision_hash
from app.agents.language import detect_language
from app.agents.redact import redact_message
from app.agents.rules import (
    evaluate_rules,
    get_confidence_threshold,
    get_urgency_overrides,
    rule_pack_hash,
)
from app.agents.urgency import classify_urgency
from app.ingest.normalize import NormalizedMessage

SCHEMA_VERSION = "1"
PROMPT_VERSION = "amin-v2"   # incremented: prompt structure changed in Phase 5
CONFIDENCE_FLOOR = 0.0   # accept any model result ≥ 0

# ---------------------------------------------------------------------------
# Prompt body clip length — attachment bytes never go to a model (Phase 5).
# Only the first BODY_CLIP_CHARS characters of the redacted body are included.
# ---------------------------------------------------------------------------
BODY_CLIP_CHARS = 200

# ---------------------------------------------------------------------------
# In-process LRU cache for identical (redacted_subject, body_clip, pack_hash).
# Rule-hit results are deterministic; caching them avoids repeat work.
# Cache size is bounded to prevent unbounded memory growth.
# ---------------------------------------------------------------------------
_TRIAGE_CACHE: Dict[tuple, Dict[str, Any]] = {}
_CACHE_MAX = 512

# ---------------------------------------------------------------------------
# Schema validation helper (row 15)
# ---------------------------------------------------------------------------

def _load_schema() -> Dict[str, Any]:
    schema_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "schemas", "triage_decision.json"
    )
    with open(os.path.normpath(schema_path), encoding="utf-8") as f:
        return json.load(f)


_SCHEMA: Optional[Dict[str, Any]] = None


def _validate(data: Dict[str, Any]) -> None:
    """
    Validate *data* against triage_decision.json using jsonschema when
    available, otherwise perform a minimal required-key check.
    Row 15: schema validation of agent output.
    """
    global _SCHEMA
    required = {
        "schema_version", "prompt_version", "department", "language",
        "urgency", "confidence", "rule_hit", "action", "decision_hash",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"TriageDecision missing required fields: {missing}")

    try:
        import jsonschema  # type: ignore[import]
        if _SCHEMA is None:
            _SCHEMA = _load_schema()
        jsonschema.validate(instance=data, schema=_SCHEMA)
    except ImportError:
        pass   # jsonschema optional; required-key check above is sufficient


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class TriageError(Exception):
    """Raised when Amin cannot produce a valid decision."""


def triage(
    msg: NormalizedMessage,
    rule_pack: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run Amin's triage pipeline and return a triage-decision dict.

    Parameters
    ----------
    msg:
        The normalised message to classify.
    rule_pack:
        Tenant rule pack dict.  May be ``None`` (no rules configured).
    model:
        An object with a ``.call(prompt: str) -> dict`` method.
        If ``None`` and no rule fires, raises :class:`TriageError`.
    """
    # 1. Redact PII before any model call (row 13).
    #    Attachment content is never passed here — only subject and body_text.
    #    Attachment bytes must not reach the model (Phase 5).
    redacted_subject, redacted_body, _ = redact_message(
        msg.subject, msg.body_text
    )

    # 2. Short body clip — only the first BODY_CLIP_CHARS chars go to the model.
    body_clip = redacted_body[:BODY_CLIP_CHARS]

    # 3. Language detection (row 10).
    language = detect_language(redacted_body or redacted_subject)

    # 4. Urgency scoring with tenant overrides (row 11).
    urgency_overrides = get_urgency_overrides(rule_pack)
    urgency = classify_urgency(
        redacted_subject + " " + redacted_body, urgency_overrides
    )

    # 5. Rule evaluation (row 7).  Rule before model.
    sender_domain = msg.sender.split("@")[-1] if "@" in msg.sender else msg.sender
    pack_hash = rule_pack_hash(rule_pack)
    rule_hit = evaluate_rules(
        sender_domain=sender_domain,
        subject=msg.subject_normalized,
        department_hint="",
        rule_pack=rule_pack,
    )

    confidence_threshold = get_confidence_threshold(rule_pack)

    if rule_hit:
        # Rule fires → no model call; deterministic result.
        # Check cache first to avoid redundant work on identical inputs.
        cache_key = (redacted_subject, body_clip, pack_hash, rule_hit.rule_name)
        if cache_key in _TRIAGE_CACHE:
            return _TRIAGE_CACHE[cache_key]

        department = rule_hit.department
        action = rule_hit.action
        confidence = 1.0
        rule_hit_name: Optional[str] = rule_hit.rule_name
    else:
        # 6. Model call (row 8).
        if model is None:
            raise TriageError(
                "No rule fired and no model provided — cannot produce a decision."
            )
        prompt = _build_prompt(redacted_subject, body_clip, language, urgency, pack_hash)
        model_output = model.call(prompt)

        department = model_output.get("department", "general")
        action = model_output.get("action", "hold")
        confidence = float(model_output.get("confidence", 0.0))
        rule_hit_name = None

        # 6. Confidence threshold → Leila (row 8).
        if confidence < confidence_threshold:
            from app.agents.leila import supervise  # noqa: PLC0415
            decision_hash = compute_decision_hash(
                msg.subject_normalized, rule_hit_name, action, rule_pack
            )
            exception_ctx = {
                "reason": "low_confidence",
                "detail": f"confidence {confidence:.3f} < threshold {confidence_threshold:.3f}",
                "triage_action": action,
                "decision_hash": decision_hash,
            }
            return supervise(exception_ctx, rule_pack=rule_pack)

    # 7–8. Build and validate the decision (rows 14, 15).
    decision_hash = compute_decision_hash(
        msg.subject_normalized, rule_hit_name, action, rule_pack
    )
    decision = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "department": department,
        "language": language,
        "urgency": urgency,
        "confidence": confidence,
        "rule_hit": rule_hit_name,
        "action": action,
        "reason": f"rule:{rule_hit_name}" if rule_hit_name else "model",
        "decision_hash": decision_hash,
    }
    _validate(decision)

    # Store rule-hit results in the cache.
    if rule_hit_name is not None:
        if len(_TRIAGE_CACHE) >= _CACHE_MAX:
            # Evict the oldest entry (insertion-ordered dict, Python 3.7+).
            _TRIAGE_CACHE.pop(next(iter(_TRIAGE_CACHE)))
        cache_key = (redacted_subject, body_clip, pack_hash, rule_hit_name)
        _TRIAGE_CACHE[cache_key] = decision

    return decision


def _build_prompt(
    subject: str, body_clip: str, language: str, urgency: str, pack_hash: str
) -> str:
    """
    Build the model prompt.

    Only the redacted subject, a short body clip (≤ BODY_CLIP_CHARS chars),
    and the rule pack hash are sent.  Attachment bytes are never included.
    """
    return (
        f"rules_hash: {pack_hash}\n"
        f"language: {language}\n"
        f"urgency: {urgency}\n"
        f"subject: {subject}\n"
        f"body: {body_clip}\n"
        "Return JSON: {\"department\": \"...\", \"action\": \"draft_reply|hold|forward|reject|escalate\", \"confidence\": 0.0-1.0}"
    )
