"""
app/agents/leila.py – Leila, the supervisor agent (row 15).

Leila accepts an exception context and returns a dict valid against
schemas/supervisor_decision.json.

Allowed actions: hold, request_human, reject, reroute.
Leila cannot send or delete (DIRECTIVE.txt §4).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from app.agents.decision_hash import compute_decision_hash

SCHEMA_VERSION = "1"
PROMPT_VERSION = "leila-v1"

# ---------------------------------------------------------------------------
# Schema validation (row 15)
# ---------------------------------------------------------------------------

def _load_schema() -> Dict[str, Any]:
    schema_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "schemas", "supervisor_decision.json"
    )
    with open(os.path.normpath(schema_path), encoding="utf-8") as f:
        return json.load(f)


_SCHEMA: Optional[Dict[str, Any]] = None


def _validate(data: Dict[str, Any]) -> None:
    required = {"schema_version", "prompt_version", "action", "reason", "decision_hash"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"SupervisorDecision missing required fields: {missing}")
    try:
        import jsonschema  # type: ignore[import]
        global _SCHEMA
        if _SCHEMA is None:
            _SCHEMA = _load_schema()
        jsonschema.validate(instance=data, schema=_SCHEMA)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Exception-to-action mapping
# ---------------------------------------------------------------------------

_REASON_TO_ACTION: Dict[str, str] = {
    "low_confidence": "request_human",
    "schema_invalid": "hold",
    "rule_exception": "hold",
    "no_department": "reroute",
    "quarantine": "reject",
}

_ALLOWED_ACTIONS = frozenset(["hold", "request_human", "reject", "reroute"])

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class SupervisionError(Exception):
    """Raised when Leila cannot produce a valid decision."""


def supervise(
    exception_ctx: Dict[str, Any],
    rule_pack: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Produce a supervisor decision for *exception_ctx*.

    Parameters
    ----------
    exception_ctx:
        Dict with at least ``reason`` (str) and optional ``detail``,
        ``triage_action``, ``decision_hash``.
    rule_pack:
        Tenant rule pack — consulted to check whether rerouting is configured.
    model:
        Optional model client.  If provided and the reason is not in the
        deterministic mapping, the model is consulted.
    """
    reason = exception_ctx.get("reason", "unknown")
    detail = exception_ctx.get("detail", "")
    incoming_hash = exception_ctx.get("decision_hash", "")

    # Deterministic mapping covers most cases.
    action = _REASON_TO_ACTION.get(reason)

    if action is None:
        if model is not None:
            prompt = (
                f"Exception reason: {reason}\n"
                f"Detail: {detail}\n"
                "Return JSON: {\"action\": \"hold|request_human|reject|reroute\", \"reroute_department\": null}"
            )
            try:
                out = model.call(prompt)
                action = out.get("action", "hold")
            except Exception:
                action = "hold"
        else:
            action = "hold"

    if action not in _ALLOWED_ACTIONS:
        action = "hold"

    reroute_department: Optional[str] = None
    if action == "reroute":
        reroute_department = exception_ctx.get("reroute_department") or "general"

    decision_hash = compute_decision_hash(
        subject_normalized=reason,
        rule_hit=None,
        action=action,
        rule_pack=rule_pack,
    )

    decision = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "action": action,
        "reason": reason,
        "detail": detail,
        "reroute_department": reroute_department,
        "decision_hash": decision_hash,
    }
    _validate(decision)
    return decision
