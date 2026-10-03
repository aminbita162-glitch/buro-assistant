"""
app/agents/rules.py – tenant rule engine (row 7).

A rule pack is a plain dict (loaded from JSON or the DB) that maps
domain/subject/department patterns to triage decisions.  When a rule hits,
the model client is not called (row 7, row 8 – confidence threshold bypass).

Rule pack schema
----------------
{
  "domain_rules": [
    {"match": "example.com", "department": "support", "action": "draft_reply"}
  ],
  "subject_rules": [
    {"match": "invoice", "department": "billing", "action": "draft_reply"}
  ],
  "department_rules": [
    {"match": "hr", "department": "hr", "action": "hold"}
  ],
  "confidence_threshold": 0.7,
  "urgency_overrides": {"invoice overdue": 120}
}
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


CONFIDENCE_THRESHOLD_DEFAULT = 0.7


class RuleHit:
    """Returned when a rule fires."""

    def __init__(self, rule_name: str, department: str, action: str) -> None:
        self.rule_name = rule_name
        self.department = department
        self.action = action

    def __repr__(self) -> str:  # pragma: no cover
        return f"RuleHit({self.rule_name!r}, {self.department!r}, {self.action!r})"


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _matches(pattern: str, text: str) -> bool:
    """Case-insensitive substring match."""
    return _normalize(pattern) in _normalize(text)


def evaluate_rules(
    sender_domain: str,
    subject: str,
    department_hint: str,
    rule_pack: Optional[Dict[str, Any]] = None,
) -> Optional[RuleHit]:
    """
    Evaluate the tenant rule pack against the message envelope.

    Returns a :class:`RuleHit` if a rule fires, ``None`` otherwise.
    Rules are evaluated in order: domain → subject → department.
    First match wins.
    """
    if not rule_pack:
        return None

    for rule in rule_pack.get("domain_rules", []):
        if _matches(rule["match"], sender_domain):
            return RuleHit(
                rule_name=f"domain:{rule['match']}",
                department=rule.get("department", "general"),
                action=rule.get("action", "draft_reply"),
            )

    for rule in rule_pack.get("subject_rules", []):
        if _matches(rule["match"], subject):
            return RuleHit(
                rule_name=f"subject:{rule['match']}",
                department=rule.get("department", "general"),
                action=rule.get("action", "draft_reply"),
            )

    for rule in rule_pack.get("department_rules", []):
        if _matches(rule["match"], department_hint):
            return RuleHit(
                rule_name=f"department:{rule['match']}",
                department=rule.get("department", department_hint),
                action=rule.get("action", "hold"),
            )

    return None


def get_confidence_threshold(rule_pack: Optional[Dict[str, Any]] = None) -> float:
    """Return the tenant confidence threshold, defaulting to 0.7."""
    if not rule_pack:
        return CONFIDENCE_THRESHOLD_DEFAULT
    return float(rule_pack.get("confidence_threshold", CONFIDENCE_THRESHOLD_DEFAULT))


def get_urgency_overrides(rule_pack: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Return tenant urgency overrides dict."""
    if not rule_pack:
        return {}
    return dict(rule_pack.get("urgency_overrides", {}))
