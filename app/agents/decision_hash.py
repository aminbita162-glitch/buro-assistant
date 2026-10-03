"""
app/agents/decision_hash.py – deterministic decision hash (row 14).

The same tenant rules, normalised subject, rule_hit name, and action always
produce the same SHA-256 hex digest — with or without a live model.  This
lets operators verify that the same input always produces the same decision.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def compute_decision_hash(
    subject_normalized: str,
    rule_hit: Optional[str],
    action: str,
    rule_pack: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return a deterministic SHA-256 hex digest.

    Inputs are sorted before hashing so dict key order does not matter.

    Parameters
    ----------
    subject_normalized:
        Normalised subject string (lower-cased, whitespace-collapsed).
    rule_hit:
        Name of the rule that fired, or ``None`` if a model call was made.
    action:
        The decided action string (e.g. ``"draft_reply"``).
    rule_pack:
        The tenant rule pack dict.  Only the stable, deterministic fields are
        included: ``domain_rules``, ``subject_rules``, ``department_rules``.
    """
    stable_pack: Dict[str, Any] = {}
    if rule_pack:
        for key in ("domain_rules", "subject_rules", "department_rules"):
            if key in rule_pack:
                stable_pack[key] = rule_pack[key]

    payload = {
        "subject_normalized": subject_normalized,
        "rule_hit": rule_hit,
        "action": action,
        "rule_pack": stable_pack,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
