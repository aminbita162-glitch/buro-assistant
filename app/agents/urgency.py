"""
app/agents/urgency.py – urgency lexicon with tenant overrides (row 11).

Base lexicon maps words/phrases → score (positive = more urgent).
Tenant rule packs may supply ``urgency_overrides`` to add or suppress terms.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Base lexicon  (term → score, higher = more urgent)
# ---------------------------------------------------------------------------

BASE_LEXICON: Dict[str, int] = {
    # critical
    "urgent": 100,
    "asap": 100,
    "immediately": 100,
    "emergency": 100,
    "critical": 100,
    "deadline today": 100,
    "overdue": 90,
    # high
    "important": 70,
    "high priority": 70,
    "time-sensitive": 70,
    "time sensitive": 70,
    "as soon as possible": 70,
    "please respond": 50,
    "action required": 70,
    "follow up": 40,
    "reminder": 30,
    # low / neutral
    "whenever": -20,
    "no rush": -30,
    "low priority": -50,
    "fyi": -20,
    "for your information": -20,
}

_LEVEL_THRESHOLDS = [
    (100, "critical"),
    (60, "high"),
    (20, "medium"),
    (0, "low"),
    (None, "low"),
]


def score_urgency(
    text: str,
    tenant_overrides: Optional[Dict[str, int]] = None,
) -> int:
    """
    Return a numeric urgency score for *text*.

    *tenant_overrides* is merged on top of the base lexicon so tenants can
    add domain-specific terms or suppress base ones (set score to 0).
    """
    lexicon = dict(BASE_LEXICON)
    if tenant_overrides:
        lexicon.update(tenant_overrides)

    lower = text.lower()
    total = 0
    for term, score in lexicon.items():
        if term in lower:
            total += score
    return total


def classify_urgency(
    text: str,
    tenant_overrides: Optional[Dict[str, int]] = None,
) -> str:
    """
    Return an urgency level string: ``critical`` | ``high`` | ``medium`` | ``low``.
    """
    score = score_urgency(text, tenant_overrides)
    for threshold, level in _LEVEL_THRESHOLDS:
        if threshold is None or score >= threshold:
            return level
    return "low"
