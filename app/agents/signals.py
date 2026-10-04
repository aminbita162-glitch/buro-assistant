"""
app/agents/signals.py – desk signals: semantic duplicate and dissatisfied tone
(Phase 6).

Semantic duplicate
------------------
A semantic duplicate is a message whose content is *near-identical* to a
recent message from the same tenant even though it does not match the exact
idempotency key (provider_message_id) or the header/subject duplicate rules
from Phase 1.  It is recorded as a boolean flag on the Message row.  No
second Message row is created; the original is kept.

The check uses a simple normalised-body overlap heuristic (Jaccard on word
sets) so that no model call is required.  The threshold is configurable and
defaults to 0.85.

Dissatisfied tone
-----------------
A dissatisfied-tone flag is set when the message body contains language
that signals customer dissatisfaction.  When flagged:
  - the message is routed to Leila (supervise) rather than Amin continuing.
  - the pipeline does not send a reply.

The check uses a keyword lexicon.  No model call is required.

Public API
----------
is_semantic_duplicate(body: str, candidates: list[str], threshold: float) -> bool
    Return True when *body* is near-identical to any string in *candidates*.

has_dissatisfied_tone(text: str) -> bool
    Return True when *text* contains dissatisfaction signals.
"""
from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Semantic duplicate
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLD = 0.85


def _word_set(text: str) -> set:
    """Return the set of lowercase words in *text*."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    """Return the Jaccard similarity of two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def is_semantic_duplicate(
    body: str,
    candidates: List[str],
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    """
    Return True when *body* is near-identical to any string in *candidates*.

    Uses Jaccard similarity on word sets.  No model is called.
    A threshold of 1.0 requires exact word-set equality.
    """
    ws = _word_set(body)
    for candidate in candidates:
        if _jaccard(ws, _word_set(candidate)) >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Dissatisfied tone
# ---------------------------------------------------------------------------

# Lexicon of dissatisfaction signals.  Matched case-insensitively as
# substrings.  Terms chosen to minimise false positives.
_DISSATISFIED_LEXICON: tuple[str, ...] = (
    "unacceptable",
    "very disappointed",
    "deeply disappointed",
    "not satisfied",
    "unsatisfied",
    "terrible service",
    "worst service",
    "worst experience",
    "demand a refund",
    "requesting a refund",
    "escalate this",
    "completely unacceptable",
    "i am appalled",
    "i am furious",
    "absolutely disgusted",
    "this is ridiculous",
    "i want to complain",
    "formal complaint",
    "not good enough",
    "extremely unhappy",
    "extremely disappointed",
    "very unhappy",
    "will never use",
    "will never return",
    "will not recommend",
    "do not recommend",
)


def has_dissatisfied_tone(text: str) -> bool:
    """
    Return True when *text* contains dissatisfaction signals.

    Uses a keyword lexicon only.  No model is called.
    A True result routes to Leila and blocks auto-send.
    """
    lower = text.lower()
    return any(term in lower for term in _DISSATISFIED_LEXICON)
