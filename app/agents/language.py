"""
app/agents/language.py – language detection (row 10).

Strategy: library first (langdetect when installed), fall back to a fast
character-set heuristic so tests never need an external package.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Fast heuristic fallback (no external dependency required)
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_FRENCH_MARKERS = frozenset([
    "bonjour", "merci", "vous", "nous", "est", "les", "des", "une", "pour",
])
_GERMAN_MARKERS = frozenset([
    "bitte", "danke", "sehr", "geehrte", "mit", "wir", "und", "auf",
])
_SPANISH_MARKERS = frozenset([
    "hola", "gracias", "por", "favor", "usted", "estamos", "una", "para",
])


def _heuristic(text: str) -> str:
    """Return an ISO 639-1 code using simple character-set heuristics."""
    if _ARABIC_RE.search(text):
        return "fa"   # Persian/Arabic script — default fa (Farsi)
    if _CJK_RE.search(text):
        return "zh"
    words = set(re.sub(r"[^\w\s]", " ", text.lower()).split())
    if words & _FRENCH_MARKERS:
        return "fr"
    if words & _GERMAN_MARKERS:
        return "de"
    if words & _SPANISH_MARKERS:
        return "es"
    return "en"


def detect_language(text: str) -> str:
    """
    Return an ISO 639-1 language code for *text*.

    Tries ``langdetect`` first (row 10: library first).  Falls back to the
    built-in heuristic if the library is not installed or the text is too
    short to classify reliably.
    """
    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect as _ld_detect  # type: ignore[import]
        from langdetect.lang_detect_exception import LangDetectException  # type: ignore[import]
        try:
            code = _ld_detect(text)
            return code if code else _heuristic(text)
        except LangDetectException:
            return _heuristic(text)
    except ImportError:
        return _heuristic(text)
