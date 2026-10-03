"""
app/agents/fake_model.py – deterministic fake model client for tests.

Usage
-----
    from app.agents.fake_model import FakeModel
    model = FakeModel(responses=[{"department": "billing", ...}])
    result = model.call(prompt="...")

The fake model never calls any external service.  It pops responses in order
and raises ``StopIteration`` if the list is exhausted.  A ``None`` entry
triggers a ``ModelCallError`` to test the low-confidence / schema-error path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ModelCallError(Exception):
    """Raised by FakeModel when a None sentinel is encountered."""


class FakeModel:
    """
    In-memory stub for the model client.

    Parameters
    ----------
    responses:
        Ordered list of dicts to return from :meth:`call`.  A ``None`` entry
        causes a :class:`ModelCallError` (simulates a failed/invalid call).
    default_confidence:
        Confidence injected into responses that do not already contain it.
    """

    def __init__(
        self,
        responses: Optional[List[Optional[Dict[str, Any]]]] = None,
        default_confidence: float = 0.9,
    ) -> None:
        self._responses: List[Optional[Dict[str, Any]]] = list(responses or [])
        self.default_confidence = default_confidence
        self.calls: List[str] = []   # recorded prompts

    def queue(self, response: Optional[Dict[str, Any]]) -> None:
        """Add a response to the back of the queue."""
        self._responses.append(response)

    def call(self, prompt: str) -> Dict[str, Any]:
        """
        Return the next queued response.

        Raises :class:`ModelCallError` if the next item is ``None``.
        Raises :class:`StopIteration` if the queue is empty.
        """
        self.calls.append(prompt)
        if not self._responses:
            raise StopIteration("FakeModel response queue is empty")
        item = self._responses.pop(0)
        if item is None:
            raise ModelCallError("FakeModel: sentinel None response")
        if "confidence" not in item:
            item = dict(item)
            item["confidence"] = self.default_confidence
        return item
