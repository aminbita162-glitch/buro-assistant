"""
app/agents/local_model.py – local model route (Phase 3).

When a tenant sets ``local_model: true`` in its policy config the pipeline
must not call the cloud model client.  This module provides:

  ``is_local_model_enabled(policy_config)``
      Returns True when the tenant flag is set.

  ``FakeLocalModel``
      Deterministic stub used in tests.  It never calls any external service.
      It behaves identically to FakeModel but is a distinct class so tests can
      assert *which* client was used.

No real local-inference vendor is shipped here and no new vendor key is
introduced in this repository.  A production operator replaces FakeLocalModel
with their own implementation that satisfies the ``.call(prompt) -> dict``
contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Flag reader
# ---------------------------------------------------------------------------

def is_local_model_enabled(policy_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Return True when the tenant's policy config has ``local_model: true``.

    A missing or falsy value returns False so existing tenants are unaffected.
    """
    if not policy_config:
        return False
    return bool(policy_config.get("local_model", False))


# ---------------------------------------------------------------------------
# Fake local client — used in tests only
# ---------------------------------------------------------------------------

class FakeLocalModel:
    """
    Deterministic stub that satisfies the ``.call(prompt) -> dict`` contract.

    Usage in tests
    --------------
        model = FakeLocalModel(responses=[{"department": "billing", ...}])
        result = model.call(prompt="...")

    The client pops responses in order and raises ``StopIteration`` when the
    queue is exhausted.  A ``None`` entry raises ``LocalModelError`` to
    exercise the error path.

    This class intentionally mirrors ``FakeModel`` so tests can assert on the
    *type* of client the pipeline chose.
    """

    def __init__(
        self,
        responses: Optional[List[Optional[Dict[str, Any]]]] = None,
        default_confidence: float = 0.9,
    ) -> None:
        self._responses: List[Optional[Dict[str, Any]]] = list(responses or [])
        self.default_confidence = default_confidence
        self.calls: List[str] = []

    def queue(self, response: Optional[Dict[str, Any]]) -> None:
        """Append a response to the back of the queue."""
        self._responses.append(response)

    def call(self, prompt: str) -> Dict[str, Any]:
        """
        Return the next queued response.

        Raises :class:`LocalModelError` when the next item is ``None``.
        Raises ``StopIteration`` when the queue is empty.
        """
        self.calls.append(prompt)
        if not self._responses:
            raise StopIteration("FakeLocalModel response queue is empty")
        item = self._responses.pop(0)
        if item is None:
            raise LocalModelError("FakeLocalModel: sentinel None response")
        if "confidence" not in item:
            item = dict(item)
            item["confidence"] = self.default_confidence
        return item


class LocalModelError(Exception):
    """Raised by FakeLocalModel when a None sentinel is encountered."""
