"""
app/workers/cost.py – cost and token fields on model calls (row 25).

Every model call made through the worker pipeline must record:
  - tokens_in   (prompt tokens)
  - tokens_out  (completion tokens)
  - cost_usd    (computed from the model's per-token price)

These fields live on the ``decisions`` table (added by migration 0005).
This module provides helpers to record costs and to read per-tenant totals.

Pricing table
-------------
Prices are approximations for planning.  The benchmark in docs/BENCHMARK.md
uses only measured values from real runs — this table is never printed as a
throughput claim.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Simple per-model price table (USD per 1 000 tokens)
# ---------------------------------------------------------------------------

_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4.1-mini": {"in": 0.000_150, "out": 0.000_600},
    "gpt-4o":        {"in": 0.005_000, "out": 0.015_000},
    "gpt-4o-mini":   {"in": 0.000_150, "out": 0.000_600},
    "fake":          {"in": 0.0,       "out": 0.0},
}

_DEFAULT_PRICE = {"in": 0.002, "out": 0.002}


def compute_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float:
    """Return estimated cost in USD for *tokens_in* + *tokens_out* on *model*."""
    prices = _PRICE_TABLE.get(model, _DEFAULT_PRICE)
    return (tokens_in * prices["in"] + tokens_out * prices["out"]) / 1000.0


# ---------------------------------------------------------------------------
# Cost record — attaches to a decision row
# ---------------------------------------------------------------------------

class CostRecord:
    """Holds cost metadata for one model call."""

    __slots__ = ("model", "tokens_in", "tokens_out", "cost_usd")

    def __init__(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        self.model = model
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = compute_cost(model, tokens_in, tokens_out)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
        }


def null_cost() -> CostRecord:
    """Return a zero-cost record for rule-hit or fake-model paths."""
    return CostRecord("fake", 0, 0)
