"""
app/policy/sla.py – SLA clock from ingest time (row 20).

SLA tiers map urgency levels to target response hours.  The clock starts at
``ingest_time`` (stored on the Message row).

SLA config dict schema
----------------------
{
  "sla_hours": {
    "critical": 1,
    "high": 4,
    "medium": 24,
    "low": 72
  }
}
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SLA_HOURS: Dict[str, int] = {
    "critical": 1,
    "high": 4,
    "medium": 24,
    "low": 72,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sla_deadline(
    ingest_time: datetime,
    urgency: str,
    sla_config: Optional[Dict[str, Any]] = None,
) -> datetime:
    """
    Return the SLA deadline for a message.

    Parameters
    ----------
    ingest_time:
        The UTC timestamp at which the message was ingested (Message.ingest_time).
    urgency:
        One of ``critical | high | medium | low``.
    sla_config:
        Optional tenant SLA configuration dict with ``sla_hours`` mapping.
        Falls back to :data:`DEFAULT_SLA_HOURS`.
    """
    if ingest_time.tzinfo is None:
        ingest_time = ingest_time.replace(tzinfo=timezone.utc)

    hours_map: Dict[str, int] = DEFAULT_SLA_HOURS.copy()
    if sla_config:
        hours_map.update(sla_config.get("sla_hours", {}))

    hours = hours_map.get(urgency, hours_map["low"])
    return ingest_time + timedelta(hours=hours)


def is_sla_breached(
    ingest_time: datetime,
    urgency: str,
    sla_config: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> bool:
    """Return True if the SLA deadline has already passed."""
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now >= sla_deadline(ingest_time, urgency, sla_config)


def sla_remaining_seconds(
    ingest_time: datetime,
    urgency: str,
    sla_config: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> float:
    """
    Return seconds remaining until the SLA deadline.
    Negative means the SLA is already breached.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    deadline = sla_deadline(ingest_time, urgency, sla_config)
    return (deadline - now).total_seconds()
