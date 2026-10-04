"""
app/policy/calendar.py – business-hours calendar per tenant (row 19).

A tenant calendar defines working days and a daily window expressed as
UTC hour offsets.  ``is_business_hours(dt, calendar)`` returns whether
*dt* falls inside that window.

Calendar dict schema
--------------------
{
  "timezone": "UTC",          # IANA name; only UTC supported at this layer
  "working_days": [0,1,2,3,4], # 0=Monday … 6=Sunday
  "start_hour": 9,             # inclusive, UTC
  "end_hour": 17               # exclusive, UTC
}
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WORKING_DAYS: List[int] = [0, 1, 2, 3, 4]   # Mon–Fri
DEFAULT_START_HOUR: int = 9
DEFAULT_END_HOUR: int = 17


def _default_calendar() -> Dict[str, Any]:
    return {
        "timezone": "UTC",
        "working_days": DEFAULT_WORKING_DAYS,
        "start_hour": DEFAULT_START_HOUR,
        "end_hour": DEFAULT_END_HOUR,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_business_hours(
    dt: Optional[datetime] = None,
    calendar: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Return True if *dt* falls within the tenant's business hours.

    *dt* defaults to ``datetime.now(timezone.utc)`` when not supplied.
    *calendar* defaults to Mon–Fri 09:00–17:00 UTC when not supplied.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Normalise to UTC-aware datetime.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    cal = calendar or _default_calendar()
    working_days: List[int] = cal.get("working_days", DEFAULT_WORKING_DAYS)
    start_hour: int = int(cal.get("start_hour", DEFAULT_START_HOUR))
    end_hour: int = int(cal.get("end_hour", DEFAULT_END_HOUR))

    # weekday(): 0=Monday, 6=Sunday
    if dt.weekday() not in working_days:
        return False

    return start_hour <= dt.hour < end_hour


def next_business_start(
    dt: Optional[datetime] = None,
    calendar: Optional[Dict[str, Any]] = None,
) -> datetime:
    """
    Return the next business-day start time on or after *dt*.

    Useful for SLA deadline calculation when *dt* is outside business hours.
    """
    from datetime import timedelta

    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    cal = calendar or _default_calendar()
    working_days: List[int] = cal.get("working_days", DEFAULT_WORKING_DAYS)
    start_hour: int = int(cal.get("start_hour", DEFAULT_START_HOUR))

    # Try up to 14 days forward to find the next business day start.
    candidate = dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if candidate < dt:
        candidate += timedelta(days=1)

    for _ in range(14):
        if candidate.weekday() in working_days:
            return candidate
        candidate += timedelta(days=1)

    # Fallback: return same day start (should never be reached)
    return candidate
