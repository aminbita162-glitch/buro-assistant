"""
app/workers/trace.py – lightweight trace context (row 30).

A Trace is a structured record of one pipeline run: ingest → decide → draft → send.
It is written to the ``traces`` table (created by migration 0005) for every
message that passes through the worker pipeline.

No external tracing service is required.  The trace is a plain DB row so
operators can inspect it from the desk without additional infrastructure.

Trace fields
------------
id, tenant_id, message_id (FK), stage (ingest|decide|draft|send),
started_at, finished_at, duration_ms, status (ok|error), detail (text)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session

from app.main import Base


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class Trace(Base):
    __tablename__ = "traces"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)

    # Stage: ingest | decide | draft | send
    stage = Column(String, nullable=False, index=True)

    started_at  = Column(DateTime(timezone=True), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)       # wall-clock milliseconds

    # ok | error
    status = Column(String, nullable=False, default="ok")
    detail = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Helper: write a single trace row
# ---------------------------------------------------------------------------

def record_trace(
    db: Session,
    tenant_id: int,
    stage: str,
    started_at: datetime,
    finished_at: Optional[datetime] = None,
    status: str = "ok",
    detail: Optional[str] = None,
    message_id: Optional[int] = None,
) -> Trace:
    """Persist one trace row.  Does not raise; swallows DB errors silently."""
    if finished_at is None:
        finished_at = datetime.now(timezone.utc)
    duration_ms = (finished_at - started_at).total_seconds() * 1000.0
    t = Trace(
        tenant_id=tenant_id,
        message_id=message_id,
        stage=stage,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        status=status,
        detail=detail,
    )
    try:
        db.add(t)
        db.commit()
        db.refresh(t)
    except Exception:   # noqa: BLE001
        db.rollback()
    return t


# ---------------------------------------------------------------------------
# Context manager convenience wrapper (row 30)
# ---------------------------------------------------------------------------

@contextmanager
def traced(
    db: Session,
    tenant_id: int,
    stage: str,
    message_id: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that times the block and writes a Trace row on exit.

    Usage::

        with traced(db, tenant_id, "ingest", message_id=msg.id):
            ...do work...

    The yielded dict is populated with ``trace_id`` after the block succeeds.
    """
    started = datetime.now(timezone.utc)
    ctx: Dict[str, Any] = {}
    try:
        yield ctx
        tr = record_trace(
            db, tenant_id, stage, started, message_id=message_id,
            status="ok",
        )
        ctx["trace_id"] = tr.id
    except Exception as exc:   # noqa: BLE001
        record_trace(
            db, tenant_id, stage, started, message_id=message_id,
            status="error", detail=str(exc),
        )
        raise
