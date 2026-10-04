"""
app/workers/intake_loop.py – worker loop for live intake (follow-up Phase 2).

Polls the configured mail provider once per tick, runs each new message through
the pipeline, and sends failed jobs to the dead-letter queue.

Provider selection
------------------
- When ``IMAP_HOST`` is present in the environment the :class:`IMAPProvider`
  is used.
- Otherwise the loop falls back to the injected *provider* argument (default:
  ``FakeProvider``).  Tests pass a ``FakeProvider`` instance directly.

Secrets
-------
All mailbox credentials are read from environment variables inside
``IMAPProvider``.  This module never reads, logs, or forwards any secret.

Usage (library, blocking)::

    from app.workers.intake_loop import run_once, IntakeLoop

    # Single tick (useful for tests):
    stats = run_once(db, tenant_id=1, provider=fake_provider)

    # Long-running loop:
    loop = IntakeLoop(tenant_id=1, interval_seconds=30)
    loop.start()   # blocking
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.ingest.normalize import NormalizedMessage
from app.ingest.providers.base import MailProvider
from app.ingest.providers.fake_provider import FakeProvider
from app.main import SessionLocal
from app.pipeline import run_pipeline, PipelineResult
from app.workers.dlq import send_to_dlq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TickStats:
    """Counts for one poll tick."""
    processed: int = 0
    failed: int = 0
    dead_lettered: int = 0
    outcomes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _make_provider() -> MailProvider:
    """
    Return the appropriate provider.

    Uses IMAPProvider when IMAP_HOST is set; FakeProvider otherwise.
    The FakeProvider starts empty — in tests, callers pre-load messages.
    """
    if os.environ.get("IMAP_HOST"):
        from app.ingest.providers.imap_provider import IMAPProvider
        return IMAPProvider()
    return FakeProvider()


# ---------------------------------------------------------------------------
# Single-tick entry point
# ---------------------------------------------------------------------------

def run_once(
    db: Session,
    tenant_id: int,
    provider: Optional[MailProvider] = None,
    rule_pack: Optional[Dict[str, Any]] = None,
    policy_config: Optional[Dict[str, Any]] = None,
) -> TickStats:
    """
    Poll the provider once and process every new message.

    Parameters
    ----------
    db:
        Active SQLAlchemy session scoped to *tenant_id*.
    tenant_id:
        Tenant being processed.
    provider:
        Mail provider instance.  Defaults to :func:`_make_provider`.
    rule_pack:
        Optional rule pack forwarded to :func:`run_pipeline`.
    policy_config:
        Optional policy config forwarded to :func:`run_pipeline`.

    Returns
    -------
    TickStats
        Counts for this tick.
    """
    if provider is None:
        provider = _make_provider()

    stats = TickStats()

    try:
        messages = list(provider.fetch_new(tenant_id))
    except Exception as exc:  # noqa: BLE001
        logger.error("intake_loop: provider fetch failed for tenant %s: %s", tenant_id, exc)
        # Provider failure is not a per-message failure; return empty stats.
        return stats

    for msg in messages:
        try:
            result: PipelineResult = run_pipeline(
                db=db,
                msg=msg,
                rule_pack=rule_pack,
                policy_config=policy_config,
            )
            stats.processed += 1
            stats.outcomes.append(result.outcome)
            logger.debug(
                "intake_loop: tenant=%s provider_msg=%s outcome=%s",
                tenant_id, msg.provider_message_id, result.outcome,
            )
        except Exception as exc:  # noqa: BLE001
            stats.failed += 1
            logger.error(
                "intake_loop: pipeline error for tenant=%s provider_msg=%s: %s",
                tenant_id, msg.provider_message_id, exc,
            )
            # Send to dead-letter queue.
            try:
                send_to_dlq(
                    db,
                    tenant_id=tenant_id,
                    priority="medium",
                    payload=msg.provider_message_id,
                    failure_reason=str(exc),
                )
                stats.dead_lettered += 1
            except Exception as dlq_exc:  # noqa: BLE001
                logger.error(
                    "intake_loop: DLQ write failed for tenant=%s: %s",
                    tenant_id, dlq_exc,
                )

    return stats


# ---------------------------------------------------------------------------
# Long-running loop
# ---------------------------------------------------------------------------

class IntakeLoop:
    """
    Blocking poll loop.  Instantiate with a tenant and interval, then call
    :meth:`start`.

    Intended for process-supervisor or thread-based deployment.
    The loop is intentionally simple: one tenant, one DB session per tick,
    no concurrency.
    """

    def __init__(
        self,
        tenant_id: int,
        interval_seconds: float = 30.0,
        provider: Optional[MailProvider] = None,
        rule_pack: Optional[Dict[str, Any]] = None,
        policy_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.interval_seconds = interval_seconds
        self.provider = provider or _make_provider()
        self.rule_pack = rule_pack
        self.policy_config = policy_config
        self._running = False

    def start(self) -> None:
        """Poll until interrupted (KeyboardInterrupt or stop())."""
        self._running = True
        logger.info(
            "IntakeLoop starting for tenant=%s interval=%.1fs provider=%s",
            self.tenant_id, self.interval_seconds, self.provider.provider_name,
        )
        while self._running:
            db = SessionLocal()
            try:
                stats = run_once(
                    db,
                    tenant_id=self.tenant_id,
                    provider=self.provider,
                    rule_pack=self.rule_pack,
                    policy_config=self.policy_config,
                )
                if stats.processed or stats.failed:
                    logger.info(
                        "IntakeLoop tick: tenant=%s processed=%d failed=%d dlq=%d",
                        self.tenant_id, stats.processed, stats.failed, stats.dead_lettered,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("IntakeLoop tick error: %s", exc)
            finally:
                db.close()

            time.sleep(self.interval_seconds)

    def stop(self) -> None:
        """Signal the loop to exit after the current tick."""
        self._running = False
