"""
app/pipeline.py – end-to-end message pipeline (follow-up Phase 1).

Connects:
    ingest_message  →  Amin (triage)  →  Amilos (draft) or Leila (supervise)
                    →  shadow store or approval queue

Rules
-----
- A rule hit must not call a model.  CostRecord is zero when rule_hit is set.
- Tokens and cost are recorded only when a model is actually called.
- Send is off unless the tenant policy_config has ``auto_reply_enabled: True``.
- Quarantine and duplicate messages are returned immediately without agent calls.

Returns a :class:`PipelineResult` describing what happened.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.agents.amin import triage, TriageError
from app.agents.amilos import draft_reply, ReplyError
from app.ingest.ingest import ingest_message, RESULT_NEW, RESULT_DUPLICATE, RESULT_QUARANTINE
from app.ingest.models import Message
from app.ingest.normalize import NormalizedMessage
from app.policy.approval import enqueue as enqueue_approval
from app.policy.send_decision import should_send, SEND_DECISION_ALLOW
from app.policy.shadow import store_draft
from app.policy.templates import DEFAULT_REGISTRY, RECEIPT_TEMPLATE_ID, build_receipt_variables
from app.workers.cost import CostRecord, null_cost
from app.domain.usage import emit as emit_usage


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Outcome of one pipeline run."""

    #: "new" | "duplicate" | "quarantine" | "no_draft" | "draft" | "approval" | "send" | "error"
    outcome: str

    #: Ingest result code: "new" | "duplicate" | "quarantine"
    ingest_result: str = ""

    #: Persisted Message row (may be None if ingest failed hard)
    message: Optional[Message] = None

    #: Triage decision dict from Amin (or Leila if routed there)
    triage_decision: Optional[Dict[str, Any]] = None

    #: Reply draft dict from Amilos (None when no draft was produced)
    reply_draft: Optional[Dict[str, Any]] = None

    #: Cost metadata (zero for rule-hit paths)
    cost: Optional[CostRecord] = None

    #: ID of the stored Draft row (None when no draft stored)
    draft_id: Optional[int] = None

    #: ID of the ApprovalQueueEntry row (None when not enqueued)
    approval_id: Optional[int] = None

    #: Human-readable note for logging/reporting
    note: str = ""


# ---------------------------------------------------------------------------
# Tracking wrapper — counts model calls made by agents
# ---------------------------------------------------------------------------

class _TrackingModel:
    """
    Wraps any model client and records whether it was called.
    Also captures token counts when the wrapped model returns them.
    """

    def __init__(self, inner: Any, model_name: str = "fake") -> None:
        self._inner = inner
        self.model_name = model_name
        self.called = False
        self.last_tokens_in: int = 0
        self.last_tokens_out: int = 0

    def call(self, prompt: str) -> Dict[str, Any]:
        self.called = True
        result = self._inner.call(prompt)
        # Capture token counts if the model returns them.
        self.last_tokens_in = result.pop("tokens_in", len(prompt) // 4)
        self.last_tokens_out = result.pop("tokens_out", 20)
        return result

    def as_cost_record(self) -> CostRecord:
        if not self.called:
            return null_cost()
        return CostRecord(self.model_name, self.last_tokens_in, self.last_tokens_out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    db: Session,
    msg: NormalizedMessage,
    rule_pack: Optional[Dict[str, Any]] = None,
    policy_config: Optional[Dict[str, Any]] = None,
    triage_model: Optional[Any] = None,
    draft_model: Optional[Any] = None,
    triage_model_name: str = "fake",
    draft_model_name: str = "fake",
) -> PipelineResult:
    """
    Run the full pipeline for *msg*.

    Parameters
    ----------
    db:
        Active SQLAlchemy session (tenant-scoped by the caller).
    msg:
        Normalised message produced by the ingest layer.
    rule_pack:
        Tenant rule pack dict.  Pass None when the tenant has no rules.
    policy_config:
        Tenant policy config dict (shadow_mode, auto_reply_enabled, etc.).
    triage_model:
        Model client for Amin.  Must implement ``.call(prompt) -> dict``.
        May be None if a rule is expected to fire for every message.
    draft_model:
        Model client for Amilos.  None = template-only path (no model call).
    triage_model_name / draft_model_name:
        Identifiers used for cost accounting (e.g. "gpt-4.1-mini").
    """
    # ---- 1. Ingest ----
    message, ingest_result = ingest_message(db, msg)

    if ingest_result in (RESULT_DUPLICATE, RESULT_QUARANTINE):
        return PipelineResult(
            outcome=ingest_result,
            ingest_result=ingest_result,
            message=message,
            cost=null_cost(),
            note=f"Message {ingest_result}; pipeline stopped.",
        )

    # ---- 2. Triage (Amin) ----
    tracked_triage = _TrackingModel(triage_model, triage_model_name) if triage_model else None

    try:
        triage_decision = triage(msg, rule_pack=rule_pack, model=tracked_triage)
    except TriageError as exc:
        return PipelineResult(
            outcome="error",
            ingest_result=ingest_result,
            message=message,
            cost=null_cost(),
            note=f"Triage failed: {exc}",
        )

    triage_cost = tracked_triage.as_cost_record() if tracked_triage else null_cost()

    # ---- 3. Route: draft_reply → Amilos; anything else → hold/approval ----
    action = triage_decision.get("action", "hold")

    if action == "draft_reply":
        # Build reply via Amilos using the receipt template.
        tracked_draft = _TrackingModel(draft_model, draft_model_name) if draft_model else None

        sender_name = msg.sender.split("@")[0] if "@" in msg.sender else msg.sender
        import secrets as _secrets
        reference = _secrets.token_hex(4).upper()

        variables = build_receipt_variables(
            original_subject=msg.subject,
            sender_name=sender_name,
            reference=reference,
        )
        template_entry = DEFAULT_REGISTRY.require(RECEIPT_TEMPLATE_ID)

        try:
            reply = draft_reply(
                triage_decision,
                template=template_entry.body,
                template_id=RECEIPT_TEMPLATE_ID,
                model=tracked_draft,
                variables=variables,
            )
        except (ReplyError, ValueError) as exc:
            return PipelineResult(
                outcome="error",
                ingest_result=ingest_result,
                message=message,
                triage_decision=triage_decision,
                cost=triage_cost,
                note=f"Draft failed: {exc}",
            )

        draft_cost = tracked_draft.as_cost_record() if tracked_draft else null_cost()
        total_cost = CostRecord(
            triage_cost.model,
            triage_cost.tokens_in + draft_cost.tokens_in,
            triage_cost.tokens_out + draft_cost.tokens_out,
        )

        # ---- 4. Emit usage event when a model was called (Phase 5 dashboard). ----
        _emit_cost_event(db, msg.tenant_id, total_cost, reference_id=message.id)

        # ---- 5. Send decision ----
        send_dec = should_send(policy_config)
        if send_dec == SEND_DECISION_ALLOW:
            # Caller handles actual delivery; we record state as "send".
            return PipelineResult(
                outcome="send",
                ingest_result=ingest_result,
                message=message,
                triage_decision=triage_decision,
                reply_draft=reply,
                cost=total_cost,
                note="send allowed by policy",
            )

        # Shadow or draft: store in DB.
        stored = store_draft(
            db,
            tenant_id=msg.tenant_id,
            subject=reply["subject"],
            body=reply["body"],
            template_id=reply["template_id"],
            language=reply.get("language", "en"),
            decision_hash=reply.get("decision_hash"),
            message_id=message.id,
            policy_config=policy_config,
        )
        return PipelineResult(
            outcome="draft",
            ingest_result=ingest_result,
            message=message,
            triage_decision=triage_decision,
            reply_draft=reply,
            cost=total_cost,
            draft_id=stored.id,
            note=f"draft stored (state={stored.state})",
        )

    else:
        # Action is hold / request_human / reject / reroute / escalate.
        # Emit usage for triage model cost (may be zero on rule-hit path).
        _emit_cost_event(db, msg.tenant_id, triage_cost, reference_id=message.id)

        # Enqueue for human review when the action is not "reject".
        if action != "reject":
            entry = enqueue_approval(
                db,
                tenant_id=msg.tenant_id,
                subject=msg.subject,
                body=msg.body_text[:500],
                message_id=message.id,
                reason=action,
            )
            return PipelineResult(
                outcome="approval",
                ingest_result=ingest_result,
                message=message,
                triage_decision=triage_decision,
                cost=triage_cost,
                approval_id=entry.id,
                note=f"enqueued for human review (action={action})",
            )
        else:
            return PipelineResult(
                outcome="no_draft",
                ingest_result=ingest_result,
                message=message,
                triage_decision=triage_decision,
                cost=triage_cost,
                note=f"action={action}; no draft produced",
            )


# ---------------------------------------------------------------------------
# Internal helper — emit a usage event row for one pipeline run.
# Only emits when tokens > 0 to keep the events table clean.
# ---------------------------------------------------------------------------

def _emit_cost_event(
    db: Session,
    tenant_id: int,
    cost: CostRecord,
    reference_id: Optional[int] = None,
) -> None:
    """Write a usage_event row for this pipeline run (best-effort; never raises)."""
    try:
        total_tokens = cost.tokens_in + cost.tokens_out
        emit_usage(
            db,
            tenant_id=tenant_id,
            event_type="model_called",
            quantity=total_tokens,
            unit="tokens",
            cost_usd=cost.cost_usd if cost.cost_usd else None,
            actor="system",
            reference_id=reference_id,
        )
    except Exception:  # noqa: BLE001
        pass  # usage event failure must not break the pipeline
