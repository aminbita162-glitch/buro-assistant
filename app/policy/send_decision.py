"""
app/policy/send_decision.py – send-only-on-allow policy (row 17).

Auto-reply is OFF by default.  The tenant must explicitly enable it by
setting ``auto_reply_enabled = True`` in its policy config.  When shadow
mode is active, the draft is stored but not sent (row 48, handled in
app/policy/shadow.py).

Public function: ``should_send(policy_config) -> bool``
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEND_DECISION_ALLOW = "allow"
SEND_DECISION_DRAFT = "draft"
SEND_DECISION_SHADOW = "shadow"


# ---------------------------------------------------------------------------
# Public entry point (row 17)
# ---------------------------------------------------------------------------

def should_send(
    policy_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return the send decision for the tenant.

    Return values
    -------------
    ``"allow"``
        Auto-reply is on; send the draft.
    ``"shadow"``
        Shadow mode is on; store the draft, do not send.
    ``"draft"``
        Default.  Store the draft for human review; do not send.

    Row 17: auto-reply is off unless ``auto_reply_enabled`` is True.
    Row 48: shadow mode stores draft and does not send.
    """
    if not policy_config:
        return SEND_DECISION_DRAFT

    if policy_config.get("shadow_mode", False):
        return SEND_DECISION_SHADOW

    if policy_config.get("auto_reply_enabled", False):
        return SEND_DECISION_ALLOW

    return SEND_DECISION_DRAFT


def is_send_allowed(policy_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True only when the tenant has explicitly enabled auto-reply."""
    return should_send(policy_config) == SEND_DECISION_ALLOW
