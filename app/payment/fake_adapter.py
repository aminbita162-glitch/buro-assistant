"""
app/payment/fake_adapter.py — Fake payment adapter.

Commercial contract, Phase 2.

Used when GATEWAY_SECRET is absent from the environment.
Every charge succeeds and returns a deterministic fake reference.
No network call is made.  Safe to use in tests and in local development.
"""
from __future__ import annotations

import secrets

from app.payment.port import ChargeResult, PaymentPort


class FakePaymentAdapter(PaymentPort):
    """
    In-process fake adapter.

    Charges always succeed.  The gateway reference is a short random hex
    string prefixed with ``fake-`` so callers can recognise it in logs.
    """

    def charge(
        self,
        tenant_id: int,
        amount_eur_cents: int,
        description: str,
    ) -> ChargeResult:
        ref = "fake-" + secrets.token_hex(8)
        return ChargeResult(success=True, gateway_reference=ref)
