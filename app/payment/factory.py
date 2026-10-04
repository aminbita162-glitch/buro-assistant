"""
app/payment/factory.py — Payment adapter factory.

Commercial contract, Phase 2.

Returns the FakePaymentAdapter when GATEWAY_SECRET is absent or empty.
Returns the LivePaymentAdapter when GATEWAY_SECRET is present.

The factory is the only place that inspects the environment variable name.
All callers obtain an adapter through get_payment_adapter() and never
read the secret directly.
"""
from __future__ import annotations

import os

from app.payment.port import PaymentPort

_SECRET_ENV_VAR = "GATEWAY_SECRET"


def get_payment_adapter() -> PaymentPort:
    """
    Return the appropriate payment adapter for the current environment.

    - If ``GATEWAY_SECRET`` is set and non-empty: returns a
      :class:`~app.payment.live_adapter.LivePaymentAdapter`.
    - Otherwise: returns a
      :class:`~app.payment.fake_adapter.FakePaymentAdapter`.

    Tests always run without ``GATEWAY_SECRET`` so they always receive the
    fake adapter.
    """
    if os.environ.get(_SECRET_ENV_VAR):
        from app.payment.live_adapter import LivePaymentAdapter
        return LivePaymentAdapter()

    from app.payment.fake_adapter import FakePaymentAdapter
    return FakePaymentAdapter()
