"""
app/payment/live_adapter.py — Live payment adapter stub.

Commercial contract, Phase 2.

The live adapter is activated only when GATEWAY_SECRET is present in the
environment.  It reads the secret once at construction time and never logs
it, prints it, or includes it in any returned object.

The actual gateway call is a stub (raises NotImplementedError) because no
live gateway has been wired up in this contract.  The stub exists so that:

  1. The factory can instantiate the adapter without error.
  2. Future wiring only needs to replace the body of _call_gateway().
  3. The secret is already handled safely (read from env, not stored in code).

SECURITY RULES enforced by this file:
  - The secret is never passed to __repr__, __str__, or any logger.
  - No card number, no private URL, and no Stripe key appears here.
  - Tests do not reach this adapter (factory uses fake when secret absent).
"""
from __future__ import annotations

import os

from app.payment.port import ChargeResult, PaymentPort

# The environment variable name is public knowledge; the value is not.
_SECRET_ENV_VAR = "GATEWAY_SECRET"


class LivePaymentAdapter(PaymentPort):
    """
    Live gateway adapter.

    Reads GATEWAY_SECRET from the environment at construction time.
    The secret is stored only in the instance attribute ``_secret`` and is
    never included in log output.
    """

    def __init__(self) -> None:
        secret = os.environ.get(_SECRET_ENV_VAR)
        if not secret:
            raise RuntimeError(
                f"LivePaymentAdapter requires {_SECRET_ENV_VAR} in the environment."
            )
        # Store as a private attribute; never pass to logging or repr.
        self._secret: str = secret

    def __repr__(self) -> str:
        # Intentionally omit _secret from repr.
        return f"LivePaymentAdapter(configured=True)"

    def charge(
        self,
        tenant_id: int,
        amount_eur_cents: int,
        description: str,
    ) -> ChargeResult:
        """
        Charge the tenant via the live gateway.

        This method calls _call_gateway() which is a stub until a gateway
        vendor is chosen and wired up.
        """
        return self._call_gateway(tenant_id, amount_eur_cents, description)

    def _call_gateway(
        self,
        tenant_id: int,
        amount_eur_cents: int,
        description: str,
    ) -> ChargeResult:
        """
        Stub: replace this body when a live gateway is integrated.

        The secret is available as self._secret.  Do NOT log it.
        """
        raise NotImplementedError(
            "Live gateway not yet wired.  "
            "Implement _call_gateway() with the chosen payment provider."
        )
