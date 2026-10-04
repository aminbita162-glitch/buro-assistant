"""
app/payment/port.py — Payment port (abstract interface).

Commercial contract, Phase 2.

Defines the PaymentPort protocol and the ChargeResult dataclass that every
adapter must return.  No gateway secret, no card number, and no private URL
appears anywhere in this file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ChargeResult:
    """
    Outcome of a charge attempt.

    Attributes
    ----------
    success:
        True when the charge was accepted by the gateway.
    gateway_reference:
        An opaque reference string returned by the gateway (e.g. a charge ID).
        None when success is False or the adapter is in fake mode.
    error_message:
        Human-readable error description.  None when success is True.
    """
    success: bool
    gateway_reference: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract port
# ---------------------------------------------------------------------------

class PaymentPort(ABC):
    """Abstract payment port.  Both adapters implement this interface."""

    @abstractmethod
    def charge(
        self,
        tenant_id: int,
        amount_eur_cents: int,
        description: str,
    ) -> ChargeResult:
        """
        Attempt to charge *amount_eur_cents* euro-cents for *tenant_id*.

        Parameters
        ----------
        tenant_id:
            The internal tenant identifier.  Passed to the gateway for
            correlation only; no card data is stored here.
        amount_eur_cents:
            Amount in euro-cents (e.g. 6500 for 65.00 EUR).
        description:
            Short human-readable description of the charge
            (e.g. "Desk plan – monthly").

        Returns
        -------
        ChargeResult
        """
