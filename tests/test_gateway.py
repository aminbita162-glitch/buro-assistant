"""
tests/test_gateway.py — Commercial Phase 2: payment gateway boundary tests.

Covers:
  - FakePaymentAdapter: charge succeeds, reference starts with "fake-"
  - FakePaymentAdapter: multiple charges return distinct references
  - FakePaymentAdapter: ChargeResult fields (success, no error_message)
  - LivePaymentAdapter: raises RuntimeError when secret absent
  - LivePaymentAdapter: repr does not contain the secret value
  - LivePaymentAdapter: charge raises NotImplementedError (stub)
  - Factory: returns FakePaymentAdapter when GATEWAY_SECRET absent
  - Factory: returns LivePaymentAdapter when GATEWAY_SECRET present
  - Factory: fake adapter satisfies PaymentPort interface
  - ChargeResult: dataclass fields accessible

All tests use the fake adapter only.  No external service is contacted.
No Stripe key, no card number, and no private URL appears in this file.
"""
from __future__ import annotations

import os
import importlib

import pytest

from app.payment.port import ChargeResult, PaymentPort
from app.payment.fake_adapter import FakePaymentAdapter


# ---------------------------------------------------------------------------
# ChargeResult
# ---------------------------------------------------------------------------


class TestChargeResult:
    def test_success_fields(self):
        r = ChargeResult(success=True, gateway_reference="ref-1")
        assert r.success is True
        assert r.gateway_reference == "ref-1"
        assert r.error_message is None

    def test_failure_fields(self):
        r = ChargeResult(success=False, error_message="declined")
        assert r.success is False
        assert r.gateway_reference is None
        assert r.error_message == "declined"

    def test_default_fields(self):
        r = ChargeResult(success=True)
        assert r.gateway_reference is None
        assert r.error_message is None


# ---------------------------------------------------------------------------
# FakePaymentAdapter
# ---------------------------------------------------------------------------


class TestFakePaymentAdapter:
    def test_charge_succeeds(self):
        adapter = FakePaymentAdapter()
        result = adapter.charge(tenant_id=1, amount_eur_cents=6500, description="Desk plan")
        assert result.success is True

    def test_charge_returns_fake_reference(self):
        adapter = FakePaymentAdapter()
        result = adapter.charge(tenant_id=1, amount_eur_cents=6500, description="Desk plan")
        assert result.gateway_reference is not None
        assert result.gateway_reference.startswith("fake-")

    def test_charge_no_error_message(self):
        adapter = FakePaymentAdapter()
        result = adapter.charge(tenant_id=1, amount_eur_cents=6500, description="Desk plan")
        assert result.error_message is None

    def test_multiple_charges_distinct_references(self):
        adapter = FakePaymentAdapter()
        r1 = adapter.charge(tenant_id=1, amount_eur_cents=6500, description="a")
        r2 = adapter.charge(tenant_id=1, amount_eur_cents=6500, description="a")
        assert r1.gateway_reference != r2.gateway_reference

    def test_different_tenants_both_succeed(self):
        adapter = FakePaymentAdapter()
        r1 = adapter.charge(tenant_id=1, amount_eur_cents=14900, description="Mail plan")
        r2 = adapter.charge(tenant_id=2, amount_eur_cents=27000, description="Agents plan")
        assert r1.success is True
        assert r2.success is True

    def test_fake_adapter_is_payment_port(self):
        adapter = FakePaymentAdapter()
        assert isinstance(adapter, PaymentPort)


# ---------------------------------------------------------------------------
# LivePaymentAdapter — secret absent
# ---------------------------------------------------------------------------


class TestLivePaymentAdapterNoSecret:
    def test_raises_when_secret_absent(self):
        """LivePaymentAdapter must raise RuntimeError if GATEWAY_SECRET is not set."""
        # Ensure the secret is absent for this test.
        env_backup = os.environ.pop("GATEWAY_SECRET", None)
        try:
            from app.payment.live_adapter import LivePaymentAdapter
            with pytest.raises(RuntimeError, match="GATEWAY_SECRET"):
                LivePaymentAdapter()
        finally:
            if env_backup is not None:
                os.environ["GATEWAY_SECRET"] = env_backup

    def test_live_adapter_charge_raises_not_implemented(self):
        """When secret is present, charge() raises NotImplementedError (stub)."""
        os.environ["GATEWAY_SECRET"] = "test-secret-value"
        try:
            # Force reimport so the module picks up the new env value.
            import app.payment.live_adapter as _mod
            importlib.reload(_mod)
            adapter = _mod.LivePaymentAdapter()
            with pytest.raises(NotImplementedError):
                adapter.charge(tenant_id=1, amount_eur_cents=6500, description="test")
        finally:
            del os.environ["GATEWAY_SECRET"]
            importlib.reload(_mod)

    def test_live_adapter_repr_does_not_contain_secret(self):
        """repr() of LivePaymentAdapter must not leak the secret value."""
        secret_value = "super-secret-gateway-key-9999"
        os.environ["GATEWAY_SECRET"] = secret_value
        try:
            import app.payment.live_adapter as _mod
            importlib.reload(_mod)
            adapter = _mod.LivePaymentAdapter()
            assert secret_value not in repr(adapter)
            assert secret_value not in str(adapter)
        finally:
            del os.environ["GATEWAY_SECRET"]
            importlib.reload(_mod)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestPaymentFactory:
    def test_factory_returns_fake_when_no_secret(self):
        """Factory must return FakePaymentAdapter when GATEWAY_SECRET is absent."""
        env_backup = os.environ.pop("GATEWAY_SECRET", None)
        try:
            import app.payment.factory as _fac
            importlib.reload(_fac)
            adapter = _fac.get_payment_adapter()
            assert isinstance(adapter, FakePaymentAdapter)
        finally:
            if env_backup is not None:
                os.environ["GATEWAY_SECRET"] = env_backup
            importlib.reload(_fac)

    def test_factory_returns_live_when_secret_present(self):
        """Factory must return LivePaymentAdapter when GATEWAY_SECRET is set."""
        os.environ["GATEWAY_SECRET"] = "any-non-empty-value"
        try:
            import app.payment.factory as _fac
            import app.payment.live_adapter as _live
            importlib.reload(_live)
            importlib.reload(_fac)
            adapter = _fac.get_payment_adapter()
            assert isinstance(adapter, _live.LivePaymentAdapter)
        finally:
            del os.environ["GATEWAY_SECRET"]
            importlib.reload(_fac)
            importlib.reload(_live)

    def test_factory_fake_is_payment_port(self):
        env_backup = os.environ.pop("GATEWAY_SECRET", None)
        try:
            import app.payment.factory as _fac
            importlib.reload(_fac)
            adapter = _fac.get_payment_adapter()
            assert isinstance(adapter, PaymentPort)
        finally:
            if env_backup is not None:
                os.environ["GATEWAY_SECRET"] = env_backup
            importlib.reload(_fac)
