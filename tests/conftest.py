"""
conftest.py – applied before any test module is imported.

Sets up in-memory SQLite with cross-thread access for all tests in this
package, and provides the OPENAI_API_KEY stub so the app module imports
without error.  Alembic is used (not create_all) to honour row 32.
"""
from __future__ import annotations

import os
import sqlalchemy
from sqlalchemy.pool import StaticPool

# Environment must be set before app.main is imported.
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")

# ---------------------------------------------------------------------------
# Patch create_engine BEFORE the app module is imported so the engine it
# builds at module level uses StaticPool and allows cross-thread access.
# ---------------------------------------------------------------------------
_orig = sqlalchemy.create_engine


def _patched(url, **kwargs):
    if str(url).startswith("sqlite"):
        kwargs.pop("pool_recycle", None)
        kwargs.pop("pool_timeout", None)
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs["poolclass"] = StaticPool
    return _orig(url, **kwargs)


sqlalchemy.create_engine = _patched  # type: ignore[assignment]

# Ensure the ingest models are registered on Base before any test creates tables.
try:
    import app.ingest.models  # noqa: F401
except Exception:
    pass

# Ensure agent modules are importable (they do not define ORM models but
# import app.main which may not yet be imported in some test orderings).
try:
    import app.agents.amin      # noqa: F401
    import app.agents.amilos    # noqa: F401
    import app.agents.leila     # noqa: F401
except Exception:
    pass

# Register policy ORM models (approval_queue, audit_log, drafts) on Base.
try:
    import app.policy.approval  # noqa: F401
    import app.policy.audit     # noqa: F401
    import app.policy.shadow    # noqa: F401
except Exception:
    pass

# Register worker ORM models (traces, work_queue, dead_letter, quotas) on Base.
try:
    import app.workers.trace   # noqa: F401
    import app.workers.quota   # noqa: F401
    import app.workers.queue   # noqa: F401
    import app.workers.dlq     # noqa: F401
except Exception:
    pass

# Register commercial ORM models (usage_events, api_keys, webhook_subscriptions) on Base.
try:
    import app.domain.usage     # noqa: F401
    import app.domain.apikeys   # noqa: F401
    import app.domain.webhooks  # noqa: F401
except Exception:
    pass

# Register Phase 2 ORM models (delivery_log) on Base.
try:
    import app.workers.webhook_delivery  # noqa: F401
except Exception:
    pass

# Register Phase 7 ORM models (buyer features).
try:
    import app.domain.buyer_features  # noqa: F401
except Exception:
    pass

# Register commercial Phase 1 ORM models (tenant_subscriptions).
try:
    import app.domain.subscription  # noqa: F401
except Exception:
    pass


# ---------------------------------------------------------------------------
# Also patch alembic's engine_from_config so migration tests work the
# same way (StaticPool, no pool_timeout).
# ---------------------------------------------------------------------------
import sqlalchemy.engine  # noqa: E402

_orig_efc = None

try:
    from sqlalchemy import engine_from_config as _orig_efc_fn
    import alembic.runtime.migration  # noqa: F401 – ensure alembic is importable

    def _patched_efc(configuration, prefix="sqlalchemy.", **kwargs):
        url = configuration.get(f"{prefix}url", "")
        if str(url).startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["poolclass"] = StaticPool
        return _orig_efc_fn(configuration, prefix=prefix, **kwargs)

    import alembic.env as _ae  # noqa: F401
    sqlalchemy.engine_from_config = _patched_efc  # type: ignore[assignment]
except Exception:
    pass
