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
