"""
conftest.py – applied before any test module is imported.

Sets up in-memory SQLite with cross-thread access for all tests in this
package, and provides the OPENAI_API_KEY stub so the app module imports
without error.
"""
import os
import sqlalchemy
from sqlalchemy.pool import StaticPool

# Environment must be set before app.main is imported.
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")

# Patch create_engine so the in-memory SQLite engine that app.main creates
# at import time uses StaticPool and allows cross-thread access.
_orig = sqlalchemy.create_engine


def _patched(url, **kwargs):
    if str(url).startswith("sqlite"):
        kwargs.pop("pool_recycle", None)
        kwargs.pop("pool_timeout", None)
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs["poolclass"] = StaticPool
    return _orig(url, **kwargs)


sqlalchemy.create_engine = _patched  # type: ignore[assignment]
