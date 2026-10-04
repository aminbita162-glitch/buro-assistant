"""
tests/test_air_gap.py — Phase 7 air-gap package

Verifies that docker-compose.yml:
  - exists at the repo root
  - names no cloud model secret (OPENAI_API_KEY)
  - names no gateway secret (GATEWAY_SECRET)
  - sets LOCAL_MODEL to disable the cloud model client
  - references the operator-hosted README note in the compose header
  - does not embed any literal API key value pattern (sk-... style)
"""

from __future__ import annotations

import re
import pathlib

import pytest

COMPOSE_PATH = pathlib.Path(__file__).parent.parent / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose_text() -> str:
    assert COMPOSE_PATH.exists(), "docker-compose.yml not found at repo root"
    return COMPOSE_PATH.read_text(encoding="utf-8")


class TestComposeFileExists:
    def test_compose_file_present(self) -> None:
        assert COMPOSE_PATH.exists(), "docker-compose.yml must exist at repo root"

    def test_compose_file_not_empty(self, compose_text: str) -> None:
        assert len(compose_text.strip()) > 0, "docker-compose.yml must not be empty"


class TestNoSecretNamed:
    """The compose file must not assign a value to secret variables."""

    def test_openai_api_key_not_assigned(self, compose_text: str) -> None:
        """OPENAI_API_KEY must not appear with an assigned value (key: value or KEY=value)."""
        # Pattern matches  OPENAI_API_KEY: sk-...  or  OPENAI_API_KEY=sk-...
        # Bare mention in a comment is allowed; value assignment is not.
        matches = re.findall(
            r'OPENAI_API_KEY\s*[:=]\s*\S+',
            compose_text,
        )
        # Strip matches that are commented out
        uncommented = [m for m in matches if not _is_comment_line(compose_text, m)]
        assert not uncommented, (
            f"docker-compose.yml must not assign OPENAI_API_KEY; found: {uncommented}"
        )

    def test_gateway_secret_not_assigned(self, compose_text: str) -> None:
        """GATEWAY_SECRET must not appear with an assigned value."""
        matches = re.findall(
            r'GATEWAY_SECRET\s*[:=]\s*\S+',
            compose_text,
        )
        uncommented = [m for m in matches if not _is_comment_line(compose_text, m)]
        assert not uncommented, (
            f"docker-compose.yml must not assign GATEWAY_SECRET; found: {uncommented}"
        )

    def test_no_sk_style_api_key_value(self, compose_text: str) -> None:
        """No literal OpenAI key pattern (sk-...) may appear anywhere in the file."""
        # OpenAI keys start with sk- followed by alphanumeric characters
        matches = re.findall(r'\bsk-[A-Za-z0-9_\-]{10,}', compose_text)
        assert not matches, (
            f"docker-compose.yml must not contain a literal API key value; found: {matches}"
        )

    def test_imap_password_not_assigned(self, compose_text: str) -> None:
        """IMAP_PASSWORD must not appear with a non-empty assigned value."""
        matches = re.findall(r'IMAP_PASSWORD\s*[:=]\s*\S+', compose_text)
        uncommented = [m for m in matches if not _is_comment_line(compose_text, m)]
        assert not uncommented, (
            f"docker-compose.yml must not assign IMAP_PASSWORD; found: {uncommented}"
        )


class TestLocalModelEnabled:
    """LOCAL_MODEL must be set so the cloud model client is not called."""

    def test_local_model_set(self, compose_text: str) -> None:
        assert re.search(r'LOCAL_MODEL\s*[:=]\s*["\']?1["\']?', compose_text), (
            "docker-compose.yml must set LOCAL_MODEL: \"1\" to disable the cloud model"
        )


class TestOperatorHostedStatement:
    """The compose file must carry the operator-hosted note."""

    def test_operator_hosted_comment(self, compose_text: str) -> None:
        assert "Operator-hosted" in compose_text or "operator-hosted" in compose_text, (
            "docker-compose.yml must state that this is operator-hosted"
        )


class TestComposeStructure:
    """Basic structural checks — services, db, app, volumes."""

    def test_has_services_block(self, compose_text: str) -> None:
        assert "services:" in compose_text

    def test_has_db_service(self, compose_text: str) -> None:
        assert "db:" in compose_text

    def test_has_app_service(self, compose_text: str) -> None:
        assert "app:" in compose_text

    def test_has_volumes_block(self, compose_text: str) -> None:
        assert "volumes:" in compose_text

    def test_no_host_network_mode(self, compose_text: str) -> None:
        """The compose file must not use host network mode (security boundary)."""
        assert "network_mode: host" not in compose_text

    def test_postgres_image_used(self, compose_text: str) -> None:
        assert "postgres" in compose_text.lower()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_comment_line(full_text: str, match_str: str) -> bool:
    """Return True if every line containing match_str is a YAML comment line."""
    for line in full_text.splitlines():
        if match_str.strip() in line or match_str in line:
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                return False
    return True
