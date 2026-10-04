"""
tests/test_differentiation.py – Follow-up Phase 8 (Section C, fifteen finish items).

Each test class corresponds to one Section C item.  Tests verify that the
required artifact, content, or behaviour exists.  No database or external
service is contacted.
"""
from __future__ import annotations

import os
import re

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_html() -> str:
    return _read("index.html")


# ---------------------------------------------------------------------------
# C1 – README opening a buyer can scan in one minute
# ---------------------------------------------------------------------------

class TestC1:
    """README must have a short at-a-glance section near the top."""

    def test_readme_has_at_a_glance(self):
        src = _read("README.md")
        assert "At a glance" in src or "at a glance" in src.lower()

    def test_readme_opening_has_five_key_points(self):
        src = _read("README.md")
        # Five numbered key-point lines should appear.
        assert "Rule before model" in src
        assert "Redaction before model" in src
        assert "Shadow mode" in src
        assert "Append-only audit log" in src
        assert "Self-hosted" in src

    def test_readme_has_author_line(self):
        src = _read("README.md")
        assert "Amin Azimi" in src
        assert "Azimi Innovation Lab" in src

    def test_readme_has_license_reference(self):
        src = _read("README.md")
        assert "LICENSE" in src or "License" in src

    def test_readme_has_version_line(self):
        src = _read("README.md")
        assert "version" in src.lower() or "1.0.0" in src


# ---------------------------------------------------------------------------
# C2 – Text architecture diagram
# ---------------------------------------------------------------------------

class TestC2:
    """docs/diagrams/architecture.txt must exist and cover all layers."""

    def test_arch_diagram_exists(self):
        assert os.path.isfile("docs/diagrams/architecture.txt")

    def test_arch_diagram_covers_ingest(self):
        src = _read("docs/diagrams/architecture.txt")
        assert "ingest" in src.lower() or "INGEST" in src

    def test_arch_diagram_covers_amin(self):
        src = _read("docs/diagrams/architecture.txt")
        assert "Amin" in src

    def test_arch_diagram_covers_amilos(self):
        src = _read("docs/diagrams/architecture.txt")
        assert "Amilos" in src

    def test_arch_diagram_covers_leila(self):
        src = _read("docs/diagrams/architecture.txt")
        assert "Leila" in src

    def test_arch_diagram_covers_policy(self):
        src = _read("docs/diagrams/architecture.txt")
        assert "policy" in src.lower() or "POLICY" in src

    def test_arch_diagram_covers_workers(self):
        src = _read("docs/diagrams/architecture.txt")
        assert "worker" in src.lower() or "WORKER" in src

    def test_arch_diagram_covers_operator_desk(self):
        src = _read("docs/diagrams/architecture.txt")
        assert "desk" in src.lower() or "DESK" in src

    def test_arch_diagram_linked_from_readme(self):
        src = _read("README.md")
        assert "docs/diagrams/architecture.txt" in src


# ---------------------------------------------------------------------------
# C3 – Designed-versus-verified table
# ---------------------------------------------------------------------------

class TestC3:
    """README must contain a designed-versus-verified table."""

    def test_designed_vs_verified_heading(self):
        src = _read("README.md")
        assert "Designed versus verified" in src or "designed versus verified" in src.lower()

    def test_table_has_designed_column(self):
        src = _read("README.md")
        assert "Designed for" in src or "designed for" in src.lower()

    def test_table_has_verified_column(self):
        src = _read("README.md")
        assert "Verified" in src

    def test_table_cites_test_files(self):
        src = _read("README.md")
        assert "tests/test_agents.py" in src
        assert "tests/test_policy.py" in src

    def test_table_marks_unverified_explicitly(self):
        src = _read("README.md")
        assert "designed for" in src.lower() and "not yet" in src.lower()


# ---------------------------------------------------------------------------
# C4 – External-links section
# ---------------------------------------------------------------------------

class TestC4:
    """README must have an external links section with only real public links."""

    def test_external_links_heading(self):
        src = _read("README.md")
        assert "External links" in src or "external links" in src.lower()

    def test_external_links_has_fastapi(self):
        src = _read("README.md")
        assert "FastAPI" in src

    def test_external_links_has_alembic(self):
        src = _read("README.md")
        assert "Alembic" in src

    def test_external_links_no_private_disclaimer(self):
        src = _read("README.md")
        assert "No private links" in src or "no private links" in src.lower()


# ---------------------------------------------------------------------------
# C5 – File map with one responsibility per package
# ---------------------------------------------------------------------------

class TestC5:
    """README file map must list all core packages with one responsibility each."""

    def test_file_map_heading(self):
        src = _read("README.md")
        assert "File map" in src or "file map" in src.lower()

    def test_file_map_has_agents(self):
        src = _read("README.md")
        assert "app/agents/" in src

    def test_file_map_has_ingest(self):
        src = _read("README.md")
        assert "app/ingest/" in src

    def test_file_map_has_policy(self):
        src = _read("README.md")
        assert "app/policy/" in src

    def test_file_map_has_workers(self):
        src = _read("README.md")
        assert "app/workers/" in src

    def test_file_map_has_web(self):
        src = _read("README.md")
        assert "app/web/" in src

    def test_file_map_has_domain(self):
        src = _read("README.md")
        assert "app/domain/" in src

    def test_file_map_has_migrations(self):
        src = _read("README.md")
        assert "migrations/" in src

    def test_file_map_has_docs(self):
        src = _read("README.md")
        assert "`docs/`" in src or "| `docs/`" in src


# ---------------------------------------------------------------------------
# C6 – Release timeline in docs/releases
# ---------------------------------------------------------------------------

class TestC6:
    """docs/releases/timeline.md must exist and list all follow-up phases."""

    def test_timeline_exists(self):
        assert os.path.isfile("docs/releases/timeline.md")

    def test_timeline_has_initial_release(self):
        src = _read("docs/releases/timeline.md")
        assert "1.0.0" in src

    def test_timeline_has_all_followup_phases(self):
        src = _read("docs/releases/timeline.md")
        for n in range(1, 9):
            assert f"Follow-up {n}" in src or f"follow-up {n}" in src.lower(), \
                f"Follow-up phase {n} missing from timeline"

    def test_timeline_mentions_remaining_phases(self):
        src = _read("docs/releases/timeline.md")
        assert "planned" in src.lower()

    def test_timeline_mentions_commercial_gap(self):
        src = _read("docs/releases/timeline.md")
        assert "commercial" in src.lower() or "sale" in src.lower()


# ---------------------------------------------------------------------------
# C7 – Comparison page against public desks
# ---------------------------------------------------------------------------

class TestC7:
    """docs/COMPARISON.md must exist, cite real products, and not invent benchmarks."""

    def test_comparison_exists(self):
        assert os.path.isfile("docs/COMPARISON.md")

    def test_comparison_cites_helpscout(self):
        src = _read("docs/COMPARISON.md")
        assert "HelpScout" in src or "helpscout" in src.lower()

    def test_comparison_cites_front(self):
        src = _read("docs/COMPARISON.md")
        assert "Front" in src

    def test_comparison_cites_freshdesk(self):
        src = _read("docs/COMPARISON.md")
        assert "Freshdesk" in src

    def test_comparison_has_honesty_note(self):
        src = _read("docs/COMPARISON.md")
        # Must say no benchmarks have been run.
        assert "no benchmarks" in src.lower() or "no throughput" in src.lower()

    def test_comparison_does_not_invent_benchmark_numbers(self):
        src = _read("docs/COMPARISON.md")
        # No numeric throughput comparisons like "3x faster" or "200 msg/s"
        assert not re.search(r"\d+\s*x\s+(faster|better|more)", src, re.IGNORECASE)
        assert not re.search(r"\d+\s*(msg|message|req|request)s?/s", src, re.IGNORECASE)

    def test_comparison_states_no_commercial_sale(self):
        src = _read("docs/COMPARISON.md")
        assert "not done" in src.lower() or "not offered" in src.lower()

    def test_comparison_linked_from_readme(self):
        src = _read("README.md")
        assert "COMPARISON.md" in src


# ---------------------------------------------------------------------------
# C8 – Empty states and error copy on the desk
# ---------------------------------------------------------------------------

class TestC8:
    """index.html must have non-empty empty-state messages for each section."""

    def test_inbound_empty_state(self):
        src = _read_html()
        assert "No messages found" in src

    def test_decisions_empty_state(self):
        src = _read_html()
        assert "No decisions recorded" in src

    def test_drafts_empty_state(self):
        src = _read_html()
        assert "No drafts found" in src

    def test_approval_empty_state(self):
        src = _read_html()
        assert "No pending entries" in src

    def test_audit_empty_state(self):
        src = _read_html()
        assert "No audit entries found" in src

    def test_cost_empty_state(self):
        src = _read_html()
        assert "No usage events recorded" in src

    def test_login_error_copy(self):
        src = _read_html()
        assert "Email and password are required" in src

    def test_network_error_copy(self):
        src = _read_html()
        assert "Network error" in src


# ---------------------------------------------------------------------------
# C9 – Keyboard path for approve and reject
# ---------------------------------------------------------------------------

class TestC9:
    """index.html must have keyboard shortcuts for approve and reject."""

    def test_keyboard_hint_text_present(self):
        src = _read_html()
        assert "kbd-hint" in src

    def test_alt_a_approve_documented(self):
        src = _read_html()
        # The hint text must reference approve and a key.
        assert "Alt" in src
        assert "approve" in src.lower()

    def test_alt_r_reject_documented(self):
        src = _read_html()
        assert "reject" in src.lower()

    def test_keyboard_event_listener_present(self):
        src = _read_html()
        assert "keydown" in src
        assert "altKey" in src or "Alt" in src

    def test_arrow_navigation_present(self):
        src = _read_html()
        assert "ArrowDown" in src or "arrowdown" in src.lower()
        assert "ArrowUp" in src or "arrowup" in src.lower()

    def test_approve_via_keyboard(self):
        src = _read_html()
        assert "btn-approve" in src and "btn.click" in src


# ---------------------------------------------------------------------------
# C10 – Consistent names Amin, Amilos, Leila
# ---------------------------------------------------------------------------

class TestC10:
    """Agent names must be consistent across all source files."""

    def test_amin_in_amin_module(self):
        src = _read("app/agents/amin.py")
        assert "Amin" in src

    def test_amilos_in_amilos_module(self):
        src = _read("app/agents/amilos.py")
        assert "Amilos" in src

    def test_leila_in_leila_module(self):
        src = _read("app/agents/leila.py")
        assert "Leila" in src

    def test_amin_in_pipeline(self):
        src = _read("app/pipeline.py")
        assert "Amin" in src

    def test_amilos_in_pipeline(self):
        src = _read("app/pipeline.py")
        assert "Amilos" in src

    def test_leila_in_pipeline(self):
        src = _read("app/pipeline.py")
        assert "Leila" in src

    def test_amin_in_readme(self):
        src = _read("README.md")
        assert "Amin" in src

    def test_amilos_in_readme(self):
        src = _read("README.md")
        assert "Amilos" in src

    def test_leila_in_readme(self):
        src = _read("README.md")
        assert "Leila" in src

    def test_no_agent_name_misspellings(self):
        # Common misspellings to guard against.
        for path in ("app/agents/amin.py", "app/agents/amilos.py",
                     "app/agents/leila.py", "app/pipeline.py", "README.md"):
            src = _read(path)
            assert "Aminos" not in src, f"Misspelling 'Aminos' in {path}"
            assert "Laila" not in src,  f"Misspelling 'Laila' in {path}"
            assert "Amim" not in src,   f"Misspelling 'Amim' in {path}"


# ---------------------------------------------------------------------------
# C11 – Test-house index
# ---------------------------------------------------------------------------

class TestC11:
    """docs/TEST_HOUSE.md must exist and contain a run index."""

    def test_test_house_exists(self):
        assert os.path.isfile("docs/TEST_HOUSE.md")

    def test_test_house_has_run_index(self):
        src = _read("docs/TEST_HOUSE.md")
        assert "Run" in src or "run" in src.lower()

    def test_test_house_records_phase_7_run(self):
        src = _read("docs/TEST_HOUSE.md")
        assert "504" in src or "Phase 7" in src or "follow-up 7" in src.lower()

    def test_test_house_has_honesty_note(self):
        src = _read("docs/TEST_HOUSE.md")
        assert "designed for" in src.lower() or "designed-for" in src.lower()

    def test_test_house_states_no_external_test_run(self):
        src = _read("docs/TEST_HOUSE.md")
        assert ("not been performed" in src or "not yet" in src.lower()
                or "not run" in src.lower() or "no external test" in src.lower())

    def test_test_house_has_test_file_index(self):
        src = _read("docs/TEST_HOUSE.md")
        assert "tests/test_security.py" in src
        assert "tests/test_agents.py" in src


# ---------------------------------------------------------------------------
# C12 – Install video script (not a video file)
# ---------------------------------------------------------------------------

class TestC12:
    """docs/INSTALL_VIDEO_SCRIPT.md must be a text script, not a binary."""

    def test_video_script_exists(self):
        assert os.path.isfile("docs/INSTALL_VIDEO_SCRIPT.md")

    def test_video_script_is_text(self):
        src = _read("docs/INSTALL_VIDEO_SCRIPT.md")
        assert len(src) > 100

    def test_video_script_is_markdown_not_binary(self):
        # File must be readable as UTF-8 text.
        src = _read("docs/INSTALL_VIDEO_SCRIPT.md")
        assert "# " in src  # Has at least one markdown heading.

    def test_video_script_covers_install_steps(self):
        src = _read("docs/INSTALL_VIDEO_SCRIPT.md")
        assert "git clone" in src or "clone" in src.lower()
        assert "pip install" in src or "requirements" in src.lower()
        assert "alembic" in src.lower() or "migration" in src.lower()

    def test_video_script_covers_health_check(self):
        src = _read("docs/INSTALL_VIDEO_SCRIPT.md")
        assert "/health" in src

    def test_video_script_has_no_real_secrets(self):
        src = _read("docs/INSTALL_VIDEO_SCRIPT.md")
        # Must not contain a real OpenAI key pattern.
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", src)

    def test_video_script_states_no_hosted_service(self):
        src = _read("docs/INSTALL_VIDEO_SCRIPT.md")
        assert "no hosted service" in src.lower() or "self-hosted" in src.lower()


# ---------------------------------------------------------------------------
# C13 – Threat model linked from README
# ---------------------------------------------------------------------------

class TestC13:
    """README must link to docs/THREAT_MODEL.md."""

    def test_threat_model_file_exists(self):
        assert os.path.isfile("docs/THREAT_MODEL.md")

    def test_threat_model_linked_from_readme(self):
        src = _read("README.md")
        assert "THREAT_MODEL.md" in src

    def test_threat_model_link_in_author_section(self):
        src = _read("README.md")
        # The link should appear near the author section or a dedicated line.
        assert "Threat model" in src or "threat model" in src.lower()

    def test_threat_model_has_stride_content(self):
        src = _read("docs/THREAT_MODEL.md")
        assert "STRIDE" in src or "Spoofing" in src


# ---------------------------------------------------------------------------
# C14 – No secret in any sample
# ---------------------------------------------------------------------------

class TestC14:
    """No sample file should contain a real secret pattern."""

    def test_env_example_has_no_real_openai_key(self):
        src = _read(".env.example")
        # Real OpenAI keys look like sk-<48+ alphanumeric chars>.
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", src)

    def test_env_example_has_no_real_password(self):
        src = _read(".env.example")
        # DATABASE_URL must use the placeholder pattern, not a real password.
        assert "password" in src.lower()  # Placeholder word is fine.
        assert not re.search(r"postgresql://\w+:[^p][^a][^s][^s]", src)

    def test_install_video_script_has_no_real_key(self):
        src = _read("docs/INSTALL_VIDEO_SCRIPT.md")
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", src)

    def test_readme_has_no_real_key(self):
        src = _read("README.md")
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", src)

    def test_install_doc_has_no_real_key(self):
        src = _read("docs/INSTALL.md")
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", src)


# ---------------------------------------------------------------------------
# C15 – Final wording that sale and hosting are not done
# ---------------------------------------------------------------------------

class TestC15:
    """README must explicitly state that sale and hosting are not done."""

    def test_no_hosted_service_claim(self):
        src = _read("README.md")
        assert "No hosted service" in src or "no hosted service" in src.lower()

    def test_no_commercial_sale_claim(self):
        src = _read("README.md")
        assert "No commercial sale" in src or "no commercial sale" in src.lower()

    def test_what_is_not_claimed_section(self):
        src = _read("README.md")
        assert "What is not claimed" in src or "what is not claimed" in src.lower()

    def test_comparison_also_states_no_sale(self):
        src = _read("docs/COMPARISON.md")
        assert "not done" in src.lower() or "out of scope" in src.lower()

    def test_threat_model_also_states_no_hosted(self):
        src = _read("docs/THREAT_MODEL.md")
        assert "not" in src.lower() and "hosted" in src.lower()
