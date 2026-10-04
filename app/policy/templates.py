"""
app/policy/templates.py – template registry with variables and forbidden phrases (row 16).

A template registry maps template IDs to TemplateEntry objects.  Each entry
carries the body string with {variable} placeholders, a set of required
variable names, and a frozenset of forbidden phrases that must never appear
in the rendered output.

The built-in receipt template satisfies row 18.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set


# ---------------------------------------------------------------------------
# Forbidden phrase checking
# ---------------------------------------------------------------------------

def check_forbidden_phrases(text: str, forbidden: FrozenSet[str]) -> List[str]:
    """
    Return a list of forbidden phrases found in *text* (case-insensitive).
    Empty list means the text is clean.
    """
    lower = text.lower()
    return [phrase for phrase in forbidden if phrase.lower() in lower]


# ---------------------------------------------------------------------------
# Template entry
# ---------------------------------------------------------------------------

@dataclass
class TemplateEntry:
    """A single registered template."""

    id: str
    subject: str
    body: str
    required_variables: Set[str] = field(default_factory=set)
    forbidden_phrases: FrozenSet[str] = field(default_factory=frozenset)
    language: str = "en"
    department: Optional[str] = None   # None = applies to all departments

    def render(self, variables: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Substitute *variables* into subject and body.

        Raises
        ------
        KeyError
            If a required variable is missing from *variables*.
        ValueError
            If the rendered text contains a forbidden phrase.
        """
        variables = variables or {}
        missing = self.required_variables - variables.keys()
        if missing:
            raise KeyError(f"Template '{self.id}' missing variables: {missing}")

        rendered_subject = self.subject.format(**variables)
        rendered_body = self.body.format(**variables)

        hits = check_forbidden_phrases(rendered_body, self.forbidden_phrases)
        if hits:
            raise ValueError(
                f"Template '{self.id}' rendered body contains forbidden phrases: {hits}"
            )
        return {"subject": rendered_subject, "body": rendered_body}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TemplateRegistry:
    """In-memory template registry.  Tenant registries are isolated by tenant_id."""

    def __init__(self) -> None:
        self._templates: Dict[str, TemplateEntry] = {}

    def register(self, entry: TemplateEntry) -> None:
        """Add or replace a template."""
        self._templates[entry.id] = entry

    def get(self, template_id: str) -> Optional[TemplateEntry]:
        """Return the template or None."""
        return self._templates.get(template_id)

    def require(self, template_id: str) -> TemplateEntry:
        """Return the template or raise KeyError."""
        entry = self.get(template_id)
        if entry is None:
            raise KeyError(f"Template '{template_id}' not found in registry")
        return entry

    def list_ids(self) -> List[str]:
        return list(self._templates.keys())


# ---------------------------------------------------------------------------
# Built-in templates (row 18 – receipt template)
# ---------------------------------------------------------------------------

#: Global default registry pre-loaded with the built-in receipt template.
DEFAULT_REGISTRY: TemplateRegistry = TemplateRegistry()

RECEIPT_TEMPLATE_ID = "receipt-v1"
RECEIPT_TEMPLATE = TemplateEntry(
    id=RECEIPT_TEMPLATE_ID,
    subject="Re: {original_subject}",
    body=(
        "Dear {sender_name},\n\n"
        "We have received your message and it will be reviewed by our team.\n\n"
        "Reference: {reference}\n\n"
        "Thank you,\n"
        "The Buro Team"
    ),
    required_variables={"original_subject", "sender_name", "reference"},
    # Row 18 constraint: must state received and will be reviewed.
    forbidden_phrases=frozenset([
        # Phrases that must never appear in a receipt reply.
        "guarantee", "warranty", "liability", "no refund", "not responsible",
    ]),
    language="en",
    department=None,
)

DEFAULT_REGISTRY.register(RECEIPT_TEMPLATE)


def build_receipt_variables(
    original_subject: str,
    sender_name: str,
    reference: str,
) -> Dict[str, str]:
    """Convenience: build the variables dict for the receipt template."""
    return {
        "original_subject": original_subject,
        "sender_name": sender_name,
        "reference": reference,
    }
