"""tests/golden/__init__.py – loader for the 50 golden messages (row 46)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

_PATH = os.path.join(os.path.dirname(__file__), "golden_messages.json")


def load_golden_messages() -> List[Dict[str, Any]]:
    """Return the list of 50 synthetic golden messages."""
    with open(_PATH, encoding="utf-8") as f:
        return json.load(f)
