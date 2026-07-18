"""Track-name sanitization and fallback ID generation.

Pure functions, no I/O — shared between the desktop annotator's recorder
(``recorder.py``) and the mobile companion server so both produce identical
track ids for the same input.
"""

from __future__ import annotations

import re
from datetime import datetime

_SANITIZE_RE = re.compile(r"[^\w\-]")


def sanitize_track_name(name: str) -> str:
    """Turn free-form user input into a filesystem-safe track id.

    Falls back to ``"recording"`` if *name* is empty or whitespace-only.
    """
    return _SANITIZE_RE.sub("_", name.strip()) or "recording"


def generate_track_id() -> str:
    """Timestamp-based fallback id, e.g. ``field_20260715_143201``.

    For quick captures where typing a name is friction (e.g. on a phone).
    """
    return datetime.now().strftime("field_%Y%m%d_%H%M%S")
