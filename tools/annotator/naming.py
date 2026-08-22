"""Track-name sanitization and fallback ID generation.

Pure functions, no I/O — shared between the desktop annotator's recorder
(``recorder.py``) and the mobile companion server so both produce identical
track ids for the same input.
"""

from __future__ import annotations

from datetime import datetime

# sanitize_track_name lives in musicality.dataformats.track_io — it's also
# needed there to build annotation paths, and musicality/ must not import
# from tools/. Re-exported here so every existing `from tools.annotator
# .naming import sanitize_track_name` call site keeps working unchanged.
from musicality.dataformats.track_io import sanitize_track_name

__all__ = ["sanitize_track_name", "generate_track_id"]


def generate_track_id() -> str:
    """Timestamp-based fallback id, e.g. ``field_20260715_143201_482913``.

    For quick captures where typing a name is friction (e.g. on a phone).
    Includes microseconds: two requests landing in the same second (e.g. a
    sync retry racing the original) would otherwise get the identical id and
    silently overwrite each other's file.
    """
    return datetime.now().strftime("field_%Y%m%d_%H%M%S_%f")
