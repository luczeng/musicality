"""Corrections for bar-position (phase) labeling errors in beat annotations."""

from __future__ import annotations

import numpy as np


def shift_beat_positions(
    beat_positions: np.ndarray, offset: int, n_beats: int
) -> np.ndarray:
    """Rotate 1-indexed bar-position labels by *offset* beats, cyclically.

    Fixes a track whose whole cyclic 1..n_beats numbering is off by a fixed
    beat count relative to the true downbeats — beat *times* are untouched,
    only which label each beat carries. offset=+1 turns 1->2, 2->3, ...,
    n_beats->1; offset=-1 is the inverse; shifts compose (two +1 calls equal
    one +2 call).
    """

    return ((beat_positions - 1 + offset) % n_beats) + 1
