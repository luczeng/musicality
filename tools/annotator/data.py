"""Track data model, beat annotation helpers, and persistence.

Pure functions (``beats_per_bar``, ``active_beat_position``, ``add_beat``,
``remove_beat``) are kept side-effect-free so they can be tested without
mirdata or a filesystem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mirdata
import numpy as np

import musicality.dataformats as dataformats

DATA_DIR = Path(__file__).parent.parent.parent / dataformats.load().data_dir
ANNOTATIONS_DIR = Path(__file__).parent.parent.parent / "annotations"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrackData:
    """All annotation data for a single track."""

    dataset_name: str
    track_id: str
    audio_path: str
    tempo: float | None
    beat_times: np.ndarray       # seconds, sorted ascending
    beat_positions: np.ndarray | None  # 1-indexed bar positions, or None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def tempo_from_beats(beat_times: np.ndarray) -> float | None:
    """Estimate tempo from beat timestamps using the median inter-beat interval."""
    if len(beat_times) < 2:
        return None
    intervals = np.diff(beat_times)
    median_interval = np.median(intervals)
    return 60.0 / median_interval


def beats_per_bar(beat_positions: np.ndarray | None) -> int:
    """Return the number of beats per bar inferred from beat positions.

    Falls back to 4 when positions are unavailable or empty.
    """
    if beat_positions is None or len(beat_positions) == 0:
        return 4
    return int(max(beat_positions))


def active_beat_position(
    beat_times: np.ndarray,
    beat_positions: np.ndarray | None,
    t: float,
) -> int | None:
    """Return the 1-indexed bar position of the most recent beat at time *t*.

    Returns ``None`` if *t* is before the first beat.
    """
    if len(beat_times) == 0:
        return None
    idx = int(np.searchsorted(beat_times, t, side="right")) - 1
    if idx < 0:
        return None
    if beat_positions is not None:
        return int(beat_positions[idx])
    n = beats_per_bar(beat_positions)
    return (idx % n) + 1


def add_beat(track: TrackData, time: float) -> TrackData:
    """Return a new :class:`TrackData` with a beat inserted at *time*.

    Beat positions are recomputed sequentially (1-based) after insertion,
    preserving the original beats-per-bar.
    """
    times = np.sort(np.append(track.beat_times, time))
    n = beats_per_bar(track.beat_positions)
    positions = np.array([(i % n) + 1 for i in range(len(times))], dtype=int)
    return TrackData(
        dataset_name=track.dataset_name,
        track_id=track.track_id,
        audio_path=track.audio_path,
        tempo=track.tempo,
        beat_times=times,
        beat_positions=positions,
    )


def remove_beat(track: TrackData, time: float, tolerance: float = 0.1) -> TrackData:
    """Return a new :class:`TrackData` with the beat nearest to *time* removed.

    Does nothing if no beat falls within *tolerance* seconds of *time*.
    """
    if len(track.beat_times) == 0:
        return track
    idx = int(np.argmin(np.abs(track.beat_times - time)))
    if abs(track.beat_times[idx] - time) > tolerance:
        return track
    times = np.delete(track.beat_times, idx)
    n = beats_per_bar(track.beat_positions)
    positions = np.array([(i % n) + 1 for i in range(len(times))], dtype=int)
    return TrackData(
        dataset_name=track.dataset_name,
        track_id=track.track_id,
        audio_path=track.audio_path,
        tempo=track.tempo,
        beat_times=times,
        beat_positions=positions,
    )


def annotation_path(track: TrackData) -> Path:
    """Return the canonical save path for a track's annotations."""
    return ANNOTATIONS_DIR / track.dataset_name / f"{track.track_id}.json"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_track(dataset_name: str, track_id: str) -> TrackData:
    """Load track metadata and beat annotations from a mirdata dataset."""
    ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
    track = ds.track(track_id)

    beat_times = np.array([])
    beat_positions = None
    if track.beats is not None:
        beat_times = track.beats.times
        beat_positions = getattr(track.beats, "positions", None)

    return TrackData(
        dataset_name=dataset_name,
        track_id=track_id,
        audio_path=track.audio_path,
        tempo=track.tempo,
        beat_times=beat_times,
        beat_positions=beat_positions,
    )


def save_annotations(track: TrackData, path: Path) -> None:
    """Persist beat annotations to a JSON file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": track.dataset_name,
        "track_id": track.track_id,
        "tempo": track.tempo,
        "beat_times": track.beat_times.tolist(),
        "beat_positions": (
            track.beat_positions.tolist() if track.beat_positions is not None else None
        ),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_annotations(path: Path) -> dict:
    """Load previously saved annotations from *path*.

    Returns a dict with keys ``beat_times`` (ndarray), ``beat_positions``
    (ndarray or None), ``tempo``, ``dataset_name``, ``track_id``.
    """
    payload = json.loads(Path(path).read_text())
    payload["beat_times"] = np.array(payload["beat_times"])
    if payload.get("beat_positions") is not None:
        payload["beat_positions"] = np.array(payload["beat_positions"], dtype=int)
    return payload
