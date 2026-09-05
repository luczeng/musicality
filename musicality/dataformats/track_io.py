"""Track data model, on-disk file I/O, and lookup helpers for this
project's own tracks/+annotations/ dataset format (see docs/source/data.rst).

Shared by the annotator (``tools/annotator/data.py``), the migration tools
(``tools/migrate_*.py``), and the training loaders (``musicality/loaders/``),
so none of them can drift out of sync about the on-disk layout.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import musicality.dataformats as dataformats

# Bumped whenever TrackMetadata's on-disk shape changes. save_metadata always
# stamps this value; TrackMetadata.schema_version defaults to 1 (the implicit
# version of every file saved before this field existed), so a file missing
# the key on load is correctly read as version 1 rather than "current".
# v2 adds annotator_id and section_aligned. v3 adds warning. v4 adds
# needs_review. v5 adds corrected.
METADATA_SCHEMA_VERSION = 5

_SANITIZE_RE = re.compile(r"[^\w\-]")


def sanitize_track_name(name: str) -> str:
    """Turn free-form user input into a filesystem-safe track id.

    Falls back to ``"recording"`` if *name* is empty or whitespace-only.
    """

    return _SANITIZE_RE.sub("_", name.strip()) or "recording"


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
    beat_times: np.ndarray  # seconds, sorted ascending
    beat_positions: np.ndarray | None  # 1-indexed bar positions, or None
    annotator_id: str | None = None  # None = the default/legacy annotation slot


@dataclass
class TrackRef:
    """One track's identity plus where to resolve its audio/annotations from.

    Returned by :func:`list_track_refs`, and the unit a
    ``TempoDataset``/``BeatDataset`` can be built from directly via its
    ``refs=`` constructor argument — e.g. a list of refs spanning several
    source datasets, such as one produced by ``tools/merge_datasets.py``.
    """

    dataset_name: str
    track_id: str
    data_home: Path


@dataclass
class TrackMetadata:
    """Free-form descriptive info about a track, separate from its beats.

    All fields optional — captured incrementally from either the desktop
    annotator or the mobile companion, never required to save a recording.
    """

    location: str | None = None
    device: str | None = None
    structure: str | None = None
    duration_s: float | None = None
    bpm_mean: float | None = None
    bpm_median: float | None = None
    bpm_std: float | None = None
    annotator_id: str | None = None  # who made this annotation, if known
    # Tapping always starts at count position 1 (see cycle_positions) — that
    # part is guaranteed, not something to confirm. section_aligned instead
    # records whether that first tap also happens to be the true start of a
    # section, vs. landing mid-section. True/False = confirmed either way,
    # None = not recorded.
    section_aligned: bool | None = None
    warning: bool = False  # flagged as a suspicious/questionable annotation
    needs_review: bool = False  # flagged as "take another look", distinct from warning
    corrected: bool = False  # a phase-shift correction has been saved for this track
    schema_version: int = 1


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def bpm_stats(
    beat_times: np.ndarray,
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Mean/median/std BPM from inter-beat intervals.

    Instantaneous tempo per interval, then averaged — the persisted-metadata
    convention used by the migration tools. Distinct from
    ``tools.annotator.data.tempo_from_beats``' single median-interval
    estimate, which the live tap-tempo UI uses instead.
    """

    if len(beat_times) < 2:
        return None, None, None

    tempos = 60.0 / np.diff(beat_times)

    return float(np.mean(tempos)), float(np.median(tempos)), float(np.std(tempos))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _annotations_slot_dir(
    dataset_name: str, annotator_id: str | None, data_home: Path | None = None
) -> Path:
    """Return the annotations directory for one annotator's slot.

    ``annotator_id=None`` is the original, unsuffixed default slot, so every
    file saved before multi-annotator support existed keeps resolving to the
    same path. A non-None id is sanitized the same way track ids are, since
    it can come from free-form input (e.g. the mobile companion).

    :param data_home: Dataset directory to use instead of the canonical
        ``dataformats.DATA_DIR / dataset_name`` (e.g. a training script's
        ``--data-home`` override). Read fresh rather than caching
        ``dataformats.DATA_DIR`` at import time, so tests can monkeypatch it
        (module-level snapshots wouldn't see the patch).
    """

    base = (data_home or dataformats.DATA_DIR / dataset_name) / (
        dataformats.FORMAT.annotations_dirname
    )
    if annotator_id:
        return base / sanitize_track_name(annotator_id)
    return base


def annotation_path(track: TrackData) -> Path:
    """Return the canonical save path for a track's annotations (.beats file).

    Nested under ``track.annotator_id``'s slot; ``None`` is the default slot.
    """

    return (
        _annotations_slot_dir(track.dataset_name, track.annotator_id)
        / f"{track.track_id}{dataformats.FORMAT.beats_suffix}"
    )


def metadata_path(
    dataset_name: str,
    track_id: str,
    annotator_id: str | None = None,
    data_home: Path | None = None,
) -> Path:
    """Return the canonical save path for a track's metadata (.meta.json file).

    Nested under *annotator_id*'s slot; ``None`` is the default slot. See
    :func:`_annotations_slot_dir` for *data_home*.
    """

    return (
        _annotations_slot_dir(dataset_name, annotator_id, data_home)
        / f"{track_id}{dataformats.FORMAT.metadata_suffix}"
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_beats_file(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a .beats file into ``(times, positions)``.

    ``<time> <position>`` per line (seconds, 1-indexed bar/phrase position) —
    see e.g. the ballroom dataset's raw annotation files, which this format
    mirrors. Falls back to bare timestamps (one per line, no position) for
    files saved before position tracking was added; in that case
    ``positions`` is ``None``.
    """

    times: list[float] = []
    positions: list[int] = []
    has_positions = True
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        times.append(float(parts[0]))
        if len(parts) >= 2:
            positions.append(int(parts[1]))
        else:
            has_positions = False

    times_arr = np.array(times, dtype=float)
    if has_positions and len(positions) == len(times):
        return times_arr, np.array(positions, dtype=int)
    return times_arr, None


def save_annotations(track: TrackData, path: Path) -> None:
    """Persist beat annotations to a .beats file — see :func:`read_beats_file`
    for the format. Falls back to bare timestamps (no position column) if the
    track has no positions.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = track.beat_positions
    if positions is not None and len(positions) == len(track.beat_times):
        lines = (f"{t:.6f} {p}" for t, p in zip(track.beat_times, positions))
    else:
        lines = (f"{t:.6f}" for t in track.beat_times)
    path.write_text("\n".join(lines))


def save_metadata(dataset_name: str, track_id: str, metadata: TrackMetadata) -> None:
    """Persist track metadata as JSON, next to that track's .beats file.

    Saved under ``metadata.annotator_id``'s slot; ``None`` is the default slot.
    """

    path = metadata_path(dataset_name, track_id, metadata.annotator_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.schema_version = METADATA_SCHEMA_VERSION
    path.write_text(json.dumps(asdict(metadata), indent=2))


def load_metadata(
    dataset_name: str,
    track_id: str,
    annotator_id: str | None = None,
    data_home: Path | None = None,
) -> TrackMetadata | None:
    """Load a track's metadata for *annotator_id*'s slot, or None if unsaved.
    See :func:`_annotations_slot_dir` for *data_home*.
    """

    path = metadata_path(dataset_name, track_id, annotator_id, data_home)
    if not path.exists():
        return None
    return TrackMetadata(**json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def list_migrated_track_ids(
    dataset_name: str, data_home: Path | None = None
) -> list[str]:
    """Return every track id with a default-slot .beats file for *dataset_name*.

    Non-recursive, so annotator subdirectories (alternate annotation slots)
    are excluded. An empty result means *dataset_name* hasn't been migrated
    to this project's own format yet. See :func:`_annotations_slot_dir` for
    *data_home*.
    """

    ann_dir = (data_home or dataformats.DATA_DIR / dataset_name) / (
        dataformats.FORMAT.annotations_dirname
    )
    if not ann_dir.is_dir():
        return []

    return sorted(p.stem for p in ann_dir.glob(f"*{dataformats.FORMAT.beats_suffix}"))


def resolve_track_audio(
    dataset_name: str, track_id: str, data_home: Path | None = None
) -> Path | None:
    """Return the on-disk audio path for *track_id* in *dataset_name*, or
    ``None`` if no ``tracks/<track_id>.wav`` exists. See
    :func:`_annotations_slot_dir` for *data_home*.
    """

    path = (
        (data_home or dataformats.DATA_DIR / dataset_name)
        / dataformats.FORMAT.tracks_dirname
        / f"{track_id}.wav"
    )
    return path if path.exists() else None


def list_track_refs(name: str, data_home: Path | None = None) -> list[TrackRef]:
    """Return every migrated track for *name*, wrapped as :class:`TrackRef`.

    Thin wrapper over :func:`list_migrated_track_ids` — kept as the loaders'
    entry point since it returns ready-to-resolve refs rather than bare
    track ids. To build a dataset from tracks spanning several source
    datasets (e.g. a merge), pass ``refs=`` directly to
    ``TempoDataset``/``BeatDataset`` instead of going through *name*.
    """

    base = data_home or dataformats.DATA_DIR / name

    return [TrackRef(name, stem, base) for stem in list_migrated_track_ids(name, base)]
