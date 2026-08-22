"""Track data model, beat annotation helpers, and persistence.

Pure functions (``beats_per_bar``, ``active_beat_position``, ``bar_indices``,
``active_bar_index``, ``add_beat``, ``remove_beat``) are kept side-effect-free
so they can be tested without mirdata or a filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import (
    TrackData,
    _annotations_slot_dir,
    annotation_path,
    metadata_path,
    read_beats_file,
)

# Re-exported for external callers (migrate_*.py, mobile_companion/server.py)
# that import these from tools.annotator.data rather than the shared module
# directly — not used within this file itself.
from musicality.dataformats.track_io import (
    METADATA_SCHEMA_VERSION as METADATA_SCHEMA_VERSION,
)
from musicality.dataformats.track_io import TrackMetadata as TrackMetadata
from musicality.dataformats.track_io import load_metadata as load_metadata
from musicality.dataformats.track_io import save_annotations as save_annotations
from musicality.dataformats.track_io import save_metadata as save_metadata

# Read dataformats.DATA_DIR fresh at every call site below rather than
# caching it in a module-level constant, so tests can monkeypatch it
# (musicality.dataformats.track_io does the same, for the same reason).

# Tapping always starts on beat 1 of an N-count phrase ("sentence" in dance
# terms — e.g. an 8-count in swing). 8 is the default phrase length.
DEFAULT_N_BEATS = 8


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DatasetInfo:
    name: str
    n_tracks: int
    n_annotations: int
    mtime: float


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


def beats_per_bar(
    beat_positions: np.ndarray | None, default: int = DEFAULT_N_BEATS
) -> int:
    """Return the number of beats per bar inferred from beat positions.

    Falls back to *default* when positions are unavailable or empty.
    """
    if beat_positions is None or len(beat_positions) == 0:
        return default
    return int(max(beat_positions))


def cycle_positions(n_total: int, n_beats: int) -> np.ndarray:
    """Return *n_total* 1-indexed positions cycling 1..n_beats, starting at 1.

    Matches the annotation convention that tapping always starts on beat 1
    of an N-count phrase.
    """
    return np.array([(i % n_beats) + 1 for i in range(n_total)], dtype=int)


def active_beat_position(
    beat_times: np.ndarray,
    beat_positions: np.ndarray | None,
    t: float,
    default_n_beats: int = DEFAULT_N_BEATS,
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
    n = beats_per_bar(beat_positions, default=default_n_beats)
    return (idx % n) + 1


def bar_indices(
    beat_positions: np.ndarray | None,
    n_total: int,
    default_n_beats: int = DEFAULT_N_BEATS,
) -> np.ndarray:
    """Return the 0-indexed bar number for each of *n_total* beats.

    Uses ``beat_positions`` (cumulative count of downbeats) when available,
    otherwise assumes a constant beats-per-bar cycle.
    """
    if n_total == 0:
        return np.array([], dtype=int)
    if beat_positions is not None:
        return np.cumsum(beat_positions == 1) - 1
    n = beats_per_bar(beat_positions, default=default_n_beats)
    return np.arange(n_total) // n


def active_bar_index(
    beat_times: np.ndarray,
    beat_positions: np.ndarray | None,
    t: float,
    default_n_beats: int = DEFAULT_N_BEATS,
) -> int | None:
    """Return the 0-indexed bar number containing the most recent beat at time *t*.

    Returns ``None`` if *t* is before the first beat.
    """
    if len(beat_times) == 0:
        return None
    idx = int(np.searchsorted(beat_times, t, side="right")) - 1
    if idx < 0:
        return None
    return int(bar_indices(beat_positions, len(beat_times), default_n_beats)[idx])


def is_accent_beat(
    position: int,
    bar_index: int,
    n_beats: int,
    accent_bars: float,
) -> bool:
    """Return whether the beat at 1-indexed *position* in *bar_index* is accented.

    *accent_bars* is the accent period in bars:
    - 0.5 accents the downbeat and the halfway beat of every bar.
    - 1 (default) accents the downbeat of every bar.
    - N >= 1 accents the downbeat only once every N bars.
    """
    if accent_bars < 1:
        halfway = n_beats // 2 + 1
        return position in (1, halfway)
    return position == 1 and bar_index % int(accent_bars) == 0


def add_beat(
    track: TrackData, time: float, n_beats: int = DEFAULT_N_BEATS
) -> TrackData:
    """Return a new :class:`TrackData` with a beat inserted at *time*.

    Positions always cycle 1..n_beats, starting at 1 on the very first
    beat — the annotation convention that tapping begins on beat 1 of an
    N-count phrase. *n_beats* is trusted as given rather than inferred from
    the track's existing positions: inferring it from a still-partial tap
    sequence (e.g. only 1 beat tapped so far) would underestimate the
    intended count. Callers that want to preserve an already-known meter
    (e.g. loaded from mirdata) should pass that meter's beat count.
    """
    times = np.sort(np.append(track.beat_times, time))
    positions = cycle_positions(len(times), n_beats)
    return TrackData(
        dataset_name=track.dataset_name,
        track_id=track.track_id,
        audio_path=track.audio_path,
        tempo=track.tempo,
        beat_times=times,
        beat_positions=positions,
        annotator_id=track.annotator_id,
    )


def remove_beat(
    track: TrackData,
    time: float,
    tolerance: float = 0.1,
    n_beats: int = DEFAULT_N_BEATS,
) -> TrackData:
    """Return a new :class:`TrackData` with the beat nearest to *time* removed.

    Does nothing if no beat falls within *tolerance* seconds of *time*.
    Remaining positions are recycled 1..n_beats, same rule as :func:`add_beat`.
    """
    if len(track.beat_times) == 0:
        return track
    idx = int(np.argmin(np.abs(track.beat_times - time)))
    if abs(track.beat_times[idx] - time) > tolerance:
        return track
    times = np.delete(track.beat_times, idx)
    positions = cycle_positions(len(times), n_beats)
    return TrackData(
        dataset_name=track.dataset_name,
        track_id=track.track_id,
        audio_path=track.audio_path,
        tempo=track.tempo,
        beat_times=times,
        beat_positions=positions,
        annotator_id=track.annotator_id,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}


def _dataset_mtime(tracks_dir: Path) -> float:
    """Most recent modification time for a dataset, used to rank by recording date
    — the newest audio file's mtime (i.e. the last time a track was added)."""
    mtimes = [
        f.stat().st_mtime
        for f in tracks_dir.iterdir()
        if f.suffix.lower() in _AUDIO_EXTENSIONS
    ]
    return max(mtimes) if mtimes else tracks_dir.stat().st_mtime


def _require_tracks_dir(dataset_name: str, tracks_dir: Path) -> None:
    """Raise if *dataset_name* hasn't been migrated to this project's own format."""
    if not tracks_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' has no tracks/ folder — migrate it first via "
            f"`uv run python tools/migrate_mirdata_dataset.py --dataset {dataset_name}` "
            f"(mirdata datasets) or "
            f"`uv run python tools/migrate_rwc_genre.py --dataset {dataset_name}` "
            f"(CSV-annotated ones)."
        )


def list_datasets() -> list[DatasetInfo]:
    """Return info for every migrated dataset found in dataformats.DATA_DIR.

    Only datasets with a ``tracks/`` subfolder are listed — see
    :func:`_require_tracks_dir` for how to migrate one that isn't yet.
    """
    infos: list[DatasetInfo] = []
    for path in sorted(dataformats.DATA_DIR.iterdir()):
        if not path.is_dir():
            continue
        tracks_dir = path / dataformats.FORMAT.tracks_dirname
        if not tracks_dir.is_dir():
            continue
        name = path.name
        n_tracks = sum(
            1 for f in tracks_dir.iterdir() if f.suffix.lower() in _AUDIO_EXTENSIONS
        )
        ann_dir = path / dataformats.FORMAT.annotations_dirname
        n_ann = (
            len(list(ann_dir.glob(f"*{dataformats.FORMAT.beats_suffix}")))
            if ann_dir.is_dir()
            else 0
        )
        mtime = _dataset_mtime(tracks_dir)
        infos.append(
            DatasetInfo(name=name, n_tracks=n_tracks, n_annotations=n_ann, mtime=mtime)
        )
    return infos


def has_annotation(dataset_name: str, track_id: str) -> bool:
    """Return True if a saved .beats annotation file exists for this track."""
    return (
        dataformats.DATA_DIR
        / dataset_name
        / dataformats.FORMAT.annotations_dirname
        / f"{track_id}{dataformats.FORMAT.beats_suffix}"
    ).exists()


def annotation_meter_label(dataset_name: str, track_id: str) -> str:
    """Return a concise "1..N" summary of a track's beat-position resolution.

    N is the highest 1-indexed bar position seen, e.g. "1..4" for a track
    annotated at full-bar resolution or "1..2" for one annotated at half-bar
    (cut-time) resolution.

    Returns "" if there's no saved beat annotation for this track, "•" if
    beats exist but carry no bar-position channel (bare timestamps only).
    """
    ann_path = (
        _annotations_slot_dir(dataset_name, None)
        / f"{track_id}{dataformats.FORMAT.beats_suffix}"
    )
    if not ann_path.exists():
        return ""

    _, positions = read_beats_file(ann_path)
    if positions is None or len(positions) == 0:
        return "•"
    return f"1..{int(max(positions))}"


def load_dataset_tracks(dataset_name: str) -> list[str]:
    """Return all track IDs for *dataset_name*."""
    tracks_dir = dataformats.DATA_DIR / dataset_name / dataformats.FORMAT.tracks_dirname
    _require_tracks_dir(dataset_name, tracks_dir)
    return [
        f.stem
        for f in sorted(tracks_dir.iterdir())
        if f.suffix.lower() in _AUDIO_EXTENSIONS
    ]


def load_track(
    dataset_name: str, track_id: str, annotator_id: str | None = None
) -> TrackData:
    """Load track audio path and beat annotations from ``tracks/``/``annotations/``.

    *annotator_id* selects which annotator's saved slot to read; ``None`` is
    the default/legacy slot.
    """
    tracks_dir = dataformats.DATA_DIR / dataset_name / dataformats.FORMAT.tracks_dirname
    _require_tracks_dir(dataset_name, tracks_dir)

    audio_path = tracks_dir / f"{track_id}.wav"
    ann_path = (
        _annotations_slot_dir(dataset_name, annotator_id)
        / f"{track_id}{dataformats.FORMAT.beats_suffix}"
    )
    beat_times: np.ndarray = np.array([])
    beat_positions: np.ndarray | None = None
    if ann_path.exists():
        beat_times, beat_positions = read_beats_file(ann_path)
    return TrackData(
        dataset_name=dataset_name,
        track_id=track_id,
        audio_path=str(audio_path),
        tempo=None,
        beat_times=beat_times,
        beat_positions=beat_positions,
        annotator_id=annotator_id,
    )


def list_annotators(dataset_name: str, track_id: str) -> list[str | None]:
    """Return every annotator slot with a saved annotation for this track.

    ``None`` (the default/legacy slot) is listed first if present, followed
    by each annotator subdirectory that has a saved ``.beats`` file, sorted
    by name.
    """
    suffix = dataformats.FORMAT.beats_suffix
    default_dir = _annotations_slot_dir(dataset_name, None)
    annotators: list[str | None] = []
    if (default_dir / f"{track_id}{suffix}").exists():
        annotators.append(None)
    if default_dir.is_dir():
        for sub in sorted(default_dir.iterdir()):
            if sub.is_dir() and (sub / f"{track_id}{suffix}").exists():
                annotators.append(sub.name)
    return annotators


def _delete_annotation_files(
    dataset_name: str, track_id: str, annotator_id: str | None
) -> None:
    """Remove one annotator slot's saved .beats + .meta.json, if present."""
    ann = (
        _annotations_slot_dir(dataset_name, annotator_id)
        / f"{track_id}{dataformats.FORMAT.beats_suffix}"
    )
    if ann.exists():
        ann.unlink()
    meta = metadata_path(dataset_name, track_id, annotator_id)
    if meta.exists():
        meta.unlink()


def delete_annotation(track: TrackData) -> None:
    """Delete this annotator's saved annotation + metadata for *track*.

    Leaves the shared audio file and any other annotator's data untouched.
    """
    _delete_annotation_files(track.dataset_name, track.track_id, track.annotator_id)


def delete_track(track: TrackData) -> None:
    """Delete a custom-dataset track: its audio file plus every annotator's
    saved annotation and metadata for it.

    Raises ValueError for mirdata datasets.
    """
    tracks_dir = (
        dataformats.DATA_DIR / track.dataset_name / dataformats.FORMAT.tracks_dirname
    )
    if not tracks_dir.is_dir():
        raise ValueError("Cannot delete tracks from a mirdata dataset.")
    audio = Path(track.audio_path)
    if audio.exists():
        audio.unlink()
    for annotator_id in list_annotators(track.dataset_name, track.track_id):
        _delete_annotation_files(track.dataset_name, track.track_id, annotator_id)


def rename_track(track: TrackData, new_id: str) -> TrackData:
    """Rename a custom-dataset track on disk and return an updated TrackData.

    Renames the shared audio file and this annotator's saved ``.beats`` file
    (if present). The audio is shared across every annotator, but only the
    caller's own slot is renamed to match — any other annotator's saved
    ``.beats``/``.meta.json`` still reference the old track_id afterward.
    Raises ValueError for mirdata datasets or if *new_id* is already taken.
    """
    tracks_dir = (
        dataformats.DATA_DIR / track.dataset_name / dataformats.FORMAT.tracks_dirname
    )
    if not tracks_dir.is_dir():
        raise ValueError("Cannot rename tracks from a mirdata dataset.")
    old_audio = Path(track.audio_path)
    new_audio = old_audio.with_stem(new_id)
    if new_audio.exists():
        raise ValueError(f"A track named '{new_id}' already exists.")
    old_audio.rename(new_audio)
    old_ann = annotation_path(track)
    if old_ann.exists():
        old_ann.rename(old_ann.with_stem(new_id))
    return TrackData(
        dataset_name=track.dataset_name,
        track_id=new_id,
        audio_path=str(new_audio),
        tempo=track.tempo,
        beat_times=track.beat_times,
        beat_positions=track.beat_positions,
        annotator_id=track.annotator_id,
    )
