"""Track data model, beat annotation helpers, and persistence.

Pure functions (``beats_per_bar``, ``active_beat_position``, ``bar_indices``,
``active_bar_index``, ``add_beat``, ``remove_beat``) are kept side-effect-free
so they can be tested without mirdata or a filesystem.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mirdata
import numpy as np

import musicality.dataformats as dataformats

from .naming import sanitize_track_name

DATA_DIR = dataformats.DATA_DIR

# Tapping always starts on beat 1 of an N-count phrase ("sentence" in dance
# terms — e.g. an 8-count in swing). 8 is the default phrase length.
DEFAULT_N_BEATS = 8

# Bumped whenever TrackMetadata's on-disk shape changes. save_metadata always
# stamps this value; TrackMetadata.schema_version defaults to 1 (the implicit
# version of every file saved before this field existed), so a file missing
# the key on load is correctly read as version 1 rather than "current".
# v2 adds annotator_id and section_aligned.
METADATA_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DatasetInfo:
    name: str
    n_tracks: int
    n_annotations: int
    mtime: float


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
    schema_version: int = 1


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


def _annotations_slot_dir(dataset_name: str, annotator_id: str | None) -> Path:
    """Return the annotations directory for one annotator's slot.

    ``annotator_id=None`` is the original, unsuffixed default slot, so every
    file saved before multi-annotator support existed keeps resolving to the
    same path. A non-None id is sanitized the same way track ids are, since
    it can come from free-form input (e.g. the mobile companion).
    """
    base = DATA_DIR / dataset_name / dataformats.FORMAT.annotations_dirname
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
    dataset_name: str, track_id: str, annotator_id: str | None = None
) -> Path:
    """Return the canonical save path for a track's metadata (.meta.json file).

    Nested under *annotator_id*'s slot; ``None`` is the default slot.
    """
    return (
        _annotations_slot_dir(dataset_name, annotator_id)
        / f"{track_id}{dataformats.FORMAT.metadata_suffix}"
    )


def save_metadata(dataset_name: str, track_id: str, metadata: TrackMetadata) -> None:
    """Persist track metadata as JSON, next to that track's .beats file.

    Saved under ``metadata.annotator_id``'s slot; ``None`` is the default slot.
    """
    path = metadata_path(dataset_name, track_id, metadata.annotator_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.schema_version = METADATA_SCHEMA_VERSION
    path.write_text(json.dumps(asdict(metadata), indent=2))


def load_metadata(
    dataset_name: str, track_id: str, annotator_id: str | None = None
) -> TrackMetadata | None:
    """Load a track's metadata for *annotator_id*'s slot, or None if unsaved."""
    path = metadata_path(dataset_name, track_id, annotator_id)
    if not path.exists():
        return None
    return TrackMetadata(**json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}


def _read_beats_file(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a .beats file into ``(times, positions)``.

    Uses mirdata's own format — ``<time> <position>`` per line (see e.g. the
    ballroom dataset's raw annotation files). Falls back to bare timestamps
    (one per line, no position) for files saved before position tracking was
    added; in that case ``positions`` is ``None``.
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


def _dataset_mtime(path: Path, tracks_dir: Path) -> float:
    """Most recent modification time for a dataset, used to rank by recording date.

    For custom recorded datasets, this is the newest audio file's mtime (i.e. the
    last time a track was recorded into it). Falls back to the folder's own mtime
    for mirdata datasets, where "recording date" isn't meaningful.
    """
    if tracks_dir.is_dir():
        mtimes = [
            f.stat().st_mtime
            for f in tracks_dir.iterdir()
            if f.suffix.lower() in _AUDIO_EXTENSIONS
        ]
        if mtimes:
            return max(mtimes)
    return path.stat().st_mtime


def list_datasets() -> list[DatasetInfo]:
    """Return info for every dataset found in DATA_DIR.

    Custom datasets (those with a ``tracks/`` subfolder) report their own
    annotation count from their ``annotations/`` subfolder.  Mirdata datasets
    report ``None`` for annotations because those are embedded in the download.
    """
    infos: list[DatasetInfo] = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        tracks_dir = path / dataformats.FORMAT.tracks_dirname
        if tracks_dir.is_dir():
            n_tracks = sum(
                1 for f in tracks_dir.iterdir() if f.suffix.lower() in _AUDIO_EXTENSIONS
            )
        else:
            try:
                ds = mirdata.initialize(name, data_home=str(path))
                n_tracks = len(ds.track_ids)
            except Exception:
                continue
        ann_dir = path / dataformats.FORMAT.annotations_dirname
        n_ann = (
            len(list(ann_dir.glob(f"*{dataformats.FORMAT.beats_suffix}")))
            if ann_dir.is_dir()
            else 0
        )
        mtime = _dataset_mtime(path, tracks_dir)
        infos.append(
            DatasetInfo(name=name, n_tracks=n_tracks, n_annotations=n_ann, mtime=mtime)
        )
    return infos


def has_annotation(dataset_name: str, track_id: str) -> bool:
    """Return True if a saved .beats annotation file exists for this track."""
    return (
        DATA_DIR
        / dataset_name
        / dataformats.FORMAT.annotations_dirname
        / f"{track_id}{dataformats.FORMAT.beats_suffix}"
    ).exists()


def has_mirdata_annotation(dataset_name: str, track_id: str) -> bool:
    """Return True if the mirdata dataset has built-in beat annotations for this track."""
    if (DATA_DIR / dataset_name / dataformats.FORMAT.tracks_dirname).is_dir():
        return False
    try:
        ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
        track = ds.track(track_id)
        return track.beats is not None and len(track.beats.times) > 0
    except Exception:
        return False


def load_dataset_tracks(dataset_name: str) -> list[str]:
    """Return all track IDs for *dataset_name*."""
    tracks_dir = DATA_DIR / dataset_name / dataformats.FORMAT.tracks_dirname
    if tracks_dir.is_dir():
        return [
            f.stem
            for f in sorted(tracks_dir.iterdir())
            if f.suffix.lower() in _AUDIO_EXTENSIONS
        ]
    ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
    return list(ds.track_ids)


def load_track(
    dataset_name: str, track_id: str, annotator_id: str | None = None
) -> TrackData:
    """Load track audio path and beat annotations.

    For custom datasets reads from ``tracks/`` and ``annotations/``.
    For mirdata datasets checks for a saved ``.beats`` file first, then
    falls back to the dataset's own annotations. *annotator_id* selects
    which annotator's saved slot to read; ``None`` is the default/legacy slot.
    """
    tracks_dir = DATA_DIR / dataset_name / dataformats.FORMAT.tracks_dirname
    if tracks_dir.is_dir():
        audio_path = tracks_dir / f"{track_id}.wav"
        ann_path = (
            _annotations_slot_dir(dataset_name, annotator_id)
            / f"{track_id}{dataformats.FORMAT.beats_suffix}"
        )
        beat_times: np.ndarray = np.array([])
        beat_positions: np.ndarray | None = None
        if ann_path.exists():
            beat_times, beat_positions = _read_beats_file(ann_path)
        return TrackData(
            dataset_name=dataset_name,
            track_id=track_id,
            audio_path=str(audio_path),
            tempo=None,
            beat_times=beat_times,
            beat_positions=beat_positions,
            annotator_id=annotator_id,
        )

    ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
    track = ds.track(track_id)

    # Prefer our own saved annotation over the dataset's built-in beats
    ann_path = (
        _annotations_slot_dir(dataset_name, annotator_id)
        / f"{track_id}{dataformats.FORMAT.beats_suffix}"
    )
    if ann_path.exists():
        beat_times, beat_positions = _read_beats_file(ann_path)
        return TrackData(
            dataset_name=dataset_name,
            track_id=track_id,
            audio_path=track.audio_path,
            tempo=None,
            beat_times=beat_times,
            beat_positions=beat_positions,
            annotator_id=annotator_id,
        )

    beat_times = np.array([])
    beat_positions = None
    if track.beats is not None:
        beat_times = track.beats.times
        beat_positions = getattr(track.beats, "positions", None)
    return TrackData(
        dataset_name=dataset_name,
        track_id=track_id,
        audio_path=track.audio_path,
        tempo=getattr(track, "tempo", None),
        beat_times=beat_times,
        beat_positions=beat_positions,
        annotator_id=annotator_id,
    )


def save_annotations(track: TrackData, path: Path) -> None:
    """Persist beat annotations to a .beats file, mirdata's own format:
    ``<time> <position>`` per line (seconds, 1-indexed bar/phrase position) —
    see e.g. the ballroom dataset's raw annotation files.

    Falls back to bare timestamps (no position column) if the track has no
    positions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = track.beat_positions
    if positions is not None and len(positions) == len(track.beat_times):
        lines = (f"{t:.6f} {p}" for t, p in zip(track.beat_times, positions))
    else:
        lines = (f"{t:.6f}" for t in track.beat_times)
    path.write_text("\n".join(lines))


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
    tracks_dir = DATA_DIR / track.dataset_name / dataformats.FORMAT.tracks_dirname
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
    tracks_dir = DATA_DIR / track.dataset_name / dataformats.FORMAT.tracks_dirname
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
