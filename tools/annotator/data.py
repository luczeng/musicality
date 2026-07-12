"""Track data model, beat annotation helpers, and persistence.

Pure functions (``beats_per_bar``, ``active_beat_position``, ``add_beat``,
``remove_beat``) are kept side-effect-free so they can be tested without
mirdata or a filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mirdata
import numpy as np

import musicality.dataformats as dataformats

DATA_DIR = Path(__file__).parent.parent.parent / dataformats.load().data_dir


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DatasetInfo:
    name: str
    n_tracks: int
    n_annotations: int


@dataclass
class TrackData:
    """All annotation data for a single track."""

    dataset_name: str
    track_id: str
    audio_path: str
    tempo: float | None
    beat_times: np.ndarray  # seconds, sorted ascending
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


def bar_indices(beat_positions: np.ndarray | None, n_total: int) -> np.ndarray:
    """Return the 0-indexed bar number for each of *n_total* beats.

    Uses ``beat_positions`` (cumulative count of downbeats) when available,
    otherwise assumes a constant beats-per-bar cycle.
    """
    if n_total == 0:
        return np.array([], dtype=int)
    if beat_positions is not None:
        return np.cumsum(beat_positions == 1) - 1
    n = beats_per_bar(beat_positions)
    return np.arange(n_total) // n


def active_bar_index(
    beat_times: np.ndarray,
    beat_positions: np.ndarray | None,
    t: float,
) -> int | None:
    """Return the 0-indexed bar number containing the most recent beat at time *t*.

    Returns ``None`` if *t* is before the first beat.
    """
    if len(beat_times) == 0:
        return None
    idx = int(np.searchsorted(beat_times, t, side="right")) - 1
    if idx < 0:
        return None
    return int(bar_indices(beat_positions, len(beat_times))[idx])


def add_beat(track: TrackData, time: float) -> TrackData:
    """Return a new :class:`TrackData` with a beat inserted at *time*.

    If the track has no bar positions (beats-only mode), positions stay None.
    Otherwise positions are recomputed sequentially preserving beats-per-bar.
    """
    times = np.sort(np.append(track.beat_times, time))
    if track.beat_positions is None:
        positions = None
    else:
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
    if track.beat_positions is None:
        positions = None
    else:
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
    """Return the canonical save path for a track's annotations (.beats file)."""
    return DATA_DIR / track.dataset_name / "annotations" / f"{track.track_id}.beats"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}


def _read_beats_file(path: Path) -> np.ndarray:
    """Read a .beats text file (one timestamp per line) into a sorted array."""
    return np.array(
        [float(line) for line in path.read_text().splitlines() if line.strip()],
        dtype=float,
    )


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
        tracks_dir = path / "tracks"
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
        ann_dir = path / "annotations"
        n_ann = len(list(ann_dir.glob("*.beats"))) if ann_dir.is_dir() else 0
        infos.append(DatasetInfo(name=name, n_tracks=n_tracks, n_annotations=n_ann))
    return infos


def has_annotation(dataset_name: str, track_id: str) -> bool:
    """Return True if a saved .beats annotation file exists for this track."""
    return (DATA_DIR / dataset_name / "annotations" / f"{track_id}.beats").exists()


def has_mirdata_annotation(dataset_name: str, track_id: str) -> bool:
    """Return True if the mirdata dataset has built-in beat annotations for this track."""
    if (DATA_DIR / dataset_name / "tracks").is_dir():
        return False
    try:
        ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
        track = ds.track(track_id)
        return track.beats is not None and len(track.beats.times) > 0
    except Exception:
        return False


def load_dataset_tracks(dataset_name: str) -> list[str]:
    """Return all track IDs for *dataset_name*."""
    tracks_dir = DATA_DIR / dataset_name / "tracks"
    if tracks_dir.is_dir():
        return [
            f.stem
            for f in sorted(tracks_dir.iterdir())
            if f.suffix.lower() in _AUDIO_EXTENSIONS
        ]
    ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
    return list(ds.track_ids)


def load_track(dataset_name: str, track_id: str) -> TrackData:
    """Load track audio path and beat annotations.

    For custom datasets reads from ``tracks/`` and ``annotations/``.
    For mirdata datasets checks for a saved ``.beats`` file first, then
    falls back to the dataset's own annotations.
    """
    tracks_dir = DATA_DIR / dataset_name / "tracks"
    if tracks_dir.is_dir():
        audio_path = tracks_dir / f"{track_id}.wav"
        ann_path = DATA_DIR / dataset_name / "annotations" / f"{track_id}.beats"
        beat_times: np.ndarray = np.array([])
        if ann_path.exists():
            beat_times = _read_beats_file(ann_path)
        return TrackData(
            dataset_name=dataset_name,
            track_id=track_id,
            audio_path=str(audio_path),
            tempo=None,
            beat_times=beat_times,
            beat_positions=None,
        )

    ds = mirdata.initialize(dataset_name, data_home=str(DATA_DIR / dataset_name))
    track = ds.track(track_id)

    # Prefer our own saved annotation over the dataset's built-in beats
    ann_path = DATA_DIR / dataset_name / "annotations" / f"{track_id}.beats"
    if ann_path.exists():
        return TrackData(
            dataset_name=dataset_name,
            track_id=track_id,
            audio_path=track.audio_path,
            tempo=None,
            beat_times=_read_beats_file(ann_path),
            beat_positions=None,
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
        tempo=track.tempo,
        beat_times=beat_times,
        beat_positions=beat_positions,
    )


def save_annotations(track: TrackData, path: Path) -> None:
    """Persist beat times to a .beats file (one timestamp per line, in seconds)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{t:.6f}" for t in track.beat_times))


def delete_track(track: TrackData) -> None:
    """Delete a custom-dataset track (audio file + annotation) from disk.

    Raises ValueError for mirdata datasets.
    """
    tracks_dir = DATA_DIR / track.dataset_name / "tracks"
    if not tracks_dir.is_dir():
        raise ValueError("Cannot delete tracks from a mirdata dataset.")
    audio = Path(track.audio_path)
    if audio.exists():
        audio.unlink()
    ann = annotation_path(track)
    if ann.exists():
        ann.unlink()


def rename_track(track: TrackData, new_id: str) -> TrackData:
    """Rename a custom-dataset track on disk and return an updated TrackData.

    Renames the audio file and the annotation file (if present).
    Raises ValueError for mirdata datasets or if *new_id* is already taken.
    """
    tracks_dir = DATA_DIR / track.dataset_name / "tracks"
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
    )
