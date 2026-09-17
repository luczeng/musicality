"""On-disk cache of a baseline's raw predictions, keyed by track.

Running a baseline is the expensive half and scoring it is the free half —
madmom's DBN is minutes per track on CPU, while re-deriving bar positions and
re-running ``mir_eval`` over 277 tracks is seconds. Separating the two means a
report can be re-cut per genre, at a different tolerance, or against a
different ``group_size`` without touching the tracker again.

It also decouples *where* a baseline runs from where it is scored. The
predictions are plain times in seconds, so a tracker that needs its own
environment can be run anywhere, on any machine, and its cache file scored
here.

The header guards the obvious mistake: a cache is bound to the baseline name
*and its configuration*, so a file written with ``checkpoint=final0`` cannot be
silently read back into a run that asked for ``single_final0``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Times are rounded before writing: 4 decimals is 0.1 ms, three orders of
# magnitude below the 70 ms matching window, and it roughly halves the file.
_TIME_DECIMALS = 4


def track_key(dataset_name: str, track_id: str) -> str:
    """Cache key for one track: ``"<corpus>/<track_id>"``.

    The same spelling a split file uses, so a cache stays valid across splits —
    predictions made for ``merge`` are reused when scoring ``ballroom`` alone.
    """

    return f"{dataset_name}/{track_id}"


class PredictionCache:
    """A baseline's ``(beats, downbeats)`` per track, persisted as JSON.

    :param path: Where the cache lives. Created on :meth:`save`.
    :param baseline: Baseline name the cache belongs to.
    :param config: The baseline's configuration (see
        :attr:`~musicality.baselines.base.Baseline.config`).
    """

    def __init__(self, path: Path, baseline: str, config: dict | None = None):
        self.path = Path(path)
        self.baseline = baseline
        self.config = config or {}
        self.tracks: dict[str, dict] = {}
        self.n_loaded = 0

    @classmethod
    def open(cls, path: Path, baseline: str, config: dict | None = None):
        """Load *path* if it exists and matches *baseline*/*config*, else start empty.

        A mismatched header is a warning and a fresh cache, not an error: the
        usual cause is deliberately re-running with a changed setting, and
        refusing to proceed would just mean deleting the file by hand.

        :returns: A cache, populated or empty.
        """

        cache = cls(path, baseline, config)

        if not cache.path.exists():
            return cache

        payload = json.loads(cache.path.read_text())

        if payload.get("baseline") != baseline:
            print(
                f"[cache] {cache.path} holds '{payload.get('baseline')}' "
                f"predictions, not '{baseline}' — ignoring it"
            )
            return cache

        if payload.get("config", {}) != cache.config:
            print(
                f"[cache] {cache.path} was written with a different "
                f"configuration ({payload.get('config')}) — ignoring it"
            )
            return cache

        cache.tracks = payload.get("tracks", {})
        cache.n_loaded = len(cache.tracks)

        return cache

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Cached prediction for *key*, or ``None`` if the track is not in it."""

        entry = self.tracks.get(key)
        if entry is None:
            return None

        downbeats = entry.get("downbeats")

        return (
            np.asarray(entry["beats"], dtype=float),
            None if downbeats is None else np.asarray(downbeats, dtype=float),
        )

    def put(self, key: str, beats: np.ndarray, downbeats: np.ndarray | None) -> None:
        """Record one track's prediction."""

        self.tracks[key] = {
            "beats": np.round(np.asarray(beats, dtype=float), _TIME_DECIMALS).tolist(),
            "downbeats": (
                None
                if downbeats is None
                else np.round(
                    np.asarray(downbeats, dtype=float), _TIME_DECIMALS
                ).tolist()
            ),
        }

    def save(self) -> None:
        """Write the cache, creating parent directories as needed."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "baseline": self.baseline,
                    "config": self.config,
                    "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "tracks": self.tracks,
                },
                indent=1,
            )
        )
