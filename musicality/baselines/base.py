"""What every baseline tracker has in common: the shape of a prediction, how
that prediction becomes bar positions our metrics can score, and which of our
corpora each baseline was trained on.

A baseline answers one question — *is our architecture the bottleneck?* — and
it only answers it if the comparison is fair in three separate ways:

1. **Same tracks.** Both sides resolve their split through
   :func:`~musicality.evaluation.build_eval_dataset`.
2. **Same scorer.** Both sides produce a ``{"time", "beat_in_bar"}`` event list
   and hand it to :func:`~musicality.evaluation.score_events`. Nothing here
   reimplements an F-measure.
3. **Same bar-position convention.** Public trackers emit *downbeat times*, not
   bar positions, so :func:`assign_bar_positions` derives the positions — once,
   here, for every baseline — rather than each wrapper inventing its own
   counting rule.

The fourth fairness question, training-set overlap, cannot be fixed by code:
see :data:`CORPUS_EXPOSURE`.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


# Whether a baseline's published weights were trained on each of our corpora.
#
# This is the single most important caveat on any number this package produces.
# Both baselines ship weights trained on public beat-tracking corpora, and most
# of our validation split *is* those corpora — so on ballroom, rwc_popular,
# rwc_classical and rwc_jazz a baseline is being asked to recall tracks it was
# fitted on, while our model is being asked to generalize. Those rows are an
# upper bound on the baseline, not a measurement of it.
#
# `gtzan` and `rwc_genre` and `jtd` are the rows that mean something.
#
# Sourcing, per baseline:
#
# - **beat_this**: authoritative. The `final*`/`small*` checkpoints are
#   documented upstream as "trained on all data except the GTZAN dataset", and
#   the datasets they used are enumerated by the file list of the released
#   annotation repo (CPJKU/beat_this_annotations) and spectrogram bundle:
#   asap, ballroom, beatles, candombe, filosax, groove_midi, gtzan (excluded
#   from training), guitarset, hainsworth, harmonix, hjdb, jaah, rwc, simac,
#   smc, tapcorrect. Their `rwc` folder holds rwc_classical, rwc_jazz,
#   rwc_popular and rwc_royalty-free — but *not* the RWC Music Genre database,
#   which is what our `rwc_genre` is.
# - **madmom**: NOT verified. The shipped beat/downbeat RNN weights carry no
#   training manifest, and this row is recalled from Böck et al.'s papers
#   rather than read off anything. Treat `SEEN` entries as "probably" and
#   `UNKNOWN` as "nobody checked". GTZAN being madmom's standard held-out test
#   set is the one part of this that is widely documented.
SEEN = "seen"
HELD_OUT = "held-out"
UNKNOWN = "unknown"

CORPUS_EXPOSURE = {
    "beat_this": {
        "ballroom": SEEN,
        "gtzan": HELD_OUT,
        "jtd": HELD_OUT,
        "rwc_classical": SEEN,
        "rwc_genre": HELD_OUT,
        "rwc_jazz": SEEN,
        "rwc_popular": SEEN,
    },
    "madmom": {
        "ballroom": SEEN,
        "gtzan": HELD_OUT,
        "jtd": UNKNOWN,
        "rwc_classical": UNKNOWN,
        "rwc_genre": UNKNOWN,
        "rwc_jazz": UNKNOWN,
        "rwc_popular": SEEN,
    },
}


def assign_bar_positions(
    beats: np.ndarray,
    downbeats: np.ndarray,
    group_size: int = 4,
) -> np.ndarray | None:
    """Number each beat within its bar, counting forward from the nearest
    preceding downbeat.

    Public trackers report two time lists — beats, and the subset of them that
    are downbeats — while every bar-position metric here wants a position
    ``1..group_size`` attached to each beat. This is the conversion, and it is
    deliberately the *only* one: both wrappers go through it, so a difference
    between two baselines' ``position_acc`` is a difference in their downbeats
    and never a difference in how their bars were counted.

    Beats *before* the first downbeat are counted backwards from it, which the
    modular arithmetic does for free: the beat immediately before a downbeat
    gets ``group_size``, the one before that ``group_size - 1``. That matches
    how a human would number an incomplete opening bar, and it keeps the track's
    leading beats scorable instead of silently unlabelled.

    Every downbeat restarts the count, so a tracker that changes bar length
    mid-track (madmom is allowed to, with ``beats_per_bar=[3, 4]``) stays
    correctly aligned after the change rather than drifting for the rest of the
    track. The flip side is that a bar whose length is not *group_size* — a
    waltz scored against ``group_size=4`` — produces positions that cycle at the
    wrong rate. That is not worth special-casing: a reference annotated in 3
    cannot be folded onto 4 either, so
    :func:`~musicality.loaders.beat_dataset.fold_positions` has already switched
    that track's position supervision off, and ``--binary-only`` drops it
    outright.

    :param beats: Estimated beat times, in seconds, ascending.
    :param downbeats: Estimated downbeat times, in seconds. Matched to the
        nearest beat rather than compared exactly, so a tracker that reports
        them to different precision than its own beat list still anchors
        correctly.
    :param group_size: Beats per bar to count over.
    :returns: Positions ``1..group_size``, one per beat — or ``None`` when
        there is nothing to anchor on (no beats, or no downbeats), which is the
        same "unlabelled" signal a beat-only readout gives
        :func:`~musicality.evaluation.score_events`.
    """

    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)

    if len(beats) == 0 or len(downbeats) == 0:
        return None

    # Downbeats as indices into the beat list. `unique` because two downbeats
    # can round onto the same beat on a track where the tracker disagrees with
    # itself, and a duplicated anchor would break the searchsorted below.
    anchors = np.unique([int(np.argmin(np.abs(beats - time))) for time in downbeats])

    indices = np.arange(len(beats))

    # For each beat, the last anchor at or before it — clamped to the first
    # anchor, which is what counts the opening beats backwards.
    slots = np.maximum(np.searchsorted(anchors, indices, side="right") - 1, 0)

    return ((indices - anchors[slots]) % group_size) + 1


def events_from_beats(
    beats: np.ndarray,
    downbeats: np.ndarray | None,
    group_size: int = 4,
) -> list[dict]:
    """Package a baseline's prediction as the event list
    :func:`~musicality.evaluation.score_events` scores.

    :param beats: Estimated beat times, in seconds.
    :param downbeats: Estimated downbeat times, or ``None`` for a beat-only
        tracker. Either way the result has one entry per beat; only
        ``beat_in_bar`` differs, exactly as it does between a beat-only and a
        beat-phase checkpoint.
    :param group_size: Beats per bar — see :func:`assign_bar_positions`.
    :returns: ``[{"time": float, "beat_in_bar": int | None}, ...]``.
    """

    positions = (
        None
        if downbeats is None
        else assign_bar_positions(beats, downbeats, group_size=group_size)
    )

    return [
        {
            "time": float(time),
            "beat_in_bar": None if positions is None else int(positions[i]),
        }
        for i, time in enumerate(np.asarray(beats, dtype=float))
    ]


class Baseline(ABC):
    """A third-party beat tracker, wrapped so it can be scored exactly like one
    of our checkpoints.

    Subclasses implement :meth:`predict` and declare :attr:`name`. Everything
    else — bar-position assignment, caching, split resolution, scoring,
    reporting — is shared, so adding a third tracker is one file and one
    registry entry.

    The heavy import (``madmom``, ``beat_this``) belongs in the subclass's
    ``__init__``, not at module scope: both are optional extras, and
    ``import musicality.baselines`` must keep working when neither is
    installed so that ``--help`` and the test suite do not depend on them.
    """

    #: Registry key, and the name written into a prediction cache.
    name: str = ""

    @property
    def config(self) -> dict:
        """Everything that changes this baseline's output, as plain JSON types.

        Stored in the prediction cache header and compared against the live
        settings on load, so a cache produced with a different checkpoint or a
        different DBN setting is rejected instead of quietly answering for one
        that was never run. Cheap to be thorough here; the failure it prevents
        is a report labelled with the wrong configuration.
        """

        return {}

    def describe(self) -> str:
        """One line naming the baseline and its configuration, for report headers."""

        settings = ", ".join(f"{key}={value}" for key, value in self.config.items())

        return f"{self.name}({settings})" if settings else self.name

    @abstractmethod
    def predict(self, audio_path: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Track one audio file.

        Implementations read the file themselves rather than taking a waveform:
        each tracker has its own sample rate and its own loader, and resampling
        into it from ours would handicap it for no reason.

        :param audio_path: Path to an audio file.
        :returns: ``(beats, downbeats)`` — times in seconds. ``downbeats`` is
            ``None`` for a tracker that does not estimate them, and must
            otherwise be a subset of (or align with) *beats*.
        """


class CachedBaseline(Baseline):
    """A baseline that exists only as a file of predictions.

    :class:`~musicality.baselines.cache.PredictionCache` deliberately stores
    plain times in seconds, so a tracker can be run wherever it is convenient —
    a GPU box, a machine with an environment this project does not want, a
    colleague's laptop — and scored here. That only works if scoring does not
    also require the tracker to be installed, which is what this is for: it
    adopts the cache file's own name and configuration, and refuses to predict.

    :param name: Baseline name, as recorded in the cache header.
    :param config: Configuration, as recorded in the cache header. Adopted
        verbatim so the header check passes.
    """

    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self._config = config or {}

    @classmethod
    def from_cache_file(cls, path: str | Path) -> "CachedBaseline":
        """Build one from an existing cache file's header.

        :raises FileNotFoundError: If *path* does not exist.
        :raises KeyError: If the file is not a prediction cache.
        """

        payload = json.loads(Path(path).read_text())

        return cls(payload["baseline"], payload.get("config", {}))

    @property
    def config(self) -> dict:
        return self._config

    def predict(self, audio_path: str):
        raise RuntimeError(
            f"No prediction cached for {audio_path}, and this is a cache-only "
            f"baseline. Re-run the real '{self.name}' tracker to fill the "
            "cache, or drop --from-cache."
        )
