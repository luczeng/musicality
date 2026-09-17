"""Third-party beat trackers, scored on our splits with our metrics.

High level
----------

Our downbeat numbers are roughly 20 points behind what the beat-tracking
literature reports, and that gap has two possible explanations that no amount
of tuning can tell apart: our *model* is behind the field, or our *data and
annotations* are harder than theirs. Running a published tracker over our own
held-out tracks separates them in one pass — if a state-of-the-art system also
struggles on a corpus, the corpus is hard; if it sails through, the model is the
problem. This package is that experiment
(``plans/08_rethinking_the_approach.md`` §6.1).

Two trackers are wrapped: **madmom**, the RNN+DBN system the field has used as
its reference for a decade, and **Beat This!**, the current state of the art.

Technical
---------

- :func:`build_baseline` constructs one by name; every heavy import happens
  inside the wrapper's ``__init__``, so importing this package never requires
  either extra to be installed.
- :class:`~musicality.baselines.base.Baseline` subclasses do one thing: turn an
  audio path into ``(beats, downbeats)``.
- :class:`~musicality.baselines.evaluator.BaselineEvaluator` does everything
  else, and shares all of it with ``tools/eval_beat.py``: the split comes from
  :func:`~musicality.evaluation.build_eval_dataset`, the scoring from
  :func:`~musicality.evaluation.score_events`, the report from
  :func:`~musicality.evaluation.summary_block`. A baseline row and a checkpoint
  row differ only in who produced the beats.
- Predictions are cached to disk per track
  (:class:`~musicality.baselines.cache.PredictionCache`), because running the
  tracker is minutes and scoring it is seconds.

Installation
------------

Neither tracker is a default dependency: madmom needs a source build with an
old-style setup, and Beat This! pulls its own transformer stack. Both live in
an optional extra::

    uv sync --extra baselines

Read :data:`~musicality.baselines.base.CORPUS_EXPOSURE` before quoting any
number this package produces — most of our validation split is in both
trackers' training data.
"""

from musicality.baselines.base import (
    CORPUS_EXPOSURE,
    Baseline,
    CachedBaseline,
    assign_bar_positions,
    events_from_beats,
)
from musicality.baselines.cache import PredictionCache, track_key
from musicality.baselines.evaluator import BaselineEvaluator, default_cache_path

#: Registry keys accepted by :func:`build_baseline`, in the order a report
#: should present them: the long-standing reference first, the current state of
#: the art second.
BASELINES = ("madmom", "beat_this")

__all__ = [
    "BASELINES",
    "CORPUS_EXPOSURE",
    "Baseline",
    "BaselineEvaluator",
    "CachedBaseline",
    "PredictionCache",
    "assign_bar_positions",
    "build_baseline",
    "default_cache_path",
    "events_from_beats",
    "track_key",
]


def build_baseline(name: str, **options) -> Baseline:
    """Construct a baseline by registry name.

    Imports the wrapper lazily, so a missing optional dependency raises only
    when that particular tracker is actually asked for — ``--help``, the test
    suite, and the other baseline all keep working.

    :param name: One of :data:`BASELINES`.
    :param options: Forwarded to the wrapper's constructor — see
        :class:`~musicality.baselines.madmom_baseline.MadmomBaseline` and
        :class:`~musicality.baselines.beat_this_baseline.BeatThisBaseline`.
    :raises ValueError: If *name* is not a known baseline.
    :raises ImportError: If the tracker's package is not installed; the message
        carries the install command.
    """

    if name == "madmom":
        from musicality.baselines.madmom_baseline import MadmomBaseline

        return MadmomBaseline(**options)

    if name == "beat_this":
        from musicality.baselines.beat_this_baseline import BeatThisBaseline

        return BeatThisBaseline(**options)

    raise ValueError(f"Unknown baseline {name!r} — expected one of {list(BASELINES)}")
