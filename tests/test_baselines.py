"""Tests for musicality.baselines — the third-party-tracker harness.

Nothing here imports madmom or beat_this. Those are optional extras, and the
part of this package that can get a benchmark *wrong* is not the two twenty-line
wrappers around someone else's tracker — it is the shared code underneath them:
turning downbeat times into bar positions, and deciding when a cached
prediction may be reused. Both are exercised directly, with a stub baseline
standing in for a real one.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from musicality.baselines import (
    BASELINES,
    CORPUS_EXPOSURE,
    Baseline,
    CachedBaseline,
    PredictionCache,
    assign_bar_positions,
    build_baseline,
    events_from_beats,
    track_key,
)
from musicality.baselines.evaluator import BaselineEvaluator, default_cache_path


class StubBaseline(Baseline):
    """A tracker that returns canned beats and counts how often it was asked."""

    name = "stub"

    def __init__(self, beats=None, downbeats=None, setting="a"):
        self.beats = np.arange(0.0, 8.0, 0.5) if beats is None else np.asarray(beats)
        self.downbeats = (
            np.arange(0.0, 8.0, 2.0) if downbeats is None else np.asarray(downbeats)
        )
        self.setting = setting
        self.n_calls = 0

    @property
    def config(self):
        return {"setting": self.setting}

    def predict(self, audio_path):
        self.n_calls += 1

        return self.beats, self.downbeats


def _fake_dataset(corpora, beat_times=None, positions=None, has_positions=True):
    """A BeatDataset stand-in: one track per entry in *corpora*."""

    beat_times = np.arange(0.0, 8.0, 0.5) if beat_times is None else beat_times
    positions = (
        np.tile([1, 2, 3, 4], len(beat_times) // 4) if positions is None else positions
    )

    dataset = MagicMock()
    dataset.samples = [
        (f"{corpus}_{i}.wav", beat_times, positions, has_positions)
        for i, corpus in enumerate(corpora)
    ]
    dataset.refs = [
        MagicMock(dataset_name=corpus, track_id=f"t{i}")
        for i, corpus in enumerate(corpora)
    ]
    dataset.__len__.return_value = len(corpora)

    return dataset


# --- assign_bar_positions ------------------------------------------------


def test_positions_count_forward_from_each_downbeat():
    beats = np.arange(0.0, 4.0, 0.5)  # 8 beats
    downbeats = np.array([0.0, 2.0])

    assert list(assign_bar_positions(beats, downbeats)) == [1, 2, 3, 4, 1, 2, 3, 4]


def test_beats_before_the_first_downbeat_count_backwards():
    """An incomplete opening bar is numbered so that it *ends* on the downbeat,
    which is how it would be counted by hand — not dropped, and not restarted
    at 1."""

    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    downbeats = np.array([1.0])

    assert list(assign_bar_positions(beats, downbeats)) == [3, 4, 1, 2, 3]


def test_a_downbeat_is_matched_to_its_nearest_beat():
    """Trackers report downbeats and beats from separate peak-picks, so an
    exact float match is not guaranteed even when they mean the same event."""

    beats = np.array([0.0, 0.51, 1.02, 1.53])
    downbeats = np.array([1.0])  # ~20 ms off the beat at 1.02

    assert list(assign_bar_positions(beats, downbeats)) == [3, 4, 1, 2]


def test_every_downbeat_restarts_the_count():
    """A tracker allowed to switch bar length mid-track must not drift for the
    rest of the track after it does."""

    beats = np.arange(0.0, 3.5, 0.5)  # 7 beats
    downbeats = np.array([0.0, 1.5])  # a 3-beat bar, then a 4-beat one

    assert list(assign_bar_positions(beats, downbeats)) == [1, 2, 3, 1, 2, 3, 4]


def test_group_size_sets_the_cycle_length():
    beats = np.arange(0.0, 4.0, 0.5)
    downbeats = np.array([0.0])

    positions = assign_bar_positions(beats, downbeats, group_size=8)

    assert list(positions) == [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize(
    "beats, downbeats",
    [
        (np.array([]), np.array([0.0])),
        (np.array([0.0, 0.5]), np.array([])),
    ],
)
def test_nothing_to_anchor_on_is_unlabelled(beats, downbeats):
    assert assign_bar_positions(beats, downbeats) is None


# --- events_from_beats ---------------------------------------------------


def test_events_carry_positions_when_downbeats_are_known():
    events = events_from_beats(np.array([0.0, 0.5, 1.0]), np.array([0.0]))

    assert [e["beat_in_bar"] for e in events] == [1, 2, 3]
    assert [e["time"] for e in events] == [0.0, 0.5, 1.0]


def test_a_beat_only_tracker_produces_unlabelled_events():
    """The same shape a beat-only checkpoint's readout produces, so
    score_events treats the two identically."""

    events = events_from_beats(np.array([0.0, 0.5]), None)

    assert [e["beat_in_bar"] for e in events] == [None, None]


# --- PredictionCache -----------------------------------------------------


def test_cache_round_trips_predictions(tmp_path):
    path = tmp_path / "stub.json"

    cache = PredictionCache(path, "stub", {"setting": "a"})
    cache.put("ballroom/t0", np.array([0.5, 1.0]), np.array([0.5]))
    cache.save()

    reopened = PredictionCache.open(path, "stub", {"setting": "a"})
    beats, downbeats = reopened.get("ballroom/t0")

    assert reopened.n_loaded == 1
    assert list(beats) == [0.5, 1.0]
    assert list(downbeats) == [0.5]


def test_cache_preserves_a_beat_only_prediction(tmp_path):
    path = tmp_path / "stub.json"

    cache = PredictionCache(path, "stub")
    cache.put("ballroom/t0", np.array([0.5]), None)
    cache.save()

    _beats, downbeats = PredictionCache.open(path, "stub").get("ballroom/t0")

    assert downbeats is None


def test_a_cache_written_with_other_settings_is_ignored(tmp_path):
    """The failure this prevents is a report labelled with a configuration that
    never produced it."""

    path = tmp_path / "stub.json"

    cache = PredictionCache(path, "stub", {"setting": "a"})
    cache.put("ballroom/t0", np.array([0.5]), None)
    cache.save()

    reopened = PredictionCache.open(path, "stub", {"setting": "b"})

    assert reopened.n_loaded == 0
    assert reopened.get("ballroom/t0") is None


def test_a_cache_from_another_baseline_is_ignored(tmp_path):
    path = tmp_path / "shared.json"

    cache = PredictionCache(path, "madmom")
    cache.put("ballroom/t0", np.array([0.5]), None)
    cache.save()

    assert PredictionCache.open(path, "beat_this").n_loaded == 0


def test_a_missing_cache_starts_empty(tmp_path):
    cache = PredictionCache.open(tmp_path / "absent.json", "stub")

    assert cache.n_loaded == 0
    assert cache.get("anything") is None


def test_track_key_matches_the_split_file_spelling():
    assert track_key("ballroom", "Media-103414") == "ballroom/Media-103414"


# --- registry ------------------------------------------------------------


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="Unknown baseline"):
        build_baseline("not_a_tracker")


def test_every_registered_baseline_has_an_exposure_row():
    """A baseline with no exposure row would score our training corpora with no
    warning at all, which is the one mistake this package exists to prevent."""

    assert set(CORPUS_EXPOSURE) == set(BASELINES)


def test_a_cache_only_baseline_adopts_the_file_header(tmp_path):
    """Scoring predictions made elsewhere must not require the tracker to be
    installed here — which only works if the config comes from the file."""

    path = tmp_path / "madmom.json"
    PredictionCache(path, "madmom", {"variant": "beat"}).save()

    baseline = CachedBaseline.from_cache_file(path)

    assert baseline.name == "madmom"
    assert baseline.config == {"variant": "beat"}


def test_a_cache_only_baseline_refuses_to_predict(tmp_path):
    path = tmp_path / "madmom.json"
    PredictionCache(path, "madmom").save()

    with pytest.raises(RuntimeError, match="cache-only"):
        CachedBaseline.from_cache_file(path).predict("some.wav")


def test_cache_path_separates_the_binary_split():
    plain = default_cache_path("madmom", "merge", "val", False)
    binary = default_cache_path("madmom", "merge", "val", True)

    assert plain != binary
    assert binary.name.endswith("-binary.json")


# --- BaselineEvaluator ---------------------------------------------------


def _evaluator(baseline, corpora, tmp_path, **kwargs):
    dataset = _fake_dataset(corpora)

    patcher = patch(
        "musicality.baselines.evaluator.build_eval_dataset",
        return_value=(dataset, list(range(len(corpora)))),
    )
    patcher.start()

    evaluator = BaselineEvaluator(
        baseline,
        dataset="merge",
        cache_path=tmp_path / "cache.json",
        verbose=False,
        **kwargs,
    )
    evaluator._patcher = patcher

    return evaluator


def test_scoring_produces_one_row_per_track_tagged_with_its_corpus(tmp_path):
    evaluator = _evaluator(StubBaseline(), ["ballroom", "gtzan"], tmp_path)

    rows = evaluator.score()

    assert [row["corpus"] for row in rows] == ["ballroom", "gtzan"]
    assert all(row["f_beat"] is not None for row in rows)

    evaluator._patcher.stop()


def test_a_perfect_tracker_scores_perfectly(tmp_path):
    """The stub returns exactly the reference beats and downbeats, so anything
    less than 1.0 means the harness, not the tracker, lost something."""

    evaluator = _evaluator(StubBaseline(), ["ballroom"], tmp_path, trim=False)

    row = evaluator.score()[0]

    assert row["f_beat"] == pytest.approx(1.0)
    assert row["position_acc"] == pytest.approx(1.0)

    evaluator._patcher.stop()


def test_predictions_are_cached_across_evaluators(tmp_path):
    first = StubBaseline()
    evaluator = _evaluator(first, ["ballroom", "gtzan"], tmp_path)
    evaluator.score()
    evaluator._patcher.stop()

    assert first.n_calls == 2

    second = StubBaseline()
    reused = _evaluator(second, ["ballroom", "gtzan"], tmp_path)
    reused.score()
    reused._patcher.stop()

    assert second.n_calls == 0


def test_refresh_re_runs_the_tracker(tmp_path):
    first = StubBaseline()
    evaluator = _evaluator(first, ["ballroom"], tmp_path)
    evaluator.score()
    evaluator._patcher.stop()

    second = StubBaseline()
    refreshed = _evaluator(second, ["ballroom"], tmp_path, refresh=True)
    refreshed.score()
    refreshed._patcher.stop()

    assert second.n_calls == 1


def test_exposure_note_names_only_the_corpora_actually_selected(tmp_path):
    baseline = StubBaseline()
    baseline.name = "beat_this"

    evaluator = _evaluator(baseline, ["ballroom", "gtzan"], tmp_path)
    note = evaluator.exposure_note()
    evaluator._patcher.stop()

    assert "ballroom" in note
    assert "gtzan" not in note


def test_exposure_note_is_silent_on_a_clean_split(tmp_path):
    baseline = StubBaseline()
    baseline.name = "beat_this"

    evaluator = _evaluator(baseline, ["gtzan", "jtd"], tmp_path)
    note = evaluator.exposure_note()
    evaluator._patcher.stop()

    assert note is None
