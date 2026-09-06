"""Tests for per-genre evaluation reporting.

A merged split blends several corpora into one number, and that number is
weighted by however many tracks each corpus happens to contribute. On the
current merge val set ballroom + jtd cast 80% of the vote and rwc_classical
casts 2.8%, so a model can be useless on classical and the blended score barely
registers it — measured: micro confusion 0.283 vs macro 0.311, with classical
``f_last`` at 0.066.

These cover the two pieces that make that visible:

- :meth:`~musicality.evaluation.BeatEvaluator.track_corpora`, which says which
  corpus each scored track came from,
- :func:`~musicality.evaluation.summarize` /
  :func:`~musicality.evaluation.group_by_corpus`, which turn per-track rows
  into per-corpus, macro and worst-corpus numbers.

Background: plans/04_beat_phase_generalization_and_data_prep.md, proposal #0.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from musicality.evaluation import BeatEvaluator, group_by_corpus, summarize

FPS = 10.0
PERIOD = 0.5
G = 4

# Evaluator settings shared by the BeatEvaluator.score() integration tests.
# Trimming is off because these synthetic tracks are shorter than mir_eval's
# 5s warm-up window.
# fps is derived from sample_rate/hop_length rather than passed in, so these
# two are what pin the synthetic grid at FPS=10.
EVALUATOR_KW = dict(
    sample_rate=int(FPS),
    hop_length=1,
    decoder="greedy",
    beat_threshold=0.3,
    min_distance_frames=1,
    gate_tolerance=0.2,
    anchor_threshold=0.5,
    group_size=G,
    tolerance=0.07,
    trim=False,
    verbose=False,
)


def _row(corpus, f_one=0.5, f_last=0.5, confusion=0.5, stability=0.5):
    return {
        "corpus": corpus,
        "f_one": f_one,
        "f_last": f_last,
        "confusion": confusion,
        "position_acc_best_offset": stability,
        "modal_offset": 0,
        "position_acc": 1.0,
    }


def _track(has_positions=True, n_beats=16):
    """A synthetic cached track with clean beat/one/last spikes on the grid."""

    beat_times = np.arange(n_beats) * PERIOD
    positions = np.array([(i % G) + 1 for i in range(n_beats)])
    n_frames = int(beat_times[-1] * FPS) + 10

    beat_p = np.full(n_frames, 0.02)
    one_p = np.full(n_frames, 0.02)
    last_p = np.full(n_frames, 0.02)

    for i, t in enumerate(beat_times):
        frame = int(round(t * FPS))
        beat_p[frame] = 0.95
        if positions[i] == 1:
            one_p[frame] = 0.95
        if positions[i] == G:
            last_p[frame] = 0.95

    probs = np.stack([beat_p, one_p, last_p])

    return (
        beat_times,
        positions if has_positions else None,
        has_positions,
        probs,
    )


class TestGroupByCorpus:
    def test_groups_rows_by_corpus(self):
        grouped = group_by_corpus([_row("ballroom"), _row("jtd"), _row("ballroom")])

        assert set(grouped) == {"ballroom", "jtd"}
        assert len(grouped["ballroom"]) == 2
        assert len(grouped["jtd"]) == 1

    def test_preserves_first_seen_order(self):
        grouped = group_by_corpus([_row("jtd"), _row("ballroom"), _row("jtd")])

        assert list(grouped) == ["jtd", "ballroom"]

    def test_untagged_rows_land_in_one_bucket(self):
        grouped = group_by_corpus([{"confusion": 0.1}, {"confusion": 0.2}])

        assert list(grouped) == [""]


class TestSummarize:
    def test_micro_is_the_per_track_mean(self):
        rows = [_row("a", confusion=0.0)] * 3 + [_row("b", confusion=1.0)]

        assert summarize(rows)["confusion"] == pytest.approx(0.25)

    def test_macro_ignores_corpus_size(self):
        """The point of the whole exercise: 3 easy tracks and 1 hard one from
        different corpora average to 0.25 per track but 0.5 per corpus."""

        rows = [_row("a", confusion=0.0)] * 3 + [_row("b", confusion=1.0)]
        summary = summarize(rows)

        assert summary["confusion"] == pytest.approx(0.25)
        assert summary["macro_confusion"] == pytest.approx(0.5)

    def test_reproduces_the_measured_merge_split(self):
        """The real numbers from plans/04 §3 #0, as a regression pin."""

        measured = [
            ("ballroom", 104, 0.271),
            ("jtd", 96, 0.278),
            ("rwc_popular", 19, 0.273),
            ("rwc_genre", 16, 0.330),
            ("rwc_classical", 7, 0.480),
            ("rwc_jazz", 7, 0.236),
        ]
        rows = [
            _row(corpus, confusion=value)
            for corpus, n, value in measured
            for _ in range(n)
        ]
        summary = summarize(rows)

        assert summary["n_tracks"] == 249
        assert summary["confusion"] == pytest.approx(0.283, abs=5e-4)
        assert summary["macro_confusion"] == pytest.approx(0.311, abs=5e-4)
        assert summary["worst_confusion"] == pytest.approx(0.480)

    def test_single_corpus_makes_macro_identical_to_micro(self):
        """Pins the no-op guarantee behind `--rank-by macro` being the default:
        on a single-corpus split it cannot change which decoder wins."""

        rows = [
            _row("ballroom", f_one=0.1, f_last=0.2, confusion=0.3, stability=0.4),
            _row("ballroom", f_one=0.7, f_last=0.8, confusion=0.9, stability=0.6),
            _row("ballroom", f_one=0.4, f_last=0.5, confusion=0.6, stability=0.5),
        ]
        summary = summarize(rows)

        for key in ("f_one", "f_last", "confusion", "position_acc_best_offset"):
            assert summary[f"macro_{key}"] == pytest.approx(summary[key])

    def test_equal_sized_corpora_make_macro_identical_to_micro(self):
        rows = [_row("a", confusion=0.2), _row("b", confusion=0.8)]
        summary = summarize(rows)

        assert summary["macro_confusion"] == pytest.approx(summary["confusion"])

    def test_worst_is_the_weakest_corpus_not_the_largest(self):
        rows = [_row("big", confusion=0.1)] * 50 + [_row("tiny", confusion=0.9)]
        summary = summarize(rows)

        assert summary["worst_confusion"] == pytest.approx(0.9)
        # ...and it is invisible in both means.
        assert summary["confusion"] < 0.12
        assert summary["macro_confusion"] == pytest.approx(0.5)

    def test_unscorable_corpus_does_not_poison_worst(self):
        rows = [
            _row("a", confusion=0.3),
            _row("b", confusion=float("nan")),
            _row("c", confusion=None),
        ]

        assert summarize(rows)["worst_confusion"] == pytest.approx(0.3)

    def test_empty_rows_are_nan_not_a_crash(self):
        summary = summarize([])

        assert summary["n_tracks"] == 0
        assert math.isnan(summary["confusion"])
        assert math.isnan(summary["macro_confusion"])
        assert math.isnan(summary["worst_confusion"])


class TestScoreTagsCorpora:
    """`BeatEvaluator.score()` is the single scoring path; these pin that it
    tags every row with the right corpus and never desyncs the two lists."""

    @staticmethod
    def _evaluator(cached, corpora):
        dataset = MagicMock()
        dataset.samples = [(f"{c}.wav", None, None, None) for c in corpora]
        dataset.refs = [MagicMock(dataset_name=c) for c in corpora]

        module = MagicMock()
        module.hparams = {}  # not a softmax bar-position checkpoint

        evaluator = BeatEvaluator(
            checkpoint="unused.ckpt", dataset="merge", **EVALUATOR_KW
        )
        evaluator._loaded = (module, "beat_phase", dataset, list(range(len(corpora))))
        evaluator._probs = cached  # skip inference entirely

        return evaluator

    def test_rows_carry_their_corpus(self):
        corpora = ["ballroom", "jtd", "jtd"]
        rows = self._evaluator([_track(), _track(), _track()], corpora).score()

        assert [r["corpus"] for r in rows] == corpora

    def test_unannotated_tracks_keep_their_beat_score_and_stay_aligned(self):
        """An unannotated track is still a scorable *beat* track, so it keeps
        its row rather than being dropped — only its position metrics are
        ``None``. Dropping it (as the old score_decoder did) also meant the
        row list no longer lined up with the track list."""

        cached = [
            _track(has_positions=True),
            _track(has_positions=False),
            _track(has_positions=True),
        ]
        rows = self._evaluator(cached, ["a", "b", "c"]).score()

        assert [r["corpus"] for r in rows] == ["a", "b", "c"]
        assert rows[1]["f_beat"] is not None
        assert rows[1]["position_acc"] is None
        # ...and it is skipped by the position aggregate rather than zeroing it.
        annotated = summarize([rows[0], rows[2]])["position_acc"]
        assert summarize(rows)["position_acc"] == pytest.approx(annotated)

    def test_rescoring_reuses_the_cached_probabilities(self):
        evaluator = self._evaluator([_track(), _track()], ["a", "b"])
        first = evaluator.score()
        second = evaluator.score(decoder="global", switch_penalty=2.0)

        assert len(first) == len(second) == 2
        assert evaluator._loaded[0].call_count == 0  # never re-ran the model


class TestTrackCorpora:
    """`track_corpora()` must line up with `compute_track_probs()` element for
    element — a silent misalignment would attribute scores to the wrong genre."""

    @staticmethod
    def _evaluator(corpora, **kwargs):
        dataset = MagicMock()
        dataset.samples = [
            (f"{c}_{i}.wav", np.array([0.5, 1.0]), None, False)
            for i, c in enumerate(corpora)
        ]
        dataset.refs = [MagicMock(dataset_name=c) for c in corpora]

        module = MagicMock(return_value=torch.zeros(1, 3, 8))
        module.hparams = {}  # not a softmax bar-position checkpoint

        evaluator = BeatEvaluator(checkpoint="unused.ckpt", dataset="merge", **kwargs)
        evaluator._loaded = (module, "beat_phase", dataset, list(range(len(corpora))))

        return evaluator, module

    def test_returns_the_corpus_of_each_selected_track(self):
        evaluator, _ = self._evaluator(["ballroom", "jtd", "jtd"])

        assert evaluator.track_corpora() == ["ballroom", "jtd", "jtd"]

    def test_stays_aligned_with_compute_track_probs(self):
        evaluator, _ = self._evaluator(["ballroom", "jtd", "rwc_classical"])

        with patch(
            "musicality.evaluation.load_track_waveform",
            return_value=torch.zeros(1000),
        ):
            cached = evaluator.compute_track_probs()

        assert len(cached) == len(evaluator.track_corpora())

    def test_respects_limit(self):
        evaluator, _ = self._evaluator(["ballroom", "jtd", "jtd"])
        # `limit` is applied inside load(); emulate the post-limit state.
        module, task, dataset, indices = evaluator._loaded
        evaluator._loaded = (module, task, dataset, indices[:2])

        assert evaluator.track_corpora() == ["ballroom", "jtd"]
