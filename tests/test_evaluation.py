"""Tests for musicality.evaluation — the single scoring path.

Two layers:

- :func:`score_events`, :func:`summarize` and :meth:`BeatEvaluator.resolve_postprocess`
  are tested directly on synthetic data, since they are pure.
- ``BeatEvaluator``'s orchestration (task-default resolution, split/limit
  handling, memoization, verbose printing) is tested with
  ``load_module``/``BeatDataset``/``indices_for_split``/``load_track_waveform``/
  ``score_events`` mocked, so no real checkpoint, audio, or inference happens.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from musicality.evaluation import (
    DATA_DIR,
    DEFAULTS,
    SCORE_KEYS,
    BeatEvaluator,
    score_events,
    summarize,
    summary_block,
)


def _fake_dataset(n_tracks, corpora=None):
    dataset = MagicMock()
    dataset.samples = [
        (f"track_{i}.wav", np.array([6.0, 6.5, 7.0, 7.5]), None, False)
        for i in range(n_tracks)
    ]
    names = corpora or ["ballroom"] * n_tracks
    dataset.refs = [MagicMock(dataset_name=name) for name in names]

    return dataset


def _blank_row(**overrides):
    row = dict.fromkeys(SCORE_KEYS)
    row["modal_offset"] = None

    return {**row, **overrides}


@contextmanager
def _mocked(task="beat_only", n_tracks=3, track_results=None, corpora=None):
    """Patch everything BeatEvaluator.score() reaches outside itself."""

    results = track_results or [_blank_row(f_beat=0.9) for _ in range(n_tracks)]

    n_frames = 10
    shape = (1, 3, n_frames) if task == "beat_phase" else (1, n_frames)
    module = MagicMock(return_value=torch.randn(*shape))
    # A plain dict, not a MagicMock attribute: `.get("group_size")` on a
    # MagicMock returns a truthy MagicMock, which would silently route the
    # decode down the softmax-head branch.
    module.hparams = {}

    with (
        patch(
            "musicality.evaluation.load_module", return_value=(module, task)
        ) as load_module,
        patch(
            "musicality.evaluation.BeatDataset",
            return_value=_fake_dataset(n_tracks, corpora),
        ) as beat_dataset,
        patch(
            "musicality.evaluation.indices_for_split",
            return_value=list(range(n_tracks)),
        ) as indices_for_split,
        patch(
            "musicality.evaluation.load_track_waveform",
            return_value=torch.zeros(1, 1000),
        ),
        patch(
            "musicality.evaluation.score_events", side_effect=results
        ) as score_events_mock,
    ):
        yield {
            "load_module": load_module,
            "BeatDataset": beat_dataset,
            "indices_for_split": indices_for_split,
            "score_events": score_events_mock,
            "module": module,
        }


# ---------------------------------------------------------------------------
# score_events
# ---------------------------------------------------------------------------


def _labelled(times, labels):
    return [{"time": t, "beat_in_bar": p} for t, p in zip(times, labels)]


class TestScoreEvents:
    def test_always_returns_the_full_key_set(self):
        times = np.arange(6.0, 14.0, 0.5)
        events = _labelled(times, [None] * len(times))
        row = score_events(times, None, False, events)

        for key in SCORE_KEYS:
            assert key in row
        assert "modal_offset" in row

    def test_perfect_beat_only_scores_beat_and_continuity(self):
        times = np.arange(6.0, 20.0, 0.5)
        events = _labelled(times, [None] * len(times))
        row = score_events(times, None, False, events)

        assert row["f_beat"] == pytest.approx(1.0)
        assert row["cmlt"] == pytest.approx(1.0)
        assert row["amlt"] == pytest.approx(1.0)

    def test_unlabelled_events_leave_position_keys_none(self):
        # The load-bearing property: a beat-only checkpoint has no bar
        # positions to get wrong, and scoring that as 0.0 would drag every
        # aggregate down.
        times = np.arange(6.0, 20.0, 0.5)
        positions = np.array([(i % 4) + 1 for i in range(len(times))])
        events = _labelled(times, [None] * len(times))

        row = score_events(times, positions, True, events)

        for key in ("f_one", "f_last", "confusion", "position_acc"):
            assert row[key] is None

    def test_missing_reference_positions_leave_position_keys_none(self):
        times = np.arange(6.0, 20.0, 0.5)
        positions = np.array([(i % 4) + 1 for i in range(len(times))])
        events = _labelled(times, positions)

        assert score_events(times, positions, False, events)["position_acc"] is None

    def test_perfect_beat_phase_scores_everything(self):
        times = np.arange(6.0, 20.0, 0.5)
        positions = np.array([(i % 4) + 1 for i in range(len(times))])
        row = score_events(times, positions, True, _labelled(times, positions))

        assert row["position_acc"] == pytest.approx(1.0)
        assert row["position_acc_best_offset"] == pytest.approx(1.0)
        assert row["anchor_error"] == pytest.approx(0.0)
        assert row["confusion"] == pytest.approx(0.0)
        assert row["modal_offset"] == 0

    def test_whole_track_rotation_shows_up_as_anchor_error(self):
        times = np.arange(6.0, 20.0, 0.5)
        positions = np.array([(i % 4) + 1 for i in range(len(times))])
        rotated = np.array([((i + 1) % 4) + 1 for i in range(len(times))])

        row = score_events(times, positions, True, _labelled(times, rotated))

        assert row["position_acc"] == pytest.approx(0.0)
        assert row["position_acc_best_offset"] == pytest.approx(1.0)
        assert row["anchor_error"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_macro_weights_corpora_equally_micro_does_not(self):
        rows = [{"corpus": "big", "position_acc": 1.0} for _ in range(9)]
        rows.append({"corpus": "small", "position_acc": 0.0})
        summary = summarize(rows)

        assert summary["position_acc"] == pytest.approx(0.9)  # micro
        assert summary["macro_position_acc"] == pytest.approx(0.5)  # macro

    def test_worst_corpus_is_named(self):
        rows = [
            {"corpus": "a", "position_acc": 0.9},
            {"corpus": "b", "position_acc": 0.2},
        ]
        summary = summarize(rows)

        assert summary["worst_corpus"] == "b"
        assert summary["worst_position_acc"] == pytest.approx(0.2)

    def test_all_none_gives_nan_not_a_crash(self):
        summary = summarize([{"corpus": "a", "position_acc": None}])

        assert np.isnan(summary["position_acc"])
        assert summary["worst_corpus"] is None

    def test_summary_block_renders_every_headline(self):
        text = summary_block(summarize([_blank_row(f_beat=0.5) | {"corpus": "a"}]))

        for token in ("f_beat", "cmlt / amlt", "position_acc", "anchor_error"):
            assert token in text


# ---------------------------------------------------------------------------
# resolve_postprocess
# ---------------------------------------------------------------------------


class TestResolvePostprocess:
    def test_beat_only_falls_back_to_task_defaults(self):
        with _mocked(task="beat_only"):
            knobs = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).resolve_postprocess()

            task_defaults = DEFAULTS["beat_only"]
            assert knobs["beat_threshold"] == task_defaults["beat_threshold"]
            assert knobs["min_distance_frames"] == task_defaults["min_distance_frames"]
            assert knobs["gate_tolerance"] == task_defaults["gate_tolerance"]
            assert knobs["anchor_threshold"] == 0.5  # no beat_only-specific default
            assert knobs["group_size"] == 4

    def test_beat_phase_falls_back_to_task_defaults(self):
        with _mocked(task="beat_phase"):
            knobs = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).resolve_postprocess()

            task_defaults = DEFAULTS["beat_phase"]
            for key in (
                "beat_threshold",
                "min_distance_frames",
                "gate_tolerance",
                "anchor_threshold",
                "group_size",
                "decoder",
                "switch_penalty",
            ):
                assert knobs[key] == task_defaults[key]

    def test_constructor_values_beat_task_defaults(self):
        with _mocked(task="beat_phase"):
            knobs = BeatEvaluator(
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
                beat_threshold=0.42,
                min_distance_frames=7,
                gate_tolerance=0.33,
                anchor_threshold=0.66,
                group_size=8,
            ).resolve_postprocess()

            assert knobs["beat_threshold"] == 0.42
            assert knobs["min_distance_frames"] == 7
            assert knobs["gate_tolerance"] == 0.33
            assert knobs["anchor_threshold"] == 0.66
            assert knobs["group_size"] == 8

    def test_explicit_override_beats_constructor(self):
        with _mocked(task="beat_phase"):
            knobs = BeatEvaluator(
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
                beat_threshold=0.42,
            ).resolve_postprocess(beat_threshold=0.11)

            assert knobs["beat_threshold"] == 0.11

    def test_explicit_none_switch_penalty_selects_the_exact_decode(self):
        # None is a meaningful switch_penalty (no mid-track resync allowed), so
        # overrides are keyed by presence rather than by value. Omitting the
        # key must fall through to the tuned default instead.
        with _mocked(task="beat_phase"):
            evaluator = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            )

            assert (
                evaluator.resolve_postprocess(switch_penalty=None)["switch_penalty"]
                is None
            )
            assert (
                evaluator.resolve_postprocess()["switch_penalty"]
                == DEFAULTS["beat_phase"]["switch_penalty"]
            )


# ---------------------------------------------------------------------------
# BeatEvaluator orchestration
# ---------------------------------------------------------------------------


class TestBeatEvaluatorDataHome:
    def test_defaults_to_data_dir_slash_dataset_name(self):
        with _mocked() as mocks:
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

            assert mocks["BeatDataset"].call_args.kwargs["data_home"] == (
                DATA_DIR / "ballroom"
            )

    def test_explicit_data_home_is_used_verbatim(self):
        with _mocked() as mocks:
            BeatEvaluator(
                checkpoint="fake.ckpt",
                dataset="ballroom",
                data_home="/some/custom/path",
                split="all",
                verbose=False,
            ).run()

            assert mocks["BeatDataset"].call_args.kwargs["data_home"] == Path(
                "/some/custom/path"
            )

    def test_group_size_threaded_into_dataset_construction(self):
        with _mocked() as mocks:
            BeatEvaluator(
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
                group_size=8,
            ).run()

            assert mocks["BeatDataset"].call_args.kwargs["group_size"] == 8


class TestBeatEvaluatorLimit:
    def test_limit_truncates_indices(self):
        with _mocked(n_tracks=5) as mocks:
            results = BeatEvaluator(
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                limit=2,
                verbose=False,
            ).run()

            assert mocks["score_events"].call_count == 2
            assert len(results) == 2

    def test_no_limit_evaluates_every_index(self):
        with _mocked(n_tracks=5) as mocks:
            results = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

            assert mocks["score_events"].call_count == 5
            assert len(results) == 5


class TestBeatEvaluatorReturnValue:
    def test_rows_carry_the_corpus_and_preserve_order(self):
        track_results = [_blank_row(f_beat=0.1), _blank_row(f_beat=0.2)]
        with _mocked(n_tracks=2, track_results=track_results, corpora=["a", "b"]):
            results = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

            assert [r["f_beat"] for r in results] == [0.1, 0.2]
            assert [r["corpus"] for r in results] == ["a", "b"]


class TestBeatEvaluatorMemoization:
    def test_load_memoized_across_repeated_calls(self):
        with _mocked(n_tracks=3) as mocks:
            evaluator = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            )
            evaluator.load()
            evaluator.load()

            assert mocks["load_module"].call_count == 1
            assert mocks["BeatDataset"].call_count == 1
            assert mocks["indices_for_split"].call_count == 1

    def test_probs_computed_once_across_repeated_scores(self):
        # This is what makes scoring N decoder variants cost one model pass.
        results = [_blank_row(f_beat=0.9) for _ in range(9)]
        with _mocked(n_tracks=3, track_results=results) as mocks:
            evaluator = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            )
            evaluator.score()
            evaluator.score()
            evaluator.score()

            assert mocks["module"].call_count == 3  # 3 tracks, once each
            assert mocks["score_events"].call_count == 9  # 3 tracks x 3 scorings


class TestBeatEvaluatorComputeTrackProbs:
    def _cached(self, task, n_frames=10, n_tracks=2):
        logits_shape = (1, 3, n_frames) if task == "beat_phase" else (1, n_frames)
        module = MagicMock(return_value=torch.randn(*logits_shape))
        module.hparams = {}

        with (
            patch("musicality.evaluation.load_module", return_value=(module, task)),
            patch(
                "musicality.evaluation.BeatDataset",
                return_value=_fake_dataset(n_tracks),
            ),
            patch(
                "musicality.evaluation.indices_for_split",
                return_value=list(range(n_tracks)),
            ),
            patch(
                "musicality.evaluation.load_track_waveform",
                return_value=torch.zeros(1, 1000),
            ),
        ):
            evaluator = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            )
            return evaluator.compute_track_probs()

    def test_beat_only_returns_raw_1d_probs(self):
        cached = self._cached(task="beat_only", n_frames=10)

        assert len(cached) == 2
        _beat_times, _positions, has_positions, probs = cached[0]
        assert probs.shape == (10,)
        assert has_positions is False

    def test_beat_phase_keeps_all_channels(self):
        cached = self._cached(task="beat_phase", n_frames=10)

        _beat_times, _positions, _has_positions, probs = cached[0]
        assert probs.shape == (3, 10)

    def test_softmax_head_concatenates_beat_and_position_block(self):
        module = MagicMock(return_value=torch.randn(1, 5, 10))
        module.hparams = {"group_size": 4}

        with (
            patch(
                "musicality.evaluation.load_module", return_value=(module, "beat_phase")
            ),
            patch("musicality.evaluation.BeatDataset", return_value=_fake_dataset(1)),
            patch("musicality.evaluation.indices_for_split", return_value=[0]),
            patch(
                "musicality.evaluation.load_track_waveform",
                return_value=torch.zeros(1, 1000),
            ),
        ):
            cached = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).compute_track_probs()

        probs = cached[0][3]
        assert probs.shape == (5, 10)
        # channels 1.. are a softmax over positions, so they sum to 1 per frame
        assert np.allclose(probs[1:].sum(axis=0), 1.0)


class TestBeatEvaluatorVerbose:
    def test_silent_when_verbose_false(self, capsys):
        with _mocked():
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

        assert capsys.readouterr().out == ""

    def test_beat_only_omits_the_position_columns(self, capsys):
        with _mocked(task="beat_only"):
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=True
            ).run()

        out = capsys.readouterr().out
        assert "beat=" in out
        assert "pos=" not in out
        assert "f_beat" in out

    def test_beat_phase_prints_the_position_block(self, capsys):
        rows = [_blank_row(f_beat=0.9, position_acc=0.5, position_acc_best_offset=0.7)]
        with _mocked(task="beat_phase", n_tracks=1, track_results=rows):
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=True
            ).run()

        out = capsys.readouterr().out
        assert "pos=" in out
        assert "best=" in out
        assert "position_acc_best_offset" in out
