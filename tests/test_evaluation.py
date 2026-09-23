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
    SCORE_KEYS,
    BeatEvaluator,
    score_events,
    summarize,
    summary_block,
)


# BeatEvaluator holds no default knobs, so every construction below supplies
# these. Fixed values rather than the shipped config: what is under test is the
# resolution order, not what the project happens to be tuned to this week.
POSTPROCESS = {
    "beat_only": {
        "beat_threshold": 0.8,
        "min_distance_frames": 4,
        "gate_tolerance": 0.1,
    },
    "beat_phase": {
        "beat_threshold": 0.5,
        "min_distance_frames": 4,
        "gate_tolerance": 0.1,
        "group_size": 4,
        "decoder": "global",
        "switch_penalty": 2.0,
        "anchor_threshold": 0.8,
    },
}


# BeatEvaluator keeps no defaults of its own, so these have to be passed at
# every construction. Values match what the class used to assume, so the tests
# below measure the same thing they did before the defaults were removed.
RUN_KW = dict(
    postprocess=POSTPROCESS,
    sample_rate=22050,
    hop_length=512,
    tolerance=0.07,
    device="cpu",
    # Not required by the class — `from_module` never resolves a split — but
    # `load()` reads all three, and every test below that reaches it needs
    # them. `split` stays out: several tests set it themselves.
    val_split=0.2,
    binary_only=False,
)


def _fake_dataset(n_tracks, corpora=None):
    dataset = MagicMock()
    dataset.samples = [
        (f"track_{i}.wav", np.array([6.0, 6.5, 7.0, 7.5]), None, False)
        for i in range(n_tracks)
    ]
    names = corpora or ["ballroom"] * n_tracks
    dataset.refs = [MagicMock(dataset_name=name) for name in names]
    dataset.__len__.return_value = n_tracks

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
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
            ).resolve_postprocess()

            task_knobs = POSTPROCESS["beat_only"]
            assert knobs["beat_threshold"] == task_knobs["beat_threshold"]
            assert knobs["min_distance_frames"] == task_knobs["min_distance_frames"]
            assert knobs["gate_tolerance"] == task_knobs["gate_tolerance"]
            # The bar-position knobs describe a stage this task does not have.
            # Nothing invents a value for them, and every caller that reads
            # them is guarded on beat_phase.
            assert knobs["anchor_threshold"] is None
            assert knobs["group_size"] is None

    def test_beat_phase_falls_back_to_task_defaults(self):
        with _mocked(task="beat_phase"):
            knobs = BeatEvaluator(
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
            ).resolve_postprocess()

            task_knobs = POSTPROCESS["beat_phase"]
            for key in (
                "beat_threshold",
                "min_distance_frames",
                "gate_tolerance",
                "anchor_threshold",
                "group_size",
                "decoder",
                "switch_penalty",
            ):
                assert knobs[key] == task_knobs[key]

    def test_constructor_values_beat_task_defaults(self):
        with _mocked(task="beat_phase"):
            knobs = BeatEvaluator(
                **RUN_KW,
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
                **RUN_KW,
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
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
            )

            assert (
                evaluator.resolve_postprocess(switch_penalty=None)["switch_penalty"]
                is None
            )
            assert (
                evaluator.resolve_postprocess()["switch_penalty"]
                == POSTPROCESS["beat_phase"]["switch_penalty"]
            )


# ---------------------------------------------------------------------------
# BeatEvaluator orchestration
# ---------------------------------------------------------------------------


class TestSplitSettingsOnlyMatterWhenResolvingASplit:
    """`split`, `val_split` and `binary_only` default to "not supplied" rather
    than to a value, so the class states no opinion the config could disagree
    with. `load()` has to name the ones it needs, and `from_module` — which
    never resolves a split — must not need them at all."""

    _MINIMAL = dict(
        postprocess=POSTPROCESS,
        sample_rate=22050,
        hop_length=512,
        tolerance=0.07,
        device="cpu",
        verbose=False,
    )

    def test_load_names_every_missing_one(self):
        evaluator = BeatEvaluator(
            checkpoint="fake.ckpt", dataset="ballroom", **self._MINIMAL
        )

        with pytest.raises(ValueError, match="split, val_split, binary_only"):
            evaluator.load()

    def test_from_module_needs_none_of_them(self):
        module = MagicMock()
        module.hparams = {"task": "beat_phase", "group_size": 4}

        evaluator = BeatEvaluator.from_module(module, _fake_dataset(2), **self._MINIMAL)

        assert evaluator.load()[1] == "beat_phase"


class TestBeatEvaluatorDataHome:
    def test_defaults_to_data_dir_slash_dataset_name(self):
        with _mocked() as mocks:
            BeatEvaluator(
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
            ).run()

            assert mocks["BeatDataset"].call_args.kwargs["data_home"] == (
                DATA_DIR / "ballroom"
            )

    def test_explicit_data_home_is_used_verbatim(self):
        with _mocked() as mocks:
            BeatEvaluator(
                **RUN_KW,
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
                **RUN_KW,
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
                **RUN_KW,
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
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
            ).run()

            assert mocks["score_events"].call_count == 5
            assert len(results) == 5


class TestBeatEvaluatorReturnValue:
    def test_rows_carry_the_corpus_and_preserve_order(self):
        track_results = [_blank_row(f_beat=0.1), _blank_row(f_beat=0.2)]
        with _mocked(n_tracks=2, track_results=track_results, corpora=["a", "b"]):
            results = BeatEvaluator(
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
            ).run()

            assert [r["f_beat"] for r in results] == [0.1, 0.2]
            assert [r["corpus"] for r in results] == ["a", "b"]


class TestBeatEvaluatorMemoization:
    def test_load_memoized_across_repeated_calls(self):
        with _mocked(n_tracks=3) as mocks:
            evaluator = BeatEvaluator(
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
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
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
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
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
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
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
            ).compute_track_probs()

        probs = cached[0][3]
        assert probs.shape == (5, 10)
        # channels 1.. are a softmax over positions, so they sum to 1 per frame
        assert np.allclose(probs[1:].sum(axis=0), 1.0)


class TestBeatEvaluatorVerbose:
    def test_silent_when_verbose_false(self, capsys):
        with _mocked():
            BeatEvaluator(
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
            ).run()

        assert capsys.readouterr().out == ""

    def test_beat_only_omits_the_position_columns(self, capsys):
        with _mocked(task="beat_only"):
            BeatEvaluator(
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=True,
            ).run()

        out = capsys.readouterr().out
        assert "beat=" in out
        assert "pos=" not in out
        assert "f_beat" in out

    def test_beat_phase_prints_the_position_block(self, capsys):
        rows = [_blank_row(f_beat=0.9, position_acc=0.5, position_acc_best_offset=0.7)]
        with _mocked(task="beat_phase", n_tracks=1, track_results=rows):
            BeatEvaluator(
                **RUN_KW,
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=True,
            ).run()

        out = capsys.readouterr().out
        assert "pos=" in out
        assert "best=" in out
        assert "position_acc_best_offset" in out


class TestFromModule:
    """`from_module` is the seam a training run scores itself through
    (musicality/callbacks/event_metrics.py): an in-memory module and an
    already-built dataset, with checkpoint loading and split resolution
    skipped. Everything downstream must behave as if `load()` had run."""

    @staticmethod
    def _from_module(*args, **kwargs):
        return BeatEvaluator.from_module(*args, **RUN_KW, **kwargs)

    @staticmethod
    def _module(task="beat_phase", group_size=4):
        module = MagicMock()
        module.hparams = {"task": task, "group_size": group_size}

        return module

    def test_detects_the_task_from_the_modules_hyperparameters(self):
        evaluator = self._from_module(self._module(), _fake_dataset(2))

        _module, task, _dataset, _indices = evaluator.load()
        assert task == "beat_phase"

    def test_explicit_task_wins_over_detection(self):
        evaluator = self._from_module(
            self._module(task="beat_phase"), _fake_dataset(2), task="beat_only"
        )

        assert evaluator.load()[1] == "beat_only"

    def test_selects_every_track_in_the_dataset(self):
        evaluator = self._from_module(self._module(), _fake_dataset(5))

        assert evaluator.load()[3] == [0, 1, 2, 3, 4]

    def test_limit_still_applies(self):
        """`load()` is bypassed, so the limit has to be applied here or it is
        silently ignored."""

        evaluator = self._from_module(self._module(), _fake_dataset(5), limit=2)

        assert evaluator.load()[3] == [0, 1]

    def test_never_touches_the_checkpoint_loader(self):
        with patch("musicality.evaluation.load_module") as load:
            evaluator = self._from_module(self._module(), _fake_dataset(2))
            evaluator.load()

        load.assert_not_called()

    def test_constructor_settings_still_reach_postprocessing(self):
        evaluator = self._from_module(
            self._module(), _fake_dataset(2), group_size=8, beat_threshold=0.42
        )
        knobs = evaluator.resolve_postprocess()

        assert knobs["group_size"] == 8
        assert knobs["beat_threshold"] == 0.42
        # ...and everything left unset still falls back to the block the
        # caller handed over.
        assert knobs["decoder"] == POSTPROCESS["beat_phase"]["decoder"]

    def test_track_corpora_still_line_up(self):
        evaluator = self._from_module(
            self._module(), _fake_dataset(3, corpora=["ballroom", "jtd", "jtd"])
        )

        assert evaluator.track_corpora() == ["ballroom", "jtd", "jtd"]
