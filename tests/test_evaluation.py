"""Lightweight tests for musicality.evaluation.BeatEvaluator — orchestration
logic only (task-default resolution, split/limit handling, verbose printing).
``load_module``/``BeatDataset``/``indices_for_split``/``evaluate_track`` are
all mocked, so no real checkpoint, audio, or model inference happens.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from musicality.evaluation import DATA_DIR, DEFAULTS, BeatEvaluator


def _fake_dataset(n_tracks):
    dataset = MagicMock()
    dataset.samples = [
        (f"track_{i}.wav", [0.5 * j for j in range(4)], None, False)
        for i in range(n_tracks)
    ]
    return dataset


@contextmanager
def _mocked(task="beat_only", n_tracks=3, track_results=None):
    """Patches the four external calls BeatEvaluator.run() makes, yielding a
    dict of the resulting mocks."""

    results = track_results or [
        {"f_beat": 0.9, "f_one": None, "f_last": None, "confusion": None}
        for _ in range(n_tracks)
    ]
    with (
        patch(
            "musicality.evaluation.load_module",
            return_value=(MagicMock(name="module"), task),
        ) as load_module,
        patch(
            "musicality.evaluation.BeatDataset",
            return_value=_fake_dataset(n_tracks),
        ) as beat_dataset,
        patch(
            "musicality.evaluation.indices_for_split",
            return_value=list(range(n_tracks)),
        ) as indices_for_split,
        patch(
            "musicality.evaluation.evaluate_track", side_effect=results
        ) as evaluate_track,
    ):
        yield {
            "load_module": load_module,
            "BeatDataset": beat_dataset,
            "indices_for_split": indices_for_split,
            "evaluate_track": evaluate_track,
        }


class TestBeatEvaluatorDefaults:
    def test_beat_only_falls_back_to_task_defaults(self):
        with _mocked(task="beat_only") as mocks:
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

            args = mocks["evaluate_track"].call_args_list[0].args
            task_defaults = DEFAULTS["beat_only"]
            assert args[11] == task_defaults["beat_threshold"]
            assert args[12] == task_defaults["min_distance_frames"]
            assert args[13] == task_defaults["gate_tolerance"]
            assert args[14] == 0.5  # anchor_threshold — no beat_only-specific default
            assert args[15] == 4  # group_size — no beat_only-specific default

    def test_beat_phase_falls_back_to_task_defaults(self):
        with _mocked(task="beat_phase") as mocks:
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

            args = mocks["evaluate_track"].call_args_list[0].args
            task_defaults = DEFAULTS["beat_phase"]
            assert args[11] == task_defaults["beat_threshold"]
            assert args[12] == task_defaults["min_distance_frames"]
            assert args[13] == task_defaults["gate_tolerance"]
            assert args[14] == task_defaults["anchor_threshold"]
            assert args[15] == task_defaults["group_size"]

    def test_explicit_values_override_task_defaults(self):
        with _mocked(task="beat_phase") as mocks:
            BeatEvaluator(
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
                beat_threshold=0.42,
                min_distance_frames=7,
                gate_tolerance=0.33,
                anchor_threshold=0.66,
                group_size=8,
            ).run()

            args = mocks["evaluate_track"].call_args_list[0].args
            assert args[11:16] == (0.42, 7, 0.33, 0.66, 8)

    def test_group_size_threaded_into_dataset_construction(self):
        with _mocked(task="beat_phase") as mocks:
            BeatEvaluator(
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                verbose=False,
                group_size=8,
            ).run()

            assert mocks["BeatDataset"].call_args.kwargs["group_size"] == 8


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


class TestBeatEvaluatorLimit:
    def test_limit_truncates_indices(self):
        with _mocked(n_tracks=5) as mocks:
            evaluator = BeatEvaluator(
                checkpoint="fake.ckpt",
                dataset="ballroom",
                split="all",
                limit=2,
                verbose=False,
            )
            results = evaluator.run()

            assert mocks["evaluate_track"].call_count == 2
            assert len(results) == 2

    def test_no_limit_evaluates_every_index(self):
        with _mocked(n_tracks=5) as mocks:
            results = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

            assert mocks["evaluate_track"].call_count == 5
            assert len(results) == 5


class TestBeatEvaluatorReturnValue:
    def test_returns_evaluate_track_results_in_order(self):
        track_results = [
            {"f_beat": 0.1, "f_one": None, "f_last": None, "confusion": None},
            {"f_beat": 0.2, "f_one": None, "f_last": None, "confusion": None},
        ]
        with _mocked(n_tracks=2, track_results=track_results):
            results = BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

            assert results == track_results


class TestBeatEvaluatorVerbose:
    def test_silent_when_verbose_false(self, capsys):
        with _mocked():
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=False
            ).run()

        assert capsys.readouterr().out == ""

    def test_beat_only_prints_beat_line_only(self, capsys):
        with _mocked(task="beat_only"):
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=True
            ).run()

        out = capsys.readouterr().out
        assert "beat=" in out
        assert "one=" not in out
        assert "mean beat F-measure" in out

    def test_beat_phase_prints_position_lines_too(self, capsys):
        with _mocked(task="beat_phase"):
            BeatEvaluator(
                checkpoint="fake.ckpt", dataset="ballroom", split="all", verbose=True
            ).run()

        out = capsys.readouterr().out
        assert "one=" in out
        assert "last=" in out
        assert "confusion=" in out
        assert "mean '1' F-measure" in out
        assert "mean phase confusion" in out
