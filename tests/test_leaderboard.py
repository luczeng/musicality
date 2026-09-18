"""What `tools/leaderboard.py` does around the evaluator: which checkpoint
represents a run, which grid points get swept, and how the board is ordered.

Scoring itself is not exercised here (tests/test_evaluation.py and
tests/test_metrics.py cover it) — the evaluator is a stub returning canned
rows, the same way tests/test_eval_beat_cli.py stubs it.
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from musicality.dataformats.track_io import TrackRef
from tools.leaderboard import (
    BOARD_COLUMNS,
    DEFAULT_BOARD,
    board_settings,
    build_payload,
    carried_rows,
    check_comparable,
    board_name,
    dvc,
    find_runs,
    load_board,
    parse_args,
    publish,
    pull_board,
    rank_rows,
    run_checkpoints,
    sweep_evaluator,
    sweep_knobs,
)


class _StubEvaluator:
    """Returns a canned row set per `score` call, recording every call's knobs."""

    def __init__(self, per_call, decoder="global"):
        self.per_call = list(per_call)
        self.calls = []
        self.decoder = decoder

    def score(self, **kwargs):
        self.calls.append(kwargs)

        return self.per_call.pop(0)

    def resolve_postprocess(self, **overrides):
        return {"decoder": self.decoder, **overrides}


def _rows(**metrics) -> list[dict]:
    return [{"corpus": "ballroom", **metrics}]


def _grid(monkeypatch, **overrides):
    """A two-point grid in place of the shipped one, so a test can count calls."""

    grid = {
        "beat_thresholds": [0.4, 0.8],
        "min_distance_frames": [2],
        "gate_tolerances": [0.1],
        "anchor_thresholds": [0.5, 0.9],
        "switch_penalties": [2.0],
        **overrides,
    }
    monkeypatch.setattr("tools.leaderboard.SWEEP_DEFAULTS", grid)

    return grid


def _touch(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("")

    return directory


class TestRunCheckpoints:
    """`save_top_k` leaves three files of one run side by side; a hand-named
    folder holds several different models. Reading one as the other either
    triples the board or drops five models from it."""

    def test_a_save_top_k_group_is_one_run_at_its_best_loss(self, tmp_path):
        run = _touch(
            tmp_path / "20260917-204539",
            "beat-phase-epoch83-valloss1.3927.ckpt",
            "beat-phase-epoch103-valloss1.3894.ckpt",
            "beat-phase-epoch107-valloss1.3908.ckpt",
        )

        ((label, checkpoint),) = run_checkpoints(run)

        assert label == str(run)
        assert checkpoint.name == "beat-phase-epoch103-valloss1.3894.ckpt"

    def test_hand_named_checkpoints_are_one_run_each(self, tmp_path):
        run = _touch(tmp_path / "checkpoints", "merge_v5.ckpt", "checkpoint_v6.ckpt")

        found = run_checkpoints(run)

        assert sorted(c.name for _label, c in found) == [
            "checkpoint_v6.ckpt",
            "merge_v5.ckpt",
        ]

    def test_a_directory_without_checkpoints_contributes_nothing(self, tmp_path):
        assert run_checkpoints(_touch(tmp_path / "empty")) == []


class TestFindRuns:
    def test_a_checkpoint_path_is_taken_as_given(self, tmp_path):
        _touch(tmp_path, "merge_v5.ckpt")

        found = find_runs([tmp_path / "merge_v5.ckpt"])

        assert [c.name for _label, c in found] == ["merge_v5.ckpt"]

    def test_a_directory_expands_to_one_entry_per_nested_run(self, tmp_path):
        """A sweep directory holds one run per learning rate, two levels down —
        so the board can be asked for the whole sweep by naming it once."""

        sweep = tmp_path / "lr_sweep-20260918-141530"
        _touch(sweep / "lr_0.0008", "beat-phase-epoch91-valloss1.4794.ckpt")
        _touch(sweep / "lr_0.002", "beat-phase-epoch95-valloss1.4757.ckpt")

        found = find_runs([sweep])

        assert [Path(label).name for label, _c in found] == ["lr_0.0008", "lr_0.002"]


class TestSweepKnobs:
    def _args(self, **overrides):
        return SimpleNamespace(
            rank_by="micro",
            rank_metric="f_beat",
            **{"sweep_rank_metric": "position_acc", **overrides},
        )

    def test_the_position_stage_ignores_f_beat(self, monkeypatch):
        """A bar-position decoder relabels beats without moving them, so every
        stage-2 candidate scores the same f_beat. Ranking stage 2 by the
        board's own --rank-metric (f_beat by default) is therefore a tie that
        the sort breaks by candidate order, pinning switch_penalty to the first
        value in the list — measurably worse: 0.73 against 0.88 position_acc on
        six ballroom tracks."""

        _grid(monkeypatch)
        evaluator = _StubEvaluator(
            [_rows(f_beat=0.5, position_acc=0.1), _rows(f_beat=0.9, position_acc=0.2)]
            # Stage 2: identical f_beat, as it always is in reality.
            + [_rows(f_beat=0.9, position_acc=p) for p in (0.3, 0.9)]
        )

        knobs = sweep_knobs(evaluator, "beat_phase", 4, self._args())

        assert knobs["switch_penalty"] == 2.0

    def test_beat_only_stops_after_the_beat_stage(self, monkeypatch):
        """A beat-only checkpoint has no bar-position heads, so sweeping a
        bar-position knob would tune something nothing reads."""

        _grid(monkeypatch)
        evaluator = _StubEvaluator([_rows(f_beat=0.5), _rows(f_beat=0.9)])

        knobs = sweep_knobs(evaluator, "beat_only", 4, self._args())

        assert len(evaluator.calls) == 2
        assert knobs == {
            "beat_threshold": 0.8,
            "min_distance_frames": 2,
            "gate_tolerance": 0.1,
        }

    def test_the_global_decoder_always_considers_the_no_resync_decode(
        self, monkeypatch
    ):
        """`switch_penalty=None` forbids mid-track resyncs — the penalty ->
        infinity limit that no finite candidate in the grid reaches."""

        _grid(monkeypatch)
        evaluator = _StubEvaluator(
            [_rows(f_beat=0.5, position_acc=0.1), _rows(f_beat=0.9, position_acc=0.2)]
            + [_rows(f_beat=0.9, position_acc=p) for p in (0.8, 0.3)]
        )

        knobs = sweep_knobs(evaluator, "beat_phase", 4, self._args())

        assert [call.get("switch_penalty") for call in evaluator.calls[2:]] == [
            None,
            2.0,
        ]
        assert knobs["switch_penalty"] is None

    def test_the_greedy_decoder_sweeps_its_own_knob_instead(self, monkeypatch):
        _grid(monkeypatch)
        evaluator = _StubEvaluator(
            [_rows(f_beat=0.9, position_acc=p) for p in (0.1, 0.2, 0.3, 0.9)],
            decoder="greedy",
        )

        knobs = sweep_knobs(evaluator, "beat_phase", 4, self._args())

        assert knobs["anchor_threshold"] == 0.9
        assert "switch_penalty" not in knobs

    def test_the_beat_stage_winner_is_held_fixed(self, monkeypatch):
        _grid(monkeypatch)
        evaluator = _StubEvaluator(
            [_rows(f_beat=0.9, position_acc=0.5), _rows(f_beat=0.5, position_acc=0.9)]
            + [_rows(f_beat=0.9, position_acc=0.5) for _ in range(2)]
        )

        sweep_knobs(evaluator, "beat_phase", 4, self._args())

        # 0.4 won stage 1 on f_beat; stage 2 must not re-open that choice just
        # because the other threshold scored better on the position metric.
        assert all(call["beat_threshold"] == 0.4 for call in evaluator.calls[2:])


class TestSweepEvaluator:
    """The knobs are tuned on their own tracks, and on a spread of corpora."""

    def _dataset(self, counts: dict):
        refs = [
            TrackRef(dataset_name=corpus, track_id=f"{corpus}-{i}", data_home=Path("."))
            for corpus, n in counts.items()
            for i in range(n)
        ]

        return SimpleNamespace(refs=refs)

    def _args(self, **overrides):
        return SimpleNamespace(
            dataset="merge",
            data_home=None,
            sweep_split="train",
            val_split=0.2,
            sample_rate=22050,
            hop_length=512,
            binary_only=False,
            tolerance=0.07,
            device="cpu",
            **{"sweep_tracks": 4, **overrides},
        )

    def _patched(self, monkeypatch, dataset):
        """`BeatEvaluator` replaced by a stub that loads *dataset* whole."""

        class _Stub:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._loaded = (
                    "module",
                    "beat_phase",
                    dataset,
                    list(range(len(dataset.refs))),
                )

            def load(self):
                return self._loaded

        monkeypatch.setattr("tools.leaderboard.BeatEvaluator", _Stub)

        return _Stub

    def test_it_reads_the_sweep_split_not_the_reported_one(self, monkeypatch):
        self._patched(monkeypatch, self._dataset({"ballroom": 2}))

        evaluator = sweep_evaluator(Path("a.ckpt"), self._args(), 4)

        assert evaluator.kwargs["split"] == "train"

    def test_the_subsample_spreads_across_corpora(self, monkeypatch):
        """A split file is written corpus by corpus, so taking the first N
        would tune every knob on whichever corpus was written first."""

        dataset = self._dataset({"ballroom": 6, "rwc_classical": 2})
        self._patched(monkeypatch, dataset)

        evaluator = sweep_evaluator(Path("a.ckpt"), self._args(), 4)
        corpora = [dataset.refs[i].dataset_name for i in evaluator.load()[3]]

        assert sorted(corpora) == [
            "ballroom",
            "ballroom",
            "rwc_classical",
            "rwc_classical",
        ]

    def test_the_narrowed_indices_still_point_into_the_same_dataset(self, monkeypatch):
        """Indices are narrowed, not the dataset — everything downstream walks
        `indices` against the dataset the evaluator loaded."""

        dataset = self._dataset({"ballroom": 5})
        self._patched(monkeypatch, dataset)

        evaluator = sweep_evaluator(Path("a.ckpt"), self._args(sweep_tracks=2), 4)
        _module, _task, loaded, indices = evaluator.load()

        assert loaded is dataset
        assert len(indices) == 2
        assert all(0 <= i < len(dataset.refs) for i in indices)

    def test_a_subsample_larger_than_the_split_keeps_every_track(self, monkeypatch):
        dataset = self._dataset({"ballroom": 3})
        self._patched(monkeypatch, dataset)

        evaluator = sweep_evaluator(Path("a.ckpt"), self._args(sweep_tracks=50), 4)

        assert len(evaluator.load()[3]) == 3


class TestRunningBoard:
    """`--append` keeps one board growing across invocations. What it must get
    right: which rows survive, and refusing to mix rows that were never
    measured the same way."""

    def _args(self, **overrides):
        return SimpleNamespace(
            dataset="ballroom",
            split="val",
            val_split=0.2,
            sample_rate=22050,
            hop_length=512,
            tolerance=0.07,
            binary_only=False,
            group_size=None,
            limit=None,
            sweep=True,
            sweep_split="train",
            sweep_tracks=50,
            rank_metric="f_beat",
            sweep_rank_metric="position_acc",
            rank_by="macro",
            commit="abc1234",
            **overrides,
        )

    def _previous(self, *runs, **eval_overrides):
        return {
            "leaderboard": [{"run": run, "f_beat": 0.5} for run in runs],
            "per_corpus": {run: {"ballroom": {}} for run in runs},
            "eval": {**board_settings(self._args()), **eval_overrides},
        }

    def test_appending_is_the_default(self, monkeypatch):
        """A board is only useful once it has more than one row on it, so
        extending one is the behaviour you get without asking."""

        monkeypatch.setattr(sys, "argv", ["leaderboard.py", "checkpoints_deeper"])

        assert parse_args().append == DEFAULT_BOARD

    def test_no_append_opts_out(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["leaderboard.py", "checkpoints_deeper", "--no-append"]
        )

        assert parse_args().append is None

    def test_an_explicit_path_wins(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["leaderboard.py", "checkpoints_deeper", "--append", "boards/x.json"],
        )

        assert parse_args().append == Path("boards/x.json")

    def test_a_missing_file_starts_an_empty_board(self, tmp_path):
        """So the first invocation of a running board is the same command as
        every later one."""

        board = load_board(tmp_path / "nothing-here.json")

        assert board["leaderboard"] == []

    def test_an_existing_board_is_read_back(self, tmp_path):
        path = tmp_path / "board.json"
        path.write_text(json.dumps(self._previous("a", "b")))

        assert len(load_board(path)["leaderboard"]) == 2

    def test_rows_not_re_measured_are_carried(self):
        carried = carried_rows(self._previous("a", "b"), [{"run": "b"}])

        assert [row["run"] for row in carried] == ["a"]

    def test_a_re_measured_run_is_replaced_not_duplicated(self):
        """Identity is the run label: re-running a folder after more training
        picks a different epoch's file, and that is the same experiment with a
        better number rather than a second entry."""

        carried = carried_rows(self._previous("a"), [{"run": "a"}])

        assert carried == []

    def test_incomparable_settings_are_refused(self):
        previous = self._previous("a", tolerance=0.05)

        with pytest.raises(SystemExit, match="tolerance"):
            check_comparable(
                previous["eval"], board_settings(self._args()), previous["leaderboard"]
            )

    def test_re_measuring_everything_lifts_the_refusal(self):
        """Nothing survives from the old board, so its settings are no longer a
        claim about anything — which is what lets the split be changed without
        needing a flag to override the check."""

        previous = self._previous("a", tolerance=0.05)

        check_comparable(previous["eval"], board_settings(self._args()), [])

    def test_sweep_settings_alone_do_not_block_an_append(self):
        """They are recorded per row (`swept_on` and the knobs themselves), so
        a swept and an unswept row are told apart by reading them."""

        previous = self._previous("a", sweep_split="val", swept=False)

        check_comparable(
            previous["eval"], board_settings(self._args()), previous["leaderboard"]
        )


class TestDvcSync:
    """The board lives in the DVC-tracked data repo so it outlives the machine
    that wrote it. Syncing must never take a run down with it: a board that
    cannot be pushed is still a board."""

    def _board(self, tmp_path, dvc_repo=True, pointer=False):
        repo = tmp_path / "musicality_db"
        (repo / "leaderboard").mkdir(parents=True)

        if dvc_repo:
            (repo / ".dvc").mkdir()
        if pointer:
            (repo / "leaderboard.dvc").write_text("")

        return repo / "leaderboard" / "leaderboard.json"

    def test_the_dvc_target_is_the_folder_not_the_file(self, tmp_path):
        """`leaderboard.dvc` tracks the directory, the way `splits.dvc` does."""

        assert board_name(self._board(tmp_path)) == "leaderboard"

    def test_a_non_dvc_directory_is_skipped_not_fatal(self, tmp_path, capsys):
        path = self._board(tmp_path, dvc_repo=False)

        assert dvc(["push", "leaderboard"], path) is False
        assert "not a DVC repo" in capsys.readouterr().out

    def test_the_first_run_does_not_pull(self, tmp_path, monkeypatch, capsys):
        """With no pointer yet there is nothing to fetch, and a doomed `dvc
        pull` prints an error that reads like a failure when nothing is wrong."""

        def _boom(*args, **kwargs):
            raise AssertionError("dvc pull should not have run")

        monkeypatch.setattr("tools.leaderboard.subprocess.run", _boom)

        pull_board(self._board(tmp_path, pointer=False))

        assert "starting a new board" in capsys.readouterr().out

    def test_an_existing_pointer_is_pulled(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "tools.leaderboard.subprocess.run",
            lambda cmd, **kwargs: (
                calls.append(cmd) or SimpleNamespace(returncode=0, stderr="")
            ),
        )

        pull_board(self._board(tmp_path, pointer=True))

        assert calls == [["dvc", "pull", "leaderboard"]]

    def test_a_failed_add_does_not_push(self, tmp_path, monkeypatch, capsys):
        """Pushing content whose pointer was never rewritten would upload the
        previous board under the new commit."""

        monkeypatch.setattr("tools.leaderboard.dvc", lambda command, path: False)

        publish(self._board(tmp_path))

        assert "git commit" not in capsys.readouterr().out

    def test_a_successful_push_says_what_to_commit(self, tmp_path, monkeypatch, capsys):
        """The pointer only becomes the shared truth once committed, and
        committing in someone else's repo is not this tool's call."""

        monkeypatch.setattr("tools.leaderboard.dvc", lambda command, path: True)

        publish(self._board(tmp_path))

        assert "git add leaderboard.dvc" in capsys.readouterr().out


class TestRankRows:
    def _rows(self):
        return [
            {"run": "a", "f_beat": 0.7, "confusion": 0.4},
            {"run": "b", "f_beat": 0.9, "confusion": 0.1},
            {"run": "c", "f_beat": 0.8, "confusion": 0.2},
        ]

    def test_higher_is_better_metrics_rank_descending(self):
        ranked = rank_rows(self._rows(), "f_beat", "micro")

        assert [row["run"] for row in ranked] == ["b", "c", "a"]

    def test_confusion_ranks_ascending(self):
        """The one lower-is-better member of the canonical set."""

        ranked = rank_rows(self._rows(), "confusion", "micro")

        assert [row["run"] for row in ranked] == ["b", "c", "a"]

    def test_macro_ranks_on_the_macro_mean(self):
        rows = [
            {"run": "a", "f_beat": 0.9, "macro_f_beat": 0.4},
            {"run": "b", "f_beat": 0.5, "macro_f_beat": 0.8},
        ]

        assert [r["run"] for r in rank_rows(rows, "f_beat", "macro")] == ["b", "a"]

    def test_an_unscorable_run_ranks_last_rather_than_first(self):
        """`jsonable` has already turned NaN into None by this point, and
        `None` must not be read as a winning score."""

        rows = [{"run": "a", "f_beat": None}, {"run": "b", "f_beat": 0.1}]

        assert [row["run"] for row in rank_rows(rows, "f_beat", "micro")] == ["b", "a"]


class TestBuildPayload:
    def _args(self):
        return SimpleNamespace(
            dataset="ballroom",
            split="val",
            val_split=0.2,
            sample_rate=22050,
            hop_length=512,
            tolerance=0.07,
            binary_only=True,
            group_size=None,
            limit=None,
            sweep=True,
            sweep_split="train",
            sweep_tracks=50,
            rank_metric="f_beat",
            sweep_rank_metric="position_acc",
            rank_by="macro",
            commit="abc1234",
        )

    def _row(self, **overrides):
        return {
            "run": "a",
            "n_tracks": 4,
            **{column: 0.5 for column in BOARD_COLUMNS},
            **overrides,
        }

    def test_the_file_parses_under_a_strict_reader(self):
        """`json.dumps` emits bare `NaN` for an unmeasurable metric, which is
        not valid JSON — the point of routing the payload through `jsonable`."""

        payload = build_payload([self._row(f_beat=math.nan)], {}, {}, self._args())

        loaded = json.loads(json.dumps(payload), parse_constant=_reject)

        assert loaded["leaderboard"][0]["f_beat"] is None

    def test_the_readable_block_holds_the_board(self):
        payload = build_payload([self._row()], {}, {}, self._args())

        assert "run" in payload["readable"]
        assert "0.500" in payload["readable"]

    def test_the_ranking_is_recorded(self):
        payload = build_payload([self._row()], {}, {}, self._args())

        assert payload["eval"]["ranked_by"] == "macro_f_beat"

    def test_the_sweep_split_is_recorded(self):
        """A board swept on the split it reports is a different claim from one
        swept on train, and the file has to say which it is."""

        payload = build_payload([self._row()], {}, {}, self._args())

        assert payload["eval"]["sweep_split"] == "train"
        assert payload["eval"]["sweep_tracks"] == 50

    def test_sweeping_on_the_reported_split_records_no_subsample(self):
        """`--sweep-split val --split val` reuses the reporting evaluator, so
        `--sweep-tracks` never applies — recording it would imply a hold-out."""

        args = self._args()
        args.sweep_split = "val"

        payload = build_payload([self._row()], {}, {}, args)

        assert payload["eval"]["sweep_split"] == "val"
        assert payload["eval"]["sweep_tracks"] is None

    def test_an_empty_board_still_produces_a_payload(self):
        payload = build_payload([], {}, {"a": "boom"}, self._args())

        assert payload["leaderboard"] == []
        assert payload["failed"] == {"a": "boom"}


def _reject(value):
    raise AssertionError(f"non-JSON constant in the payload: {value}")
