"""What `tools/leaderboard.py` does around the evaluator: which checkpoint
represents a run, which grid points get swept, how the board is ordered, what
makes a board refuse new rows, and what reaches W&B.

Scoring itself is covered by tests/test_evaluation.py; the evaluator here is a
stub returning canned rows.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from musicality.dataformats.track_io import TrackRef
from tools.leaderboard import (
    ARTIFACT,
    BOARD_COLUMNS,
    DEFAULT_BOARD,
    FILENAME,
    build_payload,
    fetch_board,
    find_runs,
    load_board,
    merge,
    parse_args,
    publish,
    training_run,
    rank_rows,
    run_checkpoints,
    settings_for,
    sweep_evaluator,
    sweep_knobs,
    write_board,
)


def _args(**overrides):
    return SimpleNamespace(
        limit=None, sweep=True, rank_metric="f_beat", commit="abc1234", **overrides
    )


def _touch(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("")

    return directory


class TestRunCheckpoints:
    """Reading a `save_top_k` group as separate models triples the board;
    reading a hand-named folder as one group drops five."""

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

        assert sorted(c.name for _label, c in run_checkpoints(run)) == [
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
        """One run per learning rate, two levels down."""

        sweep = tmp_path / "lr_sweep-20260918-141530"
        _touch(sweep / "lr_0.0008", "beat-phase-epoch91-valloss1.4794.ckpt")
        _touch(sweep / "lr_0.002", "beat-phase-epoch95-valloss1.4757.ckpt")

        found = find_runs([sweep])

        assert [Path(label).name for label, _c in found] == ["lr_0.0008", "lr_0.002"]


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


def _board(n_runs: int = 2) -> dict:
    """A payload shaped like a real board, small enough to assert on."""

    return {
        "eval": {"dataset": "merge", "split": "val", "tolerance": 0.07},
        "ranked_by": "macro_f_beat",
        "leaderboard": [
            {
                "run": f"run{i}",
                "n_tracks": 10,
                "macro_f_beat": 0.9 - i / 100,
                "f_beat": 0.5,
                "checkpoint": f"checkpoints/run{i}/epoch{i}.ckpt",
            }
            for i in range(n_runs)
        ],
        "per_corpus": {},
        "failed": {},
    }


class _PublishedBoard:
    """The artifact `fetch_board` downloads: a folder holding one board file."""

    name = "leaderboard:v7"

    def __init__(self, directory: Path, payload: dict):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / FILENAME).write_text(json.dumps(payload))

    def download(self, root=None) -> str:
        return str(self.directory)


class _StubRun:
    """The W&B run `publish` opens, remembering everything it was handed."""

    url = "https://wandb.ai/acme/musicality-leaderboard/runs/abc123"

    def __init__(self, **kwargs):
        self.opened_with = kwargs
        self.logged: dict = {}
        self.summary: dict = {}
        self.artifacts: list = []

    def log(self, data):
        self.logged.update(data)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class _StubArtifact:
    def __init__(self, name, type=None):
        self.name, self.type, self.files = name, type, {}

    def add_file(self, path, name=None):
        self.files[name] = Path(path).read_text()


@pytest.fixture
def fake_wandb(monkeypatch):
    """`tools.leaderboard.wandb`, minus the network.

    `state.artifact` is what `wandb.Api().artifact(...)` does — set it per test
    to a published board, or to a raise for a project that has none.
    """

    state = SimpleNamespace(run=None, finished=False, asked_for=None)

    def _artifact(name):
        state.asked_for = name

        return state.artifact(name)

    def _init(**kwargs):
        state.run = _StubRun(**kwargs)

        return state.run

    monkeypatch.setattr(
        "tools.leaderboard.wandb",
        SimpleNamespace(
            init=_init,
            Table=lambda columns, data: SimpleNamespace(columns=columns, data=data),
            Artifact=_StubArtifact,
            Api=lambda: SimpleNamespace(artifact=_artifact),
            finish=lambda: setattr(state, "finished", True),
        ),
    )

    return state


@pytest.fixture
def small_grid(monkeypatch):
    """A two-point grid in place of the shipped one, so a test can count calls."""

    grid = {
        "split": "train",
        "tracks": 4,
        "beat_thresholds": [0.4, 0.8],
        "min_distance_frames": [2],
        "gate_tolerances": [0.1],
        "anchor_thresholds": [0.5, 0.9],
        "switch_penalties": [2.0],
    }
    monkeypatch.setattr("tools.leaderboard.SWEEP", grid)

    return grid


class TestSweepKnobs:
    def test_beat_only_stops_after_the_beat_stage(self, small_grid):
        """No bar-position heads, so stage 2 would tune what nothing reads."""

        evaluator = _StubEvaluator([_rows(f_beat=0.5), _rows(f_beat=0.9)])

        knobs = sweep_knobs(evaluator, "beat_only", 4)

        assert len(evaluator.calls) == 2
        assert knobs == {
            "beat_threshold": 0.8,
            "min_distance_frames": 2,
            "gate_tolerance": 0.1,
        }

    def test_the_position_stage_ignores_f_beat(self, small_grid):
        """Every stage-2 candidate ties on f_beat, so ranking by it picks by
        list order — 0.73 against 0.88 position_acc on six ballroom tracks."""

        evaluator = _StubEvaluator(
            [_rows(f_beat=0.5, position_acc=0.1), _rows(f_beat=0.9, position_acc=0.2)]
            # Stage 2: identical f_beat, as it always is in reality.
            + [_rows(f_beat=0.9, position_acc=p) for p in (0.3, 0.9)]
        )

        assert sweep_knobs(evaluator, "beat_phase", 4)["switch_penalty"] == 2.0

    def test_the_global_decoder_always_considers_the_no_resync_decode(self, small_grid):
        """The penalty -> infinity limit, which no finite candidate reaches."""

        evaluator = _StubEvaluator(
            [_rows(f_beat=0.5, position_acc=0.1), _rows(f_beat=0.9, position_acc=0.2)]
            + [_rows(f_beat=0.9, position_acc=p) for p in (0.8, 0.3)]
        )

        knobs = sweep_knobs(evaluator, "beat_phase", 4)

        assert [c.get("switch_penalty") for c in evaluator.calls[2:]] == [None, 2.0]
        assert knobs["switch_penalty"] is None

    def test_the_greedy_decoder_sweeps_its_own_knob_instead(self, small_grid):
        evaluator = _StubEvaluator(
            [_rows(f_beat=0.9, position_acc=p) for p in (0.1, 0.2, 0.3, 0.9)],
            decoder="greedy",
        )

        knobs = sweep_knobs(evaluator, "beat_phase", 4)

        assert knobs["anchor_threshold"] == 0.9
        assert "switch_penalty" not in knobs

    def test_the_beat_stage_winner_is_held_fixed(self, small_grid):
        evaluator = _StubEvaluator(
            [_rows(f_beat=0.9, position_acc=0.5), _rows(f_beat=0.5, position_acc=0.9)]
            + [_rows(f_beat=0.9, position_acc=0.5) for _ in range(2)]
        )

        sweep_knobs(evaluator, "beat_phase", 4)

        # 0.4 won stage 1 on f_beat; stage 2 must not re-open that choice just
        # because the other threshold scored better on the position metric.
        assert all(c["beat_threshold"] == 0.4 for c in evaluator.calls[2:])


class TestSweepEvaluator:
    """The knobs are tuned on their own tracks, and on a spread of corpora."""

    def _dataset(self, counts: dict):
        return SimpleNamespace(
            refs=[
                TrackRef(dataset_name=c, track_id=f"{c}-{i}", data_home=Path("."))
                for c, n in counts.items()
                for i in range(n)
            ]
        )

    def _patched(self, monkeypatch, dataset):
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

    def test_it_reads_the_sweep_split_not_the_reported_one(
        self, monkeypatch, small_grid
    ):
        self._patched(monkeypatch, self._dataset({"ballroom": 2}))

        assert sweep_evaluator(Path("a.ckpt"), 4, "cpu").kwargs["split"] == "train"

    def test_the_subsample_spreads_across_corpora(self, monkeypatch, small_grid):
        """A split file is written corpus by corpus: the first N is one genre."""

        dataset = self._dataset({"ballroom": 6, "rwc_classical": 2})
        self._patched(monkeypatch, dataset)

        evaluator = sweep_evaluator(Path("a.ckpt"), 4, "cpu")
        corpora = sorted(dataset.refs[i].dataset_name for i in evaluator.load()[3])

        assert corpora == ["ballroom", "ballroom", "rwc_classical", "rwc_classical"]

    def test_the_narrowed_indices_still_point_into_the_same_dataset(
        self, monkeypatch, small_grid
    ):
        """Everything downstream walks `indices` against the loaded dataset."""

        dataset = self._dataset({"ballroom": 9})
        self._patched(monkeypatch, dataset)

        _m, _t, loaded, indices = sweep_evaluator(Path("a.ckpt"), 4, "cpu").load()

        assert loaded is dataset
        assert len(indices) == small_grid["tracks"]
        assert all(0 <= i < len(dataset.refs) for i in indices)

    def test_a_subsample_larger_than_the_split_keeps_every_track(
        self, monkeypatch, small_grid
    ):
        self._patched(monkeypatch, self._dataset({"ballroom": 3}))

        assert len(sweep_evaluator(Path("a.ckpt"), 4, "cpu").load()[3]) == 3


class TestMerge:
    """Which rows survive, and refusing rows measured a different way."""

    def _previous(self, *runs, **eval_overrides):
        return {
            "leaderboard": [{"run": run, "f_beat": 0.5} for run in runs],
            "per_corpus": {run: {} for run in runs},
            "eval": {**settings_for(_args()), **eval_overrides},
        }

    def test_rows_not_re_measured_are_carried(self):
        carried = merge(self._previous("a", "b"), [{"run": "b"}], settings_for(_args()))

        assert [row["run"] for row in carried] == ["a"]

    def test_a_re_measured_run_is_replaced_not_duplicated(self):
        """More training on a folder is the same experiment, not a second one."""

        assert merge(self._previous("a"), [{"run": "a"}], settings_for(_args())) == []

    def test_incomparable_settings_are_refused(self):
        previous = self._previous("a", tolerance=0.05)

        with pytest.raises(SystemExit, match="tolerance"):
            merge(previous, [], settings_for(_args()))

    def test_an_unswept_row_cannot_join_a_swept_board(self):
        """Different knobs, so not the same scale."""

        previous = self._previous("a", swept=False)

        with pytest.raises(SystemExit, match="swept"):
            merge(previous, [], settings_for(_args()))

    def test_re_measuring_everything_lifts_the_refusal(self):
        """Nothing survives, so the old settings claim nothing — this is how
        they get changed without an override flag."""

        previous = self._previous("a", tolerance=0.05)

        assert merge(previous, [{"run": "a"}], settings_for(_args())) == []

    def test_an_empty_board_takes_anything(self):
        assert merge({}, [{"run": "a"}], settings_for(_args())) == []


class TestBoardFile:
    def test_a_missing_board_starts_an_empty_one(self, tmp_path, capsys):
        """So the first invocation is the same command as every later one."""

        assert load_board(tmp_path / "leaderboard.json", "proj", fetch=False) == {}
        assert "starting one" in capsys.readouterr().out

    def test_an_existing_board_is_read_back(self, tmp_path):
        path = tmp_path / "leaderboard.json"
        path.write_text(json.dumps({"leaderboard": [{"run": "a"}, {"run": "b"}]}))

        assert len(load_board(path, "proj", fetch=False)["leaderboard"]) == 2

    def test_the_published_board_wins_over_the_local_copy(self, tmp_path, fake_wandb):
        """A rented instance's own copy is whatever it last happened to write;
        the board everyone extends is the published one."""

        fake_wandb.artifact = lambda name: _PublishedBoard(
            tmp_path / "artifact", {"leaderboard": [{"run": "elsewhere"}]}
        )
        board = tmp_path / "leaderboard" / FILENAME
        board.parent.mkdir()
        board.write_text(json.dumps({"leaderboard": [{"run": "stale"}]}))

        loaded = load_board(board, "musicality-leaderboard", fetch=True)

        assert fake_wandb.asked_for == f"musicality-leaderboard/{ARTIFACT}:latest"
        assert loaded["leaderboard"] == [{"run": "elsewhere"}]

    def test_nothing_published_yet_is_not_fatal(self, tmp_path, fake_wandb, capsys):
        """The first board ever has nothing to fetch, and an unreachable W&B is
        a reason to score locally rather than to refuse."""

        def _missing(name):
            raise ValueError("artifact not found")

        fake_wandb.artifact = _missing

        fetch_board(tmp_path / FILENAME, "musicality-leaderboard")

        assert "nothing fetched" in capsys.readouterr().out


class TestWriteBoard:
    def test_it_creates_its_folder_on_the_first_run(self, tmp_path):
        board = tmp_path / "leaderboard" / FILENAME

        write_board({"leaderboard": []}, board)

        assert json.loads(board.read_text()) == {"leaderboard": []}


class TestRankRows:
    def _rows(self):
        return [
            {"run": "a", "macro_f_beat": 0.7, "macro_confusion": 0.4},
            {"run": "b", "macro_f_beat": 0.9, "macro_confusion": 0.1},
            {"run": "c", "macro_f_beat": 0.8, "macro_confusion": 0.2},
        ]

    def test_higher_is_better_metrics_rank_descending(self):
        assert [r["run"] for r in rank_rows(self._rows(), "f_beat")] == ["b", "c", "a"]

    def test_confusion_ranks_ascending(self):
        """The one lower-is-better member of the canonical set."""

        assert [r["run"] for r in rank_rows(self._rows(), "confusion")] == [
            "b",
            "c",
            "a",
        ]

    def test_an_unscorable_run_ranks_last_rather_than_first(self):
        """`None` (a NaN through `jsonable`) must not read as a winning score."""

        rows = [{"run": "a", "macro_f_beat": None}, {"run": "b", "macro_f_beat": 0.1}]

        assert [r["run"] for r in rank_rows(rows, "f_beat")] == ["b", "a"]


class TestBuildPayload:
    def _row(self, **overrides):
        return {
            "run": "a",
            "n_tracks": 4,
            **{column: 0.5 for column in BOARD_COLUMNS},
            **overrides,
        }

    def test_the_file_parses_under_a_strict_reader(self):
        """`json.dumps` emits bare `NaN`, which is not valid JSON."""

        payload = build_payload([self._row(f_beat=math.nan)], {}, {}, _args())

        loaded = json.loads(json.dumps(payload), parse_constant=_reject)

        assert loaded["leaderboard"][0]["f_beat"] is None

    def test_the_readable_block_holds_the_board(self):
        payload = build_payload([self._row()], {}, {}, _args())

        assert "0.500" in payload["readable"]

    def test_an_empty_board_still_produces_a_payload(self):
        payload = build_payload([], {}, {"a": "boom"}, _args())

        assert payload["leaderboard"] == []
        assert payload["failed"] == {"a": "boom"}


class TestTrainingRun:
    """Without this, a leading row names a folder on a destroyed instance."""

    def _report(self, directory: Path, **run) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "training_report.json").write_text(json.dumps({"run": run}))

        return directory / "beat-phase-epoch91-valloss1.47.ckpt"

    def test_the_row_links_back_to_the_wandb_run(self, tmp_path):
        checkpoint = self._report(
            tmp_path / "run",
            wandb_name="eternal-feather-106",
            wandb_url="https://wandb.ai/acme/musicality-beat-phase/runs/ae65nj3v",
            git_commit="6e1d3e8",
        )

        assert training_run(checkpoint) == {
            "wandb_name": "eternal-feather-106",
            "wandb_url": "https://wandb.ai/acme/musicality-beat-phase/runs/ae65nj3v",
            "train_commit": "6e1d3e8",
        }

    def test_the_training_commit_is_not_the_evaluating_one(self, tmp_path):
        """They differ whenever a run is scored by later code, which is the
        normal case — the row carries both."""

        checkpoint = self._report(tmp_path / "run", git_commit="6e1d3e8")

        assert training_run(checkpoint)["train_commit"] == "6e1d3e8"

    def test_a_run_without_a_report_still_makes_a_row(self, tmp_path):
        """The report is written at `on_fit_end`: an interrupted run has none,
        and that must cost it a link, not its place on the board."""

        assert training_run(tmp_path / "nothing-here.ckpt") == {}


class TestPublish:
    """The two halves of a published board: the table to look at, the file to
    keep. Both come from the same payload, so neither can drift."""

    def _publish(self, tmp_path, payload):
        board = tmp_path / FILENAME
        board.write_text(json.dumps(payload))
        publish(payload, board, "musicality-leaderboard")

        return board

    def test_the_table_carries_every_column_of_every_row(self, tmp_path, fake_wandb):
        """Narrowing it here would make the board on W&B a different board from
        the one in the file."""

        payload = _board(3)

        self._publish(tmp_path, payload)
        table = fake_wandb.run.logged["leaderboard"]

        assert table.columns == list(payload["leaderboard"][0])
        assert [row[0] for row in table.data] == ["run0", "run1", "run2"]

    def test_the_columns_are_the_union_over_rows(self, tmp_path, fake_wandb):
        """A board holds rows written by different versions of this tool; the
        first row's keys would drop whatever only the newer ones carry."""

        payload = _board(2)
        payload["leaderboard"][1]["wandb_url"] = "https://wandb.ai/acme/p/runs/xyz"

        self._publish(tmp_path, payload)

        assert "wandb_url" in fake_wandb.run.logged["leaderboard"].columns

    def test_the_artifact_is_the_board_file_itself(self, tmp_path, fake_wandb):
        """It is what the next invocation fetches, so it has to be the file
        that was just written, not a second rendering of it."""

        board = self._publish(tmp_path, _board())
        [artifact] = fake_wandb.run.artifacts

        assert artifact.name == ARTIFACT
        assert artifact.files == {FILENAME: board.read_text()}

    def test_the_summary_names_the_leader(self, tmp_path, fake_wandb):
        """So the project's run list ranks itself without opening a table."""

        self._publish(tmp_path, _board(3))

        assert fake_wandb.run.summary["best/run"] == "run0"
        assert fake_wandb.run.summary["n_runs"] == 3

    def test_the_run_is_closed_and_its_url_printed(self, tmp_path, fake_wandb, capsys):
        self._publish(tmp_path, _board())

        assert fake_wandb.finished
        assert _StubRun.url in capsys.readouterr().out


class TestCliSurface:
    """The one rule argparse cannot state on its own."""

    def _parse(self, monkeypatch, *argv):
        monkeypatch.setattr("sys.argv", ["leaderboard.py", *argv])

        return parse_args()

    def test_the_default_board_is_the_shared_one(self, monkeypatch):
        args = self._parse(monkeypatch, "checkpoints_deeper")

        assert args.board == DEFAULT_BOARD
        assert args.fetch and args.publish

    def test_a_board_elsewhere_never_touches_wandb(self, monkeypatch, tmp_path):
        """A throwaway comparison published over the shared board would be
        merged into by the next run on any machine."""

        args = self._parse(
            monkeypatch, "checkpoints_deeper", "--board", str(tmp_path / "scratch.json")
        )

        assert not args.fetch and not args.publish

    def test_publishing_an_existing_board_needs_no_runs(self, monkeypatch):
        """Putting a board that is already scored on W&B costs no model pass."""

        args = self._parse(monkeypatch, "--publish-only")

        assert args.runs == []

    def test_naming_nothing_at_all_is_an_error(self, monkeypatch):
        """Otherwise it reads as a no-op run that silently scored nothing."""

        with pytest.raises(SystemExit):
            self._parse(monkeypatch)


def _reject(value):
    raise AssertionError(f"non-JSON constant in the payload: {value}")
