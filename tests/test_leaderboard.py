"""What `tools/leaderboard.py` does around the evaluator: which checkpoint
represents a run, which grid points get swept, how the board is ordered, and
what makes a board refuse new rows.

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
    BOARD_COLUMNS,
    DEFAULT_BOARD,
    build_payload,
    parse_args,
    dvc,
    find_runs,
    load_board,
    merge,
    publish,
    rank_rows,
    ranked_metric,
    section,
    render_page,
    run_checkpoints,
    write_board,
    write_page,
    settings_for,
    sweep_evaluator,
    sweep_knobs,
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


def _board(n_runs: int = 2, **overrides) -> dict:
    """A payload shaped like a real board: the best run carries every field a
    section of the page reads, the rest only what the ranking needs."""

    rows = [
        {
            "run": f"run{i}",
            "n_tracks": 10,
            "macro_f_beat": 0.9 - i / 100,
            "f_beat": 0.5,
            "checkpoint": f"checkpoints/run{i}/epoch{i}.ckpt",
        }
        for i in range(n_runs)
    ]
    rows[0] |= {
        "task": "beat_phase",
        "swept_on": "train",
        "sweep_n_tracks": 50,
        "beat_threshold": 0.8,
        "switch_penalty": None,
        "measured_utc": "2026-09-18T18:05:45+00:00",
        "git_commit": "f8a702c3d472518e44829e9f647e0da3c5135d8a",
    }

    return {
        "generated_utc": "2026-09-18T18:08:03+00:00",
        "git_commit": "f8a702c3d472518e44829e9f647e0da3c5135d8a",
        "eval": {"dataset": "merge", "split": "val", "tolerance": 0.07},
        "ranked_by": "macro_f_beat",
        "leaderboard": rows,
        "per_corpus": {
            "run0": {
                "ballroom": {"n_tracks": 6, "f_beat": 0.6},
                "jtd": {"n_tracks": 4, "f_beat": 0.95},
            },
            "run1": {"ballroom": {"n_tracks": 6, "f_beat": 0.8}},
        },
        "failed": {},
        **overrides,
    }


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

        assert load_board(tmp_path / "leaderboard" / "leaderboard.json", False) == {}
        assert "starting one" in capsys.readouterr().out

    def test_an_existing_board_is_read_back(self, tmp_path):
        path = tmp_path / "leaderboard.json"
        path.write_text(json.dumps({"leaderboard": [{"run": "a"}, {"run": "b"}]}))

        assert len(load_board(path, False)["leaderboard"]) == 2

    def test_the_first_run_does_not_pull(self, tmp_path, monkeypatch):
        """Nothing to fetch yet, and the error would read like a failure."""

        def _boom(*args, **kwargs):
            raise AssertionError("dvc pull should not have run")

        monkeypatch.setattr("tools.leaderboard.subprocess.run", _boom)

        load_board(tmp_path / "leaderboard" / "leaderboard.json", pull=True)

    def test_an_existing_pointer_is_pulled(self, tmp_path, monkeypatch):
        (tmp_path / ".dvc").mkdir()
        (tmp_path / "leaderboard.dvc").write_text("")
        board = tmp_path / "leaderboard" / "leaderboard.json"
        board.parent.mkdir()

        calls = []
        monkeypatch.setattr(
            "tools.leaderboard.subprocess.run",
            lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0),
        )

        load_board(board, pull=True)

        assert calls == [["dvc", "pull", "leaderboard"]]


class TestDvcSync:
    """Syncing must never take a run down with it."""

    def _board(self, tmp_path):
        (tmp_path / "leaderboard").mkdir()

        return tmp_path / "leaderboard" / "leaderboard.json"

    def test_a_non_dvc_directory_is_skipped_not_fatal(self, tmp_path, capsys):
        assert dvc(["push", "leaderboard"], self._board(tmp_path)) is False
        assert "not a DVC repo" in capsys.readouterr().out

    def test_a_failed_add_does_not_push(self, tmp_path, monkeypatch, capsys):
        """That would upload the previous board under the new commit."""

        monkeypatch.setattr("tools.leaderboard.dvc", lambda command, board: False)

        publish(self._board(tmp_path))

        assert "git commit" not in capsys.readouterr().out

    def test_a_successful_push_says_what_to_commit(self, tmp_path, monkeypatch, capsys):
        """The pointer is only shared truth once committed, by a human."""

        monkeypatch.setattr("tools.leaderboard.dvc", lambda command, board: True)

        publish(self._board(tmp_path))

        assert "git add leaderboard.dvc" in capsys.readouterr().out


class TestWriteBoard:
    def test_a_symlinked_board_is_replaced_not_written_through(self, tmp_path):
        """A pulled board is a symlink into DVC's read-only cache."""

        cache = tmp_path / "cache-object"
        cache.write_text('{"cached": true}')
        cache.chmod(0o444)

        board = tmp_path / "leaderboard" / "leaderboard.json"
        board.parent.mkdir()
        board.symlink_to(cache)

        write_board({"leaderboard": []}, board)

        assert not board.is_symlink()
        assert json.loads(cache.read_text()) == {"cached": True}
        assert json.loads(board.read_text()) == {"leaderboard": []}

    def test_it_leaves_no_scratch_file_behind(self, tmp_path):
        board = tmp_path / "leaderboard" / "leaderboard.json"

        write_board({"leaderboard": []}, board)

        assert [p.name for p in board.parent.iterdir()] == ["leaderboard.json"]


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


class TestSection:
    """The one shape every table on the page goes through."""

    def test_labels_align_left_and_numbers_right(self):
        lines = section("Ranking", ["run", "f_beat"], [["`a`", "0.900"]], "a note")

        assert lines[0] == "## Ranking"
        assert lines[3] == "| --- | ---: |"
        assert lines[-1] == "a note"

    def test_a_section_without_a_note_ends_at_its_table(self):
        assert section("Provenance", ["run"], [["`a`"]])[-1] == "| `a` |"


class TestRenderPage:
    """What the page says, and what it must not quietly get wrong — the
    numbers on it are the ones people quote."""

    def test_the_ranking_table_reports_the_macro_means(self):
        """Micro is what the terminal table prints, macro is what ranked the
        board — printing one under a heading ordered by the other invites
        exactly the wrong read."""

        page = render_page(_board())
        ranking = page.split("## Ranking")[1].split("##")[0]

        assert "0.900" in ranking
        assert "0.500" not in ranking

    def test_rows_keep_the_order_the_board_was_written_in(self):
        """Ranking happens in `rank_rows`; rendering must not re-sort."""

        page = render_page(_board())

        assert page.index("`run0`") < page.index("`run1`")

    def test_the_ranked_column_is_marked(self):
        page = render_page(_board())

        assert "**f_beat ↑**" in page
        assert "confuse ↓" in page

    def test_the_best_run_per_corpus_is_marked(self):
        """The macro mean averages this away, and the winner differs per
        corpus far more often than the headline number suggests."""

        page = render_page(_board())
        ballroom = next(
            line for line in page.splitlines() if line.startswith("| `ballroom`")
        )

        assert "**0.800**" in ballroom
        assert "**0.600**" not in ballroom

    def test_a_corpus_a_run_never_scored_is_a_gap_not_a_zero(self):
        page = render_page(_board())
        jtd = next(line for line in page.splitlines() if line.startswith("| `jtd`"))

        assert jtd.endswith("| n/a |")

    def test_a_none_knob_is_a_value_and_a_missing_one_is_not(self):
        """`switch_penalty: None` is the exact single-offset decode, not an
        absent setting."""

        decode = render_page(_board()).split("## Decode")[1].split("\n## ")[0]
        swept, unswept = [
            line for line in decode.splitlines() if line.startswith("| ")
        ][2:]

        assert "| none |" in swept and "| `train` (50 tracks) |" in swept
        assert "| — |" in unswept and "| config |" in unswept

    def test_an_empty_board_still_renders_a_page(self):
        """The first invocation can fail every checkpoint it was given."""

        page = render_page({"leaderboard": [], "failed": {"norm": "boom"}})

        assert "Nothing scored yet." in page
        assert "boom" in page

    def test_an_older_board_names_its_ranking_metric_elsewhere(self):
        """`ranked_by` moved out of `eval`; a board written before that is
        still the board people pull."""

        payload = _board(ranked_by=None)
        payload["eval"]["ranked_by"] = "macro_position_acc"

        assert ranked_metric(payload) == "position_acc"


class TestPagePlacement:
    """Where the page goes, and which boards get one. It is committed to this
    repo rather than written beside the JSON: the data repo is behind a `dvc
    pull` and renders nowhere, so a page written there is one nobody opens."""

    def test_every_run_is_listed_by_default(self):
        """The page is the leaderboard, not an excerpt of one."""

        page = render_page(_board(7))

        assert "`run6`" in page
        assert "of 7 shown" not in page

    def test_top_cuts_it_to_the_best_few(self):
        page = render_page(_board(7), top=5)

        assert "best 5 of 7 shown" in page
        assert "`run4`" in page
        assert "`run5`" not in page

    def test_each_row_names_the_checkpoint_behind_it(self):
        """A good number has to lead straight to the model that made it."""

        page = render_page(_board(7), top=2)

        assert "`checkpoints/run0/epoch0.ckpt`" in page
        assert "`checkpoints/run2/epoch2.ckpt`" not in page

    def test_a_throwaway_board_writes_nothing(self, tmp_path, monkeypatch):
        """A `--board` elsewhere is a throwaway comparison: committing its
        numbers would version something nobody can reproduce."""

        monkeypatch.setattr("tools.leaderboard.PAGE_PATH", tmp_path / "PAGE.md")

        write_page(_board(), tmp_path / "leaderboard.json")

        assert not (tmp_path / "PAGE.md").exists()

    def test_the_running_board_writes_it_folder_and_all(self, tmp_path, monkeypatch):
        """The page's folder is its own, and a fresh clone has neither."""

        page = tmp_path / "leaderboard" / "LEADERBOARD.md"
        monkeypatch.setattr("tools.leaderboard.PAGE_PATH", page)

        write_page(_board(), DEFAULT_BOARD.resolve())

        assert page.read_text().startswith("# Beat leaderboard")


class TestCliSurface:
    """The one rule argparse cannot state on its own: runs are optional, but
    only when there is nothing to score."""

    def _parse(self, monkeypatch, *argv):
        monkeypatch.setattr("sys.argv", ["leaderboard.py", *argv])

        return parse_args()

    def test_a_render_needs_no_runs(self, monkeypatch):
        args = self._parse(monkeypatch, "--render-only")

        assert args.runs == []
        assert args.top == 0  # the page lists every run unless asked otherwise

    def test_naming_nothing_at_all_is_an_error(self, monkeypatch):
        """Otherwise it reads as a no-op run that silently scored nothing."""

        with pytest.raises(SystemExit):
            self._parse(monkeypatch)


def _reject(value):
    raise AssertionError(f"non-JSON constant in the payload: {value}")
