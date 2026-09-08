"""Tests for tools.eval_beat's orchestration — the layer that sits between the
argparse surface and :meth:`musicality.evaluation.BeatEvaluator.score`.

Scoring itself is not exercised here (tests/test_evaluation.py and
tests/test_metrics.py cover it); the evaluator is a stub returning canned rows.
What these pin is the logic that only the CLI has: which grid entries get
scored, how the resulting tables are ordered, which decoder variants are
compared, and how a checkpoint's own ``group_size`` overrides ``--group-size``.

Replaces tests/test_sweep_beat_postprocess.py — the sweep moved into
``tools/eval_beat.py --sweep``, which routes through ``BeatEvaluator.score()``
instead of re-deriving the decode. See
plans/06_metric_calibration_and_eval_consolidation.md, Phase C.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from musicality.evaluation import SCORE_KEYS
from tools.eval_beat import (
    decoder_variants,
    header_for,
    headroom_recovered,
    print_genre_breakdown,
    rank_key,
    require_beat_phase,
    resolve_group_size,
    run_sweep,
    score_variants,
    sweep_grid,
    write_rows,
)


# Every column print_genre_breakdown renders, so a row is complete without
# each test having to spell all six out.
_METRICS = {
    "f_beat": 0.8,
    "cmlt": 0.7,
    "amlt": 0.75,
    "position_acc_best_offset": 0.85,
    "confusion": 0.1,
}


def _rows(corpus_values: dict, **metrics) -> list[dict]:
    """One scored row per corpus, carrying *metrics* verbatim.

    *corpus_values* maps a corpus name to how many tracks it contributes, which
    is what makes macro and micro differ.
    """

    return [
        {"corpus": corpus, **metrics}
        for corpus, n in corpus_values.items()
        for _ in range(n)
    ]


class _StubEvaluator:
    """A BeatEvaluator stand-in that returns a canned row set per call.

    *per_call* is consumed in order, so a grid of N entries needs N entries.
    Every call's overrides are recorded on ``calls``.
    """

    def __init__(self, per_call, module_group_size=None, task="beat_phase"):
        self.per_call = list(per_call)
        self.calls = []
        self.group_size = None
        self.load_calls = 0
        self._probs = "cached"

        module = MagicMock()
        module.hparams = (
            {} if module_group_size is None else {"group_size": module_group_size}
        )
        self._module = module
        self._task = task
        self._loaded = (module, task, MagicMock(), [0, 1])

    def load(self):
        self.load_calls += 1
        if self._loaded is None:
            self._loaded = (self._module, self._task, MagicMock(), [0, 1])

        return self._loaded

    def score(self, **kwargs):
        self.calls.append(kwargs)

        return self.per_call.pop(0)


class TestRankKey:
    def test_macro_prefixes_the_metric(self):
        assert rank_key("position_acc", "macro") == "macro_position_acc"

    def test_micro_is_the_bare_metric(self):
        assert rank_key("position_acc", "micro") == "position_acc"


class TestSweepGrid:
    def _grid(self):
        return [
            {"beat_threshold": 0.2},
            {"beat_threshold": 0.5},
            {"beat_threshold": 0.8},
        ]

    def test_higher_is_better_metrics_sort_descending(self):
        evaluator = _StubEvaluator(
            [
                _rows({"a": 1}, f_beat=0.4),
                _rows({"a": 1}, f_beat=0.9),
                _rows({"a": 1}, f_beat=0.7),
            ]
        )

        ranked = sweep_grid(
            evaluator, self._grid(), group_size=4, metric="f_beat", rank_by="micro"
        )

        assert [r["beat_threshold"] for r in ranked] == [0.5, 0.8, 0.2]

    def test_confusion_sorts_ascending(self):
        """The one lower-is-better metric in the canonical set — if the
        direction were assumed rather than looked up, the sweep would pick the
        worst decode and report it as the winner."""

        evaluator = _StubEvaluator(
            [
                _rows({"a": 1}, confusion=0.4),
                _rows({"a": 1}, confusion=0.9),
                _rows({"a": 1}, confusion=0.1),
            ]
        )

        ranked = sweep_grid(
            evaluator, self._grid(), group_size=4, metric="confusion", rank_by="micro"
        )

        assert [r["beat_threshold"] for r in ranked] == [0.8, 0.2, 0.5]

    def test_unscorable_combos_sort_last_not_first(self):
        """A combo that picks no beats at all scores NaN. It must never
        outrank a real result, in either sort direction."""

        evaluator = _StubEvaluator(
            [
                _rows({"a": 1}, f_beat=float("nan")),
                _rows({"a": 1}, f_beat=0.3),
                _rows({"a": 1}, f_beat=0.6),
            ]
        )

        ranked = sweep_grid(
            evaluator, self._grid(), group_size=4, metric="f_beat", rank_by="micro"
        )

        assert [r["beat_threshold"] for r in ranked[:2]] == [0.8, 0.5]
        assert math.isnan(ranked[-1]["f_beat"])

    def test_unscorable_combos_sort_last_for_confusion_too(self):
        evaluator = _StubEvaluator(
            [
                _rows({"a": 1}, confusion=float("nan")),
                _rows({"a": 1}, confusion=0.3),
            ]
        )

        ranked = sweep_grid(
            evaluator,
            self._grid()[:2],
            group_size=4,
            metric="confusion",
            rank_by="micro",
        )

        assert ranked[0]["confusion"] == pytest.approx(0.3)
        assert math.isnan(ranked[-1]["confusion"])

    def test_each_grid_entry_is_scored_with_its_knobs_and_group_size(self):
        evaluator = _StubEvaluator([_rows({"a": 1}, f_beat=0.5) for _ in range(3)])

        sweep_grid(
            evaluator, self._grid(), group_size=8, metric="f_beat", rank_by="micro"
        )

        assert evaluator.calls == [
            {"group_size": 8, "beat_threshold": 0.2},
            {"group_size": 8, "beat_threshold": 0.5},
            {"group_size": 8, "beat_threshold": 0.8},
        ]

    def test_rank_by_macro_ignores_corpus_size(self):
        """3 easy tracks from one corpus and 1 hard one from another average to
        0.25 per track but 0.5 per corpus — so macro and micro pick differently."""

        skewed = _rows({"a": 3}, position_acc=0.0) + _rows({"b": 1}, position_acc=1.0)
        even = _rows({"a": 1}, position_acc=0.4) + _rows({"b": 1}, position_acc=0.4)

        grid = [{"beat_threshold": 0.2}, {"beat_threshold": 0.5}]

        by_micro = sweep_grid(
            _StubEvaluator([skewed, even]),
            grid,
            group_size=4,
            metric="position_acc",
            rank_by="micro",
        )
        by_macro = sweep_grid(
            _StubEvaluator([skewed, even]),
            grid,
            group_size=4,
            metric="position_acc",
            rank_by="macro",
        )

        assert by_micro[0]["beat_threshold"] == 0.5  # 0.400 beats 0.250
        assert by_macro[0]["beat_threshold"] == 0.2  # 0.500 beats 0.400


class TestDecoderVariants:
    def test_greedy_is_always_the_first_variant(self):
        """The verdict is phrased as a delta from the greedy baseline, so it
        has to be the row the comparison starts from."""

        variants = decoder_variants(0.8, [2.0], ["index"])

        assert variants[0][1] == "greedy"
        assert "anchor=0.8" in variants[0][0]

    def test_exact_decode_plus_one_variant_per_penalty(self):
        variants = decoder_variants(0.8, [2.0, 5.0], ["index"])

        assert [(v[1], v[2]) for v in variants] == [
            ("greedy", None),
            ("global", None),
            ("global", 2.0),
            ("global", 5.0),
        ]

    def test_no_penalties_still_scores_the_exact_decode(self):
        variants = decoder_variants(0.8, [], ["index"])

        assert [(v[1], v[2]) for v in variants] == [("greedy", None), ("global", None)]

    def test_advance_modes_are_tagged_only_when_there_is_more_than_one(self):
        single = decoder_variants(0.8, [2.0], ["index"])
        both = decoder_variants(0.8, [2.0], ["index", "time"])

        assert all("[" not in name for name, *_ in single)
        assert [v[3] for v in both] == ["index", "index", "index", "time", "time"]
        assert "[time]" in both[-1][0]


class TestScoreVariants:
    def test_switch_penalty_none_is_passed_explicitly(self):
        """``None`` is a real switch_penalty — the exact single-offset decode,
        no mid-track resync allowed — not "unset". If it were dropped instead
        of passed, resolve_postprocess would silently substitute the tuned
        value and the exact decode would never actually be scored."""

        variants = [("exact", "global", None, "index")]
        evaluator = _StubEvaluator([_rows({"a": 1}, position_acc=0.5)])

        score_variants(evaluator, variants, group_size=4)

        assert "switch_penalty" in evaluator.calls[0]
        assert evaluator.calls[0]["switch_penalty"] is None

    def test_returns_summary_and_rows_per_variant_in_order(self):
        variants = [
            ("greedy", "greedy", None, "index"),
            ("global", "global", 2.0, "index"),
        ]
        rows_a = _rows({"a": 1}, position_acc=0.3)
        rows_b = _rows({"a": 1}, position_acc=0.7)
        evaluator = _StubEvaluator([rows_a, rows_b])

        results = score_variants(evaluator, variants, group_size=4)

        assert list(results) == ["greedy", "global"]
        assert results["greedy"][0]["position_acc"] == pytest.approx(0.3)
        assert results["greedy"][1] is rows_a
        assert results["global"][0]["position_acc"] == pytest.approx(0.7)
        assert results["global"][1] is rows_b

    def test_advance_is_threaded_through(self):
        variants = [("t", "global", 2.0, "time")]
        evaluator = _StubEvaluator([_rows({"a": 1}, position_acc=0.5)])

        score_variants(evaluator, variants, group_size=4)

        assert evaluator.calls[0]["advance"] == "time"


class TestResolveGroupSize:
    def test_checkpoint_without_the_hparam_keeps_the_requested_value(self):
        evaluator = _StubEvaluator([], module_group_size=None)

        assert resolve_group_size(evaluator, 4) == 4
        assert evaluator.group_size is None

    def test_matching_checkpoint_does_not_rebuild(self):
        evaluator = _StubEvaluator([], module_group_size=4)

        assert resolve_group_size(evaluator, 4) == 4
        assert evaluator._probs == "cached"

    def test_checkpoint_group_size_wins_and_forces_a_rebuild(self):
        """The checkpoint's group_size also decides how the *reference*
        annotations are folded, so honouring it without rebuilding the dataset
        would score 8-way predictions against 4-way references."""

        evaluator = _StubEvaluator([], module_group_size=8)

        assert resolve_group_size(evaluator, 4) == 8
        assert evaluator.group_size == 8
        assert evaluator._probs is None


class TestRequireBeatPhase:
    def test_beat_phase_passes(self):
        require_beat_phase("beat_phase", "--decoders")

    def test_beat_only_exits_with_the_mode_named(self):
        with pytest.raises(SystemExit, match="--decoders"):
            require_beat_phase("beat_only", "--decoders")


class TestWriteRows:
    def test_writes_a_header_from_the_first_row(self, tmp_path):
        path = tmp_path / "out.csv"

        write_rows(path, [{"corpus": "a", "f_beat": 0.5}])

        assert path.read_text().splitlines()[0] == "corpus,f_beat"

    def test_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "out.csv"

        write_rows(path, [{"corpus": "a"}])

        assert path.exists()

    def test_no_rows_writes_no_file(self, tmp_path):
        path = tmp_path / "out.csv"

        write_rows(path, [])

        assert not path.exists()


class TestHeadroomRecovered:
    """The verdict's 0.30/0.10 thresholds have to mean the same thing whichever
    metric ranks the decoders. Measuring the share of the *remaining error* that
    was recovered is what puts an error rate and an accuracy on one scale."""

    def test_halving_confusion_recovers_half_the_error(self):
        _gain, share = headroom_recovered("confusion", 0.20, 0.10)

        assert share == pytest.approx(0.5)

    def test_halving_the_position_error_also_reads_as_half(self):
        _gain, share = headroom_recovered("position_acc", 0.80, 0.90)

        assert share == pytest.approx(0.5)

    def test_a_raw_relative_change_would_have_read_as_a_tenth(self):
        """Pins the bug this replaced: 0.847 -> 0.921 position_acc is +9% on the
        level but recovers half the error, and the old formulation reported it
        as CAUSE (A), the model."""

        gain, share = headroom_recovered("position_acc", 0.847, 0.921)

        assert gain / 0.847 < 0.10
        assert share > 0.40

    def test_gain_is_signed_positive_for_both_directions(self):
        assert headroom_recovered("confusion", 0.20, 0.10)[0] == pytest.approx(0.10)
        assert headroom_recovered("position_acc", 0.80, 0.90)[0] == pytest.approx(0.10)

    def test_a_worse_decoder_gives_a_negative_share(self):
        assert headroom_recovered("confusion", 0.10, 0.20)[1] < 0
        assert headroom_recovered("position_acc", 0.90, 0.80)[1] < 0

    def test_no_headroom_is_zero_not_a_division_error(self):
        assert headroom_recovered("confusion", 0.0, 0.0)[1] == 0.0
        assert headroom_recovered("position_acc", 1.0, 1.0)[1] == 0.0


class TestHeaderFor:
    def test_the_two_position_columns_get_distinct_names(self):
        """`col[:8]` printed both `position_acc` and `position_acc_best_offset`
        as "position", making the table unreadable."""

        header = header_for(("position_acc", "position_acc_best_offset"))

        assert "pos_acc" in header
        assert "best_off" in header

    def test_every_score_key_has_a_label(self):
        header_for(SCORE_KEYS)


class TestPrintGenreBreakdown:
    """The per-genre table is the whole point of a merged split: one blended
    number hides a corpus the model is useless on."""

    def _rows(self):
        return _rows({"ballroom": 3}, **_METRICS, position_acc=0.9) + _rows(
            {"rwc_classical": 1}, **_METRICS, position_acc=0.1
        )

    def test_one_line_per_corpus_plus_macro_and_micro(self, capsys):
        print_genre_breakdown(self._rows())
        out = capsys.readouterr().out

        assert "ballroom" in out
        assert "rwc_classical" in out
        assert "MACRO (per corpus)" in out
        assert "micro (per track)" in out

    def test_macro_and_micro_differ_on_unequal_corpora(self, capsys):
        print_genre_breakdown(self._rows())
        lines = capsys.readouterr().out.splitlines()

        macro = next(line for line in lines if "MACRO" in line)
        micro = next(line for line in lines if "micro" in line)

        assert "0.500" in macro  # mean(0.9, 0.1)
        assert "0.700" in micro  # mean(0.9, 0.9, 0.9, 0.1)

    def test_names_the_weakest_corpus(self, capsys):
        print_genre_breakdown(self._rows())

        assert "Weakest corpus: rwc_classical" in capsys.readouterr().out

    def test_single_corpus_skips_the_macro_micro_footer(self, capsys):
        """With one corpus the two means are the same number by construction,
        so printing both would just be noise."""

        print_genre_breakdown(_rows({"ballroom": 3}, **_METRICS, position_acc=0.9))
        out = capsys.readouterr().out

        assert "ballroom" in out
        assert "MACRO" not in out


class TestRunSweepBeatOnly:
    """A beat-only checkpoint has no bar-position heads, so stage 2 must be
    skipped rather than sweeping a knob nothing reads."""

    def _args(self, **overrides):
        return SimpleNamespace(
            sweep_beat_thresholds=[0.4, 0.5],
            sweep_min_distances=[2],
            sweep_gate_tolerances=[0.1],
            sweep_anchor_thresholds=[0.5],
            switch_penalties=[2.0],
            rank_by="micro",
            rank_metric="position_acc",
            top=10,
            output=None,
            **overrides,
        )

    def test_skips_the_position_stage(self, capsys):
        evaluator = _StubEvaluator([_rows({"a": 1}, f_beat=0.5) for _ in range(2)])

        run_sweep(evaluator, self._args(), task="beat_only", group_size=4)
        out = capsys.readouterr().out

        assert "BEAT DETECTION SWEEP" in out
        assert "BAR-POSITION SWEEP" not in out
        assert "skipping the bar-position sweep" in out

    def test_scores_only_the_beat_grid(self):
        evaluator = _StubEvaluator([_rows({"a": 1}, f_beat=0.5) for _ in range(2)])

        run_sweep(evaluator, self._args(), task="beat_only", group_size=4)

        assert len(evaluator.calls) == 2
        assert all("switch_penalty" not in c for c in evaluator.calls)
