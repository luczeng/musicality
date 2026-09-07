#!/usr/bin/env python3
"""The one tool for evaluating a beat-only or beat-phase checkpoint (task
auto-detected from the checkpoint itself) on full-length tracks — not the
fixed-duration clips used during training.

Every mode below runs the model **once** per track and re-uses the cached frame
probabilities, so comparing decoders or sweeping thresholds costs one model pass
in total rather than one per configuration.

Modes
-----
======================  ====================================================
*(default)*             Canonical report: per-track lines, then the summary
                        block (:func:`musicality.evaluation.summary_block`).
``--per-genre``         Per-corpus breakdown, macro vs micro, weakest corpus.
                        Automatic when the split spans more than one corpus.
``--profile``           Phase-offset profile: the per-track dominant offset
                        histogram and within-track phase stability.
``--decoders``          Score every bar-position decoder variant against the
                        same probabilities, then call model-vs-decoder.
``--sweep``             Grid-search the postprocessing knobs.
``--output <csv>``      Write the run's per-track rows (or, under ``--sweep``,
                        its grid tables) to CSV.
======================  ====================================================

``--per-genre`` and ``--profile`` are report sections and combine with the
default report or with ``--decoders``. ``--decoders`` and ``--sweep`` are
different computations and are mutually exclusive.

Usage
-----
    # the canonical report
    uv run python tools/eval_beat.py \\
        --checkpoint <path-to-ckpt> --dataset ballroom

    # a merged split, per genre, with the phase-offset profile
    uv run python tools/eval_beat.py --checkpoint ... \\
        --dataset merge --split val --binary-only --per-genre --profile

    # is the bar-position error the model's fault or the decoder's?
    uv run python tools/eval_beat.py --checkpoint ... \\
        --dataset merge --split val --binary-only --decoders

    # try other Viterbi resync costs, and both position-advance rules
    uv run python tools/eval_beat.py --checkpoint ... \\
        --decoders --switch-penalties 2 5 10 20 --advance both

    # re-tune configs/eval_beat.yaml's postprocessing defaults
    uv run python tools/eval_beat.py --checkpoint ... \\
        --dataset merge --split val --binary-only --sweep

    # quick check on a handful of tracks, rows saved for further analysis
    uv run python tools/eval_beat.py --checkpoint ... --limit 20 --output eval.csv

``--binary-only`` must match how the checkpoint's split was created; without it
a large share of "val" can be training data. ``--checkpoint`` is a named flag —
passed positionally it exits on an argparse error that a piped ``tail`` swallows.
"""

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np

from musicality.evaluation import (
    DATA_DIR,
    SCORE_KEYS,
    BeatEvaluator,
    _fmt,
    _mean,
    group_by_corpus,
    summarize,
)
from musicality.evaluation import DEFAULTS as EVAL_DEFAULTS

SWEEP_DEFAULTS = EVAL_DEFAULTS["sweep"]

# Columns of the per-genre table: the canonical headline metrics, narrow enough
# to sit side by side. The full set is always in the CSV.
_GENRE_COLUMNS = (
    "f_beat",
    "cmlt",
    "amlt",
    "position_acc",
    "position_acc_best_offset",
    "confusion",
)

# Columns of the decoder-comparison table. `f_beat`/`cmlt`/`amlt` are omitted
# because a bar-position decoder cannot move a beat time — they are identical
# across every row by construction.
_DECODER_COLUMNS = (
    "position_acc",
    "position_acc_best_offset",
    "anchor_error",
    "f_one",
    "f_last",
    "confusion",
)

# Table headers. Truncating the key would print `position_acc` and
# `position_acc_best_offset` as the same column name.
_LABELS = {
    "f_beat": "f_beat",
    "cmlt": "cmlt",
    "amlt": "amlt",
    "position_acc": "pos_acc",
    "position_acc_best_offset": "best_off",
    "anchor_error": "anchor",
    "f_one": "f_one",
    "f_last": "f_last",
    "confusion": "confuse",
}

# Headers for the swept knob columns, same reason.
_KNOB_LABELS = {
    "beat_threshold": "beat_thr",
    "min_distance_frames": "min_dist",
    "gate_tolerance": "gate_tol",
    "anchor_threshold": "anchor_thr",
    "switch_penalty": "switch_pen",
}

# Which direction wins, per metric. `confusion` is the only lower-is-better
# member of the canonical set; everything else is a rate or an F-measure.
_BETTER = {key: (min if key == "confusion" else max) for key in SCORE_KEYS}

# What --rank-metric offers for picking a decoder or a bar-position knob.
# Stage 1 of the sweep is not on this list: it ranks by `f_beat`, which is
# the only thing beat-detection knobs can move.
_RANK_CHOICES = ("position_acc", "confusion")


def header_for(columns) -> str:
    """The right-aligned column header for a metric table."""

    return "".join(f" {_LABELS[c]:>9}" for c in columns)


def rank_key(metric: str, rank_by: str) -> str:
    """The :func:`~musicality.evaluation.summarize` key that ranks by *metric*.

    ``macro`` weights each corpus equally, ``micro`` each track. On a
    single-corpus split the two are the same number, so the choice can only
    matter on a merged split.
    """

    return f"macro_{metric}" if rank_by == "macro" else metric


def resolve_group_size(evaluator: BeatEvaluator, requested: int) -> int:
    """The group size actually in force, rebuilding the dataset if it differs.

    A softmax bar-position checkpoint carries its own ``group_size``
    hyperparameter, and that one is authoritative — it is what the model's
    position channels mean, and it is what
    :func:`~musicality.inference.run_inference` obeys. ``--group-size`` only
    speaks for the older three-channel one/last head, which has no such hparam.

    The rebuild matters: ``group_size`` also decides how
    :func:`~musicality.loaders.beat_dataset.fold_positions` folds the
    *reference* annotations. Warning about a mismatch without rebuilding would
    score 8-way predictions against 4-way references.
    """

    module, _task, _dataset, _indices = evaluator.load()
    module_group_size = getattr(module, "hparams", {}).get("group_size")

    if module_group_size is None or module_group_size == requested:
        return requested

    print(
        f"[eval] checkpoint is a softmax bar-position head with "
        f"group_size={module_group_size} — overriding --group-size={requested} "
        f"and rebuilding the dataset so the references fold the same way"
    )
    evaluator.group_size = module_group_size
    evaluator._loaded = None
    evaluator._probs = None
    evaluator.load()

    return module_group_size


def require_beat_phase(task: str, mode: str) -> None:
    if task != "beat_phase":
        raise SystemExit(
            f"Checkpoint task is {task!r} — {mode} only applies to beat_phase "
            "checkpoints (a beat-only model has no bar-position heads)."
        )


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write *rows* as CSV, taking the header from the first row."""

    if not rows:
        print(f"[eval] nothing to write to {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[eval] {len(rows)} row(s) written to {path}")


def print_genre_breakdown(rows: list[dict]) -> None:
    """Print per-corpus scores, then the macro mean, the micro mean and the
    weakest corpus.

    Macro averages the corpora; micro averages the tracks. They agree only when
    every corpus is the same size, and the gap between them is itself the
    diagnostic: it measures how much the blended number is being carried by the
    largest corpus. A model that is excellent on the biggest corpus and useless
    on the smallest still scores well on micro.
    """

    grouped = group_by_corpus(rows)

    print(f"\n  {'corpus':<18} {'n':>5}{header_for(_GENRE_COLUMNS)}")

    def _line(label: str, n: int, stats: dict) -> None:
        print(
            f"  {label:<18} {n:5d}"
            + "".join(f" {_fmt(stats[col]):>9}" for col in _GENRE_COLUMNS)
        )

    per_corpus = {}
    for corpus, rs in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        per_corpus[corpus] = {k: _mean([r.get(k) for r in rs]) for k in SCORE_KEYS}
        _line(corpus or "<unknown>", len(rs), per_corpus[corpus])

    if len(per_corpus) < 2:
        return

    print("  " + "-" * (24 + 10 * len(_GENRE_COLUMNS)))

    _line(
        "MACRO (per corpus)",
        len(per_corpus),
        {k: _mean([s[k] for s in per_corpus.values()]) for k in SCORE_KEYS},
    )
    _line(
        "micro (per track)",
        len(rows),
        {k: _mean([r.get(k) for r in rows]) for k in SCORE_KEYS},
    )

    summary = summarize(rows)
    if summary["worst_corpus"]:
        print(
            f"\n  Weakest corpus: {summary['worst_corpus']} "
            f"(position_acc {_fmt(summary['worst_position_acc'])}, "
            f"confusion up to {_fmt(summary['worst_confusion'])}) — the number "
            f"that gates 'works everywhere'."
        )


def print_offset_profile(rows: list[dict], group_size: int) -> None:
    """Print the per-track modal-offset histogram and within-track stability."""

    offsets = [r["modal_offset"] for r in rows if r.get("modal_offset") is not None]
    stabilities = [
        r["position_acc_best_offset"]
        for r in rows
        if r.get("position_acc_best_offset") is not None
    ]

    if not offsets:
        print("  (no track produced a resolvable phase offset)")
        return

    n = len(offsets)
    half = group_size // 2

    print(f"\n  Dominant phase offset per track (n={n}):")
    for offset in range(group_size):
        count = offsets.count(offset)
        if offset == 0:
            tag = "correct"
        elif offset == half:
            tag = "HALF-CYCLE"
        else:
            tag = "off-by-%d" % offset
        bar = "#" * int(round(40 * count / n))
        print(
            f"    offset {offset} {tag:>11s} : {count:4d} ({100 * count / n:5.1f}%) {bar}"
        )

    stab = np.array(stabilities)
    print(
        f"\n  Within-track phase stability (1.0 = never changes phase, n={len(stab)}):"
    )
    print(
        f"    mean {stab.mean():.3f}   median {np.median(stab):.3f}   "
        f"p10 {np.percentile(stab, 10):.3f}"
    )
    print(
        f"    stable   (>=0.95): {int((stab >= 0.95).sum()):4d} "
        f"({100 * (stab >= 0.95).mean():5.1f}%)"
    )
    print(
        f"    flipping (< 0.80): {int((stab < 0.80).sum()):4d} "
        f"({100 * (stab < 0.80).mean():5.1f}%)"
    )


def decoder_variants(
    anchor_threshold: float, switch_penalties: list[float], advance_modes: list[str]
) -> list[tuple]:
    """The decoder configurations ``--decoders`` compares.

    Always the greedy count-forward decoder first (the historical baseline the
    verdict is phrased against), then the exact single-offset global decode and
    one Viterbi variant per resync cost, repeated per advance mode.

    :returns: ``(name, decoder, switch_penalty, advance)`` tuples.
    """

    variants = [(f"greedy (anchor={anchor_threshold:g})", "greedy", None, "index")]

    for advance in advance_modes:
        tag = f" [{advance}]" if len(advance_modes) > 1 else ""
        variants.append((f"global (exact, no resync){tag}", "global", None, advance))
        variants += [
            (f"global + viterbi (switch={p:g}){tag}", "global", p, advance)
            for p in switch_penalties
        ]

    return variants


def score_variants(
    evaluator: BeatEvaluator, variants: list[tuple], group_size: int
) -> dict[str, tuple[dict, list[dict]]]:
    """Score every decoder variant against the same cached probabilities.

    ``switch_penalty`` is passed by keyword **presence**, since ``None`` is
    itself a meaningful value here (the exact single-offset decode, no mid-track
    resync allowed) rather than "unset".

    :returns: ``{name: (summary, rows)}``, in variant order.
    """

    results = {}
    for name, decoder, switch_penalty, advance in variants:
        rows = evaluator.score(
            advance=advance,
            decoder=decoder,
            switch_penalty=switch_penalty,
            group_size=group_size,
        )
        results[name] = (summarize(rows), rows)

    return results


def headroom_recovered(metric: str, baseline: float, best: float) -> tuple:
    """How much of the baseline decoder's remaining error the best one removes.

    The verdict's thresholds have to mean the same thing whichever metric is
    ranking, and a raw relative change does not: ``confusion`` is an error rate,
    so halving it is a 50% relative move, while ``position_acc`` is an accuracy,
    where that same halving of the error reads as a few percent. Both go onto
    one scale by dividing the gain by the headroom that was actually there —
    ``baseline`` itself for an error rate, ``1 - baseline`` for an accuracy.

    :returns: ``(absolute_gain, share_of_the_headroom_recovered)``, both signed
        so that positive means better.
    """

    if metric == "confusion":
        headroom, gain = baseline, baseline - best
    else:
        headroom, gain = 1.0 - baseline, best - baseline

    return gain, (gain / headroom if headroom else 0.0)


def print_verdict(
    results: dict, baseline_name: str, best_name: str, metric: str, rank_by: str
) -> None:
    """Translate the decoder comparison into an explicit model-or-decoder call.

    The question is whether re-decoding the *same* probabilities recovers the
    error. If it does, the model's per-beat evidence was already there and the
    decoder was throwing it away; if it doesn't, no amount of postprocessing
    will help and the work belongs in the loss or the architecture.
    """

    baseline, baseline_rows = results[baseline_name]
    best, _ = results[best_name]
    key = rank_key(metric, rank_by)

    gain, relative = headroom_recovered(metric, baseline[key], best[key])

    stabilities = [
        r["position_acc_best_offset"]
        for r in baseline_rows
        if r.get("position_acc_best_offset") is not None
    ]
    mean_stability = float(np.mean(stabilities)) if stabilities else float("nan")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    print(
        f"\n  {metric} ({rank_by}): {baseline_name} {_fmt(baseline[key])} -> "
        f"{best_name} {_fmt(best[key])}  [{gain:+.3f}, "
        f"{100 * relative:+.0f}% of the remaining error]"
    )
    print(
        f"  Mean within-track phase stability under the baseline decoder: "
        f"{_fmt(mean_stability)}"
    )

    if relative >= 0.30:
        print(
            "\n  -> CAUSE (B): THE DECODER.\n"
            "     A better decode of the *same* probabilities recovers a large part of\n"
            "     the error, so the model's per-beat evidence was already there and\n"
            "     the baseline decoder was discarding it. Switch readout to\n"
            f"     '{best_name}' and re-tune. No retraining needed."
        )
    elif relative >= 0.10:
        print(
            "\n  -> MIXED, leaning (B).\n"
            "     The global decode helps meaningfully but doesn't close the gap. Take\n"
            "     the free decoder win, then move on to the loss/parameterization work\n"
            "     in docs/beat_phase_improvement_review.md."
        )
    else:
        print(
            "\n  -> CAUSE (A): THE MODEL.\n"
            "     Decoding the same probabilities optimally over the whole track barely\n"
            "     helps, so the per-beat evidence itself is wrong. Postprocessing is not\n"
            "     the bottleneck — go to the loss/parameterization work in\n"
            "     docs/beat_phase_improvement_review.md."
        )

    if math.isnan(mean_stability):
        return

    if mean_stability >= 0.90:
        print(
            "\n  Supporting evidence: phase is highly stable within tracks, so errors\n"
            "  are whole-track offsets rather than mid-track flips — consistent with\n"
            "  an acoustic/model limitation."
        )
    elif mean_stability < 0.80:
        print(
            "\n  Supporting evidence: phase is unstable within tracks (it flips partway\n"
            "  through), which is the signature of a decoder losing information rather\n"
            "  than a model that cannot hear downbeats."
        )


def sweep_grid(
    evaluator: BeatEvaluator,
    grid: list[dict],
    *,
    group_size: int,
    metric: str,
    rank_by: str,
) -> list[dict]:
    """Score one list of knob dicts, best first.

    Each entry of *grid* is a set of overrides for
    :meth:`~musicality.evaluation.BeatEvaluator.score`. Because
    ``compute_track_probs`` is memoized, every entry re-runs only the decoder.

    :returns: One row per grid entry — the knobs it used, plus every micro and
        macro aggregate :func:`~musicality.evaluation.summarize` produces —
        sorted by *metric* in its own better-is direction.
    """

    key = rank_key(metric, rank_by)
    higher_is_better = _BETTER[metric] is max

    scored = [
        {**knobs, **summarize(evaluator.score(group_size=group_size, **knobs))}
        for knobs in grid
    ]

    def _rank(row: dict) -> float:
        """Map every metric onto one ascending "worse is larger" axis, so a
        single plain sort serves both directions and NaN (nothing scorable
        under these knobs) always lands last rather than winning."""

        value = row[key]
        if math.isnan(value):
            return math.inf

        return -value if higher_is_better else value

    return sorted(scored, key=_rank)


def _knob_cell(value) -> str:
    """One swept knob value, right-aligned. ``None`` prints as ``null`` — it is
    a real ``switch_penalty`` (forbid mid-track resyncs), not a missing value."""

    if value is None:
        return f"{'null':>13}"

    return f"{value:>13.2f}" if isinstance(value, float) else f"{value:>13}"


def print_sweep_table(rows: list[dict], knobs: list[str], columns, top: int) -> None:
    knob_header = "".join(f"{_KNOB_LABELS[k]:>13}" for k in knobs)
    print(f"\n{knob_header}{header_for(columns)}")

    for row in rows[:top]:
        print(
            "".join(_knob_cell(row[k]) for k in knobs)
            + "".join(f" {_fmt(row[c]):>9}" for c in columns)
        )


def run_sweep(evaluator: BeatEvaluator, args, task: str, group_size: int) -> None:
    """Two-stage grid search over the postprocessing knobs.

    Stage 1 sweeps beat detection (``beat_threshold``/``min_distance_frames``/
    ``gate_tolerance``) and ranks by ``f_beat``. Stage 2 holds the winner fixed
    and sweeps the one bar-position knob the resolved decoder actually reads —
    ``anchor_threshold`` for ``greedy``, ``switch_penalty`` for ``global``.

    Sweeping the position knob of the *other* decoder is what the old
    ``sweep_beat_postprocess.py`` did: it hardcoded the one/last channels and
    never passed ``decoder``, so it tuned ``anchor_threshold`` for a greedy
    decode that the shipped config does not use. Routing through
    :meth:`~musicality.evaluation.BeatEvaluator.score` makes that impossible —
    the sweep now decodes exactly the way evaluation does.
    """

    beat_grid = [
        {
            "beat_threshold": bt,
            "min_distance_frames": md,
            "gate_tolerance": gt,
        }
        for bt, md, gt in itertools.product(
            args.sweep_beat_thresholds,
            args.sweep_min_distances,
            args.sweep_gate_tolerances,
        )
    ]

    print("=" * 78)
    print(f"BEAT DETECTION SWEEP  ({len(beat_grid)} combinations, ranked by f_beat)")
    print("=" * 78)

    beat_rows = sweep_grid(
        evaluator,
        beat_grid,
        group_size=group_size,
        metric="f_beat",
        rank_by=args.rank_by,
    )
    print_sweep_table(
        beat_rows, list(beat_grid[0]), ("f_beat", "cmlt", "amlt"), args.top
    )

    best_beat = {k: beat_rows[0][k] for k in beat_grid[0]}
    print(f"\n  Best: {best_beat} -> f_beat {_fmt(beat_rows[0]['f_beat'])}")

    if task != "beat_phase":
        print(
            "\n[eval] checkpoint is beat_only — skipping the bar-position sweep "
            "(no bar-position heads)"
        )
        if args.output is not None:
            write_rows(args.output, beat_rows)
        return

    decoder = evaluator.resolve_postprocess()["decoder"]

    if decoder == "greedy":
        knob = "anchor_threshold"
        values = args.sweep_anchor_thresholds
    else:
        # The exact single-offset decode is always in the running: it is the
        # switch_penalty -> infinity limit, and no finite value reaches it.
        knob = "switch_penalty"
        values = [None] + list(args.switch_penalties)

    position_grid = [{**best_beat, knob: value} for value in values]

    print("\n" + "=" * 78)
    print(
        f"BAR-POSITION SWEEP  (decoder={decoder}, {len(position_grid)} values of "
        f"{knob}, ranked by {args.rank_metric})"
    )
    print("=" * 78)

    position_rows = sweep_grid(
        evaluator,
        position_grid,
        group_size=group_size,
        metric=args.rank_metric,
        rank_by=args.rank_by,
    )
    print_sweep_table(position_rows, [knob], _DECODER_COLUMNS, args.top)

    best = position_rows[0]
    print("\n  Update configs/eval_beat.yaml under `beat_phase:`:")
    for key in (*best_beat, "decoder", knob):
        value = decoder if key == "decoder" else best.get(key, best_beat.get(key))
        print(f"    {key}: {'null' if value is None else value}")

    if args.output is not None:
        write_rows(args.output, beat_rows)
        write_rows(
            args.output.with_name(f"{args.output.stem}_position.csv"), position_rows
        )


def run_decoders(evaluator: BeatEvaluator, args, group_size: int) -> list[dict]:
    """Score every decoder variant, print the comparison, verdict and (when
    asked) the per-genre and phase-offset sections for the winner.

    :returns: The per-track rows of every variant, tagged with ``decoder``, for
        ``--output``.
    """

    advance_modes = ["index", "time"] if args.advance == "both" else [args.advance]
    knobs = evaluator.resolve_postprocess()
    variants = decoder_variants(
        knobs["anchor_threshold"], args.switch_penalties, advance_modes
    )

    print("=" * 78)
    print(f"DECODER COMPARISON  (split={args.split}, same cached probabilities)")
    print("=" * 78)

    results = score_variants(evaluator, variants, group_size)

    print(f"\n  {'decoder':<36}{header_for(_DECODER_COLUMNS)}")
    for name, (summary, _rows) in results.items():
        print(
            f"  {name:<36}"
            + "".join(f" {_fmt(summary[c]):>9}" for c in _DECODER_COLUMNS)
        )

    baseline_name = variants[0][0]
    key = rank_key(args.rank_metric, args.rank_by)
    best_name = _BETTER[args.rank_metric](
        results, key=lambda name: results[name][0][key]
    )
    best_rows = results[best_name][1]

    if args.profile:
        print("\n" + "=" * 78)
        print(f"PHASE-OFFSET PROFILE  (under {baseline_name}, split={args.split})")
        print("=" * 78)
        print_offset_profile(results[baseline_name][1], group_size)

    if args.per_genre:
        print("\n" + "=" * 78)
        print(f"PER-GENRE BREAKDOWN  (under {best_name}, split={args.split})")
        print("=" * 78)
        print_genre_breakdown(best_rows)

    print_verdict(results, baseline_name, best_name, args.rank_metric, args.rank_by)

    return [
        {"decoder": name, **row} for name, (_s, rows) in results.items() for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a beat-only or beat-phase checkpoint (task auto-detected) "
            "on full-length tracks."
        )
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a Lightning .ckpt file"
    )
    parser.add_argument(
        "--dataset", default=EVAL_DEFAULTS["dataset"], help="mirdata dataset name"
    )
    parser.add_argument(
        "--data-home", default=None, help=f"Defaults to {DATA_DIR}/<dataset>"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "all"], default=EVAL_DEFAULTS["split"]
    )
    parser.add_argument("--val-split", type=float, default=EVAL_DEFAULTS["val_split"])
    parser.add_argument("--sample-rate", type=int, default=EVAL_DEFAULTS["sample_rate"])
    parser.add_argument("--hop-length", type=int, default=EVAL_DEFAULTS["hop_length"])
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "Beats per group: 4 for bar position (default), 8 for phrase "
            "position. beat-phase only, and ignored when the checkpoint "
            "carries its own group_size."
        ),
    )
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help=(
            "Drop tracks whose beats-per-bar isn't a multiple of 2 (e.g. "
            "ballroom's waltz tracks). Must match how the split was created."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=EVAL_DEFAULTS["tolerance"],
        help="F-measure matching window, seconds",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable mir_eval's standard 5s warm-up trim (use for short clips)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N selected tracks",
    )
    parser.add_argument("--device", default=EVAL_DEFAULTS["device"])

    knobs = parser.add_argument_group(
        "postprocessing",
        "Each defaults to the tuned value for the checkpoint's detected task "
        "(configs/eval_beat.yaml).",
    )
    knobs.add_argument("--beat-threshold", type=float, default=None)
    knobs.add_argument("--min-distance-frames", type=int, default=None)
    knobs.add_argument("--gate-tolerance", type=float, default=None)
    knobs.add_argument(
        "--anchor-threshold",
        type=float,
        default=None,
        help="beat-phase only, and only when --decoder greedy",
    )
    knobs.add_argument(
        "--decoder",
        choices=["greedy", "global"],
        default=None,
        help="Bar-position stage (beat-phase only)",
    )
    knobs.add_argument(
        "--switch-penalty",
        type=float,
        default=None,
        help="--decoder global only: log-cost of a mid-track phase resync",
    )
    knobs.add_argument(
        "--advance",
        choices=["index", "time", "both"],
        default="index",
        help=(
            "How the global decoder advances bar position per beat: 'index' "
            "(one position per detected beat), 'time' (derived from the "
            "elapsed time, so a missed or spurious detection does not shift "
            "the grid), or 'both' to score them side by side under --decoders."
        ),
    )

    report = parser.add_argument_group("report sections")
    report.add_argument(
        "--per-genre",
        action="store_true",
        help="Per-corpus breakdown. Automatic when the split spans >1 corpus.",
    )
    report.add_argument(
        "--profile",
        action="store_true",
        help="Phase-offset histogram and within-track stability. beat-phase only.",
    )
    report.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write per-track rows to this CSV path. Under --sweep, writes the "
            "grid tables instead (bar position to <stem>_position.csv)."
        ),
    )

    mode = parser.add_argument_group("modes").add_mutually_exclusive_group()
    mode.add_argument(
        "--decoders",
        action="store_true",
        help="Compare every bar-position decoder variant. beat-phase only.",
    )
    mode.add_argument(
        "--sweep",
        action="store_true",
        help="Grid-search the postprocessing knobs.",
    )

    ranking = parser.add_argument_group("ranking (--decoders / --sweep)")
    ranking.add_argument(
        "--rank-metric",
        choices=_RANK_CHOICES,
        default="position_acc",
        help=(
            "Which metric picks the winner. 'position_acc' is the canonical "
            "headline (higher is better); 'confusion' is the half-cycle rate "
            "(lower is better) that every comparison in plans/04 and plans/05 "
            "is quoted in."
        ),
    )
    ranking.add_argument(
        "--rank-by",
        choices=["macro", "micro"],
        default="macro",
        help=(
            "Which mean picks the winner on a multi-corpus split: 'macro' "
            "weights each corpus equally, 'micro' weights each track (which "
            "lets the largest corpus choose for all of them). No effect on a "
            "single-corpus split, where the two are the same number."
        ),
    )
    ranking.add_argument(
        "--switch-penalties",
        type=float,
        nargs="*",
        default=SWEEP_DEFAULTS["switch_penalties"],
        help=(
            "Viterbi resync costs (log-units) to try, in addition to the exact "
            "no-resync global decode. Pass none to skip them."
        ),
    )

    sweep = parser.add_argument_group("sweep grid (--sweep)")
    sweep.add_argument(
        "--sweep-beat-thresholds",
        type=float,
        nargs="+",
        default=SWEEP_DEFAULTS["beat_thresholds"],
        help="pick_peaks `threshold` candidates",
    )
    sweep.add_argument(
        "--sweep-min-distances",
        type=int,
        nargs="+",
        default=SWEEP_DEFAULTS["min_distance_frames"],
        help="pick_peaks `min_distance` candidates",
    )
    sweep.add_argument(
        "--sweep-gate-tolerances",
        type=float,
        nargs="+",
        default=SWEEP_DEFAULTS["gate_tolerances"],
        help="gate_periodicity `tolerance` candidates",
    )
    sweep.add_argument(
        "--sweep-anchor-thresholds",
        type=float,
        nargs="+",
        default=SWEEP_DEFAULTS["anchor_thresholds"],
        help="label_bar_position `anchor_threshold` candidates (--decoder greedy only)",
    )
    sweep.add_argument(
        "--top",
        type=int,
        default=SWEEP_DEFAULTS["top"],
        help="Print only the top N combinations per table",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    evaluator = BeatEvaluator(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        data_home=args.data_home,
        split=args.split,
        val_split=args.val_split,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        group_size=args.group_size,
        binary_only=args.binary_only,
        tolerance=args.tolerance,
        trim=not args.no_trim,
        beat_threshold=args.beat_threshold,
        min_distance_frames=args.min_distance_frames,
        gate_tolerance=args.gate_tolerance,
        anchor_threshold=args.anchor_threshold,
        decoder=args.decoder,
        switch_penalty=args.switch_penalty,
        limit=args.limit,
        device=args.device,
        verbose=not (args.decoders or args.sweep),
    )

    _module, task, _dataset, indices = evaluator.load()

    if args.decoders:
        require_beat_phase(task, "--decoders")
    if args.profile:
        require_beat_phase(task, "--profile")

    group_size = resolve_group_size(
        evaluator, args.group_size if args.group_size is not None else 4
    )

    if args.decoders or args.sweep:
        print(
            f"[eval] caching frame probabilities for {len(indices)} track(s) from "
            f"'{args.dataset}' (split={args.split}, task={task})"
        )

    evaluator.compute_track_probs()  # memoized: every mode below reuses this

    corpora = evaluator.track_corpora()
    n_corpora = len(set(corpora))
    if n_corpora > 1:
        args.per_genre = True
        print(f"[eval] {n_corpora} source corpora — reporting per genre\n")

    if args.sweep:
        run_sweep(evaluator, args, task, group_size)
        return

    if args.decoders:
        rows = run_decoders(evaluator, args, group_size)
    else:
        rows = evaluator.run()

        if args.profile:
            print("\n" + "=" * 78)
            print(f"PHASE-OFFSET PROFILE  (split={args.split})")
            print("=" * 78)
            print_offset_profile(rows, group_size)

        if args.per_genre:
            print("\n" + "=" * 78)
            print(f"PER-GENRE BREAKDOWN  (split={args.split})")
            print("=" * 78)
            print_genre_breakdown(rows)

    if args.output is not None:
        write_rows(args.output, rows)


if __name__ == "__main__":
    main()
