#!/usr/bin/env python3
"""Diagnose *why* a beat-phase checkpoint gets bar position wrong.

Answers the question raised in ``docs/beat_phase_improvement_review.md``: is
the ~25% half-cycle confusion rate caused by

- **(A) the model** — its per-beat evidence genuinely doesn't say where the
  downbeat is, so the whole track locks onto one wrong phase; or
- **(B) the decoder** — the evidence is there on average, but
  :func:`musicality.postprocess.label_bar_position`'s greedy count-forward
  throws it away and lets the phase flip mid-track.

The two have opposite fixes (retrain with a better objective vs. change a
postprocessing function), so this runs two complementary experiments over the
*same* cached model output:

1. **Decoder comparison** — scores the greedy decoder against
   :func:`musicality.postprocess.label_bar_position_global`, a whole-track
   maximum-likelihood decode, plus optional Viterbi variants that allow a
   penalized mid-track resync. Same probabilities, no retraining. If the
   global decode fixes it, it was (B).
2. **Phase-offset profile** — per track, the dominant prediction-vs-reference
   phase offset and how *stable* that offset is within the track (see
   :func:`musicality.metrics.position_accuracy.position_accuracy`). Stable-but-
   wrong points at (A); unstable points at (B). This explains *why* the
   decoder comparison came out the way it did.

Run it on the **train** split first: a model that can't place downbeats on
tracks it has already memorized is a fit problem, not a generalization one.

Usage
-----
    uv run python tools/diagnose_beat_phase.py \\
        --checkpoint "checkpoints_beat/loss=1.6565.ckpt" \\
        --dataset ballroom --binary-only --split train

    # both splits, and a quick run on a handful of tracks
    uv run python tools/diagnose_beat_phase.py --checkpoint ... \\
        --dataset ballroom --binary-only --split val --limit 20

    # try other Viterbi resync costs
    uv run python tools/diagnose_beat_phase.py --checkpoint ... \\
        --switch-penalties 2 5 10 20

    # save per-track rows for further analysis
    uv run python tools/diagnose_beat_phase.py --checkpoint ... --output diag.csv
"""

import argparse
import csv
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


def print_genre_breakdown(rows: list[dict], decoder_name: str) -> None:
    """Print per-corpus scores, then the macro mean, the micro mean and the
    worst corpus.

    Macro averages the corpora; micro averages the tracks. They agree only when
    every corpus is the same size, and the gap between them is itself the
    diagnostic: it measures how much the blended number is being carried by the
    largest corpus. A model that is excellent on the biggest corpus and useless
    on the smallest still scores well on micro.
    """

    grouped = group_by_corpus(rows)

    print(f"\n  decoder: {decoder_name}\n")
    print(
        f"  {'corpus':<18} {'n':>5} {'f_one':>7} {'f_last':>7} "
        f"{'confuse':>8} {'best_off':>10}"
    )

    def _row(label: str, n: int, stats: dict) -> None:
        print(
            f"  {label:<18} {n:5d} {_fmt(stats['f_one']):>7} "
            f"{_fmt(stats['f_last']):>7} {_fmt(stats['confusion']):>8} "
            f"{_fmt(stats['position_acc_best_offset']):>10}"
        )

    per_corpus = {}
    for corpus, rs in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        per_corpus[corpus] = {k: _mean([r[k] for r in rs]) for k in SCORE_KEYS}
        _row(corpus or "<unknown>", len(rs), per_corpus[corpus])

    if len(per_corpus) < 2:
        return

    print("  " + "-" * 58)

    _row(
        "MACRO (per corpus)",
        len(per_corpus),
        {k: _mean([s[k] for s in per_corpus.values()]) for k in SCORE_KEYS},
    )
    _row(
        "micro (per track)",
        len(rows),
        {k: _mean([r[k] for r in rows]) for k in SCORE_KEYS},
    )

    ranked = [
        (corpus, stats)
        for corpus, stats in per_corpus.items()
        if not math.isnan(stats["confusion"])
    ]
    if ranked:
        name, stats = max(ranked, key=lambda kv: kv[1]["confusion"])
        print(
            f"\n  Worst corpus by confusion: {name} "
            f"({_fmt(stats['confusion'])}) — the number that gates "
            f"'works everywhere'."
        )


def print_offset_profile(rows: list[dict], group_size: int) -> None:
    """Print the per-track modal-offset histogram and within-track stability."""

    offsets = [r["modal_offset"] for r in rows if r["modal_offset"] is not None]
    stabilities = [
        r["position_acc_best_offset"]
        for r in rows
        if r["position_acc_best_offset"] is not None
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


def print_verdict(
    greedy: dict,
    best_name: str,
    best: dict,
    greedy_rows: list[dict],
    group_size: int,
) -> None:
    """Translate the two experiments into an explicit A-or-B call."""

    stabilities = [
        r["position_acc_best_offset"]
        for r in greedy_rows
        if r["position_acc_best_offset"] is not None
    ]
    mean_stability = float(np.mean(stabilities)) if stabilities else float("nan")

    delta = greedy["confusion"] - best["confusion"]
    relative = delta / greedy["confusion"] if greedy["confusion"] else 0.0

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    print(
        f"\n  Greedy confusion {greedy['confusion']:.3f} -> best decoder "
        f"({best_name}) {best['confusion']:.3f}  "
        f"[{delta:+.3f}, {100 * relative:+.0f}% relative]"
    )
    print(
        f"  Mean within-track phase stability under the greedy decoder: {mean_stability:.3f}"
    )

    if relative >= 0.30:
        print(
            "\n  -> CAUSE (B): THE DECODER.\n"
            "     A better decode of the *same* probabilities recovers a large part of\n"
            "     the error, so the model's per-beat evidence was already there and\n"
            f"     `label_bar_position` was discarding it. Switch readout to\n"
            f"     decoder='{best_name}' and re-tune. No retraining needed."
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

    if not math.isnan(mean_stability):
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


def parse_args() -> argparse.Namespace:
    defaults = EVAL_DEFAULTS["beat_phase"]

    parser = argparse.ArgumentParser(
        description="Diagnose whether beat-phase errors come from the model or the decoder."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a Lightning .ckpt file"
    )
    parser.add_argument("--dataset", default=EVAL_DEFAULTS["dataset"])
    parser.add_argument(
        "--data-home", default=None, help=f"Defaults to {DATA_DIR}/<dataset>"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="train",
        help="Defaults to train — a fit problem shows up there first.",
    )
    parser.add_argument("--val-split", type=float, default=EVAL_DEFAULTS["val_split"])
    parser.add_argument("--sample-rate", type=int, default=EVAL_DEFAULTS["sample_rate"])
    parser.add_argument("--hop-length", type=int, default=EVAL_DEFAULTS["hop_length"])
    parser.add_argument("--group-size", type=int, default=defaults.get("group_size", 4))
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help="Must match how the checkpoint's split was created.",
    )
    parser.add_argument("--tolerance", type=float, default=EVAL_DEFAULTS["tolerance"])
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable mir_eval's standard 5s warm-up trim",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N selected tracks"
    )
    parser.add_argument("--device", default=EVAL_DEFAULTS["device"])
    parser.add_argument(
        "--beat-threshold", type=float, default=defaults["beat_threshold"]
    )
    parser.add_argument(
        "--min-distance-frames", type=int, default=defaults["min_distance_frames"]
    )
    parser.add_argument(
        "--gate-tolerance", type=float, default=defaults["gate_tolerance"]
    )
    parser.add_argument(
        "--anchor-threshold",
        type=float,
        default=defaults["anchor_threshold"],
        help="Greedy decoder only — the global decoder has no threshold.",
    )
    parser.add_argument(
        "--switch-penalties",
        type=float,
        nargs="*",
        default=[5.0, 20.0],
        help=(
            "Viterbi resync costs (log-units) to try in addition to the exact "
            "no-resync global decode. Pass none to skip them."
        ),
    )
    parser.add_argument(
        "--advance",
        choices=["index", "time", "both"],
        default="index",
        help=(
            "How the global decoder advances bar position per beat: 'index' "
            "(one position per detected beat), 'time' (derived from the "
            "elapsed time, so a missed or spurious detection does not shift "
            "the grid), or 'both' to score them side by side."
        ),
    )
    parser.add_argument(
        "--rank-by",
        choices=["macro", "micro"],
        default="macro",
        help=(
            "Which mean picks the best decoder on a multi-corpus split: "
            "'macro' weights each corpus equally, 'micro' weights each track "
            "(which lets the largest corpus choose the decoder for all of "
            "them). No effect on a single-corpus split, where the two are "
            "the same number."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write per-track rows for every decoder variant to this CSV path",
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
        limit=args.limit,
        device=args.device,
        verbose=False,
    )
    module, task, _dataset, indices = evaluator.load()

    if task != "beat_phase":
        raise SystemExit(
            f"Checkpoint task is {task!r} — this diagnostic only applies to "
            "beat_phase checkpoints (a beat-only model has no bar-position heads)."
        )

    # A softmax bar-position checkpoint's own group_size is authoritative —
    # mirrors musicality.inference.run_inference. Falls back to --group-size
    # for the older three-channel one/last head, which has no such hparam.
    module_group_size = getattr(module, "hparams", {}).get("group_size")
    group_size = module_group_size if module_group_size is not None else args.group_size
    if module_group_size is not None and module_group_size != args.group_size:
        print(
            f"[diagnose] checkpoint is a softmax bar-position head with "
            f"group_size={module_group_size} — overriding --group-size={args.group_size}"
        )

    print(
        f"[diagnose] caching frame probabilities for {len(indices)} track(s) "
        f"from '{args.dataset}' (split={args.split})"
    )
    evaluator.compute_track_probs()  # memoized: every variant below reuses this
    corpora = evaluator.track_corpora()

    n_corpora = len(set(corpora))
    if n_corpora > 1:
        print(
            f"[diagnose] {n_corpora} source corpora — reporting per genre, "
            f"ranking decoders by {args.rank_by}\n"
        )

    advance_modes = ["index", "time"] if args.advance == "both" else [args.advance]

    variants = [(f"greedy (anchor={args.anchor_threshold:g})", "greedy", None, "index")]
    for adv in advance_modes:
        tag = f" [{adv}]" if len(advance_modes) > 1 else ""
        variants.append((f"global (exact, no resync){tag}", "global", None, adv))
        variants += [
            (f"global + viterbi (switch={p:g}){tag}", "global", p, adv)
            for p in args.switch_penalties
        ]

    print(f"[diagnose] scoring {len(variants)} decoder variant(s)...\n")

    print("=" * 78)
    print(f"DECODER COMPARISON  (split={args.split}, same cached probabilities)")
    print("=" * 78)
    print(
        f"\n  {'decoder':<36} {'f_one':>7} {'f_last':>7} {'confuse':>8} {'best_off':>10}"
    )

    results = {}
    all_rows = []

    for name, decoder, switch_penalty, adv in variants:
        # switch_penalty is passed by keyword *presence*, since None is itself
        # a meaningful value here (the exact single-offset decode).
        rows = evaluator.score(
            advance=adv,
            decoder=decoder,
            switch_penalty=switch_penalty,
            group_size=group_size,
        )
        summary = summarize(rows)
        results[name] = (summary, rows)

        print(
            f"  {name:<36} {_fmt(summary['f_one']):>7} {_fmt(summary['f_last']):>7} "
            f"{_fmt(summary['confusion']):>8} {_fmt(summary['position_acc_best_offset']):>10}"
        )

        for row in rows:
            all_rows.append({"decoder": name, **row})

    greedy_name = variants[0][0]
    greedy_summary, greedy_rows = results[greedy_name]

    print("\n" + "=" * 78)
    print(f"PHASE-OFFSET PROFILE  (under the greedy decoder, split={args.split})")
    print("=" * 78)
    print_offset_profile(greedy_rows, group_size)

    # On a single-corpus split "macro_confusion" == "confusion" by
    # construction, so the default ranking is unchanged for those runs.
    rank_key = "macro_confusion" if args.rank_by == "macro" else "confusion"
    best_name = min(results, key=lambda k: results[k][0][rank_key])

    print("\n" + "=" * 78)
    print(f"PER-GENRE BREAKDOWN  (split={args.split})")
    print("=" * 78)
    print_genre_breakdown(results[best_name][1], best_name)

    print_verdict(
        greedy_summary, best_name, results[best_name][0], greedy_rows, group_size
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[diagnose] per-track rows written to {args.output}")


if __name__ == "__main__":
    main()
