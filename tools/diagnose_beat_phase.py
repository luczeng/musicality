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
   :func:`musicality.metrics.phase_offset.phase_offset_profile`). Stable-but-
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

from musicality.evaluation import DATA_DIR, BeatEvaluator
from musicality.evaluation import DEFAULTS as EVAL_DEFAULTS
from musicality.metrics.confusion import confusion_half_cycle_rate
from musicality.metrics.f_measure import downbeat_f_measures
from musicality.metrics.phase_offset import phase_offset_profile
from musicality.postprocess import readout


# Decoders compared by default. `switch_penalty=None` means the exact
# single-offset decode (no mid-track resync allowed at all).
BASE_VARIANTS = ("greedy", "global")


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]

    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None or math.isnan(value) else f"{value:.3f}"


def score_decoder(
    cached: list[tuple],
    fps: float,
    *,
    decoder: str,
    switch_penalty: float | None,
    advance: str = "index",
    beat_threshold: float,
    min_distance_frames: int,
    gate_tolerance: float,
    anchor_threshold: float,
    group_size: int,
    tolerance: float,
    trim: bool,
) -> tuple[dict, list[dict]]:
    """Score one bar-position decoder across every position-annotated cached track.

    :returns: ``(summary, per_track)`` — aggregate means, and one row per
        track carrying both its F-measures and its phase-offset profile.
    """

    rows = []

    for beat_times, positions, has_positions, probs in cached:
        if not has_positions:
            continue

        events = readout(
            probs[0],
            probs[1],
            probs[2],
            fps=fps,
            beat_threshold=beat_threshold,
            min_distance_frames=min_distance_frames,
            gate_tolerance=gate_tolerance,
            anchor_threshold=anchor_threshold,
            group_size=group_size,
            decoder=decoder,
            switch_penalty=switch_penalty,
            advance=advance,
        )

        f_one, f_last = downbeat_f_measures(
            beat_times,
            positions,
            events,
            tolerance=tolerance,
            trim=trim,
            group_size=group_size,
        )
        confusion = confusion_half_cycle_rate(
            beat_times, positions, events, tolerance=tolerance, group_size=group_size
        )
        profile = phase_offset_profile(
            beat_times, positions, events, tolerance=tolerance, group_size=group_size
        )

        rows.append(
            {
                "f_one": f_one,
                "f_last": f_last,
                "confusion": confusion,
                "modal_offset": profile["modal_offset"] if profile else None,
                "stability": profile["stability"] if profile else None,
                "correct_fraction": profile["correct_fraction"] if profile else None,
            }
        )

    summary = {
        "n_tracks": len(rows),
        "f_one": _mean([r["f_one"] for r in rows]),
        "f_last": _mean([r["f_last"] for r in rows]),
        "confusion": _mean([r["confusion"] for r in rows]),
        "stability": _mean([r["stability"] for r in rows]),
    }
    summary["objective"] = (summary["f_one"] + summary["f_last"]) / 2

    return summary, rows


def print_offset_profile(rows: list[dict], group_size: int) -> None:
    """Print the per-track modal-offset histogram and within-track stability."""

    offsets = [r["modal_offset"] for r in rows if r["modal_offset"] is not None]
    stabilities = [r["stability"] for r in rows if r["stability"] is not None]

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

    stabilities = [r["stability"] for r in greedy_rows if r["stability"] is not None]
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
            "     (steps 2-3 of docs/beat_phase_improvement_review.md)."
        )
    else:
        print(
            "\n  -> CAUSE (A): THE MODEL.\n"
            "     Decoding the same probabilities optimally over the whole track barely\n"
            "     helps, so the per-beat evidence itself is wrong. Postprocessing is not\n"
            "     the bottleneck — go to steps 2-3 of\n"
            "     docs/beat_phase_improvement_review.md (beat-conditioned phase loss,\n"
            "     then a group_size-way softmax over positions)."
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
        limit=args.limit,
        device=args.device,
        verbose=False,
    )
    _module, task, _dataset, indices = evaluator.load()

    if task != "beat_phase":
        raise SystemExit(
            f"Checkpoint task is {task!r} — this diagnostic only applies to "
            "beat_phase checkpoints (a beat-only model has no bar-position heads)."
        )

    print(
        f"[diagnose] caching frame probabilities for {len(indices)} track(s) "
        f"from '{args.dataset}' (split={args.split})"
    )
    cached = evaluator.compute_track_probs()
    fps = args.sample_rate / args.hop_length

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
        f"\n  {'decoder':<36} {'f_one':>7} {'f_last':>7} {'confuse':>8} {'stability':>10}"
    )

    results = {}
    all_rows = []

    for name, decoder, switch_penalty, adv in variants:
        summary, rows = score_decoder(
            cached,
            fps,
            decoder=decoder,
            switch_penalty=switch_penalty,
            advance=adv,
            beat_threshold=args.beat_threshold,
            min_distance_frames=args.min_distance_frames,
            gate_tolerance=args.gate_tolerance,
            anchor_threshold=args.anchor_threshold,
            group_size=args.group_size,
            tolerance=args.tolerance,
            trim=not args.no_trim,
        )
        results[name] = (summary, rows)

        print(
            f"  {name:<36} {_fmt(summary['f_one']):>7} {_fmt(summary['f_last']):>7} "
            f"{_fmt(summary['confusion']):>8} {_fmt(summary['stability']):>10}"
        )

        for row in rows:
            all_rows.append({"decoder": name, **row})

    greedy_name = variants[0][0]
    greedy_summary, greedy_rows = results[greedy_name]

    print("\n" + "=" * 78)
    print(f"PHASE-OFFSET PROFILE  (under the greedy decoder, split={args.split})")
    print("=" * 78)
    print_offset_profile(greedy_rows, args.group_size)

    best_name = min(results, key=lambda k: results[k][0]["confusion"])
    print_verdict(
        greedy_summary, best_name, results[best_name][0], greedy_rows, args.group_size
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
