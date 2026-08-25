#!/usr/bin/env python3
"""Sweep postprocessing hyperparameters against a checkpoint, task
auto-detected (like :mod:`tools.eval_beat`).

Prints two tables:

- **Beat detection** (``beat_threshold``, ``min_distance_frames``,
  ``gate_tolerance`` — :func:`~musicality.postprocess.pick_peaks`/
  :func:`~musicality.postprocess.gate_periodicity`), scored by mean beat
  F-measure. Identical for both tasks — a beat-phase checkpoint's beat
  channel is scored the same way as a beat-only checkpoint's single output.
- **Bar position** (adds ``anchor_threshold`` —
  :func:`~musicality.postprocess.label_bar_position`), scored by
  mean("1" F-measure, "last" F-measure), with phase-confusion rate reported
  alongside. Beat-phase checkpoints only — skipped for beat-only checkpoints,
  which have no bar-position heads.

Runs the model once per track (the expensive part) and caches the resulting
frame probabilities, then re-scores every parameter combination in both grids
against those cached probabilities (cheap — just peak-picking, gating,
bar-position labeling, and mir_eval), so the sweep only pays the
model-inference cost once regardless of how many combinations are tried.

Usage
-----
    uv run python tools/sweep_beat_postprocess.py \\
        --checkpoint "checkpoints_beat_only/beat-only-epoch=89-val/loss=0.6449.ckpt" \\
        --binary-only --dataset ballroom

    # narrower/custom grid
    uv run python tools/sweep_beat_postprocess.py --checkpoint ... \\
        --beat-thresholds 0.2 0.3 0.4 --min-distance-frames 1 2 --gate-tolerances 0.15 0.2 0.3

    # also narrow the bar-position grid (beat-phase checkpoints only)
    uv run python tools/sweep_beat_postprocess.py --checkpoint ... \\
        --anchor-thresholds 0.4 0.5 0.6

    # save the full comparison tables to disk
    uv run python tools/sweep_beat_postprocess.py --checkpoint ... \\
        --output sweep_postprocess.csv --output-phase sweep_postprocess_phase.csv
"""

import argparse
import csv
import itertools
import math
from pathlib import Path

import yaml

import musicality.dataformats as dataformats
from musicality.evaluation import DATA_DIR, BeatEvaluator
from musicality.evaluation import DEFAULTS as EVAL_DEFAULTS
from musicality.metrics.confusion import confusion_half_cycle_rate
from musicality.metrics.f_measure import beat_f_measure, downbeat_f_measures
from musicality.postprocess import readout, readout_beat_only

# Default CLI values — see configs/sweep_beat_postprocess.yaml for what each means.
DEFAULTS = yaml.safe_load(
    (dataformats.ROOT / "configs" / "sweep_beat_postprocess.yaml").read_text()
)


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None or math.isnan(value) else f"{value:.3f}"


def score_combo(
    cached: list[tuple],
    fps: float,
    beat_threshold: float,
    min_distance_frames: int,
    gate_tolerance: float,
    tolerance: float,
    trim: bool,
) -> float:
    """Mean beat F-measure across cached tracks for one postprocessing combo."""

    scores = [
        beat_f_measure(
            beat_times,
            readout_beat_only(
                probs[0] if probs.ndim == 2 else probs,
                fps=fps,
                beat_threshold=beat_threshold,
                min_distance_frames=min_distance_frames,
                gate_tolerance=gate_tolerance,
            ),
            tolerance=tolerance,
            trim=trim,
        )
        for beat_times, _positions, _has_positions, probs in cached
    ]

    return sum(scores) / len(scores) if scores else float("nan")


def score_phase_combo(
    cached: list[tuple],
    fps: float,
    beat_threshold: float,
    min_distance_frames: int,
    gate_tolerance: float,
    anchor_threshold: float,
    group_size: int,
    tolerance: float,
    trim: bool,
) -> tuple[float, float, float]:
    """Mean f_one/f_last/confusion across cached tracks with position
    annotations, for one postprocessing combo. Tracks with no position
    annotations are skipped entirely (nothing to score bar-position
    against). Returns ``(mean_f_one, mean_f_last, mean_confusion)`` — the
    caller derives the sort objective (``mean(f_one, f_last)``) from the
    first two."""

    f_ones, f_lasts, confusions = [], [], []
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
        f_ones.append(f_one)
        f_lasts.append(f_last)
        if confusion is not None:
            confusions.append(confusion)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return _mean(f_ones), _mean(f_lasts), _mean(confusions)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep beat-detection and bar-position postprocessing hyperparameters against a beat-only or beat-phase checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a Lightning .ckpt file"
    )
    parser.add_argument(
        "--dataset", default=DEFAULTS["dataset"], help="mirdata dataset name"
    )
    parser.add_argument(
        "--data-home", default=None, help=f"Defaults to {DATA_DIR}/<dataset>"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "all"], default=DEFAULTS["split"]
    )
    parser.add_argument("--val-split", type=float, default=DEFAULTS["val_split"])
    parser.add_argument("--sample-rate", type=int, default=DEFAULTS["sample_rate"])
    parser.add_argument("--hop-length", type=int, default=DEFAULTS["hop_length"])
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "Beats per group for bar-position scoring: defaults to the tuned "
            "beat-phase value in configs/eval_beat.yaml. beat-phase only."
        ),
    )
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help="Must match how the checkpoint's split was created.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULTS["tolerance"],
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
    parser.add_argument("--device", default=DEFAULTS["device"])
    parser.add_argument(
        "--beat-thresholds",
        type=float,
        nargs="+",
        default=DEFAULTS["beat_thresholds"],
        help="Passed to pick_peaks as `threshold`",
    )
    parser.add_argument(
        "--min-distance-frames",
        type=int,
        nargs="+",
        default=DEFAULTS["min_distance_frames"],
        help="Passed to pick_peaks as `min_distance`",
    )
    parser.add_argument(
        "--gate-tolerances",
        type=float,
        nargs="+",
        default=DEFAULTS["gate_tolerances"],
        help="Passed to gate_periodicity as `tolerance`",
    )
    parser.add_argument(
        "--anchor-thresholds",
        type=float,
        nargs="+",
        default=DEFAULTS["anchor_thresholds"],
        help="Passed to label_bar_position as `anchor_threshold` (beat-phase checkpoints only)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULTS["top"],
        help="Print only the top N combinations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the full beat-detection comparison table to this CSV path",
    )
    parser.add_argument(
        "--output-phase",
        type=Path,
        default=None,
        help="Write the full bar-position comparison table to this CSV path (beat-phase checkpoints only)",
    )
    args = parser.parse_args()

    evaluator = BeatEvaluator(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        data_home=args.data_home,
        split=args.split,
        val_split=args.val_split,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        binary_only=args.binary_only,
        limit=args.limit,
        device=args.device,
    )
    _module, task, _dataset, indices = evaluator.load()

    print(
        f"[sweep_postprocess] caching frame probabilities for {len(indices)} "
        f"track(s) from '{args.dataset}' (split={args.split}, task={task})"
    )
    cached = evaluator.compute_track_probs()

    fps = args.sample_rate / args.hop_length
    grid = list(
        itertools.product(
            args.beat_thresholds, args.min_distance_frames, args.gate_tolerances
        )
    )
    print(f"[sweep_postprocess] scoring {len(grid)} beat-detection combination(s)...")

    results = [
        (
            beat_threshold,
            min_distance_frames,
            gate_tolerance,
            score_combo(
                cached,
                fps,
                beat_threshold,
                min_distance_frames,
                gate_tolerance,
                args.tolerance,
                not args.no_trim,
            ),
        )
        for beat_threshold, min_distance_frames, gate_tolerance in grid
    ]
    results.sort(key=lambda r: r[3], reverse=True)

    print(f"\n{'beat_thresh':>11}  {'min_dist':>8}  {'gate_tol':>8}  {'mean_F':>7}")
    for beat_threshold, min_distance_frames, gate_tolerance, mean_f in results[
        : args.top
    ]:
        print(
            f"{beat_threshold:>11.2f}  {min_distance_frames:>8d}  "
            f"{gate_tolerance:>8.2f}  {mean_f:>7.3f}"
        )

    best_threshold, best_min_distance, best_gate_tolerance, best_f = results[0]
    print(
        f"\nBest: beat_threshold={best_threshold} min_distance_frames={best_min_distance} "
        f"gate_tolerance={best_gate_tolerance} -> mean beat F-measure={best_f:.3f}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "beat_threshold",
                    "min_distance_frames",
                    "gate_tolerance",
                    "mean_f_measure",
                ]
            )
            writer.writerows(results)
        print(f"[sweep_postprocess] full results written to {args.output}")

    if task != "beat_phase":
        print(
            "\n[sweep_postprocess] checkpoint is beat_only — skipping "
            "anchor_threshold sweep (no bar-position heads)"
        )
        return

    group_size = (
        args.group_size
        if args.group_size is not None
        else EVAL_DEFAULTS["beat_phase"].get("group_size", 4)
    )
    phase_grid = list(
        itertools.product(
            args.beat_thresholds,
            args.min_distance_frames,
            args.gate_tolerances,
            args.anchor_thresholds,
        )
    )
    print(
        f"\n[sweep_postprocess] scoring {len(phase_grid)} bar-position combination(s)..."
    )

    phase_results = []
    for (
        beat_threshold,
        min_distance_frames,
        gate_tolerance,
        anchor_threshold,
    ) in phase_grid:
        f_one, f_last, confusion = score_phase_combo(
            cached,
            fps,
            beat_threshold,
            min_distance_frames,
            gate_tolerance,
            anchor_threshold,
            group_size,
            args.tolerance,
            not args.no_trim,
        )
        objective = (f_one + f_last) / 2
        phase_results.append(
            (
                beat_threshold,
                min_distance_frames,
                gate_tolerance,
                anchor_threshold,
                f_one,
                f_last,
                confusion,
                objective,
            )
        )
    phase_results.sort(key=lambda r: r[-1], reverse=True)

    print(
        f"\n{'beat_thresh':>11}  {'min_dist':>8}  {'gate_tol':>8}  {'anchor':>7}  "
        f"{'f_one':>7}  {'f_last':>7}  {'confuse':>7}"
    )
    for (
        beat_threshold,
        min_distance_frames,
        gate_tolerance,
        anchor_threshold,
        f_one,
        f_last,
        confusion,
        _objective,
    ) in phase_results[: args.top]:
        print(
            f"{beat_threshold:>11.2f}  {min_distance_frames:>8d}  "
            f"{gate_tolerance:>8.2f}  {anchor_threshold:>7.2f}  "
            f"{_fmt(f_one)}  {_fmt(f_last)}  {_fmt(confusion)}"
        )

    (
        best_p_threshold,
        best_p_min_distance,
        best_p_gate_tolerance,
        best_anchor,
        best_f_one,
        best_f_last,
        best_confusion,
        best_objective,
    ) = phase_results[0]
    print(
        f"\nBest (phase): beat_threshold={best_p_threshold} "
        f"min_distance_frames={best_p_min_distance} "
        f"gate_tolerance={best_p_gate_tolerance} anchor_threshold={best_anchor} -> "
        f"mean('1','last') F-measure={best_objective:.3f} "
        f"(f_one={_fmt(best_f_one)} f_last={_fmt(best_f_last)} confusion={_fmt(best_confusion)})"
    )

    if args.output_phase is not None:
        args.output_phase.parent.mkdir(parents=True, exist_ok=True)
        with args.output_phase.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "beat_threshold",
                    "min_distance_frames",
                    "gate_tolerance",
                    "anchor_threshold",
                    "f_one",
                    "f_last",
                    "confusion",
                    "objective",
                ]
            )
            writer.writerows(phase_results)
        print(f"[sweep_postprocess] full phase results written to {args.output_phase}")


if __name__ == "__main__":
    main()
