#!/usr/bin/env python3
"""Sweep beat-only postprocessing hyperparameters (``beat_threshold``,
``min_distance_frames``, ``gate_tolerance``) against a checkpoint, scoring
each combination by mean beat F-measure.

Runs the model once per track (the expensive part) and caches the resulting
frame probabilities, then re-scores every parameter combination against
those cached probabilities (cheap — just peak-picking, gating, and
mir_eval), so the sweep only pays the model-inference cost once regardless
of how many combinations are tried.

Usage
-----
    uv run python tools/sweep_beat_postprocess.py \\
        --checkpoint "checkpoints_beat_only/beat-only-epoch=89-val/loss=0.6449.ckpt" \\
        --binary-only --dataset ballroom

    # narrower/custom grid
    uv run python tools/sweep_beat_postprocess.py --checkpoint ... \\
        --beat-thresholds 0.2 0.3 0.4 --min-distance-frames 1 2 --gate-tolerances 0.15 0.2 0.3

    # save the full comparison table to disk
    uv run python tools/sweep_beat_postprocess.py --checkpoint ... --output sweep_postprocess.csv
"""

import argparse
import csv
import itertools
from pathlib import Path

import torch

import musicality.dataformats as dataformats
from musicality.loaders.beat_dataset import BeatDataset, select_indices
from musicality.metrics.f_measure import beat_f_measure
from musicality.postprocess import readout_beat_only
from tools.eval_beat_only import load_module, load_track_waveform

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir


@torch.no_grad()
def compute_track_probs(
    module,
    dataset: BeatDataset,
    indices: list[int],
    sample_rate: int,
    device: torch.device,
) -> list[tuple]:
    """Run the model once per track, returning cached ``(beat_times, probs)`` pairs."""

    cached = []
    for i in indices:
        audio_path, beat_times, _positions, _has_positions = dataset.samples[i]

        wav = load_track_waveform(audio_path, sample_rate).unsqueeze(0).to(device)
        logits = module(wav)  # (1, T')
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # (T',)

        cached.append((beat_times, probs))

    return cached


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
                probs,
                fps=fps,
                beat_threshold=beat_threshold,
                min_distance_frames=min_distance_frames,
                gate_tolerance=gate_tolerance,
            ),
            tolerance=tolerance,
            trim=trim,
        )
        for beat_times, probs in cached
    ]

    return sum(scores) / len(scores) if scores else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep beat-only postprocessing hyperparameters against a checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a Lightning .ckpt file"
    )
    parser.add_argument("--dataset", default="ballroom", help="mirdata dataset name")
    parser.add_argument("--data-home", default=None, help="Defaults to data/<dataset>")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help="Must match how the checkpoint's split was created.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.07,
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--beat-thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6],
        help="Passed to pick_peaks as `threshold`",
    )
    parser.add_argument(
        "--min-distance-frames",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Passed to pick_peaks as `min_distance`",
    )
    parser.add_argument(
        "--gate-tolerances",
        type=float,
        nargs="+",
        default=[0.1, 0.15, 0.2, 0.3],
        help="Passed to gate_periodicity as `tolerance`",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Print only the top N combinations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the full comparison table to this CSV path",
    )
    args = parser.parse_args()

    data_home = Path(args.data_home) if args.data_home else DATA_DIR / args.dataset
    dataset = BeatDataset(
        name=args.dataset,
        data_home=data_home,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        binary_only=args.binary_only,
    )

    indices = select_indices(
        dataset, args.dataset, args.split, args.val_split, args.binary_only
    )
    if args.limit is not None:
        indices = indices[: args.limit]

    device = torch.device(args.device)
    module = load_module(args.checkpoint, device)
    module.eval()
    module.to(device)

    print(
        f"[sweep_postprocess] caching frame probabilities for {len(indices)} "
        f"track(s) from '{args.dataset}' (split={args.split})"
    )
    cached = compute_track_probs(module, dataset, indices, args.sample_rate, device)

    fps = args.sample_rate / args.hop_length
    grid = list(
        itertools.product(
            args.beat_thresholds, args.min_distance_frames, args.gate_tolerances
        )
    )
    print(f"[sweep_postprocess] scoring {len(grid)} combination(s)...")

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


if __name__ == "__main__":
    main()
