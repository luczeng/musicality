#!/usr/bin/env python3
"""Evaluate a beat-only or beat-phase checkpoint (task auto-detected from the
checkpoint itself) on full-length tracks — not the fixed-duration clips used
during training: beat F-measure, plus "1"/"last" F-measure and phase-confusion
rate for beat-phase checkpoints.

Usage
-----
    uv run python tools/eval_beat.py \\
        --checkpoint <path-to-ckpt> --dataset ballroom

    # evaluate the training split too, or just the first N tracks for a quick check
    uv run python tools/eval_beat.py --checkpoint ... --dataset ballroom --split all --limit 20
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

import musicality.dataformats as dataformats
from musicality.inference import load_module, load_track_waveform, run_inference
from musicality.loaders.beat_dataset import BeatDataset, indices_for_split
from musicality.metrics.confusion import confusion_half_cycle_rate
from musicality.metrics.f_measure import beat_f_measure, downbeat_f_measures

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir

# Default CLI values — see configs/eval_beat.yaml for what each means.
DEFAULTS = yaml.safe_load((dataformats.ROOT / "configs" / "eval_beat.yaml").read_text())


def evaluate_track(
    module,
    task: str,
    audio_path: str,
    beat_times: np.ndarray,
    positions: np.ndarray | None,
    has_positions: bool,
    sample_rate: int,
    hop_length: int,
    device: str,
    tolerance: float,
    trim: bool,
    beat_threshold: float,
    min_distance_frames: int,
    gate_tolerance: float,
    anchor_threshold: float,
    group_size: int,
) -> dict:
    """Run the model on one full track and score its readout against the
    reference. Always returns the same four keys — ``f_one``/``f_last``/
    ``confusion`` stay ``None`` for a beat-only task (no bar-position heads to
    score) or when the track has no position annotations."""

    wav = load_track_waveform(audio_path, sample_rate)
    fps = sample_rate / hop_length

    result = run_inference(
        module,
        task,
        wav,
        fps,
        device,
        beat_threshold=beat_threshold,
        min_distance_frames=min_distance_frames,
        gate_tolerance=gate_tolerance,
        anchor_threshold=anchor_threshold,
        group_size=group_size,
    )

    events = result if task == "beat_phase" else None
    pred_times = [e["time"] for e in events] if events is not None else result

    out = {
        "f_beat": beat_f_measure(
            beat_times, pred_times, tolerance=tolerance, trim=trim
        ),
        "f_one": None,
        "f_last": None,
        "confusion": None,
    }

    if task == "beat_phase" and has_positions:
        positions = np.asarray(positions)
        f_one, f_last = downbeat_f_measures(
            beat_times,
            positions,
            events,
            tolerance=tolerance,
            trim=trim,
            group_size=group_size,
        )
        out["f_one"] = f_one
        out["f_last"] = f_last
        out["confusion"] = confusion_half_cycle_rate(
            beat_times, positions, events, tolerance=tolerance, group_size=group_size
        )

    return out


def _mean(results: list[dict], key: str) -> float | None:
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "  n/a"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a beat-only or beat-phase checkpoint (auto-detected) on full-length tracks."
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
        help="Beats per group: 4 for bar position (default), 8 for phrase position. beat-phase only.",
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
        default=DEFAULTS["tolerance"],
        help="F-measure matching window, seconds",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable mir_eval's standard 5s warm-up trim (use for short clips)",
    )
    parser.add_argument(
        "--beat-threshold",
        type=float,
        default=None,
        help="Defaults to the tuned value for the checkpoint's detected task (see configs/eval_beat.yaml)",
    )
    parser.add_argument(
        "--min-distance-frames",
        type=int,
        default=None,
        help="Defaults to the tuned value for the checkpoint's detected task",
    )
    parser.add_argument(
        "--gate-tolerance",
        type=float,
        default=None,
        help="Defaults to the tuned value for the checkpoint's detected task",
    )
    parser.add_argument(
        "--anchor-threshold", type=float, default=None, help="beat-phase only"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N selected tracks",
    )
    parser.add_argument("--device", default=DEFAULTS["device"])
    args = parser.parse_args()

    module, task = load_module(args.checkpoint, args.device)
    task_defaults = DEFAULTS[task]

    group_size = (
        args.group_size
        if args.group_size is not None
        else task_defaults.get("group_size", 4)
    )
    beat_threshold = (
        args.beat_threshold
        if args.beat_threshold is not None
        else task_defaults["beat_threshold"]
    )
    min_distance_frames = (
        args.min_distance_frames
        if args.min_distance_frames is not None
        else task_defaults["min_distance_frames"]
    )
    gate_tolerance = (
        args.gate_tolerance
        if args.gate_tolerance is not None
        else task_defaults["gate_tolerance"]
    )
    anchor_threshold = (
        args.anchor_threshold
        if args.anchor_threshold is not None
        else task_defaults.get("anchor_threshold", 0.5)
    )

    data_home = Path(args.data_home) if args.data_home else DATA_DIR / args.dataset
    dataset = BeatDataset(
        name=args.dataset,
        data_home=data_home,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        group_size=group_size,
        binary_only=args.binary_only,
    )

    indices = indices_for_split(
        dataset, args.dataset, args.split, args.val_split, args.binary_only
    )
    if args.limit is not None:
        indices = indices[: args.limit]

    print(
        f"[eval] {len(indices)} track(s) from '{args.dataset}' (split={args.split}, task={task})"
    )

    results = []
    for i in indices:
        audio_path, beat_times, positions, has_positions = dataset.samples[i]

        r = evaluate_track(
            module,
            task,
            audio_path,
            beat_times,
            positions,
            has_positions,
            args.sample_rate,
            args.hop_length,
            args.device,
            args.tolerance,
            not args.no_trim,
            beat_threshold,
            min_distance_frames,
            gate_tolerance,
            anchor_threshold,
            group_size,
        )
        results.append(r)

        label = Path(audio_path).stem
        if task == "beat_phase":
            print(
                f"[{label:30s}] beat={_fmt(r['f_beat'])}  one={_fmt(r['f_one'])}  "
                f"last={_fmt(r['f_last'])}  confusion={_fmt(r['confusion'])}"
            )
        else:
            print(f"[{label:30s}] beat={_fmt(r['f_beat'])}")

    print("-" * 70)
    print(f"n_tracks={len(results)}")
    print(f"mean beat F-measure:  {_fmt(_mean(results, 'f_beat'))}")
    if task == "beat_phase":
        print(f"mean '1' F-measure:   {_fmt(_mean(results, 'f_one'))}")
        print(f"mean 'last' F-measure: {_fmt(_mean(results, 'f_last'))}")
        print(f"mean phase confusion:  {_fmt(_mean(results, 'confusion'))}")


if __name__ == "__main__":
    main()
