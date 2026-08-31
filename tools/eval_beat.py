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

from musicality.evaluation import DATA_DIR, DEFAULTS, BeatEvaluator


def parse_args() -> argparse.Namespace:
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
        "--anchor-threshold",
        type=float,
        default=None,
        help="beat-phase only, and only when --decoder greedy",
    )
    parser.add_argument(
        "--decoder",
        choices=["greedy", "global"],
        default=None,
        help=(
            "Bar-position stage (beat-phase only). Defaults to the tuned value "
            "in configs/eval_beat.yaml."
        ),
    )
    parser.add_argument(
        "--switch-penalty",
        type=float,
        default=None,
        help="--decoder global only: log-cost of a mid-track phase resync",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N selected tracks",
    )
    parser.add_argument("--device", default=DEFAULTS["device"])

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
    )

    evaluator.run()


if __name__ == "__main__":
    main()
