#!/usr/bin/env python3
"""Evaluate a trained beat-phase checkpoint: beat / "1" / "last" F-measure and
the half-cycle phase-confusion rate, over full-length tracks (not the
fixed-duration clips used during training).

Usage
-----
    uv run python tools/eval_beat_phase.py \\
        --checkpoint checkpoints_beat/beat-phase-epoch=05-val_loss=0.1234.ckpt \\
        --dataset ballroom

    # evaluate the training split too, or just the first N tracks for a quick check
    uv run python tools/eval_beat_phase.py --checkpoint ... --dataset ballroom --split all --limit 20

    # a phrase-position (1-8) dataset instead of the default bar-position (1-4) one
    uv run python tools/eval_beat_phase.py --checkpoint ... --dataset my_phrase_dataset --group-size 8
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

import musicality.dataformats as dataformats
from musicality.loaders.beat_dataset import BeatDataset
from musicality.metrics import (
    beat_f_measure,
    confusion_half_cycle_rate,
    downbeat_f_measures,
)
from musicality.postprocess import readout
from musicality.splits.splitter import Splitter
from musicality.trainers.beat_phase_module import BeatPhaseModule

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir


def load_track_waveform(audio_path: str, sample_rate: int) -> torch.Tensor:
    """Load a full track (no cropping/padding), mono, at ``sample_rate``."""

    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        wav = T.Resample(sr, sample_rate)(wav)

    return wav  # (1, N)


def select_indices(
    dataset: BeatDataset,
    dataset_name: str,
    split: str,
    val_split: float,
    binary_only: bool = False,
) -> list[int]:
    """Return dataset indices for ``split``, reusing the training run's cached
    train/val split so "val" means genuinely held-out tracks."""

    if split == "all":
        return list(range(len(dataset)))

    _fmt = dataformats.load()
    splits_dir = dataformats.ROOT / _fmt.splits_dir
    suffix = "-binary" if binary_only else ""

    train_ds, val_ds = Splitter(
        dataset, splits_dir, f"beat_phase-{dataset_name}{suffix}", val_split
    ).run()

    return list((val_ds if split == "val" else train_ds).indices)


@torch.no_grad()
def evaluate_track(
    module: BeatPhaseModule,
    audio_path: str,
    beat_times: np.ndarray,
    positions: np.ndarray | None,
    has_positions: bool,
    sample_rate: int,
    hop_length: int,
    device: torch.device,
    tolerance: float,
    trim: bool,
    beat_threshold: float,
    min_distance_frames: int,
    gate_tolerance: float,
    anchor_threshold: float,
    group_size: int,
) -> dict:
    """Run the model on one full track and score its readout against the reference."""

    wav = (
        load_track_waveform(audio_path, sample_rate).unsqueeze(0).to(device)
    )  # (1, 1, N)
    logits = module(wav)  # (1, 3, T')
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # (3, T')
    fps = sample_rate / hop_length

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
    pred_times = [e["time"] for e in events]

    result = {
        "f_beat": beat_f_measure(
            beat_times, pred_times, tolerance=tolerance, trim=trim
        ),
        "f_one": None,
        "f_last": None,
        "confusion": None,
    }

    if has_positions:
        positions = np.asarray(positions)
        f_one, f_last = downbeat_f_measures(
            beat_times,
            positions,
            events,
            tolerance=tolerance,
            trim=trim,
            group_size=group_size,
        )
        result["f_one"] = f_one
        result["f_last"] = f_last
        result["confusion"] = confusion_half_cycle_rate(
            beat_times, positions, events, tolerance=tolerance, group_size=group_size
        )

    return result


def _mean(results: list[dict], key: str) -> float | None:
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "  n/a"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a beat-phase checkpoint on full-length tracks."
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
        "--group-size",
        type=int,
        default=4,
        help="Beats per group: 4 for bar position (default), 8 for phrase position",
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
        default=0.07,
        help="F-measure matching window, seconds",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable mir_eval's standard 5s warm-up trim (use for short clips)",
    )
    parser.add_argument("--beat-threshold", type=float, default=0.3)
    parser.add_argument("--min-distance-frames", type=int, default=1)
    parser.add_argument("--gate-tolerance", type=float, default=0.2)
    parser.add_argument("--anchor-threshold", type=float, default=0.5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N selected tracks",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_home = Path(args.data_home) if args.data_home else DATA_DIR / args.dataset
    dataset = BeatDataset(
        name=args.dataset,
        data_home=data_home,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        group_size=args.group_size,
        binary_only=args.binary_only,
    )

    indices = select_indices(
        dataset, args.dataset, args.split, args.val_split, args.binary_only
    )
    if args.limit is not None:
        indices = indices[: args.limit]

    device = torch.device(args.device)
    module = BeatPhaseModule.load_from_checkpoint(
        args.checkpoint, map_location=device, weights_only=False
    )
    module.eval()
    module.to(device)

    print(
        f"[eval] {len(indices)} track(s) from '{args.dataset}' (split={args.split}, group_size={args.group_size})"
    )

    results = []
    for i in indices:
        audio_path, beat_times, positions, has_positions = dataset.samples[i]

        r = evaluate_track(
            module,
            audio_path,
            beat_times,
            positions,
            has_positions,
            args.sample_rate,
            args.hop_length,
            device,
            args.tolerance,
            not args.no_trim,
            args.beat_threshold,
            args.min_distance_frames,
            args.gate_tolerance,
            args.anchor_threshold,
            args.group_size,
        )
        results.append(r)

        label = Path(audio_path).stem
        print(
            f"[{label:30s}] beat={_fmt(r['f_beat'])}  one={_fmt(r['f_one'])}  "
            f"last={_fmt(r['f_last'])}  confusion={_fmt(r['confusion'])}"
        )

    print("-" * 70)
    print(f"n_tracks={len(results)}")
    print(f"mean beat F-measure:  {_fmt(_mean(results, 'f_beat'))}")
    print(f"mean '1' F-measure:   {_fmt(_mean(results, 'f_one'))}")
    print(f"mean 'last' F-measure: {_fmt(_mean(results, 'f_last'))}")
    print(f"mean phase confusion:  {_fmt(_mean(results, 'confusion'))}")


if __name__ == "__main__":
    main()
