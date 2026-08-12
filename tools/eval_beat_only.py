#!/usr/bin/env python3
"""Evaluate a trained beat-only checkpoint: beat F-measure over full-length
tracks (not the fixed-duration clips used during training).

Usage
-----
    uv run python tools/eval_beat_only.py \\
        --checkpoint checkpoints_beat_only/beat-only-epoch=05-val_loss=0.1234.ckpt \\
        --dataset ballroom

    # evaluate the training split too, or just the first N tracks for a quick check
    uv run python tools/eval_beat_only.py --checkpoint ... --dataset ballroom --split all --limit 20
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

import musicality.dataformats as dataformats
from musicality.loaders.beat_dataset import BeatDataset
from musicality.metrics.f_measure import beat_f_measure
from musicality.postprocess import readout_beat_only
from musicality.splits.splitter import Splitter
from musicality.trainers.beat_module import BeatModule

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir


def load_module(checkpoint_path: str, device: torch.device) -> BeatModule:
    """Load a ``BeatModule`` checkpoint, migrating checkpoints saved before
    ``TCNTempoNet``'s frame head gained a dropout layer (``Conv1d`` ->
    ``Sequential(Dropout, Conv1d)``), which shifted its state dict keys from
    ``frame_head.{weight,bias}`` to ``frame_head.1.{weight,bias}``. Safe to
    delete once no pre-dropout checkpoints are still in use.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    for suffix in ("weight", "bias"):
        legacy_key = f"model.frame_head.{suffix}"
        if legacy_key in state_dict:
            state_dict[f"model.frame_head.1.{suffix}"] = state_dict.pop(legacy_key)

    module = BeatModule(**checkpoint["hyper_parameters"])
    module.load_state_dict(state_dict)

    return module


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
    train/val split so "val" means genuinely held-out tracks.

    Namespaced the same way as :mod:`musicality.trainers.common` builds it for
    both beat-phase and beat-only training (``beat_phase-<name>[-binary]``),
    since the two share the exact same held-out split.
    """

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
    module: BeatModule,
    audio_path: str,
    beat_times: np.ndarray,
    sample_rate: int,
    hop_length: int,
    device: torch.device,
    tolerance: float,
    trim: bool,
    beat_threshold: float,
    min_distance_frames: int,
    gate_tolerance: float,
) -> dict:
    """Run the model on one full track and score its readout against the reference."""

    wav = (
        load_track_waveform(audio_path, sample_rate).unsqueeze(0).to(device)
    )  # (1, 1, N)
    logits = module(wav)  # (1, T')
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # (T',)
    fps = sample_rate / hop_length

    pred_times = readout_beat_only(
        probs,
        fps=fps,
        beat_threshold=beat_threshold,
        min_distance_frames=min_distance_frames,
        gate_tolerance=gate_tolerance,
    )

    return {
        "f_beat": beat_f_measure(
            beat_times, pred_times, tolerance=tolerance, trim=trim
        ),
    }


def _mean(results: list[dict], key: str) -> float | None:
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "  n/a"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a beat-only checkpoint on full-length tracks."
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

    print(f"[eval] {len(indices)} track(s) from '{args.dataset}' (split={args.split})")

    results = []
    for i in indices:
        audio_path, beat_times, _positions, _has_positions = dataset.samples[i]

        r = evaluate_track(
            module,
            audio_path,
            beat_times,
            args.sample_rate,
            args.hop_length,
            device,
            args.tolerance,
            not args.no_trim,
            args.beat_threshold,
            args.min_distance_frames,
            args.gate_tolerance,
        )
        results.append(r)

        label = Path(audio_path).stem
        print(f"[{label:30s}] beat={_fmt(r['f_beat'])}")

    print("-" * 70)
    print(f"n_tracks={len(results)}")
    print(f"mean beat F-measure: {_fmt(_mean(results, 'f_beat'))}")


if __name__ == "__main__":
    main()
