#!/usr/bin/env python3
"""Visualize the frame-level beat/one/last targets produced by BeatDataset.

Plots the waveform of a single dataset clip alongside its three Gaussian-smeared
target curves (beat, one, last), so the smearing width and position-annotation
coverage can be sanity-checked by eye.

Usage
-----
    # random clip from ballroom, default clip settings (bar position, 1-4)
    uv run python tools/plot_beat_targets.py --dataset ballroom

    # a specific dataset index, custom clip duration/hop/smear width
    uv run python tools/plot_beat_targets.py --dataset hainsworth --index 3 --duration 8 --sigma 2.0

    # a phrase-position (1-8) dataset
    uv run python tools/plot_beat_targets.py --dataset my_phrase_dataset --group-size 8
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

from musicality.loaders.beat_dataset import BeatDataset


def plot_targets(wav, target, sample_rate: int, hop_length: int, title: str):

    beat, one, last, mask = target.numpy()

    wav_times = np.arange(wav.shape[-1]) / sample_rate
    frame_times = np.arange(target.shape[-1]) * hop_length / sample_rate

    fig, (ax_wav, ax_targets) = plt.subplots(
        2, 1, figsize=(14, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
    )

    ax_wav.plot(wav_times, wav.squeeze(0).numpy(), color="steelblue", linewidth=0.4)
    ax_wav.set_ylabel("Amplitude")
    ax_wav.set_title(title)

    ax_targets.plot(frame_times, beat, color="#FFD700", label="beat", linewidth=1.2)
    ax_targets.plot(frame_times, one, color="#E74C3C", label="one", linewidth=1.4)
    ax_targets.plot(frame_times, last, color="#44CC44", label="last", linewidth=1.4)
    ax_targets.set_ylim(-0.05, 1.05)
    ax_targets.set_xlabel("Time (s)")
    ax_targets.set_ylabel("Target amplitude")
    ax_targets.legend(loc="upper right", fontsize=8)

    has_positions = bool(mask.max() > 0)
    ax_targets.text(
        0.01,
        0.92,
        f"bar-position annotations available: {has_positions}",
        transform=ax_targets.transAxes,
        fontsize=8,
        color="gray",
    )

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot BeatDataset smeared targets for one clip."
    )
    parser.add_argument(
        "--dataset", required=True, help="mirdata dataset name (e.g. ballroom)"
    )
    parser.add_argument(
        "--index", type=int, default=None, help="Dataset index (random if omitted)"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Clip duration in seconds"
    )
    parser.add_argument(
        "--hop-length", type=int, default=512, help="Frame hop length in samples"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050, help="Target sample rate"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.5, help="Gaussian smear width, in frames"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Beats per group: 4 for bar position (default), 8 for phrase position",
    )
    args = parser.parse_args()

    ds = BeatDataset(
        name=args.dataset,
        sample_rate=args.sample_rate,
        duration=args.duration,
        hop_length=args.hop_length,
        group_size=args.group_size,
        sigma_frames=args.sigma,
    )

    if len(ds) == 0:
        print(f"No usable tracks found in '{args.dataset}'.")
        return

    index = args.index if args.index is not None else random.randrange(len(ds))
    wav, target = ds[index]

    print(f"[plot_beat_targets] dataset={args.dataset} index={index}/{len(ds)}")

    plot_targets(
        wav,
        target,
        args.sample_rate,
        args.hop_length,
        title=f"{args.dataset} — index {index}",
    )


if __name__ == "__main__":
    main()
