#!/usr/bin/env python3
"""Plot a beat-phase checkpoint's output over time: the beat probability and
the probability of each bar position, one row each, for the tracks you name.

The annotated beats are drawn as grey lines — every beat on the beat row, and
on each position row the beats annotated at that position — so a row should
peak on its own lines. The position head is only trained at beats, so what it
says in between means nothing.

Usage
-----
    uv run python tools/plot_probabilities.py \\
        --checkpoint checkpoints/v7/beat-phase-epoch123-valloss0.8799.ckpt \\
        --track ballroom/Albums-Cafe_Paradiso-02 gtzan/rock_00016

    # open on 20 s from the two-minute mark
    uv run python tools/plot_probabilities.py --checkpoint ... \\
        --track rwc_genre/RWC_G057 --start 120 --duration 20

Tracks are named ``corpus/track_id``, as in the split files. Each opens in its
own window. The whole track is plotted; ``--start``/``--duration`` only set the
initial view, and the toolbar pans and zooms from there.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef
from musicality.inference import load_module, load_track_waveform
from musicality.loaders.beat_dataset import BeatDataset
from musicality.splits.splitter import verify_refs_present


def track_ref(name: str) -> TrackRef:
    """``corpus/track_id`` to the track it names."""

    corpus, track_id = name.split("/", 1)

    return TrackRef(corpus, track_id, dataformats.DATA_DIR / corpus)


@torch.no_grad()
def probabilities(module, wav: torch.Tensor, device: str) -> np.ndarray:
    """``(1 + G, T)``: the beat sigmoid, then the softmax over bar positions."""

    logits = module(wav[None].to(device))[0]

    beat = torch.sigmoid(logits[:1])
    positions = torch.softmax(logits[1:], dim=0)

    return torch.cat([beat, positions]).cpu().numpy()


def plot_track(title: str, probs: np.ndarray, fps: float, beat_times, positions):
    """One row per output on a shared time axis, with the annotated beats."""

    times = np.arange(probs.shape[1]) / fps
    names = ["beat"] + [f"position {p}" for p in range(1, len(probs))]

    fig, axes = plt.subplots(
        len(names),
        1,
        sharex=True,
        sharey=True,
        figsize=(14, 1.5 * len(names) + 1),
        layout="constrained",
    )

    for row, (ax, name) in enumerate(zip(axes, names)):
        if row == 0:
            marks = beat_times
        elif positions is not None:
            marks = beat_times[positions == row]
        else:
            marks = []

        ax.vlines(marks, 0, 1, color="0.8", linewidth=1)
        ax.plot(times, probs[row], color="#2a78d6", linewidth=1.2)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")

    axes[0].set_ylim(0, 1)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot a beat-phase checkpoint's beat and bar-position probabilities over time."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--track",
        type=track_ref,
        nargs="+",
        required=True,
        help="One or more corpus/track_id, as in the split files",
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Initial view start, seconds"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Initial view length, seconds"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    module, task = load_module(args.checkpoint, args.device)
    group_size = module.hparams.get("group_size")

    if task != "beat_phase" or group_size is None:
        raise SystemExit(f"{args.checkpoint} has no bar-position softmax head.")

    sample_rate = module.hparams["model"]["sample_rate"]
    fps = sample_rate / module.hparams["model"]["hop_length"]

    # Annotations come through BeatDataset so that a longer annotated bar is
    # folded onto the head's positions, exactly as in training.
    verify_refs_present(args.track)
    dataset = BeatDataset(refs=args.track, group_size=group_size)

    for ref, (audio_path, beat_times, positions, _) in zip(
        dataset.refs, dataset.samples
    ):
        wav = load_track_waveform(audio_path, sample_rate)
        probs = probabilities(module, wav, args.device)

        title = (
            f"{ref.dataset_name}/{ref.track_id} — "
            f"{args.checkpoint.parent.name}/{args.checkpoint.name}"
        )
        fig = plot_track(title, probs, fps, beat_times, positions)
        fig.axes[0].set_xlim(args.start, args.start + args.duration)

    plt.show()


if __name__ == "__main__":
    main()
