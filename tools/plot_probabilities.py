#!/usr/bin/env python3
"""Plot a beat-phase checkpoint's output over time: the beat probability and
the probability of each bar position, one row each.

Pick a track in the tree on the left — the annotator's own — and its plot
appears on the right. The annotated beats are drawn as grey lines — every beat
on the beat row, and on each position row the beats annotated at that
position — so a row should peak on its own lines. The position head is only
trained at beats, so what it says in between means nothing. When the track has
bar-position annotations, a row on top shows its beats: downbeats in green,
every other beat in orange.

Usage
-----
    uv run python tools/plot_probabilities.py \\
        --checkpoint checkpoints/v7/beat-phase-epoch123-valloss0.8799.ckpt

    # open every plot on 20 s from the two-minute mark
    uv run python tools/plot_probabilities.py --checkpoint ... --start 120 --duration 20

The whole track is plotted; ``--start``/``--duration`` only set the initial
view, and the toolbar above the plot pans and zooms from there.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from musicality.inference import load_module, load_track_waveform
from musicality.loaders.beat_dataset import fold_positions
from tools.annotator.data import load_track
from tools.annotator.track_tree import TrackTree

# The annotator's colours: downbeats green, every other beat orange.
DOWNBEAT_COLOR, BEAT_COLOR = "#44cc44", "#cc7700"


@torch.no_grad()
def probabilities(module, wav: torch.Tensor, device: str) -> np.ndarray:
    """``(1 + G, T)``: the beat sigmoid, then the softmax over bar positions."""

    logits = module(wav[None].to(device))[0]

    beat = torch.sigmoid(logits[:1])
    positions = torch.softmax(logits[1:], dim=0)

    return torch.cat([beat, positions]).cpu().numpy()


def bar_positions(positions: np.ndarray | None, group_size: int) -> np.ndarray | None:
    """Annotated bar positions folded onto the head's ``group_size``, as
    training folds them — ``None`` when there are none, or when the meter
    cannot be folded (a 6-beat bar against a 4-way head)."""

    if positions is None or not np.any(positions > 0):
        return None

    return fold_positions(positions, group_size)


def plot_track(
    fig: Figure, title: str, probs: np.ndarray, fps: float, beat_times, positions
):
    """One row per output on a shared time axis, with the annotated beats.

    When *positions* is given, a thinner row on top shows the annotated beats,
    the downbeats in their own colour.
    """

    times = np.arange(probs.shape[1]) / fps
    names = ["beat"] + [f"position {p}" for p in range(1, len(probs))]
    annotated = positions is not None

    heights = ([0.5] if annotated else []) + [1] * len(names)
    axes = fig.subplots(
        len(heights),
        1,
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": heights},
    )

    if annotated:
        top, axes = axes[0], axes[1:]
        downbeat = positions == 1

        top.vlines(
            beat_times[downbeat],
            0,
            1,
            color=DOWNBEAT_COLOR,
            linewidth=2,
            label="position 1",
        )
        top.vlines(
            beat_times[~downbeat],
            0,
            1,
            color=BEAT_COLOR,
            linewidth=2,
            label="other positions",
        )

        top.tick_params(left=False, labelleft=False)
        top.set_ylabel("annotation", rotation=0, ha="right", va="center")
        top.legend(
            ncol=2,
            loc="lower right",
            bbox_to_anchor=(1, 1),
            frameon=False,
        )

    for row, (ax, name) in enumerate(zip(axes, names)):
        if row == 0:
            marks = beat_times
        elif annotated:
            marks = beat_times[positions == row]
        else:
            marks = []

        ax.vlines(marks, 0, 1, color="0.8", linewidth=1)
        ax.plot(times, probs[row], color="#2a78d6", linewidth=1.2)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")

    axes[0].set_ylim(0, 1)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)


class Viewer(QMainWindow):
    """The annotator's track tree on the left; clicking a track plots it on
    the right."""

    def __init__(
        self, module, checkpoint: Path, device: str, start: float, duration: float
    ):
        super().__init__()

        self._module = module
        self._device = device
        self._view = (start, start + duration)

        hparams = module.hparams
        self._group_size = hparams["group_size"]
        self._sample_rate = hparams["model"]["sample_rate"]
        self._fps = self._sample_rate / hparams["model"]["hop_length"]

        self._figure = Figure(layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)

        tree = TrackTree()
        tree.track_clicked.connect(self._show_track)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.addWidget(NavigationToolbar2QT(self._canvas, plot_panel))
        plot_layout.addWidget(self._canvas)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tree)
        splitter.addWidget(plot_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 1100])

        self.setCentralWidget(splitter)
        self.setWindowTitle(f"{checkpoint.parent.name}/{checkpoint.name}")

    def _show_track(self, dataset_name: str, track_id: str) -> None:

        # Loading and a full-track forward pass take a moment on long tracks.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            track = load_track(dataset_name, track_id)
            wav = load_track_waveform(track.audio_path, self._sample_rate)
            probs = probabilities(self._module, wav, self._device)
        finally:
            QApplication.restoreOverrideCursor()

        self._figure.clear()
        plot_track(
            self._figure,
            f"{dataset_name}/{track_id}",
            probs,
            self._fps,
            track.beat_times,
            bar_positions(track.beat_positions, self._group_size),
        )
        self._figure.axes[0].set_xlim(*self._view)
        self._canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a beat-phase checkpoint's beat and bar-position probabilities "
            "over time, for tracks picked in the annotator's track tree."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--start", type=float, default=0.0, help="Initial view start, seconds"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Initial view length, seconds"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    module, task = load_module(args.checkpoint, args.device)

    if task != "beat_phase" or module.hparams.get("group_size") is None:
        raise SystemExit(f"{args.checkpoint} has no bar-position softmax head.")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # the annotator's style, so the tree looks the same

    viewer = Viewer(module, args.checkpoint, args.device, args.start, args.duration)
    viewer.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
