"""Entry point for the interactive track annotator.

Usage
-----
    # random track
    uv run python -m tools.annotator --dataset ballroom

    # specific track
    uv run python -m tools.annotator --dataset ballroom --track Media-105901
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import mirdata
from PySide6.QtWidgets import QApplication

import musicality.dataformats as dataformats
from .data import list_datasets, load_dataset_tracks
from .main_window import MainWindow

DATA_DIR = Path(__file__).parent.parent.parent / dataformats.load().data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive track annotator.")
    parser.add_argument("--dataset", default=None, help="mirdata dataset name (e.g. ballroom)")
    parser.add_argument("--track", default=None, help="Track ID (random if omitted)")
    args = parser.parse_args()

    if args.dataset is not None:
        dataset_name = args.dataset
        track_ids = load_dataset_tracks(dataset_name)
        if args.track is not None:
            if args.track not in track_ids:
                print(f"Track '{args.track}' not found in '{dataset_name}'.")
                print(f"Available IDs (sample): {track_ids[:5]}")
                sys.exit(1)
            index = track_ids.index(args.track)
        else:
            index = random.randrange(len(track_ids))
    else:
        datasets = list_datasets()
        if not datasets:
            print("No datasets found in data/.")
            sys.exit(1)
        dataset_name = datasets[0].name
        track_ids = load_dataset_tracks(dataset_name)
        index = random.randrange(len(track_ids))

    print(f"[annotator] {dataset_name} / {track_ids[index]}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(dataset_name, track_ids, index)
    window.showMaximized()

    sys.exit(app.exec())


main()
