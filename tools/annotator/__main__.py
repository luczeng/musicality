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
from .data import load_track
from .main_window import MainWindow

DATA_DIR = Path(__file__).parent.parent.parent / dataformats.load().data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive track annotator.")
    parser.add_argument("--dataset", required=True, help="mirdata dataset name (e.g. ballroom)")
    parser.add_argument("--track", default=None, help="Track ID (random if omitted)")
    args = parser.parse_args()

    ds = mirdata.initialize(args.dataset, data_home=str(DATA_DIR / args.dataset))

    track_id = args.track or random.choice(ds.track_ids)
    if track_id not in ds.track_ids:
        print(f"Track '{track_id}' not found in '{args.dataset}'.")
        print(f"Available IDs (sample): {ds.track_ids[:5]}")
        sys.exit(1)

    print(f"[annotator] {args.dataset} / {track_id}")
    track = load_track(args.dataset, track_id)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(track)
    window.show()

    sys.exit(app.exec())


main()
