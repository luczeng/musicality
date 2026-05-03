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
from .main_window import MainWindow

DATA_DIR = Path(__file__).parent.parent.parent / dataformats.load().data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive track annotator.")
    parser.add_argument("--dataset", required=True, help="mirdata dataset name (e.g. ballroom)")
    parser.add_argument("--track", default=None, help="Track ID (random if omitted)")
    args = parser.parse_args()

    ds = mirdata.initialize(args.dataset, data_home=str(DATA_DIR / args.dataset))
    track_ids = ds.track_ids

    if args.track is not None:
        if args.track not in track_ids:
            print(f"Track '{args.track}' not found in '{args.dataset}'.")
            print(f"Available IDs (sample): {track_ids[:5]}")
            sys.exit(1)
        index = track_ids.index(args.track)
    else:
        index = random.randrange(len(track_ids))

    print(f"[annotator] {args.dataset} / {track_ids[index]}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(args.dataset, track_ids, index)
    window.show()

    sys.exit(app.exec())


main()
