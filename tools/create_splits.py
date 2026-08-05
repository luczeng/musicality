#!/usr/bin/env python3
"""Create train/val splits for datasets in data/, saved under data/splits/.

These files are what Splitter.run() reads at train/eval time — it never
generates a split itself, so the files here are the ground truth. They're
meant to be version-controlled (e.g. via DVC) so every machine trains and
evaluates against the exact same split.

Creates both a tempo split (`<name>`) from TempoDataset and a beat-phase
split (`beat_phase-<name>`) from BeatDataset for each dataset, skipping
whichever one has no samples for that dataset.

Usage
-----
    uv run python tools/create_splits.py
    uv run python tools/create_splits.py --datasets ballroom brid
    uv run python tools/create_splits.py --val-split 0.15 --force
"""

import argparse

import mirdata

import musicality.dataformats as dataformats
from musicality.loaders.beat_dataset import BeatDataset
from musicality.loaders.tempo_dataset import TempoDataset
from musicality.splits.splitter import Splitter

_fmt = dataformats.load()
DATA_DIR = dataformats.ROOT / _fmt.data_dir
SPLITS_DIR = dataformats.ROOT / _fmt.splits_dir


def split_exists(name: str) -> bool:

    split_dir = SPLITS_DIR / name

    return (split_dir / "train.txt").exists() and (split_dir / "val.txt").exists()


def create_split(name: str, dataset, val_split: float, force: bool) -> None:

    if len(dataset) == 0:
        print(f"[create_splits] '{name}': no samples found, skipping")
        return

    if split_exists(name) and not force:
        print(
            f"[create_splits] '{name}': split already exists, skipping (use --force to regenerate)"
        )
        return

    Splitter(dataset, SPLITS_DIR, name, val_split).create()


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to split (default: all datasets found in data/)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of each dataset held out for validation",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing splits"
    )
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help=(
            "For beat-phase splits, drop tracks whose beats-per-bar isn't a multiple "
            "of 2 (e.g. ballroom's waltz/Viennese waltz tracks, which are in triple "
            "meter — 1, 2, 3 — rather than binary meter)"
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Beats per group for the beat-phase 'last' target: 4 for bar position (default), 8 for phrase position",
    )
    args = parser.parse_args()

    available = mirdata.list_datasets()
    names = args.datasets or sorted(
        e.name for e in DATA_DIR.iterdir() if e.is_dir() and e.name in available
    )

    if not names:
        print("No recognised mirdata datasets found in data/.")
        return

    beat_phase_name_suffix = "-binary" if args.binary_only else ""

    for name in names:
        data_home = DATA_DIR / name

        tempo_ds = TempoDataset(name=name, data_home=data_home)
        create_split(name, tempo_ds, args.val_split, args.force)

        beat_ds = BeatDataset(
            name=name,
            data_home=data_home,
            group_size=args.group_size,
            binary_only=args.binary_only,
        )
        create_split(
            f"beat_phase-{name}{beat_phase_name_suffix}",
            beat_ds,
            args.val_split,
            args.force,
        )


if __name__ == "__main__":
    main()
