#!/usr/bin/env python3
"""Create train/val splits for datasets in the configured data directory,
saved under its splits subdirectory (``data_dir``/``splits_dir`` in
``musicality/dataformats/dataformat.yaml``).

These files are what Splitter.run() reads at train/eval time — it never
generates a split itself, so the files here are the ground truth. They're
meant to be version-controlled (e.g. via DVC) so every machine trains and
evaluates against the exact same split.

One split per dataset serves every task: `<name>`, or `<name>-binary`
with `--binary-only`. Tempo and beat runs read the same file — a track's
tempo label is derived from its beat annotation, so the two tasks can never
disagree about which tracks are usable (see
``musicality.splits.splitter.split_name``).

``--contains`` splits only the tracks whose id contains a given substring,
into their own separately named split (`<name>-<substring>`) — for datasets
that encode a category in the track id, e.g. gtzan's `blues_00001`. The
result is an ordinary split: train on it with `data.input=gtzan-blues`, or
combine several with `tools/merge_datasets.py`.

Usage
-----
    uv run python tools/create_splits.py
    uv run python tools/create_splits.py --datasets ballroom brid
    uv run python tools/create_splits.py --val-split 0.15 --force
    uv run python tools/create_splits.py --datasets gtzan --contains blues
"""

import argparse

import mirdata

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import list_track_refs, sanitize_track_name
from musicality.loaders.beat_dataset import BeatDataset
from musicality.splits.splitter import Splitter, is_flagged, split_name

_fmt = dataformats.load()
DATA_DIR = dataformats.ROOT / _fmt.data_dir
SPLITS_DIR = dataformats.ROOT / _fmt.splits_dir


def split_base_name(dataset_name: str, contains: str | None) -> str:
    """Name the split a ``--contains`` run writes to: ``<dataset>-<substring>``.

    A filtered split gets its own name so it never overwrites the dataset's
    full split, and so it stays self-describing downstream — ``gtzan-blues``
    reads as what it holds in a training config, a W&B run name, or a merge.
    The substring is sanitized the same way track ids are, so any shell-legal
    filter still yields a valid directory name.
    """

    if not contains:
        return dataset_name

    return f"{dataset_name}-{sanitize_track_name(contains)}"


def split_exists(name: str) -> bool:

    split_path = SPLITS_DIR / name

    return (split_path / "train.txt").exists() and (split_path / "val.txt").exists()


def split_size(name: str) -> int:
    """Number of tracks an existing split lists, both sides together."""

    split_path = SPLITS_DIR / name

    return sum(
        len(
            [
                line
                for line in (split_path / side).read_text().splitlines()
                if line.strip()
            ]
        )
        for side in ("train.txt", "val.txt")
    )


def create_split(name: str, dataset, val_split: float, force: bool) -> None:

    if len(dataset) == 0:
        print(f"[create_splits] '{name}': no samples found, skipping")
        return

    if split_exists(name) and not force:
        # A split is a file, not a view of the data directory: tracks
        # annotated since it was written stay out of it until it is
        # regenerated. Say so, so a skipped dataset is never mistaken for an
        # up-to-date one after an annotation session.
        n_eligible = sum(1 for ref in dataset.refs if not is_flagged(ref))
        n_listed = split_size(name)

        drift = (
            ""
            if n_listed == n_eligible
            else (
                f" — it lists {n_listed} track(s), {n_eligible} are annotated "
                f"and unflagged now"
            )
        )

        print(
            f"[create_splits] '{name}': split already exists, skipping "
            f"(use --force to regenerate){drift}"
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
        help=f"Dataset names to split (default: all datasets found in {DATA_DIR}/)",
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
        "--contains",
        default=None,
        help=(
            "Only split tracks whose id contains this substring (case-insensitive), "
            "into a separately named split '<dataset>-<substring>' — e.g. "
            "`--datasets gtzan --contains blues` writes splits/gtzan-blues/"
        ),
    )
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help=(
            "Drop tracks whose beats-per-bar isn't a multiple of 2 (e.g. ballroom's "
            "waltz/Viennese waltz tracks, which are in triple meter — 1, 2, 3 — "
            "rather than binary meter), into the split's '-binary' variant"
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Beats per group for the bar-position target: 4 for bar position (default), 8 for phrase position. Does not affect which tracks the split holds",
    )
    args = parser.parse_args()

    available = mirdata.list_datasets()
    names = args.datasets or sorted(
        e.name for e in DATA_DIR.iterdir() if e.is_dir() and e.name in available
    )

    if not names:
        print(f"No recognised mirdata datasets found in {DATA_DIR}/.")
        return

    for name in names:
        data_home = DATA_DIR / name

        # Filter at the ref level, before the dataset is built, so a narrowed
        # run never pays to resolve the tracks it's about to drop.
        refs = list_track_refs(name, data_home, contains=args.contains)
        base_name = split_base_name(name, args.contains)

        if args.contains:
            print(
                f"[create_splits] '{name}': {len(refs)} track(s) match "
                f"--contains '{args.contains}' → '{base_name}'"
            )

        dataset = BeatDataset(
            refs=refs,
            group_size=args.group_size,
            binary_only=args.binary_only,
        )

        create_split(
            split_name(base_name, args.binary_only),
            dataset,
            args.val_split,
            args.force,
        )


if __name__ == "__main__":
    main()
