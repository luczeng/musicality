#!/usr/bin/env python3
"""Merge several datasets' existing train/val splits into one combined
split, in this project's own splits/ layout (see docs/source/data.rst's
"Merging datasets" section).

Every source dataset must already have a split — produced by
``tools/create_splits.py`` — before this tool can merge it; a missing split
fails fast, before anything is written, rather than silently merging a
partial result.

This tool writes no dataset directory and no manifest of its own — only a
new split, ``splits_dir/<output>/{train,val}.txt``, each line a
``<dataset_name>/<track_id>`` track drawn from the matching split (train or
val) of every requested source dataset. Audio and annotations are read
straight from each source dataset's own directory when the merged split is
later loaded via ``TempoDataset``/``BeatDataset``'s ``refs=`` argument (see
``musicality.splits.splitter.Splitter.load_refs``) — no merged dataset
directory is ever created, so there's no dataformat for a merge to violate.

The merge inherits whatever train/val ratio each source was already split
at; it never generates a fresh split itself.

Usage
-----
    uv run python tools/merge_datasets.py --datasets ballroom brid --output ballroom_brid
    uv run python tools/merge_datasets.py --datasets rwc_jazz rwc_classical rwc_popular --output rwc_all --force
    uv run python tools/merge_datasets.py --datasets ballroom brid --output ballroom_brid --binary-only
"""

import argparse

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef
from musicality.splits.splitter import Splitter, split_name

_fmt = dataformats.load()
SPLITS_DIR = dataformats.ROOT / _fmt.splits_dir


def merge(
    dataset_names: list[str],
    output_name: str,
    binary_only: bool,
    force: bool,
) -> None:

    sources = [split_name(name, binary_only) for name in dataset_names]
    merged_name = split_name(output_name, binary_only)

    # Fail fast: every source must already have a split, and the output must
    # not already exist without --force, before this tool writes anything —
    # so a merge is never silently partial.
    missing = [
        source for source in sources if not (SPLITS_DIR / source / "train.txt").exists()
    ]
    if missing:
        raise RuntimeError(
            f"No split found for: {', '.join(missing)} in {SPLITS_DIR}. Run "
            f"`uv run python tools/create_splits.py` first."
        )

    if (SPLITS_DIR / merged_name / "train.txt").exists() and not force:
        raise FileExistsError(
            f"Split '{merged_name}' already exists — pass --force to overwrite."
        )

    train_refs: list[TrackRef] = []
    val_refs: list[TrackRef] = []

    for source in sources:
        source_train, source_val = Splitter.load_refs(SPLITS_DIR, source)
        train_refs.extend(source_train)
        val_refs.extend(source_val)

    train_refs.sort(key=lambda r: f"{r.dataset_name}/{r.track_id}")
    val_refs.sort(key=lambda r: f"{r.dataset_name}/{r.track_id}")

    Splitter.save_refs(SPLITS_DIR, merged_name, train_refs, val_refs)

    print(
        f"[merge] '{merged_name}': {len(train_refs)} train / {len(val_refs)} val "
        f"track(s) from {len(dataset_names)} dataset(s) ({', '.join(dataset_names)})"
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names to merge — each must already have a split (see tools/create_splits.py)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Name for the merged split",
    )
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help=(
            "Merge the '-binary' split variant — must match how each source's "
            "split was created (see tools/create_splits.py)"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing split at --output"
    )
    args = parser.parse_args()

    merge(args.datasets, args.output, args.binary_only, args.force)


if __name__ == "__main__":
    main()
