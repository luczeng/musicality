#!/usr/bin/env python3
"""Merge several datasets into one manifest of track names, in this
project's own tracks/annotations layout (see docs/source/data.rst's "Data
format" section).

Every source dataset must already be migrated to that layout — i.e. have at
least one ``.beats`` file under its ``annotations/`` directory, produced by
``tools/migrate_mirdata_dataset.py`` (mirdata datasets) or
``tools/migrate_rwc_genre.py`` (CSV-annotated ones). A dataset with no
``.beats`` files at all is treated as not migrated: this tool crashes before
writing anything, rather than silently merging a partial or empty dataset.

This tool writes no audio and no annotation files of its own — only a
``tracks.txt`` manifest under ``<output>/``, one ``<dataset>/<track_id>``
name per line. Audio and annotations are left exactly where they already
live, under each source dataset's own ``tracks/``/``annotations/``
directories; a loader consuming this manifest is expected to resolve each
name back against its source dataset (a second step, not part of this
tool).

Only tracks with a saved ``.beats`` annotation, and a resolvable audio file,
are listed — a track lacking either, even in an otherwise-migrated dataset,
is skipped (same as the migration tools).

Usage
-----
    uv run python tools/merge_datasets.py --datasets ballroom brid --output ballroom_brid
    uv run python tools/merge_datasets.py --datasets rwc_jazz rwc_classical rwc_popular --output rwc_all --force
"""

import argparse
from pathlib import Path

import musicality.dataformats as dataformats

DATA_DIR = dataformats.DATA_DIR

MANIFEST_FILENAME = "tracks.txt"


def _migrated_beats_files(dataset_name: str) -> list[Path]:
    """Return every default-slot ``.beats`` file for *dataset_name*.

    Non-recursive, so annotator subdirectories (alternate annotation slots)
    are excluded — merging only ever draws from the default slot.
    """

    ann_dir = DATA_DIR / dataset_name / dataformats.FORMAT.annotations_dirname
    if not ann_dir.is_dir():
        return []

    return sorted(ann_dir.glob(f"*{dataformats.FORMAT.beats_suffix}"))


def _audio_stems(dataset_name: str) -> set[str]:
    """Return every audio file's stem under ``tracks/`` for *dataset_name*.

    Migration (``tools/migrate_mirdata_dataset.py`` / ``tools/migrate_rwc_genre.py``)
    always places audio under ``tracks/`` alongside a track's ``.beats`` file,
    so a migrated dataset's ``tracks/`` folder is the single source of truth
    here — no mirdata fallback needed.
    """

    tracks_dir = DATA_DIR / dataset_name / dataformats.FORMAT.tracks_dirname
    if not tracks_dir.is_dir():
        return set()

    return {f.stem for f in tracks_dir.iterdir() if f.suffix.lower() == ".wav"}


def merge(dataset_names: list[str], output_name: str, force: bool) -> None:

    # Fail fast: every source dataset must already be migrated before this
    # tool writes anything, so a merge is never silently partial.
    unmigrated = [name for name in dataset_names if not _migrated_beats_files(name)]
    if unmigrated:
        raise RuntimeError(
            f"Dataset(s) not migrated to this project's tracks/annotations format "
            f"(no .beats files found under annotations/): {', '.join(unmigrated)}. "
            f"Run tools/migrate_mirdata_dataset.py --dataset <name> (mirdata "
            f"datasets) or tools/migrate_rwc_genre.py --dataset <name> "
            f"(CSV-annotated ones) first."
        )

    manifest_path = DATA_DIR / output_name / MANIFEST_FILENAME
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} already exists — pass --force to overwrite it."
        )

    names: list[str] = []
    n_no_audio = 0

    for dataset_name in dataset_names:
        audio_stems = _audio_stems(dataset_name)

        for beats_path in _migrated_beats_files(dataset_name):
            stem = beats_path.stem

            if stem not in audio_stems:
                print(
                    f"[merge] '{dataset_name}/{stem}': couldn't locate its audio "
                    f"file, skipping"
                )
                n_no_audio += 1
                continue

            names.append(f"{dataset_name}/{stem}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(sorted(names)) + "\n")

    print(
        f"[merge] '{output_name}': listed {len(names)} track(s) from "
        f"{len(dataset_names)} dataset(s) ({', '.join(dataset_names)}) in "
        f"{manifest_path}  •  {n_no_audio} skipped (audio not found)"
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names under the data dir to merge together",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Name for the merged dataset's manifest directory under the data dir",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing manifest at this --output name",
    )
    args = parser.parse_args()

    merge(args.datasets, args.output, args.force)


if __name__ == "__main__":
    main()
