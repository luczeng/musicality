#!/usr/bin/env python3
"""Migrate GTZAN's beat annotations into this project's own tracks/annotations
layout (see docs/source/data.rst's "Data format" section).

GTZAN isn't wrapped by mirdata here (the `gtzan_genre` mirdata loader's
audio host is a dead link — see configs/download.yaml), so its audio was
downloaded separately via tools/dl_gtzan.py (HuggingFace's marsyas/gtzan
mirror) into ``<genre>/<genre>.<NNNNN>.wav`` subfolders, 1-indexed. Its beat
annotations come from a separate community effort (CPJKU's
beat_this_annotations) already dropped in as this project's own
``<time> <position>`` .beats format, under
``annotations/beats/gtzan_<genre>_<NNNNN>.beats`` — but 0-indexed, one
below the matching audio's own index. So ``gtzan_pop_00037.beats`` belongs
to ``pop/pop.00038.wav``, not ``pop.00037.wav``.

This tool resolves that offset, moves each matched audio file into
``tracks/`` (named after the audio file's own stem, not the annotation's),
and rewrites the annotation + duration/tempo metadata alongside it — same
outcome as migrate_mirdata_dataset.py, just bridging a filename offset
instead of reading mirdata Tracks. The original annotations/beats/*.beats
files are left in place.

Usage
-----
    uv run python tools/migrate_gtzan.py
    uv run python tools/migrate_gtzan.py --force
"""

import argparse
import re
from pathlib import Path

import soundfile

import musicality.dataformats as dataformats
import tools.annotator.data as annotator_data
from musicality.dataformats.track_io import (
    bpm_stats,
    read_beats_file,
    sanitize_track_name,
)

DATA_DIR = dataformats.DATA_DIR

_ANN_STEM_RE = re.compile(r"^gtzan_([a-z]+)_(\d+)$")


def migrate(dataset_name: str, force: bool) -> None:

    data_home = DATA_DIR / dataset_name
    ann_paths = sorted(
        data_home.glob(f"{dataformats.FORMAT.annotations_dirname}/beats/*.beats")
    )

    n_migrated = 0
    n_audio_moved = 0
    n_no_audio = 0
    n_skipped_existing = 0
    n_bad_name = 0

    for ann_path in ann_paths:
        match = _ANN_STEM_RE.match(ann_path.stem)
        if not match:
            print(f"[migrate] '{ann_path.stem}': unexpected filename, skipping")
            n_bad_name += 1
            continue

        genre, ann_idx = match.group(1), int(match.group(2))
        stem = sanitize_track_name(f"{genre}.{ann_idx + 1:05d}")

        target_audio_path = (
            data_home / dataformats.FORMAT.tracks_dirname / f"{stem}.wav"
        )
        source_audio_path = data_home / genre / f"{genre}.{ann_idx + 1:05d}.wav"

        if target_audio_path.exists():
            resolved_audio_path = target_audio_path
        elif source_audio_path.exists():
            target_audio_path.parent.mkdir(parents=True, exist_ok=True)
            source_audio_path.rename(target_audio_path)
            resolved_audio_path = target_audio_path
            n_audio_moved += 1
        else:
            print(f"[migrate] '{ann_path.stem}': no matching audio file, skipping")
            n_no_audio += 1
            continue

        beats_path = (
            data_home
            / dataformats.FORMAT.annotations_dirname
            / f"{stem}{dataformats.FORMAT.beats_suffix}"
        )
        if beats_path.exists() and not force:
            n_skipped_existing += 1
            continue

        beat_times, beat_positions = read_beats_file(ann_path)

        track_data = annotator_data.TrackData(
            dataset_name=dataset_name,
            track_id=stem,
            audio_path=str(resolved_audio_path),
            tempo=None,
            beat_times=beat_times,
            beat_positions=beat_positions,
            annotator_id=None,
        )
        annotator_data.save_annotations(track_data, beats_path)

        bpm_mean, bpm_median, bpm_std = bpm_stats(beat_times)
        metadata = annotator_data.TrackMetadata(
            duration_s=soundfile.info(resolved_audio_path).duration,
            bpm_mean=bpm_mean,
            bpm_median=bpm_median,
            bpm_std=bpm_std,
        )
        annotator_data.save_metadata(dataset_name, stem, metadata)

        n_migrated += 1

    print(
        f"[migrate] '{dataset_name}': migrated {n_migrated}  •  "
        f"{n_audio_moved} audio file(s) moved into tracks/  •  "
        f"{n_bad_name} skipped (unexpected annotation filename)  •  "
        f"{n_no_audio} skipped (audio not found)  •  "
        f"{n_skipped_existing} skipped (already migrated, use --force to redo)"
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="gtzan",
        help="Dataset name under the data dir to migrate (default: gtzan)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-migrate tracks whose .beats file already exists",
    )
    args = parser.parse_args()

    migrate(args.dataset, args.force)


if __name__ == "__main__":
    main()
