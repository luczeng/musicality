#!/usr/bin/env python3
"""Migrate a dataset's hand-corrected raw beat CSVs into this project's own
tracks/annotations layout (see README's "Data format" section).

Written for rwc_genre, which isn't wrapped by mirdata (unlike
rwc_classical/jazz/popular) — there's no mirdata Track to read beats from,
so migrate_mirdata_dataset.py doesn't apply. Instead its audio sits flat
under tracks/ (e.g. RWC_G001.wav) and its raw annotations are
semicolon-delimited CSVs under annotations/ (header "t;beat", one row per
beat) with stems that already match the audio exactly.

The same "t;beat" CSV convention is also used to drop in hand-corrected
annotations for datasets that otherwise come from mirdata (e.g. rwc_jazz) —
point --dataset at one of those and this tool overwrites whatever
.beats/.meta.json migrate_mirdata_dataset.py (or an earlier run of this
tool) produced, using the corrected CSV instead. Either way, this tool just
rewrites each CSV as a ``.beats`` file in this project's own format
(``<time> <position>`` per line) plus a ``.meta.json`` alongside it — the
CSVs themselves are left in place.

Usage
-----
    uv run python tools/migrate_rwc_genre.py
    uv run python tools/migrate_rwc_genre.py --force
    uv run python tools/migrate_rwc_genre.py --dataset rwc_jazz --force
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile

import musicality.dataformats as dataformats
import tools.annotator.data as annotator_data
from musicality.dataformats.track_io import bpm_stats

DATA_DIR = dataformats.DATA_DIR


def _read_beat_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a "t;beat" CSV into ``(times, positions)``."""
    times: list[float] = []
    positions: list[int] = []
    for line in path.read_text().splitlines()[1:]:
        t, position = line.split(";")
        times.append(float(t))
        positions.append(int(position))
    return np.array(times, dtype=float), np.array(positions, dtype=int)


def migrate(dataset_name: str, force: bool) -> None:

    data_home = DATA_DIR / dataset_name
    csv_paths = sorted(
        data_home.glob(f"{dataformats.FORMAT.annotations_dirname}/*.csv")
    )

    n_migrated = 0
    n_no_audio = 0
    n_skipped_existing = 0

    for csv_path in csv_paths:
        stem = csv_path.stem

        audio_path = data_home / dataformats.FORMAT.tracks_dirname / f"{stem}.wav"
        if not audio_path.exists():
            print(f"[migrate] '{stem}': no matching audio file, skipping")
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

        beat_times, beat_positions = _read_beat_csv(csv_path)

        track_data = annotator_data.TrackData(
            dataset_name=dataset_name,
            track_id=stem,
            audio_path=str(audio_path),
            tempo=None,
            beat_times=beat_times,
            beat_positions=beat_positions,
            annotator_id=None,
        )
        annotator_data.save_annotations(track_data, beats_path)

        bpm_mean, bpm_median, bpm_std = bpm_stats(beat_times)
        metadata = annotator_data.TrackMetadata(
            duration_s=soundfile.info(audio_path).duration,
            bpm_mean=bpm_mean,
            bpm_median=bpm_median,
            bpm_std=bpm_std,
        )
        annotator_data.save_metadata(dataset_name, stem, metadata)

        n_migrated += 1

    print(
        f"[migrate] '{dataset_name}': migrated {n_migrated}  •  "
        f"{n_no_audio} skipped (no matching audio)  •  "
        f"{n_skipped_existing} skipped (already migrated, use --force to redo)"
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="rwc_genre",
        help="Dataset name under the data dir whose annotations/*.csv to migrate "
        "(default: rwc_genre)",
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
