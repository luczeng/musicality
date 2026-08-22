#!/usr/bin/env python3
"""Migrate a mirdata dataset into this project's own tracks/annotations
layout (see docs/source/data.rst's "Data format" section), so every other
tool (annotator, mobile companion, training loaders) can read it like a
homemade one — without ever touching mirdata again.

Writes annotations/*.beats + annotations/*.meta.json, and moves each
track's audio into tracks/<stem>.wav (renamed out of mirdata's own
download layout, not copied — mirdata is never read again for a migrated
track, so there's nothing left to keep the original copy for).

Works out of the box for any mirdata dataset whose track.audio_path already
resolves to the file on disk (e.g. ballroom, brid, groove_midi) — no config
needed; re-running migration later (audio already moved into tracks/) is
handled by the exact-track_id fallback in tools/mirdata_audio.py's
DATASET_CONFIGS. Some datasets need a couple lines in DATASET_CONFIGS
instead: either their mirdata Track class exposes audio under a different
attribute name (e.g. guitarset's audio_mic_path), or track_id doesn't equal
the audio filename's stem at all (e.g. rwc_jazz's mirdata id RM-J001 vs. its
audio's own RWC_J001 — piece_number matches reliably there instead). See
DatasetConfig's docstring for exactly what to add for a new dataset.

Tracks with no beat annotation, or whose audio file can't be located
either way, are skipped and counted in the summary.

Usage
-----
    uv run python tools/migrate_mirdata_dataset.py --dataset ballroom
    uv run python tools/migrate_mirdata_dataset.py --dataset rwc_jazz --force
"""

import argparse

import mirdata

import musicality.dataformats as dataformats
import tools.annotator.data as annotator_data
from musicality.dataformats.track_io import bpm_stats
from tools.annotator.naming import sanitize_track_name
from tools.mirdata_audio import index_audio, resolve_audio_path

DATA_DIR = dataformats.DATA_DIR


def migrate(dataset_name: str, force: bool) -> None:

    data_home = DATA_DIR / dataset_name
    ds = mirdata.initialize(dataset_name, data_home=str(data_home))

    # Always built: every dataset relies on the exact-track_id fallback tier
    # to re-resolve audio a prior run already moved into tracks/ (see
    # mirdata_audio's module docstring). A one-time rglob cost per migration
    # run, not per training run.
    audio_index = index_audio(data_home)

    n_migrated = 0
    n_audio_moved = 0
    n_no_beats = 0
    n_unresolved_audio = 0
    n_skipped_existing = 0

    track_ids = ds.track_ids
    n_tracks = len(track_ids)
    progress_every = max(1, n_tracks // 10)

    for i, track_id in enumerate(track_ids, start=1):
        if i % progress_every == 0 or i == n_tracks:
            print(f"[migrate] '{dataset_name}': {i}/{n_tracks} tracks processed")

        track = ds.track(track_id)

        if track.beats is None or len(track.beats.times) == 0:
            n_no_beats += 1
            continue

        audio_path = resolve_audio_path(track, dataset_name, audio_index)
        if audio_path is None:
            print(f"[migrate] '{track_id}': couldn't locate its audio file, skipping")
            n_unresolved_audio += 1
            continue
        stem = sanitize_track_name(audio_path.stem)

        # Always ensure the audio ends up under tracks/, independent of
        # whether its annotations need (re)writing below — so a track's
        # audio still gets moved even if a previous run already wrote its
        # .beats file (e.g. an interrupted run, or migrating with an older
        # version of this tool that didn't move audio yet). A no-op once
        # already moved: resolve_audio_path's fallback tier already found
        # it sitting in tracks/ above, so audio_path == target here.
        target_audio_path = (
            data_home / dataformats.FORMAT.tracks_dirname / f"{stem}.wav"
        )
        if audio_path != target_audio_path:
            target_audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.rename(target_audio_path)
            n_audio_moved += 1

        beats_path = (
            data_home
            / dataformats.FORMAT.annotations_dirname
            / f"{stem}{dataformats.FORMAT.beats_suffix}"
        )
        if beats_path.exists() and not force:
            n_skipped_existing += 1
            continue

        track_data = annotator_data.TrackData(
            dataset_name=dataset_name,
            track_id=stem,
            audio_path=str(target_audio_path),
            tempo=None,
            beat_times=track.beats.times,
            beat_positions=getattr(track.beats, "positions", None),
            annotator_id=None,
        )
        annotator_data.save_annotations(track_data, beats_path)

        bpm_mean, bpm_median, bpm_std = bpm_stats(track.beats.times)
        duration = getattr(track, "duration", None)
        metadata = annotator_data.TrackMetadata(
            duration_s=float(duration) if duration is not None else None,
            bpm_mean=bpm_mean,
            bpm_median=bpm_median,
            bpm_std=bpm_std,
        )
        annotator_data.save_metadata(dataset_name, stem, metadata)

        n_migrated += 1

    print(
        f"[migrate] '{dataset_name}': migrated {n_migrated}  •  "
        f"{n_audio_moved} audio file(s) moved into tracks/  •  "
        f"{n_no_beats} skipped (no beat annotation)  •  "
        f"{n_unresolved_audio} skipped (audio not found)  •  "
        f"{n_skipped_existing} skipped (already migrated, use --force to redo)"
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="mirdata dataset name to migrate (see DatasetConfig for onboarding a new one)",
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
