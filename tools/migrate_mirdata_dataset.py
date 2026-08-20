#!/usr/bin/env python3
"""Migrate a mirdata dataset's beat annotations into this project's own
tracks/annotations layout (see README's "Data format" section), so tools
that only understand that layout (the desktop annotator, mobile companion)
can read the dataset like a homemade one.

Only annotations/*.beats and annotations/*.meta.json are written here —
audio is left untouched. Point --dataset at a mirdata dataset whose audio/
folder you're renaming to tracks/ yourself; this tool just needs to name
each .beats/.meta.json file with the same stem the corresponding .wav file
already has (or will have) on disk, so the annotator's tracks/<id>.wav +
annotations/<id>.beats pairing lines up.

Works out of the box for any mirdata dataset whose track.audio_path already
resolves to the file on disk (e.g. ballroom, brid, groove_midi) — no config
needed. Some datasets need a couple lines in DATASET_CONFIGS below instead:
either their mirdata Track class exposes audio under a different attribute
name (e.g. guitarset's audio_mic_path), or the on-disk layout was flattened
after download and doesn't match mirdata's own index (e.g. this repo's
rwc_jazz has audio sitting flat as rwc_jazz/audio/RWC_J001.wav rather
than the nested rwc-j-m01/1.wav mirdata's index expects) — in which case
this tool falls back to locating the actual file by matching its trailing
number against a track attribute you name (piece_number for the RWC
datasets), rather than guessing a filename convention. See DatasetConfig's
docstring for exactly what to add for a new dataset.

Tracks with no beat annotation, or whose audio file can't be located
either way, are skipped and counted in the summary.

Usage
-----
    uv run python tools/migrate_mirdata_dataset.py --dataset ballroom
    uv run python tools/migrate_mirdata_dataset.py --dataset rwc_jazz --force
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import mirdata
import numpy as np

import musicality.dataformats as dataformats
import tools.annotator.data as annotator_data
from tools.annotator.naming import sanitize_track_name

DATA_DIR = dataformats.DATA_DIR


@dataclass
class DatasetConfig:
    """Per-dataset quirks for locating a track's on-disk audio file.

    Onboarding a new mirdata dataset:
      - Most datasets need NO entry here — mirdata's own ``track.audio_path``
        already resolves (true for ballroom, brid, groove_midi, and any new
        dataset by default).
      - If the dataset's mirdata Track class exposes its primary audio under
        a different attribute (e.g. guitarset has no ``audio_path`` at all,
        only ``audio_mic_path``/``audio_mix_path``/...), set
        ``audio_path_attr`` to that name.
      - If mirdata's index doesn't match how the audio actually sits on disk
        in this repo (e.g. flattened/renamed after download — see rwc_jazz's
        ``RWC_J001.wav`` vs. mirdata's nested ``rwc-j-m01/1.wav``), set
        ``fallback_match_attr`` to a track attribute (e.g. ``piece_number``)
        whose trailing digits match the on-disk filename's trailing digits.
    """

    audio_path_attr: str = "audio_path"
    fallback_match_attr: str | None = None


# hainsworth isn't listed: no bundled index / no data downloaded here yet,
# and it has no piece_number to fall back on either — the default config
# already handles it safely (every track just gets skipped as "audio not
# found" rather than crashing).
DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "rwc_classical": DatasetConfig(fallback_match_attr="piece_number"),
    "rwc_jazz": DatasetConfig(fallback_match_attr="piece_number"),
    "rwc_popular": DatasetConfig(fallback_match_attr="piece_number"),
    "guitarset": DatasetConfig(audio_path_attr="audio_mic_path"),
}


def _index_audio_by_trailing_number(data_home: Path) -> dict[int, str]:
    """Map each ``.wav``'s trailing number (e.g. ``RWC_J025.wav`` -> 25) to its stem.

    Searches every ``.wav`` under *data_home*, including ``tracks/`` — this
    tool never writes audio itself, so any file there is either a homemade
    recording or the dataset's own audio, already renamed into place by hand.
    """
    by_number: dict[int, str] = {}
    for wav in data_home.rglob("*.wav"):
        match = re.search(r"(\d+)$", wav.stem)
        if match:
            by_number[int(match.group(1))] = wav.stem
    return by_number


def _resolve_stem(
    track, config: DatasetConfig, audio_by_number: dict[int, str]
) -> str | None:
    """Return the on-disk audio filename stem for *track*, or None if unresolved.

    Trusts mirdata's own audio-path attribute first (``config.audio_path_attr``,
    ``"audio_path"`` by default). Falls back to matching the track attribute
    named by ``config.fallback_match_attr`` (its trailing digits) against
    *audio_by_number*, for datasets whose audio was reorganized after
    download (see DatasetConfig's docstring).
    """
    audio_path = getattr(track, config.audio_path_attr)
    if audio_path is not None and Path(audio_path).exists():
        return Path(audio_path).stem

    if config.fallback_match_attr is None:
        return None

    match_value = getattr(track, config.fallback_match_attr, None)
    if not match_value:
        return None
    match = re.search(r"(\d+)", str(match_value))
    if not match:
        return None

    return audio_by_number.get(int(match.group(1)))


def _bpm_stats(
    beat_times: np.ndarray,
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Mean/median/std BPM from inter-beat intervals, matching the convention
    used elsewhere in the annotator (tap_tempo_widget.py, main_window.py)."""
    if len(beat_times) < 2:
        return None, None, None

    tempos = 60.0 / np.diff(beat_times)

    return float(np.mean(tempos)), float(np.median(tempos)), float(np.std(tempos))


def migrate(dataset_name: str, force: bool) -> None:

    config = DATASET_CONFIGS.get(dataset_name, DatasetConfig())

    data_home = DATA_DIR / dataset_name
    ds = mirdata.initialize(dataset_name, data_home=str(data_home))

    # Only pay the rglob cost when a dataset actually needs the fallback —
    # notably skips it for groove_midi (1150 tracks, slow .beats access).
    audio_by_number = (
        _index_audio_by_trailing_number(data_home) if config.fallback_match_attr else {}
    )

    n_migrated = 0
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

        stem = _resolve_stem(track, config, audio_by_number)
        if stem is None:
            print(f"[migrate] '{track_id}': couldn't locate its audio file, skipping")
            n_unresolved_audio += 1
            continue
        stem = sanitize_track_name(stem)

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
            audio_path=str(
                data_home / dataformats.FORMAT.tracks_dirname / f"{stem}.wav"
            ),
            tempo=None,
            beat_times=track.beats.times,
            beat_positions=getattr(track.beats, "positions", None),
            annotator_id=None,
        )
        annotator_data.save_annotations(track_data, beats_path)

        bpm_mean, bpm_median, bpm_std = _bpm_stats(track.beats.times)
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
        f"{n_no_beats} skipped (no beat annotation)  •  "
        f"{n_unresolved_audio} skipped (audio not found)  •  "
        f"{n_skipped_existing} skipped (already migrated, use --force to redo)"
    )
    print(
        f"[migrate] Audio itself wasn't touched — make sure every stem above "
        f"has a matching {data_home}/tracks/<stem>.wav for the annotator to load it."
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
