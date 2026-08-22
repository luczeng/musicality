"""Locate a mirdata track's audio file on disk.

Most mirdata datasets have ``track.audio_path`` resolve directly to the file
on disk — that's always tried first. When it doesn't (typically: after
``tools/migrate_mirdata_dataset.py`` has already moved a track's audio into
this project's own ``tracks/`` layout, so mirdata's own nested path no
longer exists), resolution falls back in two tiers:

1. **Exact match by track_id.** Safe and unambiguous — ``track_id`` is
   guaranteed unique within a dataset — but only works for datasets where
   ``track_id`` already equals the audio filename's stem (true for
   ballroom, e.g. ``Media-105901`` -> ``Media-105901.wav``; verified, not
   assumed — see the DatasetConfig docstring for what breaks this
   assumption).
2. **Trailing-digit match on a configured attribute**
   (``DatasetConfig.fallback_match_attr``), for datasets where tier 1
   can't work at all because ``track_id`` doesn't resemble the filename
   (e.g. rwc_jazz's mirdata id ``RM-J001`` vs. its audio's own
   ``RWC_J001``). This tier is opt-in per dataset, not a default: matching
   by trailing digits is only safe when the chosen attribute's numbers are
   unique across the *whole* dataset. ``track_id`` itself is a tempting
   default here but is NOT safe for this — e.g. ballroom has 16+ different
   track_ids across different albums all ending in "...-04" (per-album
   track numbering, not a global id) — which is exactly why tier 1 matches
   the full track_id, never just a trailing number.

Onboarding a new mirdata dataset needs no config at all unless one of these
tiers doesn't apply — see ``DatasetConfig``'s docstring.

Used by ``tools/migrate_mirdata_dataset.py`` (annotation + audio migration)
and ``tools/merge_datasets.py``'s mirdata fallback. Training
(``musicality/loaders/{tempo_dataset,beat_dataset}.py``) no longer touches
mirdata at all — it reads this project's own tracks/+annotations/ format
directly, which is why this module lives under ``tools/`` rather than
``musicality/``: only migration tooling still needs mirdata.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from musicality.dataformats.track_io import sanitize_track_name


@dataclass
class DatasetConfig:
    """Per-dataset quirks for locating a track's on-disk audio file.

    Onboarding a new mirdata dataset:
      - Most datasets need NO entry here — mirdata's own ``track.audio_path``
        resolves on first migration, and the exact-track_id fallback tier
        (see module docstring) already covers re-resolving audio a prior
        migration run moved into ``tracks/`` (true for ballroom, and any
        new dataset by default).
      - If the dataset's mirdata Track class exposes its primary audio under
        a different attribute (e.g. guitarset has no ``audio_path`` at all,
        only ``audio_mic_path``/``audio_mix_path``/...), set
        ``audio_path_attr`` to that name.
      - If ``track_id`` doesn't equal the audio filename's stem at all (e.g.
        rwc_jazz's mirdata id ``RM-J001`` vs. its audio's own ``RWC_J001``),
        set ``fallback_match_attr`` to a track attribute (e.g.
        ``piece_number``) whose trailing digits do match the filename's —
        and whose values are unique across the whole dataset (a
        per-album-restarting counter is NOT safe here; verify before
        adding).
    """

    audio_path_attr: str = "audio_path"
    fallback_match_attr: str | None = None


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "rwc_classical": DatasetConfig(fallback_match_attr="piece_number"),
    "rwc_jazz": DatasetConfig(fallback_match_attr="piece_number"),
    "rwc_popular": DatasetConfig(fallback_match_attr="piece_number"),
    "guitarset": DatasetConfig(audio_path_attr="audio_mic_path"),
}


@dataclass
class AudioIndex:
    """On-disk ``.wav`` index for the fallback tiers in :func:`resolve_audio_path`."""

    by_stem: dict[str, Path]
    by_trailing_number: dict[int, Path]


def index_audio(data_home: Path) -> AudioIndex:
    """Index every ``.wav`` under *data_home* (including ``tracks/``, where
    audio reorganized into this project's own layout lives) by both its
    full filename stem and its trailing number, in a single scan.
    """

    by_stem: dict[str, Path] = {}
    by_trailing_number: dict[int, Path] = {}

    for wav in data_home.rglob("*.wav"):
        by_stem[wav.stem] = wav
        match = re.search(r"(\d+)$", wav.stem)
        if match:
            by_trailing_number[int(match.group(1))] = wav

    return AudioIndex(by_stem=by_stem, by_trailing_number=by_trailing_number)


def resolve_audio_path(
    track,
    dataset_name: str,
    audio_index: AudioIndex | None = None,
) -> Path | None:
    """Return the on-disk audio path for *track*, or ``None`` if unresolved.

    Trusts mirdata's own audio-path attribute first
    (``DatasetConfig.audio_path_attr``, ``"audio_path"`` by default). Then
    tries the two fallback tiers described in this module's docstring:
    exact match by ``track_id`` against *audio_index*, then — only for
    datasets with a configured ``fallback_match_attr`` — trailing-digit
    match on that attribute.

    :param audio_index: Precomputed by :func:`index_audio`. Required for
        either fallback tier to find anything; omit to only ever try the
        primary ``audio_path_attr`` tier.
    """

    config = DATASET_CONFIGS.get(dataset_name, DatasetConfig())

    audio_path = getattr(track, config.audio_path_attr, None)
    if audio_path is not None and Path(audio_path).exists():
        return Path(audio_path)

    if audio_index is None:
        return None

    exact = audio_index.by_stem.get(sanitize_track_name(str(track.track_id)))
    if exact is not None:
        return exact

    if config.fallback_match_attr is None:
        return None

    match_value = getattr(track, config.fallback_match_attr, None)
    if not match_value:
        return None

    match = re.search(r"(\d+)$", str(match_value))
    if not match:
        return None

    return audio_index.by_trailing_number.get(int(match.group(1)))
