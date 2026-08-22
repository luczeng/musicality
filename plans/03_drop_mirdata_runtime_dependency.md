# Part 3 — Drop mirdata as a runtime dependency

Builds on [Part 1](01_centralize_annotation_data_format.md)'s
`tracks_dirname`/`annotations_dirname`/`beats_suffix`/`metadata_suffix`
config and [Part 2](02_add_annotator_and_phrase_metadata.md)'s annotator-slot
layout. Part 2 explicitly scoped out bridging annotation data into
training ("`BeatDataset`/`TempoDataset` still read exclusively through
mirdata... bridging annotator-produced data into training is separate
future work") — this part is that bridging work, and goes further: mirdata
is removed from the read path entirely, not just supplemented.

## Context

Fixing a `TempoDataset` crash this session surfaced a much bigger problem:
`BeatDataset` reads beat annotations straight from mirdata's raw
`Track.beats` object, never from this project's own migrated `.beats`
files. Concretely, for `RWC_J001`, mirdata's raw beats are
`[0.03, 0.81, 1.51, 2.27, 3.04]` (sparse downbeats) while the hand-corrected
`.beats` file already migrated has `[0.03, 0.42, 0.81, 1.16, 1.51, ...]`
(every beat) — meaning **all the CSV annotation corrections for
rwc_jazz/rwc_classical/rwc_popular currently have zero effect on
training.** Separately, resolving mirdata's own audio path required a fair
amount of fallback-matching machinery
(`musicality/loaders/mirdata_audio.py`) purely because mirdata's expected
on-disk layout doesn't match how this repo's audio actually sits on disk.

Both problems disappear if every dataset — mirdata-sourced or homemade — is
required to go through one canonical on-disk format before anything else
touches it: `tracks/<id>.wav` + `annotations/<id>.beats` +
`annotations/<id>.meta.json` (already documented in `docs/source/data.rst`,
already what the annotator/mobile companion read for custom datasets).
mirdata becomes purely a *fetch* tool (`tools/download_dataset.py`); a
migration tool bridges anything it fetches into this format. Training
loaders, the annotator, the mobile companion, and `tools/merge_datasets.py`
never import `mirdata` again.

**Decisions locked in during planning:**
- Migration **physically moves** each track's audio into `tracks/<stem>.wav`
  (no symlinks — consistent with recent guidance on `merge_datasets.py`; no
  duplication like a copy would cause).
- The annotator's "built-in mirdata annotation" indicator column
  (`main_window.py:685-688`, backed by `has_mirdata_annotation()`) is
  **removed** entirely — mirdata is never consulted at runtime
  post-migration, so it would always read "no".
- Tempo label: reuse the `bpm_median` field already written to
  `.meta.json` by both migration tools today (mean/median/std are already
  computed and saved) — `TempoDataset` just needs to start reading it
  instead of mirdata's raw `track.tempo`. No new metadata field needed.
- Out of scope: `tools/summarize_datasets.py`,
  `tools/plot_tempo_histograms.py`, `tools/inspect_track.py` stay
  mirdata-based — ad hoc pre-migration inspection tools, not part of the
  annotation/training pipeline.

## Design

### 1. Shared track I/O primitives move into `musicality/`

New module `musicality/dataformats/track_io.py`, moved out of
`tools/annotator/data.py`: `TrackData`, `TrackMetadata`,
`METADATA_SCHEMA_VERSION`, `read_beats_file` (promote the current private
`_read_beats_file`), `save_annotations`, `save_metadata`, `load_metadata`,
`annotation_path`/`metadata_path`/`_annotations_slot_dir`, plus two new
helpers:
- `list_migrated_track_ids(dataset_name)` — stems with a default-slot
  `.beats` file (consolidates the ad hoc `_migrated_beats_files` logic
  already duplicated in `tools/merge_datasets.py`).
- `resolve_track_audio(dataset_name, stem)` — `tracks/<stem>.wav`
  existence check.
- `bpm_stats(beat_times)` — consolidates the `_bpm_stats()` currently
  duplicated verbatim in `tools/migrate_mirdata_dataset.py` and
  `tools/migrate_rwc_genre.py`.

This fixes the layering direction: `musicality/loaders/*.py` needs these
primitives but must never import from `tools/` (tools/ builds on
`musicality/`, not the reverse). `tools/annotator/data.py` re-imports
everything from here, keeping only its annotator-specific pure functions
(`cycle_positions`, `add_beat`, `remove_beat`, `active_beat_position`,
`bar_indices`, `active_bar_index`, `is_accent_beat`, `beats_per_bar`,
`tempo_from_beats` for the live tap-tempo UI estimate) and its higher-level
dataset-browsing functions, simplified in step 4.

### 2. Migration tools produce fully self-sufficient datasets

- `tools/migrate_mirdata_dataset.py`: after resolving each track's audio
  (still needs mirdata + the `DATASET_CONFIGS`/`resolve_audio_path`
  fallback-matching logic — **relocate**
  `musicality/loaders/mirdata_audio.py` → `tools/mirdata_audio.py`, since
  it's now purely a migration-time concern), **move** the resolved audio
  file into `tracks/<stem>.wav` instead of leaving it under mirdata's own
  layout. Reuse the new shared `bpm_stats()` instead of its local copy.
  Update the module docstring to drop the "audio is left untouched, rename
  it yourself" language.
- `tools/migrate_rwc_genre.py`: reuse the shared `bpm_stats()`; no
  audio-placement change needed (CSV-annotated datasets already require
  `tracks/<stem>.wav` to pre-exist).

### 3. Loaders read only tracks/+annotations/

- `musicality/loaders/tempo_dataset.py`: remove `import mirdata`; iterate
  `list_migrated_track_ids(name)`, resolve audio via
  `resolve_track_audio`, read tempo from
  `load_metadata(name, stem).bpm_median` (skip a track if metadata or
  `bpm_median` is missing — same skip-if-unavailable philosophy as today).
- `musicality/loaders/beat_dataset.py`: remove `import mirdata`; iterate
  the same migrated-id list, resolve audio the same way, call
  `read_beats_file(annotation_path)` for times/positions. Existing
  target-construction logic (`gaussian_smear`, `group_size`, `binary_only`
  filtering) is unchanged — only the data-acquisition front half changes.
- Delete `musicality/loaders/mirdata_audio.py` (moved to `tools/` in step
  2) — `musicality/loaders/` no longer imports `mirdata` anywhere.

### 4. Simplify the annotator and merge tool

- `tools/annotator/data.py`: delete `has_mirdata_annotation()`; remove the
  mirdata-fallback branches in `list_datasets()`, `annotation_meter_label()`,
  `load_dataset_tracks()`, `load_track()` — every dataset directory is now
  expected to already have `tracks/`; raise a clear error naming the right
  `migrate_*.py` tool if it doesn't, instead of silently falling back to
  mirdata.
- `tools/annotator/main_window.py`: remove the "built-in mirdata
  annotation" indicator column (~lines 681-691) and its
  `has_mirdata_annotation` import; keep only the "has a saved annotation
  from this app" indicator.
- `tools/merge_datasets.py`: drop the mirdata-fallback branch in
  `_audio_stems()` (always reads `tracks/` now); drop its
  `mirdata`/`mirdata_audio` imports.
- `tools/mobile_companion/server.py` already goes through
  `tools.annotator.data` exclusively — no direct change expected, just
  benefits from the simplified functions underneath.

### 5. Docs

Rewrite `docs/source/data.rst`'s "Data format" section: mirdata is
acquisition-only (`tools/download_dataset.py`); this project's own
`tracks/`+`annotations/` format is the only thing every other tool
(annotator, mobile companion, training loaders, splits) reads. Remove the
now-false "not yet bridged into training" caveat from Part 2 — it's bridged
as of this part.

## Files to modify

- `musicality/dataformats/track_io.py` — new
- `tools/annotator/data.py` — trim to annotator-specific logic, re-import
  shared primitives
- `musicality/loaders/tempo_dataset.py`, `musicality/loaders/beat_dataset.py`
  — drop mirdata, read tracks/+annotations/ directly
- `musicality/loaders/mirdata_audio.py` → `tools/mirdata_audio.py` (moved)
- `tools/migrate_mirdata_dataset.py`, `tools/migrate_rwc_genre.py` — reuse
  shared `bpm_stats()`; mirdata tool also moves audio into `tracks/`
- `tools/merge_datasets.py` — drop mirdata fallback
- `tools/annotator/main_window.py` — remove mirdata-annotation column
- `docs/source/data.rst`
- `tests/test_tempo_dataset.py`, `tests/test_beat_dataset.py` — replace
  `mirdata.initialize` mocking with real temp-directory
  tracks/+annotations/ fixtures
- `tests/test_annotator_data.py` — remove/update tests for deleted
  mirdata-fallback paths
- New test coverage for `musicality/dataformats/track_io.py`

## Verification

- `uv run pytest tests/` — full suite green, including rewritten loader
  tests.
- `uv run python tools/migrate_mirdata_dataset.py --dataset ballroom --force`
  — confirm it now creates `tracks/*.wav` (moved, not symlinked) alongside
  `annotations/*.beats`+`*.meta.json`.
- `uv run python tools/create_splits.py --datasets rwc_jazz rwc_classical rwc_popular ballroom --force`
  — confirm splits still generate correctly with the new loaders.
- `grep -rn "import mirdata" musicality/` — zero results once done
  (mirdata fully confined to `tools/`).
- Launch `uv run python -m tools.annotator` briefly and confirm the track
  list still loads, and the annotation indicator column behaves correctly
  for a migrated dataset.
