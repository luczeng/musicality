# Part 2 — Add annotator identity + section-alignment metadata

Depends on [Part 1](01_centralize_annotation_data_format.md) — reuses the
centralized dirname/suffix config and the `schema_version` field it
introduces.

## Context

Adds the two pieces of metadata originally requested: (1) support for
multiple independent annotations per track, tagged by who made them, and
(2) a per-annotation flag recording whether the first tapped beat — which
always starts count position 1, that part is guaranteed — also happens to
be the true start of a section, vs. landing mid-section (today this
distinction isn't recorded at all).

## Design

**Directory-based annotator slots**, built on Part 1's
`annotations_dirname`/`beats_suffix`/`metadata_suffix` config:

```
data/<dataset>/<annotations_dirname>/<track_id><beats_suffix>                    # legacy/default slot (annotator_id=None)
data/<dataset>/<annotations_dirname>/<track_id><metadata_suffix>
data/<dataset>/<annotations_dirname>/<annotator_id>/<track_id><beats_suffix>     # additional annotator
data/<dataset>/<annotations_dirname>/<annotator_id>/<track_id><metadata_suffix>
```

`annotator_id = None` is the existing unsuffixed "default" slot, so every
file already on disk today resolves exactly as before — zero migration.
Confirmed safe to nest under the existing `annotations/` folder: mirdata's
own per-dataset files live under a versioned subfolder (e.g.
`ballroom/B_1.0/...`), never directly inside our `annotations/` dir.

**Schema changes** in `tools/annotator/data.py` (bump
`METADATA_SCHEMA_VERSION` to `2`):

- `TrackData` (`data.py:39-48`): add `annotator_id: str | None = None`.
- `TrackMetadata` (`data.py:51-65`): add `annotator_id: str | None = None`
  and `section_aligned: bool | None = None` (`None` = unknown/unrecorded —
  matches all existing data; `True` = confirmed the first tap is the true
  start of a section; `False` = annotator flagged a mid-section start).

**Path/IO functions to update**:

- `annotation_path(track)` — nest under `track.annotator_id` when set.
- `metadata_path(dataset_name, track_id, annotator_id=None)` — add the
  parameter, nest when set.
- `save_metadata`/`load_metadata` — thread `annotator_id` through
  (`save_metadata` reads it off `metadata.annotator_id`).
- `load_track(dataset_name, track_id, annotator_id=None)` — add the
  parameter, thread into both the custom-dataset and mirdata-shadowing
  branches, set it on the returned `TrackData`.
- New `list_annotators(dataset_name, track_id) -> list[str | None]` —
  enumerates existing annotation slots for a track (`None` if the legacy
  flat file exists, plus each annotator subdirectory containing a
  `<track_id>.beats`).
- Reuse `tools/annotator/naming.py:sanitize_track_name` to sanitize
  `annotator_id`, since the mobile companion accepts it from network input
  and it becomes a path component.

**Edge case: shared audio, per-annotator sidecars.** Split `delete_track`
into two operations so deleting one annotator's take can't orphan another's:
- `delete_annotation(track)` — removes only this annotator's `.beats` +
  `.meta.json`.
- `delete_track(track)` — removes the audio file plus *every* annotator's
  data for that track_id (via `list_annotators`) — an explicit "delete
  everything" action.
`rename_track` gets a doc comment noting it renames the shared audio for
all annotators, not just the caller's slot.

**Not in scope**: `BeatDataset`/`TempoDataset` still read exclusively
through `mirdata` and continue to ignore these sidecars, same as today —
bridging annotator-produced data into training is separate future work.
UI wiring (annotator picker, identity input, section-aligned toggle in
`main_window.py`, mobile companion upload form) is follow-on work built on
top of this `data.py` API, not designed here.

## Files to modify

- `tools/annotator/data.py` (schema + path/IO changes above)
- `tests/test_annotator_metadata.py`, `tests/test_annotator_data.py` —
  round-trip save/load for a named annotator alongside the default slot
  staying untouched, `list_annotators` covering legacy-only / named-only /
  both-present cases, `delete_annotation` vs `delete_track`, and
  `schema_version` bumping to `2` on new saves while old files still load.

## Verification

- `uv run pytest tests/test_annotator_metadata.py tests/test_annotator_data.py -v`
- Manual check against `data/swing`: load the default slot for a known
  track (unchanged), save a second annotation under a new `annotator_id`,
  confirm `list_annotators` reports both, confirm the original default-slot
  file is byte-for-byte untouched.
- Confirm `BeatDataset`/`TempoDataset` still instantiate/iterate normally
  against a real mirdata dataset (e.g. `ballroom`) — this change is a no-op
  for the training path.
