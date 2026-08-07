# Part 1 — Centralize the annotation data format

## Context

Today the physical layout of annotator-produced data — folder names
(`"tracks"`, `"annotations"`) and file suffixes (`.beats`, `.meta.json`) —
is hardcoded as string literals scattered across multiple functions in
`tools/annotator/data.py` (`annotation_path:232-234`, `metadata_path:237-239`,
`list_datasets:309-338`, `has_annotation:341-343`,
`has_mirdata_annotation:346-355`, `load_dataset_tracks:358-368`,
`load_track:371-423`, `delete_track:444-457`, `rename_track:460-484`).
Worse, it's already drifted: `tools/mobile_companion/server.py:96,113`
hardcodes its own `"tracks"` literal instead of reusing a shared constant,
even though the module already imports `tools.annotator.data` directly for
everything else. There's no single declared place that says "this is the
on-disk annotation format," and no version marker on the metadata schema,
so a future format change (like Part 2) has no way to detect old files or
migrate them.

The repo already has an established pattern for exactly this kind of thing:
`musicality/dataformats/dataformat.yaml` + the `DataFormat` dataclass
(`musicality/dataformats/__init__.py`), currently covering only `data_dir`
and `splits_dir`, loaded via `dataformats.load()` and consumed by
`BeatDataset`, `TempoDataset`, `Splitter`, and the annotator tool. Extending
this existing single-YAML pattern is preferable to inventing a second,
parallel config mechanism — the metadata *schema itself* (field names/types)
stays as Python dataclasses in `data.py`, since both consumers (desktop
annotator, mobile companion) already import that one module directly, so
there's no drift risk there to solve.

## Design

**Extend `musicality/dataformats/dataformat.yaml`** with the annotation
layout:

```yaml
data_dir: data
splits_dir: data/splits
tracks_dirname: tracks
annotations_dirname: annotations
beats_suffix: .beats
metadata_suffix: .meta.json
```

**Extend `DataFormat`** (`musicality/dataformats/__init__.py:14-23`) with
matching required fields (`tracks_dirname: str`, `annotations_dirname: str`,
`beats_suffix: str`, `metadata_suffix: str`) — no defaults needed since this
is a single checked-in file, not a per-environment override.

**Update `tools/annotator/data.py`** to source these from
`dataformats.load()` (module-level `_fmt = dataformats.load()`, next to the
existing `DATA_DIR` line at `data.py:19`) instead of inline literals, in:
`annotation_path`, `metadata_path`, `list_datasets` (the `tracks_dir`/
`ann_dir` construction and the `*.beats` glob), `has_annotation`,
`has_mirdata_annotation`, `load_dataset_tracks`, `load_track`,
`delete_track`, `rename_track`. `_AUDIO_EXTENSIONS` (audio file extensions)
is a separate concern and stays as-is — out of scope here.

**Fix the drift**: update `tools/mobile_companion/server.py:96,113` to reuse
the same constant (e.g. `annotator_data.TRACKS_DIRNAME` or by calling a
`data.py` helper) instead of its own `"tracks"` literal.

**Add a schema version marker**: a module-level constant in `data.py`
(e.g. `METADATA_SCHEMA_VERSION = 1`) and a `schema_version: int` field on
`TrackMetadata` (`data.py:51-65`), always written by `save_metadata` and
read (with a `.get`-style default of `1` for pre-existing files that predate
the field) by `load_metadata`. This is the field Part 2 will bump to `2`.

## Files to modify

- `musicality/dataformats/dataformat.yaml`, `musicality/dataformats/__init__.py`
- `tools/annotator/data.py`
- `tools/mobile_companion/server.py` (remove the duplicated `"tracks"` literal)
- `tests/test_annotator_metadata.py`, `tests/test_annotator_data.py` (adjust
  any hardcoded `"annotations"`/`"tracks"`/`.beats` path assumptions to go
  through the same config, add a small test for `DataFormat` picking up the
  new fields and for `schema_version` round-tripping)

## Verification

- `uv run pytest tests/ -v` (full suite — this touches shared path-building
  code used by many tests via `DATA_DIR` monkeypatching)
- Manually run `uv run python -m tools.annotator --dataset swing` and confirm
  it still lists/loads/saves tracks against the real `data/swing` fixture
  data unchanged (paths must resolve identically to before).
- Start the mobile companion (`tools/mobile_companion/server.py`) and
  confirm a track upload still lands at the same `tracks/<id>.wav` path.
