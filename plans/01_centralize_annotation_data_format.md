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

## Follow-up: centralizing `dataformats.FORMAT` (post-implementation cleanup)

After the design above shipped, `tools/annotator/data.py` still re-exported
its own copies of `TRACKS_DIRNAME`/`ANNOTATIONS_DIRNAME`/`BEATS_SUFFIX`/
`METADATA_SUFFIX` as module constants (derived once from `dataformats.load()`
at import time). That forced other consumers — notably
`tools/mobile_companion/server.py` — to `import tools.annotator.data` just to
read `annotator_data.TRACKS_DIRNAME`, even though the value conceptually
belongs to `musicality.dataformats`, not the annotator. `beat_dataset.py` and
`tempo_dataset.py` also independently re-derived `DATA_DIR` from
`dataformats.load()` the same way.

Fixed by adding a load-once singleton directly to
`musicality/dataformats/__init__.py`:

```python
FORMAT = load()
DATA_DIR = ROOT / FORMAT.data_dir
```

Every consumer (`data.py`, `server.py`, `beat_dataset.py`, `tempo_dataset.py`)
now reads `dataformats.FORMAT.<field>` / `dataformats.DATA_DIR` directly
instead of holding its own copy or reaching through a sibling module.

### The monkeypatch quirk this leaves behind

`data.py` ends up with two different binding patterns for config values, and
it's worth remembering why before "fixing" it later:

**`DATA_DIR` — copied once at import time, into `data.py`'s own namespace:**

```python
# tools/annotator/data.py, top of file
DATA_DIR = dataformats.DATA_DIR   # copies the Path object; no ongoing link

def load_track(dataset_name, track_id):
    tracks_dir = DATA_DIR / dataset_name / ...   # reads data.py's own module global
```

After this line runs, `data.py.DATA_DIR` and `dataformats.DATA_DIR` are two
separate names that happen to point at the same object. Every function in
`data.py` reads the bare name `DATA_DIR`, resolved against `data.py`'s own
module globals — so `monkeypatch.setattr(annotator_data, "DATA_DIR",
tmp_path)` works because it overwrites exactly the slot those functions read.

**`tracks_dirname` etc. — read fresh from the source on every call:**

```python
def load_track(dataset_name, track_id):
    tracks_dir = DATA_DIR / dataset_name / dataformats.FORMAT.tracks_dirname
```

No local copy — every call does the attribute lookup live, through the
`dataformats` module reference. So the corresponding tests patch the object
actually being read: `monkeypatch.setattr(dataformats.FORMAT,
"tracks_dirname", "notes")` mutates the one shared `FORMAT` instance, and
every consumer sees it immediately.

**Why this wasn't unified**: `annotator_data.DATA_DIR` is already
monkeypatched in ~20 existing tests across `test_annotator_data.py`,
`test_annotator_metadata.py`, and `test_mobile_companion_server.py`. Making
`data.py` read `dataformats.DATA_DIR` live (the same way as the format
fields) would silently break all of those — the functions would stop reading
the attribute those tests patch, so tests could start touching the real
`data/` directory instead of `tmp_path` without erroring. The two new
format-field tests were low-risk to redirect since they were written in the
same session as this cleanup; rewriting 20 pre-existing call sites was out of
scope for a request that was specifically about the format fields.

If full consistency is ever wanted (`DATA_DIR` also read live from
`dataformats.DATA_DIR` everywhere, no local copy in `data.py`), that requires
updating those ~20 monkeypatch sites to target `dataformats.DATA_DIR`
instead — a deliberate, separate follow-up, not a byproduct of this one.
