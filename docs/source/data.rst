Data formats & splits
=====================

Every dataset — whether fetched via mirdata or recorded by hand — is read
through one on-disk format: ``../musicality_db/<name>/tracks/`` (audio) and
``../musicality_db/<name>/annotations/`` (``.beats`` + ``.meta.json``), a
plain directory layout centralized in ``musicality.dataformats``
(``musicality/dataformats/dataformat.yaml``, ``data_dir`` currently
``../musicality_db`` — a sibling git+dvc repo, cloned by
``tools/setup_remote.sh``). ``TempoDataset``/``BeatDataset``, the desktop
annotator, and the mobile companion all read exclusively through this
format via ``musicality.dataformats.track_io`` — none of them import
mirdata.

mirdata is acquisition-only: ``tools/download_dataset.py`` fetches a
dataset (ballroom, brid, hainsworth, rwc_classical, rwc_jazz, rwc_popular,
groove_midi, guitarset, ...) into its own raw layout. A migration tool then
bridges it into this project's format before anything else touches it:

- ``tools/migrate_mirdata_dataset.py`` — for mirdata datasets. Writes
  ``annotations/*.beats``/``*.meta.json`` from mirdata's own beat
  annotations, and moves (not copies) each track's audio into ``tracks/``
  — mirdata is never read again for a migrated track. Handles datasets
  whose on-disk audio layout doesn't match mirdata's own index (see
  ``tools/mirdata_audio.py``).
- ``tools/migrate_rwc_genre.py`` — for CSV-annotated datasets (e.g.
  hand-corrected RWC annotations) that already have audio under
  ``tracks/``.
- ``tools/migrate_gtzan.py`` — for GTZAN. Its audio is downloaded
  separately, via the data dir's own ``dl_gtzan.py`` (HuggingFace's
  marsyas/gtzan mirror — mirdata's ``gtzan_genre`` audio host is a dead
  link), and its beat annotations come from CPJKU's beat_this_annotations
  project, already dropped in as this project's own ``.beats`` format.
  The two sources number tracks with a one-off mismatch; this tool
  resolves that offset, then moves audio into ``tracks/`` same as the
  other tools.

A dataset with no ``tracks/`` folder is treated as not-yet-migrated: the
annotator won't list it, and ``tools.annotator.data`` raises an error
naming the migration command to run, rather than silently falling back to
mirdata.

Data format
-----------

``.beats`` files are ``<time> <position>`` per line — seconds, then the
1-indexed bar/count position, when annotated:

.. code-block:: text

   10.949773 1
   11.247052 2
   11.653333 3

Count across whatever you consider one bar — the count does not have to match
the ``group_size`` a model is trained at. A bar counted ``1..8`` against a
4-beat model is folded to ``1,2,3,4,1,2,3,4`` when the dataset is loaded, so
beat 5 correctly trains as a downbeat (see
:func:`musicality.loaders.beat_dataset.fold_positions`). A count that is *not*
a whole multiple of ``group_size`` — 6 against 4, say — has no consistent
folding; those tracks still contribute their beats, but their bar positions are
masked out of the loss rather than folded wrongly.

``.meta.json`` carries fields with no place in ``.beats`` — all optional,
filled in incrementally as an annotation is worked on, and forward-compatible
(a file missing a newer field just falls back to that field's default on
load):

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``location``
     - Where the track was recorded/found
   * - ``device``
     - Recording device (e.g. phone model, hostname)
   * - ``structure``
     - Free-text song structure notes
   * - ``duration_s``
     - Audio duration, in seconds
   * - ``bpm_mean`` / ``bpm_median`` / ``bpm_std``
     - Tempo statistics derived from the beat annotation itself (see
       ``track_io.bpm_stats``) — there's no separate ground-truth tempo
       field in this format. ``TempoDataset`` reads ``bpm_median`` as its
       tempo label.
   * - ``annotator_id``
     - Who made this annotation — ``null`` for the original/default slot
   * - ``section_aligned``
     - Whether the first tapped beat is the true start of a section
       (``true``/``false``), or ``null`` if not recorded
   * - ``schema_version``
     - Metadata schema version (currently ``2``)

Multiple people can annotate the same track independently:
``annotator_id: null`` is the original, unsuffixed slot (every file saved
before multi-annotator support existed still resolves here — no migration
needed), and each named annotator gets their own subdirectory holding a
parallel ``.beats``/``.meta.json`` pair for the same ``track_id``.
``TempoDataset``/``BeatDataset`` read only the default (``null``) slot per
track — picking a specific annotator's take for training is separate future
work.

Splits
------

``tools/create_splits.py`` builds a train/val split for a migrated dataset
and saves it under ``splits_dir/<name>/{train,val}.txt`` — one
``<dataset_name>/<track_id>`` line per track, not a positional index, so a
split's contents can be read back, concatenated, or merged with another
dataset's split independently of any one dataset instance's ordering (see
``musicality.splits.splitter.Splitter``). It creates both a tempo split
(``<name>``) and a beat-phase split (``beat_phase-<name>``, or
``beat_phase-<name>-binary`` with ``--binary-only``) per dataset.
``Splitter.load_refs``/``.save_refs`` read and write this format directly;
``TempoDataset``/``BeatDataset`` accept it straight via their ``refs=``
argument, or ``Splitter(...).run()`` for the ``Subset``-returning form used
during training.

Splitting a subset of a dataset: ``--contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some datasets encode a category in the track id — gtzan's tracks are
``blues_00001``, ``jazz_00042``, ``rock_00007``, and so on. To work with one
of those groups alone, pass a substring::

    uv run python tools/create_splits.py --datasets gtzan --contains blues

That keeps only the tracks whose id contains ``blues`` (matched
case-insensitively) and writes them to their own split name,
``splits_dir/gtzan-blues/`` — plus ``beat_phase-gtzan-blues`` for the beat
kind. The dataset's full split, if it has one, is left untouched.

The filter applies at creation time only, and deliberately so: what it
produces is an ordinary split file, indistinguishable downstream from any
other. Train on it with ``data.input=gtzan-blues``, evaluate it by that
name, and combine groups with ``tools/merge_datasets.py --datasets
gtzan-blues gtzan-jazz --output gtzan-blues_jazz`` — no consumer needs to
know the split was ever narrowed, and nothing has to re-derive the filter
to reproduce it.

The filtering itself is ``musicality.dataformats.track_io.list_track_refs``'
``contains=`` argument, applied *before* either dataset is built, so a
narrowed run doesn't pay to resolve the tracks it's about to drop. Substring
matching is intentionally all it does — for anything finer, assemble a
folder of ``train.txt``/``val.txt`` by hand and point ``data.input`` at the
path (see below).

Flagged tracks are excluded from splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A track whose metadata has ``warning`` (the annotator's ⚠ Flag — "this
annotation looks wrong") or ``needs_review`` (❓ To Review — "take another
look") set to ``True`` is skipped by the splitter, and so never reaches
training or validation. The filter runs on both sides: ``create()``
excludes flagged tracks from the pool before splitting, so they're written
into neither ``train.txt`` nor ``val.txt``, and every read path
(``run()``, ``load_refs``, ``load_refs_from_dir``) drops them again on
load. That second half is what makes flagging usable day to day — a track
flagged in the annotator *after* a split was generated falls out of the
next training run on its own, with no need to regenerate the split file or
re-run ``create_splits.py``. Only the default annotation slot's metadata is
consulted, matching which slot the loaders read. Clearing the flag in the
annotator puts the track straight back in.

Merging datasets
-----------------

``tools/merge_datasets.py --datasets ballroom brid --output ballroom_brid``
combines several datasets' *existing* splits into one merged split, keeping
train and val separate: each source dataset's train tracks go into the
merged train split, and its val tracks into the merged val split — so a
track held out for one dataset stays held out in the merge. It writes
nothing under the source datasets' own directories, and no merged dataset
directory of any kind — only
``splits_dir/ballroom_brid/{train,val}.txt``, in the same format
``create_splits.py`` produces. There's no ``--val-split`` argument: the
merge inherits whatever ratio each source was already split at, and fails
fast, before writing anything, if a requested source doesn't have a split
yet (run ``tools/create_splits.py`` first).

A merged split name is then usable anywhere a real one is — e.g.
``data.input: ballroom_brid`` in a training config — since
``build_dataloaders``/``build_beat_dataloaders`` load a split's refs and
construct ``TempoDataset``/``BeatDataset`` directly via ``refs=``, with no
distinction between a plain dataset's split and a merged one.

Telling training which split to use: ``data.input``
-----------------------------------------------------

Training configs (``configs/train.yaml``, ``beat_train.yaml``,
``beat_only_train.yaml``) have one field, ``data.input``, for naming the
split to train on. It's read two ways, told apart by whether it contains a
``/``:

- **A bare name** — e.g. ``data.input=ballroom`` or
  ``data.input=ballroom_brid`` — looked up under the canonical
  ``splits_dir``: ``splits_dir/<input>/{train,val}.txt``. This is the normal
  case: everything ``create_splits.py`` and ``merge_datasets.py`` produce
  lands under ``splits_dir`` automatically, so a name is all you need.
- **A path** — e.g. ``data.input=../musicality_db/splits/ballroom`` or
  ``data.input=/anywhere/my_split`` — used directly as the folder to read
  ``train.txt``/``val.txt`` from, bypassing ``splits_dir`` entirely. Use
  this for a split that isn't registered under ``splits_dir`` — one
  assembled by hand, or one living outside this repo's data directory.

See ``musicality.trainers.common.resolve_split_refs``, shared by both the
tempo and beat trainers, for the exact dispatch.

API reference
-------------

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.dataformats
   ~musicality.splits.splitter
