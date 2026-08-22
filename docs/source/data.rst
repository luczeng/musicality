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

Merging datasets
-----------------

``tools/merge_datasets.py --datasets ballroom brid --output ballroom_brid``
combines several already-migrated datasets into one logical dataset,
without moving or copying any audio/annotation files. It writes a single
manifest — ``../musicality_db/ballroom_brid/tracks.txt``, one
``<dataset_name>/<track_id>`` line per line (the filename comes from
``dataformat.yaml``'s ``manifest_filename``) — and fails fast if any source
dataset hasn't been migrated yet (no ``.beats`` files at all).

``TempoDataset``/``BeatDataset`` resolve this manifest transparently via
``musicality.dataformats.track_io.list_track_refs``: passing
``name="ballroom_brid"`` reads every ``<dataset_name>/<track_id>`` entry and
loads each track's audio/annotations from its own source dataset's
directory rather than from ``ballroom_brid/`` itself. A merged dataset name
is then usable anywhere a real dataset name is — including
``data.name: ballroom_brid`` in a training config, and
``tools/create_splits.py --datasets ballroom_brid`` to give it its own
train/val split — with no further code changes.

The one constraint: a merged dataset's source datasets are resolved
*sibling-relative* to its own directory (``ballroom_brid``'s parent, i.e.
``../musicality_db``), not copied or referenced by absolute path. This is
already how ``merge_datasets.py`` always writes a merged output, so it's
never a concern in the normal workflow — it only matters if a manifest is
relocated by hand.

API reference
-------------

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.dataformats
   ~musicality.splits.splitter
