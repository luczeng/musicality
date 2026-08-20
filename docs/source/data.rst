Data formats & splits
=====================

Two sources feed training data, both read identically by every loader/tool in
this repo:

- **mirdata datasets** — publicly available beat/tempo-annotated datasets
  (ballroom, brid, hainsworth, rwc_classical, rwc_jazz, rwc_popular,
  groove_midi, guitarset), fetched via
  `mirdata <https://mirdata.readthedocs.io>`_.
- **Homemade datasets** — audio recorded and beat-tapped by hand with the
  annotation apps (e.g. a ``swing`` dataset of hand-recorded dance tracks).
  These live under ``data/<name>/tracks/`` (audio) and
  ``data/<name>/annotations/*.beats`` (tapped beats), a plain directory
  layout rather than a mirdata dataset definition.

Data format
-----------

**mirdata datasets** are read entirely through
`mirdata <https://mirdata.readthedocs.io>`_'s own API and on-disk layout —
``TempoDataset``/``BeatDataset`` never touch the files directly, just
``track.audio_path``, ``track.tempo``, ``track.beats.times``, and
``track.beats.positions`` (1-indexed bar/count position per beat, when the
dataset annotates it).

**Homemade datasets** (recorded via the annotation apps) use a parallel,
hand-rolled layout under ``data/<dataset>/``, centralized in
``musicality.dataformats`` (``musicality/dataformats/dataformat.yaml``).

``.beats`` files use the same ``<time> <position>`` per-line format as
mirdata's own raw beat annotations (e.g. ballroom's), so homemade and mirdata
tracks read identically once loaded — one line per beat, seconds then
1-indexed bar/count position:

.. code-block:: text

   10.949773 1
   11.247052 2
   11.653333 3

``.meta.json`` carries fields mirdata has no place for — all optional, filled
in incrementally as an annotation is worked on, and forward-compatible (a
file missing a newer field just falls back to that field's default on load):

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
     - Tempo statistics derived from the tapped beats
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
parallel ``.beats``/``.meta.json`` pair for the same ``track_id``. Not yet
used to feed training — ``BeatDataset``/``TempoDataset`` still read
exclusively through mirdata; bridging homemade annotations into training is
separate future work.

API reference
-------------

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.dataformats
   ~musicality.splits.splitter
