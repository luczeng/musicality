Augmentations
=============

Training-time transforms applied to the **train split only** — validation
always sees a fixed, unaugmented clip, so eval numbers stay comparable
epoch to epoch. Configured under each training config's ``augmentations:``
block (see :doc:`configuration`); the top-level ``augmentations.enabled``
flag gates all of them at once.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Augmentation
     - What it does
     - Config path
   * - Random crop
     - Draws the fixed-``duration`` clip at a random offset into the track
       on every access, instead of always the first ``duration`` seconds.
       Beat/one/last annotation times are shifted to stay aligned with the
       cropped window. Beat-phase/beat-only training only
       (:class:`~musicality.loaders.beat_dataset.BeatDataset`); the tempo
       pipeline doesn't crop (whole-clip regression). Lives outside the
       ``augmentations:`` block and is implemented at the dataset level
       rather than in the augmenter pipeline below, since it needs the
       full, uncropped waveform to pick an offset from.
     - ``data.random_crop`` (beat configs only, default ``true``)
   * - Time stretch
     - Randomly speeds up or slows down the clip (via resampling) by a
       rate drawn from ``[min_rate, max_rate]``. Pitch shifts as a
       side effect. Tempo pipeline: scales the scalar tempo label by the
       same rate. Beat-phase pipeline: resamples the frame target's time
       axis by the same rate so beat/one/last/mask events stay aligned
       with the stretched audio.
     - ``augmentations.time_stretch.{enabled,min_rate,max_rate}``
   * - Gain
     - Scales waveform amplitude by a random gain drawn uniformly from
       ``[min_db, max_db]``.
     - ``augmentations.gain.{enabled,min_db,max_db}``
   * - Noise
     - Adds white Gaussian noise at a fixed standard deviation ``std``
       (relative to full-scale ±1 audio). Disabled by default in every
       training config.
     - ``augmentations.noise.{enabled,std}``

Two augmenter classes apply the waveform ops (time stretch, gain, noise) in
that order: :class:`~musicality.augmentations.TempoAugmenter` for the
scalar-label tempo pipeline, :class:`~musicality.augmentations.BeatPhaseAugmenter`
for the frame-level beat pipeline — the latter additionally carries the
frame target through the time-stretch step. Both are built from the same
``augmentations:`` config shape via
:func:`~musicality.augmentations.build_augmenter` /
:func:`~musicality.augmentations.build_beat_phase_augmenter`, which return
``None`` (skip wrapping entirely) if nothing ends up enabled.

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.augmentations
