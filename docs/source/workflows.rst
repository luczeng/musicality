Training & inference workflows
================================

Currently, three independent tasks, each with its own training entry point, checkpoint
directory, and (for the two beat tasks) a shared inference/postprocessing
path. This page is the map; the linked pages below are the reference detail.

Training
--------

All three train the same backbone, :class:`~musicality.models.tcn.TCNTempoNet`
— only ``frame_level``/``n_outputs`` differ, forced by the wrapping
``LightningModule`` regardless of what the model config says (see each
``configs/model/*.yaml``'s comment). What differs is the head, the loss, and
the target format.

.. list-table::
   :header-rows: 1

   * - Task
     - Entry point
     - Config
     - Head(s) / loss
     - Checkpoint dir
   * - Tempo
     - ``uv run python tools/train_tempo.py``
     - ``configs/train.yaml``
     - Pooled regression or classification (BPM bin) — see
       :doc:`losses`
     - ``checkpoints/``
   * - Beat-only
     - ``uv run python tools/train_beat_only.py``
     - ``configs/beat_only_train.yaml``
     - Single frame-level ``beat`` head, BCE —
       :class:`~musicality.trainers.beat_module.BeatModule`
     - ``checkpoints_beat_only/``
   * - Beat-phase
     - ``uv run python tools/train_beat.py``
     - ``configs/beat_train.yaml``
     - Three frame-level heads (``beat``/``one``/``last``), BCE —
       :class:`~musicality.trainers.beat_phase_module.BeatPhaseModule`
     - ``checkpoints_beat/``

``BeatModule``/``BeatPhaseModule`` are siblings, not a base/subclass pair —
see ``BeatModule``'s own docstring — so beat-only and beat-phase duplicate a
little training-loop boilerplate rather than share it through inheritance.
Both save an explicit ``task:`` field (``"beat_only"``/``"beat_phase"``,
declared in their own training config) into the checkpoint's
hyperparameters — this is what makes task auto-detection at inference time
possible; see below. The tempo trainer has no such field and isn't part of
that auto-detection system at all (see *Inference* below).

Full, line-by-line config reference: :doc:`configuration`.

Inference
---------

Beat-only and beat-phase checkpoints share one inference path, keyed off the
``task:`` field every such checkpoint carries:

- :func:`musicality.inference.load_module` loads a ``.ckpt`` and reads back
  its ``task`` via :func:`musicality.inference.detect_task`, returning the
  right module class already in eval mode — callers never need to know in
  advance whether a checkpoint is beat-only or beat-phase. A checkpoint
  saved before this field existed raises rather than guessing.
- :func:`musicality.inference.run_inference` runs the model on a waveform
  and decodes the resulting probability curve(s) into events via
  :mod:`musicality.postprocess` — :func:`~musicality.postprocess.readout_beat_only`
  for beat-only, :func:`~musicality.postprocess.readout` (adds bar-position
  labeling) for beat-phase.

**Tempo checkpoints have no equivalent tool today** —
:mod:`musicality.inference` only knows about ``beat_only``/``beat_phase``
(``TempoModule`` isn't part of its module-class registry), so there's
currently no auto-detecting load/run path for a tempo checkpoint.

Where this gets used:

- ``tools/eval_beat.py`` (:class:`musicality.evaluation.BeatEvaluator`) —
  full-track evaluation for either beat task, task auto-detected: beat
  F-measure always, plus "1"/"last" F-measure and phase-confusion for
  beat-phase.
- The annotator (``uv run python -m tools.annotator``) — on-demand assist
  inference in the GUI, via ``tools.annotator.inference.infer_beats``.
  Beat-phase only (it displays bar positions, which a beat-only checkpoint
  has no concept of).

Postprocessing tools
---------------------

Turning per-frame probabilities into discrete beat/bar-position events —
and tuning the thresholds that step depends on — is its own small pipeline,
documented in full at :doc:`postprocess`:

- :mod:`musicality.postprocess` is the algorithm itself: peak-picking, then
  periodicity-based gating, then (beat-phase only) bar-position labeling.
  Two labelers are available, selected by :func:`~musicality.postprocess.readout`'s
  ``decoder`` argument:

  - :func:`~musicality.postprocess.label_bar_position_global` (**the default**)
    scores the whole track at once, maximising the total log-likelihood of the
    soft ``one``/``last`` probabilities over every candidate bar phase. A
    finite ``switch_penalty`` allows a penalised mid-track resync; ``None``
    forbids one entirely, reducing the decode to an exact single-offset argmax.
    Its ``advance`` argument controls whether the bar count moves one position
    per detected beat (``"index"``) or by elapsed time via
    :func:`~musicality.postprocess.phase_advances` (``"time"``), the latter
    being robust to a missed or spurious detection shifting the grid.
  - :func:`~musicality.postprocess.label_bar_position` is the older
    count-forward labeler, resyncing on any above-threshold anchor vote. It
    discards evidence below ``anchor_threshold`` and cannot revise a bad
    resync, which measurably costs phase accuracy — see
    ``docs/beat_phase_improvement_review.md``.

- ``configs/eval_beat.yaml`` holds the *tuned* postprocessing defaults per
  task (``beat_threshold``/``min_distance_frames``/``gate_tolerance``, plus
  ``decoder``/``switch_penalty``/``group_size`` for beat-phase, and
  ``anchor_threshold`` for the greedy decoder) — selected automatically by
  ``load_module``'s detected task unless overridden.
- ``tools/sweep_beat_postprocess.py`` produces the beat-detection values: it
  runs the model once per track (cached via
  :meth:`~musicality.evaluation.BeatEvaluator.compute_track_probs`), then
  cheaply grid-searches against the cached probabilities — a beat-detection
  grid for both tasks (scored by mean beat F-measure) and, for beat-phase
  checkpoints, ``anchor_threshold`` swept jointly with it. Note the latter only
  affects the non-default greedy decoder; see :doc:`postprocess` for why
  ``anchor_threshold`` has a real interior optimum rather than "higher is
  always better."
- ``tools/diagnose_beat_phase.py`` is the beat-phase counterpart, and what
  produces the tuned ``decoder``/``switch_penalty``. Against one cached set of
  probabilities it scores every decoder variant side by side, reports a
  per-track phase-offset profile
  (:func:`~musicality.metrics.position_accuracy.position_accuracy`), and states
  whether the phase error is coming from the model or from the decoder.

API reference
-------------

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.inference
