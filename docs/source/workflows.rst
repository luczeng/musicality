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
- ``tools/eval_beat.py --sweep`` produces those values. It runs the model once
  per track (cached via
  :meth:`~musicality.evaluation.BeatEvaluator.compute_track_probs`), then
  grid-searches cheaply against the cached probabilities, in two stages: the
  beat-detection grid first, ranked by ``f_beat``, then the one bar-position
  knob the resolved decoder actually reads — ``switch_penalty`` under
  ``global``, ``anchor_threshold`` under ``greedy`` — held against the winning
  beat settings. See :doc:`postprocess` for why ``anchor_threshold`` has a real
  interior optimum rather than "higher is always better."

  The sweep decodes through
  :meth:`~musicality.evaluation.BeatEvaluator.score`, the same path evaluation
  takes, so it cannot tune a decoder the reported numbers do not use. Its
  predecessor could and did: it hardcoded the one/last channels and never
  passed ``decoder``, so it swept ``anchor_threshold`` for a greedy decode
  while the shipped config ran ``global``.

- ``tools/eval_beat.py --decoders`` produces the tuned
  ``decoder``/``switch_penalty``. Against one cached set of probabilities it
  scores every decoder variant side by side, and states whether the phase error
  is coming from the model or from the decoder — by how much of the baseline's
  remaining error a better decode of the *same* probabilities recovers. Add
  ``--profile`` for the per-track phase-offset profile
  (:func:`~musicality.metrics.position_accuracy.position_accuracy`), which says
  whether a wrong phase is a stable whole-track offset (the model cannot hear
  downbeats) or a mid-track flip (the decoder is losing information).

Comparing runs
--------------

``tools/eval_beat.py`` answers "how good is this checkpoint". Ranking several
against each other is a different question, and the numbers already lying
around cannot answer it: each run's ``training_report.json`` scores that run,
on whatever split and postprocessing it was configured with, at whatever epoch
it stopped. Comparing them compares those settings as much as the models.

``tools/leaderboard.py`` re-scores instead. Point it at checkpoint directories
and every run it finds is evaluated on one common split, through
:class:`musicality.evaluation.BeatEvaluator` — the same path
``tools/eval_beat.py`` takes, so a leaderboard row and a later single-checkpoint
report are the same number.

.. code-block:: bash

    uv run python tools/leaderboard.py checkpoints_deeper checkpoints_norm

The split it lands on is ``configs/eval_beat.yaml``'s ``dataset`` +
``binary_only`` + ``split`` — defaulting to what ``beat_train.yaml`` trains on,
so no flags are needed to evaluate against the split a checkpoint was held out
against.

Three things it does that a loop over ``eval_beat.py`` would not:

- **It sweeps each checkpoint's own postprocessing** before scoring it (stage 1
  beat detection, stage 2 the resolved decoder's bar-position knob — the two
  stages ``--sweep`` runs, against the same cached probabilities). The shipped
  ``beat_phase`` knobs are marked UNVERIFIED in ``configs/eval_beat.yaml``, and
  re-sweeping them has been worth more than a retrain, so scoring every
  checkpoint at one stale threshold would rank the models by how well that
  threshold happens to suit them. ``--no-sweep`` scores at the config's values
  instead.

- **It sweeps on a different split from the one it reports.** The knobs are
  tuned on ``sweep.split`` (``train``), against a corpus-stratified subsample of
  ``sweep.tracks`` (50) of it, so the val numbers on the board stay held out.
  ``tools/eval_beat.py --sweep`` does not do this — it tunes and reports on the
  same tracks, which makes its numbers optimistic by an amount that is not
  equal across checkpoints: 60-odd grid points against ~50 tracks give more
  room to whichever model's probability curves happen to suit some threshold.

  The cost is one extra model pass per checkpoint. The stratification matters
  because a split file is written corpus by corpus: the first N tracks of a
  merged train split are N tracks of whichever corpus was written first, so the
  knobs would be tuned for one genre.

- **It keeps one running board, on W&B rather than in a checkout.** The board
  is a single ``leaderboard.json`` holding the whole comparison — per-run
  metrics, the knobs each row was scored at, the per-corpus breakdown, a
  rendered table in ``readable`` — written the way ``training_report.json`` is,
  ``NaN`` as ``null`` so a strict parser accepts it.

W&B is what makes it a *running* board rather than a per-machine one. Training
happens on rented instances that are torn down, so every invocation fetches the
published board before reading and publishes the merged one after writing
(``--no-fetch`` / ``--no-publish`` to skip). The first invocation finds nothing
published, says so, and starts the board.

Every invocation extends that board, so the second and every later command names
only what is new:

.. code-block:: bash

    uv run python tools/leaderboard.py checkpoints_new

Every row for a run not named on the command line is carried over, and the
merged board is written back — so adding one experiment costs one experiment's
evaluation, not the whole board's. A run that *is* named is re-measured and
replaces its old row; identity is the run label rather than the checkpoint
filename, because more training on the same folder produces a different epoch's
file for the same experiment. ``--board`` keeps the board somewhere else, which is how a throwaway
comparison is made without disturbing the running one: any path but the default
is local by definition, neither fetched from W&B nor published to it.

Rows that were never measured the same way are refused rather than merged: if
anything in ``configs/eval_beat.yaml``'s run block, the ``--limit``, or whether
the knobs were swept differs from what the existing board recorded, the run
stops before any model pass and names the difference. Re-measuring every run in the old board lifts the refusal, since
nothing then survives to be incomparable — which is how those settings get
changed without a flag to override the check. Each row also carries its own
``measured_utc``, ``git_commit`` and the knobs it was scored at, so a board says
where each of its numbers came from.

It carries where the *model* came from too. A checkpoint path names a folder on
an instance that has since been torn down, which is no way back to an
experiment, so each row also takes ``wandb_name``, ``wandb_url`` and
``train_commit`` from the ``training_report.json`` that
:class:`~musicality.callbacks.training_report.TrainingReportLogger` wrote beside
those checkpoints — the leading row on the board links to the training run that
produced it. That report is written at ``on_fit_end``, so a run that was
interrupted has none and its row goes up without a link rather than without a
place.

Three separate rankings are involved, and conflating any two of them is a bug:
``--rank-metric`` (default ``f_beat``) orders the **board**; ``f_beat`` always
picks the winner of the sweep's **beat-detection** stage, the only thing those
knobs can move; and ``position_acc`` always picks the winner of its
**bar-position** stage. That last one cannot be ``f_beat``: a decoder relabels
beats without moving them, so every candidate scores an identical ``f_beat`` and
ranking that stage by it is a tie the sort breaks by candidate order — pinning
``switch_penalty`` to the first value in the list. All three rank on the macro
mean, which weights each corpus equally instead of letting the largest one
decide for all of them.

Which checkpoint stands for a run: a directory whose checkpoints carry a
``valloss`` in their filename is one run's ``save_top_k`` group, represented by
its best-scoring file; a directory of hand-named checkpoints is one entry per
file. A checkpoint that fails to load is reported under ``failed`` and the rest
of the board still runs — the model pass is the expensive part here.

Picking by ``val/loss`` is weaker than it looks for beat-phase: it tracks the
position head's confidence rather than decoded accuracy, so the lowest-loss
epoch is not reliably the best-decoding one. Scoring all three of a
``save_top_k`` group would cost three model passes per run.

A published board is two objects, from the same payload. The **table** is the
board to look at: every column of every row, sortable and filterable in the
browser, which is the only place a board this wide reads well. The **artifact**
is the board to keep — the ``leaderboard.json`` that was just written, one
version per publish. ``:latest`` is what the next invocation fetches, so a board
travels between machines without either of them holding it, and the same file
travels to a person as one download.

They go to a W&B project of their own (``--project``, default
``musicality-leaderboard``). A leaderboard is a different kind of object from a
training run, and a board buried among a few hundred training runs is a board
nobody finds. Each publish also writes the leader and the run count into the
run summary, so the project's own run list is already a history of what led
when.

The copy in this checkout (``leaderboard/leaderboard.json``, gitignored) is a
working file: what the merge reads before scoring and writes after, kept out of
git because publishing is what shares a board here.

Publishing reads that file and nothing else, so it costs no model pass and can
happen on its own:

.. code-block:: bash

    uv run python tools/leaderboard.py --publish-only

That is how a board that already exists gets onto W&B — a board from before
this was published anywhere, or one whose upload failed — and it is deliberately
not preceded by a fetch, which would replace the board being published with the
one already up there.

API reference
-------------

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.inference
