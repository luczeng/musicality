Configuration
=============

Every run — training, evaluation, dataset download — is driven by a Hydra
config under ``configs/``. The files themselves are deliberately bare: a key,
a value, and a section heading. **This page is where each key is explained**,
including the ones whose values cannot be chosen independently of each other,
and the measurements behind the defaults. If a config file and this page
disagree about what a default *is*, the file wins; if they disagree about what
it *means*, this page wins.

Technically: ``configs/*.yaml`` are Hydra config groups composed at the
entry-point scripts in ``tools/``, with ``configs/model/*.yaml`` selected
through each config's ``defaults:`` list. Any key is overridable on the command
line with dotted paths (``uv run python tools/train_beat.py lr=3e-4
data.input=ballroom trainer.max_epochs=40``). ``configs/eval_beat.yaml`` is the
exception to "config file": it is also read at import time by
:mod:`musicality.evaluation` and by ``tools/annotator/inference.py``, so its
values are library defaults, not just CLI defaults.

Which config drives what
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - File
     - Entry point
     - Selects
   * - ``beat_train.yaml``
     - ``tools/train_beat.py``, ``tools/sweep_lr.py``
     - ``model/tcn_frames.yaml``
   * - ``beat_only_train.yaml``
     - ``tools/train_beat_only.py``
     - ``model/tcn_frames_beat.yaml``
   * - ``train.yaml``
     - ``tools/train_tempo.py``
     - ``model/tcn.yaml``
   * - ``eval_beat.yaml``
     - ``tools/eval_beat.py``, :mod:`musicality.evaluation`, the annotator
     - —
   * - ``download.yaml``
     - ``tools/download_dataset.py``
     - —

Beat-phase training (``configs/beat_train.yaml``)
-------------------------------------------------

Task and head
~~~~~~~~~~~~~

``task`` — ``beat_phase``
    Saved into the checkpoint's hyperparameters by
    :class:`~musicality.trainers.beat_phase_module.BeatPhaseModule` and read
    back by :func:`musicality.inference.detect_task`, which picks the module
    class and the postprocessing block at eval/inference time. A checkpoint
    trained before this field existed cannot be detected and must be retrained.

``phase_conditioning`` — ``beat``
    Which frames the phase terms are averaged over.

    ``mask``
        Every frame of a position-annotated track (the original behaviour).
    ``beat``
        Only frames at or near a beat, weighted by the beat target channel.

    ``beat`` matches how the heads are actually read at inference:
    :mod:`musicality.postprocess`'s bar-position stage samples the phase
    channels *only* at detected beat times. Under ``mask``, ~96% of the phase
    gradient goes into re-learning "is this a beat at all" — which the beat
    head already does at 0.92 F. See ``docs/beat_phase_improvement_review.md``
    step 2.

``target_layout`` — ``positions``
    How bar position is parameterised.

    ``one_last``
        Two independent sigmoids, for position 1 and for position
        ``group_size``. Positions in between get identical supervision
        (negative on both), so the model is never asked "is this a 1 or a 3?".
    ``positions``
        One logit per bar position with a softmax over them, so the positions
        compete and 1-vs-3 becomes a single decision. Widens the head to
        ``1 + group_size`` channels and switches the loss to
        :func:`~musicality.losses.beat_position_loss`.

    Measured on ballroom val (binary, viterbi=2), ``one_last`` → ``positions``:
    confusion 0.185 → 0.130, ``f_one`` 0.756 → 0.774, ``f_last`` 0.730 → 0.769.
    See ``plans/04_beat_phase_generalization_and_data_prep.md`` §2.2 and
    ``docs/beat_phase_improvement_review.md`` section 3.

``group_size`` — ``4``
    Beats per group that the dataset's position annotations count across: 4 for
    a bar-position (1–4) dataset such as ballroom, 8 for a phrase-position
    (1–8) dataset. It only changes what the "last" target/head means (position
    ``== group_size``); "one" is always position 1. Annotations counting a
    *longer* bar are folded onto ``1..group_size`` at load time
    (:func:`~musicality.loaders.beat_dataset.fold_positions`); meters that
    cannot fold evenly have their position supervision masked off instead.

``n_mels`` — ``128``, ``hop_length`` — ``512``, ``sigma_frames`` — ``1.5``
    Front-end and target shape. ``n_mels`` and ``hop_length`` are interpolated
    into the selected ``model/*.yaml`` via ``${n_mels}`` / ``${hop_length}``,
    which is why they live at the config root rather than inside a section.
    ``sigma_frames`` is the standard deviation of the Gaussian smearing applied
    to each beat/position target, in frames.

Loss
~~~~

``pos_weight`` — ``auto``
    Positive-class weight for the BCE heads: a scalar (shared across heads), a
    3-element list (per head), or ``auto``.

    ``auto`` derives the weight per sample from the beat target, as
    ``alpha * (1 - mean(beat_y)) / mean(beat_y)``. A fixed value is only
    correct at one tempo — the true negative:positive ratio is 2.6 on jtd
    (193 BPM), 4.5 on ballroom (125) and 11.3 at classical's 10th percentile
    (56 BPM), because the smeared target carries a constant ~3.75 frames of
    mass per beat while the beat period does not. Deriving it also keeps it
    correct under ``time_stretch``, which invalidates a hand-tuned value on
    every augmented clip.

    Measured negative:positive mass on ballroom (``binary_only``, 150 tracks,
    16 s clips), for a fixed value:

    .. list-table::
       :header-rows: 1
       :widths: 20 20 20 40

       * - head
         - ``mask``
         - ``beat``
         - note
       * - one
         - 20.2
         - 4.7
         -
       * - last
         - 20.0
         - 4.6
         -
       * - beat
         - 4.3
         - 4.3
         - unconditioned either way

    Note 4.7, not the naive 3.0 you get from "1 beat in 4 is a downbeat": the
    weight is the smeared beat channel and the target is the smeared "one"
    channel, so the positive mass is a product of two Gaussians (sum ~2.66 per
    downbeat) while the total weight is a single one (sum ~3.76 per beat) — the
    positive side loses relatively more mass than the weight does. Hand-set
    values follow the convention of shading ~10% under the measured ratio;
    exact inverse-frequency weighting on a soft target tends to overshoot into
    false positives.

    .. warning::

       ``pos_weight`` is coupled to **both** ``phase_conditioning`` and
       ``target_layout``; change them together or not at all.

       - ``target_layout: positions`` takes a scalar or ``auto``. A 3-element
         list is rejected at construction with an error naming both keys.
       - ``target_layout: one_last`` takes the 3-element list ``[5, 4, 4]``.
         ``auto`` is **not** supported here, and the failure is worse than an
         error: it dies inside torch as ``TypeError: new(): invalid data type
         'str'``, which names neither config key.

       So a ``one_last`` run is
       ``target_layout=one_last pos_weight=[5,4,4]``.

``pos_weight_alpha`` — ``1.11``
    Scale factor for ``pos_weight: auto``. ``1.11`` reproduces the fixed value
    of 5 at ballroom's median tempo, so switching to ``auto`` is a pure
    cross-tempo recalibration — neutral on the corpus every previous
    measurement was taken on. ``1.0`` is exact inverse-frequency weighting.

``position_norm`` — ``per_item``
    How the bar-position cross-entropy is averaged. ``target_layout:
    positions`` only.

    ``global``
        One weighted mean over the batch, i.e. a micro-average over *beats*. A
        clip's influence is then proportional to its beat count: measured on
        ``merge``, jtd takes 73.7% of the position gradient against 63.8% of
        the tracks, and ballroom 18.0% against 24.1%. The gradient-share ratio
        between two corpora is exactly their tempo ratio.
    ``per_item``
        Normalize each clip by its own beat weight first, so every annotated
        clip carries ``1/n_valid`` regardless of tempo. That is a macro-average
        over tracks, matching the per-genre metric the project is judged on
        (``tools/eval_beat.py --per-genre``).

    .. note::

       ``position_norm`` changes the *scale* of ``val/loss``, which
       ``ModelCheckpoint`` monitors and splices into checkpoint filenames. Runs
       from before and after the switch are not loss-comparable.

``balanced`` — ``true``
    Balances the logged ``acc_one``/``acc_last`` metrics (the average of the
    true-positive and true-negative rates) instead of a pooled mean, which
    would be dominated by the true-negative rate since phase frames are a small
    minority. Reaches the ``one_last`` layout only — under ``target_layout:
    positions`` nothing reads it. The beat head is scored by
    :func:`~musicality.metrics.frame_accuracy.peak_f_measure` (logged as
    ``f_beat``), which peak-picks before matching and so has no true-negative
    term to balance.

Optimisation
~~~~~~~~~~~~

``lr`` — ``5e-4``
    Adam learning rate. :meth:`BeatPhaseModule.configure_optimizers
    <musicality.trainers.beat_phase_module.BeatPhaseModule.configure_optimizers>`
    wraps it in ``ReduceLROnPlateau(patience=5, factor=0.5)`` monitoring
    ``val/loss``, stepped at ``trainer.check_val_every_n_epoch``.

``weight_decay`` — ``1e-4``
    L2 regularisation, passed to the same Adam.

``batch_size`` — ``16``
    Clips per batch, for both loaders.

``trainer.*``
    Passed straight to ``lightning.Trainer``: ``max_epochs``, ``accelerator``
    (``cpu | gpu | auto``), ``devices``, ``log_every_n_steps``,
    ``check_val_every_n_epoch``. ``trainer.save_top_k`` (default 3) is read by
    :func:`~musicality.trainers.common.build_checkpoint_callback` rather than
    by Lightning.

Data
~~~~

``data.input`` — ``merge``
    One field serving two purposes, told apart by whether it contains a ``/``:

    - A bare name (``ballroom``, ``merge``) is looked up under the canonical
      splits directory from :mod:`musicality.dataformats`, after
      :func:`~musicality.loaders.beat_dataset.beat_split_name` applies the
      naming convention ``beat_phase-<name>[-binary]``.
    - A path (``../musicality_db/splits/ballroom``, or anywhere else on disk)
      is used directly as the split folder, bypassing the splits directory and
      the naming convention entirely — so a split can be trained on without
      being registered under a canonical name first.

    Tracks whose annotation is flagged in the annotator (``warning`` or
    ``needs_review``) are dropped from both halves on read, so a split file's
    line count is not the dataset size — see
    :func:`musicality.splits.splitter.is_flagged`.

``binary_only`` — ``true``
    Drop tracks whose beats-per-bar isn't a multiple of 2 — e.g. ballroom's
    waltz and Viennese waltz, which are in triple meter (1, 2, 3) rather than
    binary meter.

    .. warning::

       Must match whatever the split was built with (``--binary-only`` on
       ``tools/create_splits.py``), since it changes dataset length. It is
       folded into the split name, so flipping it reads a *different* split
       file. Without it, ~58% of what is called "val" is training data.

``data.sample_rate`` — ``22050``, ``data.duration`` — ``16.0``
    Audio sample rate, and clip length in seconds. The clip length interacts
    with the model's receptive field: the TCN trunk reaches
    ``1 + (k-1)(2^n - 1)`` = 511 frames ≈ 11.87 s at the default depth, so
    there is little context beyond it in a 16 s clip.

``data.random_crop`` — ``true``
    Train only: draw a random offset window per track on each access. The
    validation dataset is always built with ``random_crop=False``, taking a
    fixed window from the track's middle so eval is reproducible — and,
    deliberately, so it avoids intros. That makes frame metrics measured on the
    *easiest* 16 seconds of each track, which is one of the reasons they sit
    above the event metrics; see ``docs/frame_vs_event_metrics.md``.

``data.num_workers`` — ``4``
    DataLoader workers. Above 0 also enables ``persistent_workers``.

``train_subsample`` — ``null``
    Fraction of the training split to use (e.g. ``0.2``), for quick smoke runs.

``augmentations.*``
    ``enabled`` gates the whole block. ``time_stretch`` (``min_rate`` /
    ``max_rate``) resamples the clip and rescales the annotation times; ``gain``
    (``min_db`` / ``max_db``) and ``noise`` (``std``) each have their own
    ``enabled`` flag. Time-stretch is safe on the target side and is what
    ``pos_weight: auto`` exists to stay calibrated against.

Logging and outputs
~~~~~~~~~~~~~~~~~~~

``checkpoint_dir`` — ``checkpoints_beat/``
    Parent directory. Each run writes into its own subdirectory below it, named
    after ``wandb.run_name`` or — since W&B only generates a name after the
    logger connects, which is after callbacks are built — a timestamp. Each
    subdirectory holds that run's ``trainer.save_top_k`` best checkpoints;
    without the per-run split, consecutive runs interleave and ``save_top_k``
    cannot prune another run's files.

``event_metrics.*``
    Event-level validation metrics, logged as ``val_event/*``: beats and bar
    positions decoded on **full tracks** and scored through the same path as
    ``tools/eval_beat.py``, so a number seen in W&B during training and the
    same number recomputed afterwards cannot disagree.

    ``enabled``
        Off by omission as well as by ``false``.
    ``n_tracks`` — ``50``
        Tracks scored per pass, drawn by a corpus-stratified fixed-seed sample;
        ``null`` scores the whole validation split. Cost is roughly one
        full-track model pass per track.
    ``every_n_epochs`` — ``5``
        Scored on epoch 0, then every N — plus the final epoch, always.

    .. note::

       Postprocessing is **not** configured here. It comes from
       ``configs/eval_beat.yaml``, resolved by the detected task, which is what
       keeps the training-time and eval-time decoders from drifting apart. See
       :mod:`musicality.callbacks.event_metrics` for why these numbers differ
       from the frame metrics beside them.

``training_report.enabled`` — ``true``
    Writes one ``training_report.json`` per run beside that run's checkpoints
    and uploads it to the W&B run's Files tab. Holds final/best metrics, the
    per-epoch history, the per-track and per-corpus event scores, the resolved
    config, and the run's identity — so a run can be handed over as a single
    attachment instead of a dashboard link. It decodes nothing; it reuses the
    event-metrics callback's last scoring pass.

``wandb.*``
    ``project``, ``run_name`` (``null`` lets W&B generate one), ``tags``.

.. literalinclude:: ../../configs/beat_train.yaml
   :language: yaml
   :caption: configs/beat_train.yaml

Beat-only training (``configs/beat_only_train.yaml``)
------------------------------------------------------

The same scaffolding with the phase heads removed: one output channel, one BCE
term. Keys behave as above except:

``task`` — ``beat_only``
    Selects :class:`~musicality.trainers.beat_module.BeatModule` at load time.

``pos_weight`` — ``6.0``
    A plain scalar here — there is only one head. ~6:1 negative:positive on
    ballroom, measured directly from ``BeatDataset``; see
    ``docs/beat_phase_pos_weight_notes.md``.

``balanced``
    Unused, for the reason given above. Kept because it is stored in every
    existing checkpoint's hyperparameters and passed back to the module on load.

``binary_only`` — ``false``
    Meter does not affect a beat-only detector, so the filter is a data-selection
    lever rather than a requirement. It still has to match the split.

.. literalinclude:: ../../configs/beat_only_train.yaml
   :language: yaml
   :caption: configs/beat_only_train.yaml

Tempo training (``configs/train.yaml``)
----------------------------------------

``loss`` — ``classification``
    ``absolute`` and ``relative`` regress BPM directly; ``classification``
    discretizes tempo into bins with a Gaussian target distribution and is the
    default. Only ``classification`` reads the ``classification:`` block
    (``bpm_min``, ``bpm_max``, ``n_bins``, ``sigma``).

Everything else (``lr``, ``batch_size``, ``trainer.*``, ``data.*``,
``augmentations.*``, ``wandb.*``) matches the beat configs.

.. literalinclude:: ../../configs/train.yaml
   :language: yaml
   :caption: configs/train.yaml

Model backbones (``configs/model/``)
-------------------------------------

Selected through each training config's ``defaults:`` list, or overridden on the
command line (``model=tcn``). All three are the same dilated TCN trunk
(:class:`~musicality.models.tcn.TCNTempoNet`) at different output shapes.

``tcn.yaml``
    Clip-level tempo regression. Used by ``train.yaml``.

``tcn_frames.yaml``
    Frame-level, 3 outputs. Used by ``beat_train.yaml``. ``frame_level`` and
    ``n_outputs`` are forced to ``True``/``3`` by ``BeatPhaseModule`` regardless
    of what is set here; they are listed for documentation only.

``tcn_frames_beat.yaml``
    Frame-level, 1 output. Used by ``beat_only_train.yaml``. ``frame_level`` and
    ``n_outputs`` are likewise forced, to ``True``/``1``, by ``BeatModule``.

``use_self_attention`` — ``false``
    Adds a self-attention head over the phase channels only; the beat channel
    always reads straight off the trunk. ``n_attn_layers`` and ``n_attn_heads``
    size it.

    .. warning::

       Currently **off on purpose**, not merely unexercised.
       ``plans/05_beat_phase_overfitting.md`` §2 rules it out: +0.79M parameters
       (+45%) for ≤0.023 macro confusion, itself confounded with the
       position-folding fix. ``docs/beat_phase_improvement_review.md`` §4 lists
       four unresolved defects in the block — absolute sinusoidal positional
       encoding is actively harmful under ``random_crop``, the softmax flattens
       at full-track length, post-LN sits on an unnormalized input, and there is
       no dropout inside the block. It also needs clips longer than the trunk's
       own receptive field (~11.9 s) to have any long-range context to draw on.

Evaluation (``configs/eval_beat.yaml``)
----------------------------------------

Defaults for ``tools/eval_beat.py``, every key overridable via the matching
``--flag`` (e.g. ``--beat-threshold 0.4``). Also loaded at import time as
:data:`musicality.evaluation.DEFAULTS` and by the annotator, so these are the
project-wide postprocessing defaults.

Top level
~~~~~~~~~

``dataset``, ``split`` (``train | val | all``), ``val_split``, ``sample_rate``,
``hop_length``, ``tolerance``, ``device``. ``sample_rate`` and ``hop_length``
must match the checkpoint's training config; ``val_split`` must match how the
split was created; ``tolerance`` is the F-measure matching window in seconds.

``beat_only:``
~~~~~~~~~~~~~~

Postprocessing for a beat-only checkpoint, selected automatically by
:func:`musicality.inference.detect_task`. Tuned via ``--sweep`` against a real
checkpoint, mean beat F-measure 0.735 → 0.896.

``beat_threshold``
    ``pick_peaks``: minimum probability to be considered a peak at all.
``min_distance_frames``
    ``pick_peaks``: minimum frame gap enforced between returned peaks.
``gate_tolerance``
    ``gate_periodicity``: relative slack around an integer multiple of the
    current beat period.

``beat_phase:``
~~~~~~~~~~~~~~~

The same three beat knobs, plus ``group_size`` and the bar-position stage.

.. warning::

   The four beat-detection values in this block are **unverified**. They were
   produced by the old ``tools/sweep_beat_postprocess.py``, which hardcoded the
   probability channels and never passed the decoder, switch penalty or
   position probabilities — so it was sweeping the *greedy* decoder against a
   two-sigmoid ``one_last`` head, neither of which ``beat_train.yaml`` trains
   any more. Re-sweep before trusting them::

       uv run python tools/eval_beat.py --checkpoint <ckpt> \
           --dataset merge --split val --binary-only --sweep

   Reported at the time: mean beat F-measure 0.916 on
   ``checkpoints_beat/loss=1.6565.ckpt``, ``binary_only=True``, ballroom val.

``decoder`` — ``global``
    Bar-position stage. ``global`` runs
    :func:`~musicality.postprocess.label_bar_position_global`, a whole-track
    maximum-likelihood decode over the soft position probabilities; ``greedy``
    runs the older count-forward
    :func:`~musicality.postprocess.label_bar_position`. Measured on the same
    checkpoint with no retraining (``tools/eval_beat.py --decoders``, switch
    penalty tuned on train and reported on val):

    .. list-table::
       :header-rows: 1
       :widths: 40 20 20 20

       * - decoder
         - ``f_one``
         - ``f_last``
         - confusion
       * - greedy (anchor=0.8)
         - 0.697
         - 0.692
         - 0.253
       * - global (switch=2.0)
         - 0.756
         - 0.730
         - 0.185

    See ``docs/beat_phase_improvement_review.md`` for why the greedy decoder
    loses.

``switch_penalty`` — ``2.0``
    Log-cost of a mid-track phase resync in the global decoder. ``null``
    forbids resyncs entirely (an exact single-offset decode); lower values
    resync more eagerly. The optimum is interior — on val, 0.25 scores 0.274
    confusion and 40.0 scores 0.219, both worse than 2.0's 0.185. See
    ``docs/switch_penalty_explained.md``.

``anchor_threshold`` — ``0.8``
    Minimum probability for a beat to be trusted as a confident "1"/"last"
    anchor. Read by the ``greedy`` decoder only.

``sweep:``
~~~~~~~~~~

Grid searched by ``tools/eval_beat.py --sweep``, overridable per run via
``--sweep-beat-thresholds`` / ``--sweep-min-distances`` /
``--sweep-gate-tolerances`` / ``--sweep-anchor-thresholds`` /
``--switch-penalties`` / ``--top``.

The sweep runs in two stages rather than over the full cartesian product: the
beat grid is scored first, and the winning beat knobs are then held fixed while
the bar-position knob is swept on top of them. Bar-position decoding consumes
whatever beats the peak-picker found, so a beat setting that loses on ``f_beat``
cannot win on ``position_acc`` — which makes the joint product (60 × 7 = 420
combinations) mostly wasted work for the same answer as 60 + 7.

``anchor_thresholds`` is swept only when the resolved decoder is ``greedy``, and
``switch_penalties`` only when it is ``global``; the latter always scores the
no-resync decode (``null``) alongside, and doubles as the variant list for
``--decoders``.

.. literalinclude:: ../../configs/eval_beat.yaml
   :language: yaml
   :caption: configs/eval_beat.yaml

Dataset download (``configs/download.yaml``)
---------------------------------------------

``data_home``
    Where ``tools/download_dataset.py`` writes. Defaults to the sibling
    ``../musicality_db`` git+dvc repo.
``datasets``
    Names passed to ``mirdata``.

``gtzan_genre`` is deliberately absent: mirdata's loader has a dead audio link
(``opihi.cs.uvic.ca``). Use ``data_home/dl_gtzan.py`` (the HuggingFace
``marsyas/gtzan`` mirror) to populate ``data_home/gtzan/``, drop beat
annotations under ``gtzan/annotations/beats/``, then run
``tools/migrate_gtzan.py``.

.. literalinclude:: ../../configs/download.yaml
   :language: yaml
   :caption: configs/download.yaml
