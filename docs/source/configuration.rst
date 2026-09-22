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
        :func:`~musicality.losses.beat_position.beat_position_loss`.

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

``tolerance_frames`` — ``0``
    Half-width, in frames, of the timing error the loss forgives. ``0`` is off,
    and the loss is then bit-identical to the one every existing checkpoint was
    trained with. ``3`` is ±69.7 ms at 43.07 fps — the same tolerance
    ``mir_eval`` scores at, so the loss forgives exactly what the metric
    forgives. From Beat This! (ISMIR 2024) §3.3; see
    :mod:`musicality.losses.shift_tolerance`.

    It means two different things to the two heads, because the decoder reads
    them two different ways. The ``beat`` head is *scanned* over time by
    :func:`~musicality.postprocess.pick_peaks`, so where its peak sits is the
    answer — tolerance forgives a peak a frame or two off, and in exchange the
    model is free to make that peak sharp instead of hedging with a wide bump.
    The ``position`` head is never scanned: the decoder rounds a beat time to a
    frame and reads that one column. Its answer is a label, not a time, so
    there is no peak to forgive; instead its supervision is *widened* across
    the window the lookup might land in — gate and target together.

    .. warning::

       Coupled to ``sigma_frames``. Smearing the target and forgiving the
       prediction are two answers to the same problem, and they stack rather
       than compose: ``sigma_frames: 1.5`` already smears over ±4 frames, so
       the pair forgives ±162 ms against a ±70 ms metric. Beat This! rejects
       smearing outright, on the grounds that it mitigates slow convergence
       without fixing the blurred peaks it causes. Set ``sigma_frames: 0``
       alongside a non-zero ``tolerance_frames``;
       :func:`~musicality.trainers.common.build_beat_dataloaders` warns if you
       do not. The two are left independent so the pair can be swept.

    Sharp targets also move ``pos_weight``. Dropping the Gaussian cuts the
    positive mass ~3.75x, so the derived ``auto`` ratio climbs — which is why
    ``AUTO_POS_WEIGHT_RANGE``'s ceiling is 60 rather than 20:

    .. list-table::
       :header-rows: 1
       :widths: 30 18 18 18

       * - corpus
         - smeared
         - sharp
         - sharp, ``ignore_frames: 4``
       * - rwc_classical (10th pct, 56 BPM)
         - 12.5
         - 49.9
         - 41.0
       * - ballroom (median, 125 BPM)
         - 5.1
         - 22.1
         - 13.2
       * - jtd (median, 193 BPM)
         - 2.9
         - 13.9
         - 5.0

``ignore_frames`` — ``null``
    Half-width of the band around each beat where the ``beat`` term's negative
    half is switched off. ``null`` derives it as ``2 * tolerance_frames``,
    which is Beat This!'s rule. Read only when ``tolerance_frames > 0``, and it
    does not touch the position term, which has no negative class.

    The band exists because the two halves of the loss otherwise contradict
    each other: the positive half accepts a peak ``r`` frames off the
    annotation while the negative half is simultaneously calling that same
    frame a mistake. A peak at ``+r``, pooled over ``±r``, reaches ``+2r``,
    hence the default.

    .. warning::

       That default is too wide at our frame rate. It ignores ``4r + 1`` frames
       per beat, and jtd's 193 BPM leaves only 13.4 frames *between* beats — so
       ``r = 3`` retains almost no negatives, on 63.8% of ``merge``'s tracks.
       The negative term all but vanishes and a model that fires everywhere
       scores well. Frames surviving the band:

       .. list-table::
          :header-rows: 1
          :widths: 40 30 30

          * - corpus
            - ``ignore_frames: 6``
            - ``ignore_frames: 4``
          * - rwc_classical (10th pct)
            - 71.7%
            - 80.4%
          * - ballroom (median)
            - 37.7%
            - 56.9%
          * - jtd (median)
            - 3.8%
            - 33.4%

       Beat This! does not hit this: 50 fps gives more frames per beat at the
       same tempo, and their corpora skew slower. The pairing to start from
       here is ``tolerance_frames: 3`` (to keep ±70 ms) with ``ignore_frames:
       4``, which accepts a mild contradiction at the edge of the window — and
       that biases peaks towards its centre, which is no bad thing.

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
      :func:`~musicality.splits.splitter.split_name` applies the naming
      convention ``<name>[-binary]``.
    - A path (``../musicality_db/splits/ballroom``, or anywhere else on disk)
      is used directly as the split folder, bypassing the splits directory and
      the naming convention entirely — so a split can be trained on without
      being registered under a canonical name first.

    Tracks whose annotation is flagged in the annotator (``warning`` or
    ``needs_review``) are dropped from both halves on read, so a split file's
    line count is not the dataset size — see
    :func:`musicality.splits.splitter.is_flagged`.

``data.exclude`` — ``[]``
    Corpora to subtract from whichever split ``data.input`` names, by the
    name a split line carries before its ``/`` — ``data.exclude=[jtd]`` on the
    command line, or ``exclude: [jtd, gtzan]`` in the file. Empty means train
    on the whole split.

    This is a *subtraction from* a split, not a new split: every remaining
    track keeps the train/val side it was already drawn into, so a run
    excluding a corpus stays comparable with a run over the whole split, and
    no second split file has to exist. Use it to ask what one corpus is
    contributing (``data.exclude=[jtd]`` against an unmodified baseline)
    rather than editing ``splits/``.

    Both halves are dropped, not just training: a corpus the model was never
    shown has no business moving ``val/loss`` or the event metrics. The
    exclusion is applied in
    :func:`~musicality.trainers.common.resolve_split_refs`, which the
    validation dataloader and
    :class:`~musicality.callbacks.event_metrics.EventMetricsLogger` both read
    their tracks through, so the two cannot disagree about what "val" means.

    A name the split doesn't hold raises, listing the corpora it does hold —
    a typo must not silently train on everything. So does an exclusion that
    would leave no training tracks.

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
    with the model's receptive field in *both* directions. The trunk reaches
    ``1 + (kernel_size - 1) * sum(dilations)`` frames — 1023 frames ≈ 23.8 s at
    the default ``n_layers: 9``, so the receptive field now exceeds the 16 s
    crop rather than falling short of it as it did at eight layers (511 frames
    ≈ 11.9 s). That is the intended state: edge frames seeing some padding is
    normal. The hard limit is per layer, not per stack — a layer whose *own*
    dilation exceeds the crop has both off-centre taps in padding at every
    frame and degenerates into a 1x1 convolution. At ``n_layers: 9`` the
    deepest dilation is 256 frames (5.9 s), comfortably inside a 16 s crop;
    ``n_layers: 11`` would put two layers past it and silently waste them. See
    ``plans/08_rethinking_the_approach.md`` §3.1.

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

``pos_weight_alpha`` — ``1.11``
    As above, and read only when ``pos_weight`` is ``auto``. Present because
    this task's ``pos_weight`` is a plain scalar today but ``auto`` works here
    too, and the beat-only head has the same tempo-dependent imbalance.

``tolerance_frames`` — ``0`` / ``ignore_frames`` — ``null``
    As above, minus the position half — there is no position head here, so
    ``tolerance_frames`` only ever forgives the beat peak. The coupling to
    ``sigma_frames`` and the ``ignore_frames`` warning apply unchanged.

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

``channels`` — ``32`` / ``n_layers`` — ``9``
    Trunk width and depth. Both frame-level backbones moved here from
    ``256``/``8`` on 2026-09-17; ``tcn.yaml`` (tempo) did **not** — see below.

    **Width.** ``channels`` is quadratic in parameters, and 256 was buying
    memorisation rather than accuracy. Three measurements agree:
    ``plans/07_beat_phase_v6_and_next_moves.md`` §1.2 shows the train/val gap on
    ``pos_acc`` widening 0.026 → 0.194 while the gap on ``f_beat`` barely moves
    (0.024 → 0.038); a training run reached 0.007 half-cycle confusion on train
    against 0.130 on val; and at 256 channels the model held 1.61 M parameters
    against 1.28 M labelled frames per epoch — more parameters than data points.
    At ``32`` the same backbone is 40,773 parameters, a 40x cut, still above the
    tens-of-thousands the reference convolutional beat trackers use.

    **Depth.** ``n_layers`` is linear in parameters and doubles the receptive
    field each time, so eight layers reached 11.9 s — less than the 16 s crop
    being trained on. Nine reaches 23.8 s and every layer stays fully live (see
    ``data.duration``). Nine is the ceiling for a 16 s crop, not a free
    parameter: ten is half-wasted and eleven is two dead layers. Going deeper
    needs a repeating dilation schedule rather than a longer crop —
    ``plans/08_rethinking_the_approach.md`` §3.1 specifies it; it is not
    implemented.

    .. note::

       Existing checkpoints are unaffected. ``musicality.inference.load_module``
       reads a checkpoint's own saved ``hyper_parameters`` and never consults
       these files, so ``checkpoint_v6.ckpt`` and friends still load and
       evaluate at 256x8. Only new training runs change shape.

    .. warning::

       A run at these defaults changes **three** things at once against v6 —
       width, depth, and the ``conv2d_stem`` below — so its result attributes to
       none of them individually. To separate them, override one at a time:
       ``model.channels=256 model.n_layers=8`` isolates the stem,
       ``model.conv2d_stem=false`` isolates the resize, and
       ``model.channels=64`` / ``model.channels=16`` walk the width ladder.

``tcn.yaml`` is deliberately left at ``256``/``8``
    The evidence above is entirely beat-phase: a frame-level position head, its
    own overfitting signature, and a receptive-field argument that assumes
    per-frame outputs. Tempo regression pools globally over time, so neither the
    depth argument nor the measured train/val gap transfers. Resizing it would
    be extrapolation from another task's data.

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
       own receptive field (~23.8 s at the default ``n_layers: 9``) to have any
       long-range context to draw on — which a 16 s crop no longer is.

``fixed_norm`` — ``true``
    Normalise the log-mel with frozen per-band statistics instead of statistics
    taken over the input tensor.

    Off, the model takes one mean and one std over *both* axes of whatever it is
    handed. Training passes a 16 s crop; ``run_inference`` passes a whole track.
    The same bars therefore arrive at a different scale depending on what
    surrounds them. Measured over 100 tracks, the shift is 0.04 sigma on gtzan
    and 0.33 (worst 1.61) on rwc_classical — the corpus where every tracker
    collapses. ``plans/08_rethinking_the_approach.md`` §2.1 item 4 has the full
    table.

    On, one mean and std per mel band are measured once at the start of training
    by :func:`~musicality.trainers.common.fit_input_stats` and stored as
    buffers, so they travel in the checkpoint and inference normalises exactly
    as training did. The window dependence is then gone by construction —
    verified on real audio, where a 16 s crop's network input is bit-identical
    to the same window read out of a 60 s pass.

    Two caveats. The model no longer adapts to a recording's overall level, so
    more rests on gain augmentation (±6 dB). And this fixes only the
    normalisation term: convolution padding is a separate train/inference
    difference, and at ``n_layers: 9`` the receptive field exceeds the crop, so
    a whole-track pass is still not equivalent to a crop end to end.

    Default-off in the code so checkpoints predating it reconstruct their own
    behaviour; on in both frame-level configs. ``tcn.yaml`` (tempo) keeps it
    off, on the same reasoning as the trunk size.

``conv2d_stem`` — ``true``
    Runs a :class:`~musicality.models.tcn.Conv2dStem` between the log-mel and
    the dilated trunk instead of projecting the raw bands straight through a
    1x1 convolution.

    **What it changes.** With the stem off, the model's first operation is
    ``Conv1d(n_mels, channels, kernel_size=1)`` — at every frame, a fixed linear
    mixture of all mel bands — after which the band axis is gone and every
    remaining layer convolves over time only. Nothing in the network ever sees a
    time-frequency neighbourhood. With it on, three ``Conv2d`` blocks with
    frequency-only max-pooling run first, and the surviving frequency bins are
    folded into channels for the trunk.

    **Why.** An onset is a local, *relative* event: energy rising, within a
    limited band, over ~20 ms. A 1x1 mixer is frequency-*absolute* — it learns
    one weight per band applied identically at every frame, so a kick drum and a
    walking bass note are the same event shape it has to detect twice, in
    separate output channels. Worse, mixing before differencing lets onsets
    cancel: with the bands summed first, one band rising while another falls
    produces a flat mixture, which is exactly a harmonic change with no
    percussive attack — the dominant downbeat cue wherever no drum marks the
    bar. ``plans/08_rethinking_the_approach.md`` §2.1 is the long version, and
    §1.3 there is the measurement that motivates it: Beat This! reaches 0.935
    ``f_beat`` / 0.818 ``position_acc`` on gtzan with *no decoder at all*, so its
    advantage lives in the front end and trunk.

    **Cost.** At the current ``channels: 32``, 32,805 parameters to 40,773 — the
    stem itself is 4.9 k and the rest is ``input_proj`` widening from 128 to 224
    inputs. It was measured at the old ``channels: 256`` as 1.613 M to 1.643 M;
    the absolute cost is the same, which is why it *composes* with the resize
    above rather than competing with it.

    **On by default since 2026-09-17**, for the frame-level backbones only.
    ``tcn.yaml`` (tempo) keeps it off, on the same reasoning as the trunk size.
    Turning it off is still a supported comparison
    (``model.conv2d_stem=false``), and checkpoints trained before the stem
    existed load into identical parameter shapes either way, since inference
    reads their saved hyperparameters rather than this file.

``stem_channels`` — ``16`` / ``stem_layers`` — ``3`` / ``stem_freq_pool`` — ``3``
    Read only when ``conv2d_stem`` is on. ``stem_layers`` ``Conv2d`` blocks, with
    frequency pooled by ``stem_freq_pool`` after every block **except the last**
    — so the default schedule takes 128 bands to 42 to 14, and 14 × 16 = 224
    channels reach the trunk.

    Two invariants worth not breaking. **Time is never pooled**: the frame rate
    is the output resolution, and ±1 frame of quantisation is already 23.2 ms
    against a 70 ms tolerance. And the frequency axis is *folded into channels*
    rather than pooled away, because which band a feature fired in is
    information the bar-position head has no other route to. A schedule that
    would leave fewer than one frequency bin raises at construction rather than
    failing on the first batch.

Evaluation (``configs/eval_beat.yaml``)
----------------------------------------

Defaults for ``tools/eval_beat.py``, every key overridable via the matching
``--flag`` (e.g. ``--beat-threshold 0.4``). Also loaded at import time as
:data:`musicality.evaluation.DEFAULTS` and by the annotator, so these are the
project-wide postprocessing defaults.

Top level
~~~~~~~~~

``dataset``, ``split`` (``train | val | all``), ``val_split``, ``binary_only``,
``sample_rate``, ``hop_length``, ``tolerance``, ``device``. ``sample_rate`` and
``hop_length`` must match the checkpoint's training config; ``val_split`` must
match how the split was created; ``tolerance`` is the F-measure matching window
in seconds.

``dataset`` and ``binary_only`` together name the split that gets evaluated —
:func:`musicality.splits.splitter.split_name` folds the second into the
directory name, so ``merge`` + ``binary_only: true`` reads ``merge-binary``. They default to what ``configs/beat_train.yaml``
trains on, so evaluating a checkpoint needs no flags to land on the split it was
held out against. ``--no-binary-only`` (or ``binary_only: false``) evaluates the
meter-mixed split instead, which only makes sense for a checkpoint trained on
it: the beat-phase ``one``/``last`` targets assume a binary meter, so a waltz
scored against them is being asked the wrong question.

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
