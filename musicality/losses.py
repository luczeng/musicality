"""Tempo loss functions."""

import torch
import torch.nn.functional as F


def relative_tempo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    factors: tuple = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    r"""MAE loss invariant to metrical octave errors.

    For each sample, computes the absolute error between the prediction and
    each factor × target, then takes the minimum. Predicting double or half
    the annotated tempo incurs zero penalty — both are musically valid
    metrical interpretations of the same groove.

    .. math::

        \mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \min_{f \in \text{factors}}
        \left| \hat{y}_i - f \cdot y_i \right|

    :param pred: Predicted BPM values, shape ``(B,)``.
    :param target: Ground-truth BPM values, shape ``(B,)``.
    :param factors: Metrical multiples to consider (default: 0.5×, 1×, 2×).
    :returns: Scalar mean loss, shape ``()``.
    """

    errors = torch.stack(
        [(pred - f * target).abs() for f in factors],
        dim=1,
    )  # (B, n_factors)
    return errors.min(dim=1).values.mean()


def absolute_tempo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    r"""Plain MAE between predicted and target BPM values.

    Unlike :func:`relative_tempo_loss`, this penalises octave errors in full,
    so the model is pushed to predict the exact annotated tempo.

    .. math::

        \mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \left| \hat{y}_i - y_i \right|

    :param pred: Predicted BPM values, shape ``(B,)``.
    :param target: Ground-truth BPM values, shape ``(B,)``.
    :returns: Scalar mean loss, shape ``()``.
    """

    return (pred - target).abs().mean()


def gaussian_soft_target(
    tempo: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    r"""Soft target distribution over tempo bins.

    For each sample, places a Gaussian centred on the true tempo across the
    discrete bin grid, then normalises to a probability distribution. Bins
    near the true tempo receive non-zero target mass, which gives the model
    a smoother gradient than a one-hot target and bakes in the ordinal
    structure of the bin grid.

    .. math::

        p_{i,j} = \frac{\exp\left(-\frac{1}{2}\left(\frac{c_j - y_i}{\sigma}\right)^2\right)}
        {\sum_{k=1}^{n_{\text{bins}}} \exp\left(-\frac{1}{2}\left(\frac{c_k - y_i}{\sigma}\right)^2\right)}

    where :math:`c_j` is the centre of bin :math:`j` and :math:`y_i` is the
    true tempo of sample :math:`i`.

    :param tempo: True BPM values, shape ``(B,)``.
    :param bin_centers: BPM at the centre of each bin, shape ``(n_bins,)``.
    :param sigma: Gaussian standard deviation in BPM units.
    :returns: Soft target distribution, shape ``(B, n_bins)``.
    """

    diff = bin_centers.unsqueeze(0) - tempo.unsqueeze(1)  # (B, n_bins)

    return F.softmax(-((diff / sigma) ** 2) / 2, dim=-1)


def beat_phase_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | float = 8.0,
    phase_conditioning: str = "mask",
) -> torch.Tensor:
    r"""Masked multi-head frame-wise BCE loss for beat-phase detection.

    Sums three per-frame binary-cross-entropy terms (beat, one, last). ``beat``
    is supervised on every frame. ``one``/``last`` are supervised on a weighted
    subset, ``w``, controlled by ``phase_conditioning``: always gated by the
    target's ``mask`` channel, since not every dataset carries position
    annotations (see :class:`musicality.loaders.beat_dataset.BeatDataset`), and
    under ``phase_conditioning="beat"`` additionally weighted by the ``beat``
    target so only frames at or near a beat count.

    Both phase terms are normalized by the *sum of those weights* rather than
    the total frame count. That keeps them on the same scale as the beat term
    regardless of how much weight there is, so neither a batch light on
    position-annotated tracks nor a slow track with few beats has its phase
    loss silently shrink toward zero.

    .. math::

        \mathcal{L} = \underbrace{\frac{1}{BT} \sum_{i,t} \ell(\hat{b}_{i,t}, b_{i,t})}_{\text{beat}}
        + \underbrace{\frac{\sum_{i,t} w_{i,t} \, \ell(\hat{o}_{i,t}, o_{i,t})}{\sum_{i,t} w_{i,t}}}_{\text{one}}
        + \underbrace{\frac{\sum_{i,t} w_{i,t} \, \ell(\hat{l}_{i,t}, l_{i,t})}{\sum_{i,t} w_{i,t}}}_{\text{last}}

    where :math:`\ell` is per-frame weighted binary cross-entropy (with
    ``pos_weight``) and the phase weight is

    .. math::

        w_{i,t} = \begin{cases}
            m_{i,t} & \text{phase\_conditioning} = \text{"mask"} \\
            m_{i,t} \, b_{i,t} & \text{phase\_conditioning} = \text{"beat"}
        \end{cases}

    with :math:`m` the target's ``mask`` channel and :math:`b` its ``beat``
    channel.

    :param logits: Raw per-frame model output, shape ``(B, 3, T)`` — beat/one/last,
        unactivated (see :class:`musicality.models.tcn.TCNTempoNet` with
        ``frame_level=True``).
    :param target: Ground-truth target, shape ``(B, 4, T)`` — beat/one/last/mask
        channels.
    :param pos_weight: Positive-class weight applied to every head's BCE term,
        compensating for beat/one/last frames being a small fraction of all
        frames. Scalar (shared across heads) or shape ``(3,)`` for a per-head
        weight. Default ``8.0`` is a rough starting point, not tuned per dataset.
    :param phase_conditioning: Which frames the ``one``/``last`` terms are
        averaged over.

        - ``"mask"`` (default): every frame of a position-annotated track.
        - ``"beat"``: only frames at or near a beat, weighted by the ``beat``
          target channel.

        ``"mask"`` optimises the phase heads on a distribution they are never
        read at — :func:`musicality.postprocess.label_bar_position` samples
        ``one``/``last`` *only* at detected beat times, so ~96% of the
        gradient goes into re-learning "is this a beat at all", which the beat
        head already does at 0.92 F. ``"beat"`` restricts the terms to the
        frames the decoder actually reads, turning 1-vs-3 from a rare-event
        detection problem (~1 positive frame in 23) into a balanced
        classification problem (~1 beat in 4). See
        docs/beat_phase_improvement_review.md step 2.

        .. note::
           ``pos_weight`` for the phase heads must be retuned alongside this —
           the imbalance it compensates for largely disappears. Measured
           neg:pos mass on ballroom is ~20:1 under ``"mask"`` but ~4.7:1 under
           ``"beat"`` (not 3:1 — the positive mass is a product of the smeared
           beat weight and the smeared one/last target, so it decays faster
           than the weight alone). ``configs/beat_train.yaml`` uses ``18`` and
           ``4`` respectively.
    :returns: Scalar mean loss, shape ``()``.
    """

    if phase_conditioning not in ("mask", "beat"):
        raise ValueError(
            f"Unknown phase_conditioning {phase_conditioning!r} — "
            "expected 'mask' or 'beat'"
        )

    beat_logits, one_logits, last_logits = logits[:, 0], logits[:, 1], logits[:, 2]
    beat_y, one_y, last_y, mask = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    pos_weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    pos_weight = pos_weight.expand(3) if pos_weight.ndim == 0 else pos_weight

    beat_loss = F.binary_cross_entropy_with_logits(
        beat_logits, beat_y, pos_weight=pos_weight[0], reduction="none"
    )
    one_loss = F.binary_cross_entropy_with_logits(
        one_logits, one_y, pos_weight=pos_weight[1], reduction="none"
    )
    last_loss = F.binary_cross_entropy_with_logits(
        last_logits, last_y, pos_weight=pos_weight[2], reduction="none"
    )

    # Per-frame weight for the one/last terms. `beat_y` is already
    # Gaussian-smeared (peak 1.0 at a beat, exactly 0 more than ~5 frames
    # away), so it doubles as a soft "near a beat" weight with no threshold
    # or window size to invent.
    phase_w = mask * beat_y if phase_conditioning == "beat" else mask

    # Normalising by the weight *sum* rather than the frame count makes each
    # term a weighted mean, so it stays on the same scale as `beat_term`
    # however many frames carry weight — otherwise a fast track, having more
    # beats, would contribute a proportionally larger phase loss.
    n_weighted = phase_w.sum().clamp(min=1.0)

    beat_term = beat_loss.mean()
    one_term = (one_loss * phase_w).sum() / n_weighted
    last_term = (last_loss * phase_w).sum() / n_weighted

    return beat_term + one_term + last_term


# Bounds on a *derived* pos_weight (see :func:`beat_pos_weight`). Neither end
# binds on real music: the slowest corpus in the collection sits at 12.5 and
# stays under 15 even after a 0.85 time-stretch. They exist for degenerate
# crops — a window holding a single beat derives ~200, and one holding none is
# bounded only by the epsilon in the denominator.
AUTO_POS_WEIGHT_RANGE = (1.0, 20.0)

# Reproduces the hand-tuned pos_weight of 5 at ballroom's median tempo, where
# the derived neg:pos ratio is 4.51. Anchoring there makes self-calibration a
# pure *cross-tempo* change — neutral on the corpus every previous measurement
# was taken on, differing only where the tempo differs.
AUTO_POS_WEIGHT_ALPHA = 1.11


def beat_pos_weight(
    beat_y: torch.Tensor,
    pos_weight: torch.Tensor | float | str,
    alpha: float = AUTO_POS_WEIGHT_ALPHA,
) -> torch.Tensor:
    r"""Positive-class weight for a beat BCE term — passed through, or derived
    per sample from the target when ``pos_weight`` is ``"auto"``.

    A fixed ``pos_weight`` is only correct at one tempo. The beat target is a
    Gaussian smeared to peak 1.0, so its mass per beat is a constant ~3.75
    frames regardless of tempo, while the beat *period* is not: 20.7 frames at
    ballroom's 125 BPM, 13.4 at jtd's 193, 46.1 at classical's 10th percentile
    of 56. The true neg:pos ratio therefore spans 2.6–11.3 across the corpora
    we train on, against a single configured 5.

    Since the ratio is just a function of the target already in hand, derive it
    rather than tune it:

    .. math::

        w_i = \alpha \, \frac{1 - \bar{b}_i}{\bar{b}_i},
        \qquad \bar{b}_i = \frac{1}{T} \sum_t b_{i,t}

    Deriving it per sample also means it tracks time-stretch augmentation,
    which silently invalidates a hand-tuned value on every augmented clip.

    :param beat_y: Beat target channel, shape ``(B, T)``, values in ``[0, 1]``.
    :param pos_weight: A number (or tensor) to pass through unchanged, or the
        string ``"auto"`` to derive one per sample.
    :param alpha: Scale on the derived ratio. Defaults to
        :data:`AUTO_POS_WEIGHT_ALPHA`; ``1.0`` is exact inverse-frequency
        weighting.
    :returns: Scalar tensor when passed through, shape ``(B, 1)`` when derived
        — which broadcasts against ``(B, T)`` inside
        :func:`~torch.nn.functional.binary_cross_entropy_with_logits`.
    """

    if not isinstance(pos_weight, str):
        return torch.as_tensor(pos_weight, device=beat_y.device, dtype=beat_y.dtype)

    if pos_weight != "auto":
        raise ValueError(
            f"Unknown pos_weight {pos_weight!r} — expected a number or 'auto'"
        )

    pos_frac = beat_y.mean(dim=-1, keepdim=True)  # (B, 1)
    weight = alpha * (1.0 - pos_frac) / pos_frac.clamp(min=1e-6)

    return weight.clamp(*AUTO_POS_WEIGHT_RANGE)


def beat_position_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | float | str = 5.0,
    phase_conditioning: str = "beat",
    pos_weight_alpha: float = AUTO_POS_WEIGHT_ALPHA,
    position_norm: str = "global",
) -> torch.Tensor:
    r"""Beat BCE plus a softmax cross-entropy over bar position.

    The successor to :func:`beat_phase_loss`. That loss models bar position as
    two *independent* binary detectors (``one`` and ``last``), which leaves the
    positions in between with identical supervision — negative on both heads —
    so the model is never asked the discriminative question the metric
    measures, "is this beat a 1 or a 3?". Here every position gets its own
    logit and they compete inside a single softmax, so raising the score for
    position 1 necessarily lowers position 3.

    ``beat`` stays an independent sigmoid: "is there a beat here?" is a
    genuine binary question over time, not a pick-one-of-``G``, and folding it
    into the softmax would couple a head that already works to one that
    doesn't.

    .. math::

        \mathcal{L} = \underbrace{\frac{1}{BT} \sum_{i,t} \ell(\hat{b}_{i,t}, b_{i,t})}_{\text{beat}}
        - \underbrace{\frac{\sum_{i,t} w_{i,t} \sum_{p} q_{i,t,p} \log \hat{q}_{i,t,p}}{\sum_{i,t} w_{i,t}}}_{\text{position}}

    where :math:`\ell` is weighted binary cross-entropy, :math:`q` is the
    target's normalized position block, :math:`\hat{q}` the softmax over the
    model's position logits, and :math:`w` the per-frame phase weight (see
    ``phase_conditioning``).

    No ``pos_weight`` is needed on the position term: bar positions occur
    equally often, so the softmax is already balanced. ``pos_weight`` here is
    a scalar for the ``beat`` head alone.

    :param logits: Raw per-frame model output, shape ``(B, 1 + G, T)`` — beat
        first, then one logit per bar position (see
        :class:`musicality.models.tcn.TCNTempoNet` with ``frame_level=True``
        and ``n_outputs=1 + G``).
    :param target: Ground-truth target, shape ``(B, 2 + G, T)`` — beat, the
        normalized position block, then mask. Built by
        :class:`~musicality.loaders.beat_dataset.BeatDataset` with
        ``target_layout="positions"``.
    :param pos_weight: Positive-class weight for the ``beat`` BCE term only.
        A number, or ``"auto"`` to derive one per sample from the target — see
        :func:`beat_pos_weight`.
    :param phase_conditioning: ``"beat"`` (default) weights the position term
        by ``mask * beat``, so it is optimized only where a beat actually is —
        which is where :mod:`musicality.postprocess` reads it. ``"mask"``
        weights by ``mask`` alone, supervising every frame. See
        :func:`beat_phase_loss` for why the former matters.
    :param pos_weight_alpha: Scale on the derived ``pos_weight``. Read only
        when ``pos_weight == "auto"``.
    :param position_norm: How the position term is averaged.

        - ``"global"`` (default): one weighted mean over the whole batch. That
          makes it a micro-average over *beats*, so a clip's influence is
          proportional to how many beats it happens to contain — and a 16 s
          crop holds 51 beats at 193 BPM but 15 at 56. Measured on the merged
          split, jtd takes 73.7% of the position gradient against 63.8% of the
          tracks, while ballroom gets 18.0% against 24.1%.
        - ``"per_item"``: normalize each clip by its own weight first, then
          average over clips. Every annotated clip carries exactly
          ``1/n_valid`` whatever its tempo, which is a macro-average over
          tracks — the same shape as the per-genre metric this is graded by.

        See plans/04_beat_phase_generalization_and_data_prep.md §2.6a.
    :returns: Scalar mean loss, shape ``()``.
    """

    if phase_conditioning not in ("mask", "beat"):
        raise ValueError(
            f"Unknown phase_conditioning {phase_conditioning!r} — "
            "expected 'mask' or 'beat'"
        )

    if position_norm not in ("global", "per_item"):
        raise ValueError(
            f"Unknown position_norm {position_norm!r} — expected 'global' or 'per_item'"
        )

    beat_logits, position_logits = logits[:, 0], logits[:, 1:]
    beat_y, position_y, mask = target[:, 0], target[:, 1:-1], target[:, -1]

    if position_logits.shape[1] != position_y.shape[1]:
        raise ValueError(
            f"logits carry {position_logits.shape[1]} position channels but the "
            f"target carries {position_y.shape[1]} — logits should be "
            "(B, 1 + G, T) against a (B, 2 + G, T) target"
        )

    beat_term = F.binary_cross_entropy_with_logits(
        beat_logits,
        beat_y,
        pos_weight=beat_pos_weight(beat_y, pos_weight, pos_weight_alpha),
    )

    # Soft-target cross-entropy over the position axis, per frame.
    log_q = F.log_softmax(position_logits, dim=1)
    position_ce = -(position_y * log_q).sum(dim=1)  # (B, T)

    phase_w = mask * beat_y if phase_conditioning == "beat" else mask

    if position_norm == "global":
        n_weighted = phase_w.sum().clamp(min=1.0)

        return beat_term + (position_ce * phase_w).sum() / n_weighted

    # Divide each clip by its own weight before averaging, so tempo stops
    # buying influence. Clips with no position annotation have `den == 0` and
    # therefore `num == 0` too — both are sums of `phase_w`-weighted terms —
    # so the clamp lets them contribute exactly zero with a well-defined
    # gradient. That keeps a fully unannotated batch finite (it reduces to the
    # beat term) without boolean indexing, which would make the shape depend
    # on the data.
    num = (position_ce * phase_w).sum(dim=-1)  # (B,)
    den = phase_w.sum(dim=-1)  # (B,)
    n_valid = (den > 1e-6).sum().clamp(min=1)

    return beat_term + (num / den.clamp(min=1e-6)).sum() / n_valid


def classification_tempo_loss(
    logits: torch.Tensor,
    tempo: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    r"""Cross-entropy between predicted softmax and Gaussian soft target.

    .. math::

        \mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{n_{\text{bins}}}
        p_{i,j} \log \hat{p}_{i,j}

    where :math:`p_{i,j}` is the Gaussian soft target from
    :func:`gaussian_soft_target` and :math:`\hat{p}_{i,j}` is the model's
    softmax probability for bin :math:`j`.

    :param logits: Model logits over BPM bins, shape ``(B, n_bins)``.
    :param tempo: True BPM values, shape ``(B,)``.
    :param bin_centers: BPM at the centre of each bin, shape ``(n_bins,)``.
    :param sigma: Gaussian standard deviation in BPM units.
    :returns: Scalar mean loss, shape ``()``.
    """

    target = gaussian_soft_target(tempo, bin_centers, sigma)
    log_probs = F.log_softmax(logits, dim=-1)

    return -(target * log_probs).sum(dim=-1).mean()
