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
           the imbalance it compensates for largely disappears. ``18`` is
           right for ``"mask"``; roughly ``3`` for ``"beat"``.
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
