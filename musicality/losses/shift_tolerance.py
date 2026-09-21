"""Forgiving the beat head a few frames of timing error.

Beat annotations are not precise down to a frame — annotators disagree, players
are not synchronous, and human perception has limits. The evaluation already
accepts this: :func:`~musicality.metrics.f_measure.beat_f_measure` scores a
prediction as correct anywhere within ±70 ms. The training loss does not, so it
punishes predictions the metric would have accepted. [BT24]_ fixes that by
comparing the *max-pooled* prediction to the label instead of the prediction
itself.

The side effect is the point. Under a plain BCE the cheapest way to cover an
imprecise annotation is a wide, blurred peak; here only the largest prediction
in the window is read, so a single sharp peak is optimal — and sharp peaks are
what :func:`musicality.postprocess.pick_peaks` needs. Our alternative so far has
been to blur the *target* instead (``sigma_frames`` in
:class:`~musicality.loaders.beat_dataset.BeatDataset`), which [BT24]_ names and
rejects as mitigating slow convergence without helping with the blur.

The two are alternatives, not additions: ±3 frames of pooling on top of ±5
frames of smearing is ±186 ms of combined tolerance against a ±70 ms metric.
Pair a non-zero ``tolerance_frames`` with ``sigma_frames: 0``. See
``plans/09_lessons_from_literature.md`` §2.1.

.. [BT24] Foscarin, Schlüter and Widmer, "Beat this! Accurate beat tracking
   without DBN postprocessing", ISMIR 2024, §3.3.
"""

import torch
import torch.nn.functional as F

from musicality.losses.pos_weight import AUTO_POS_WEIGHT_ALPHA, beat_pos_weight

# ±3 frames is ±69.7 ms at our 43.07 fps (22050/512) — the mir_eval tolerance
# `musicality.metrics.f_measure.beat_f_measure` scores at, so the loss forgives
# exactly what the metric forgives. Not a default anywhere: the losses default
# to 0 (off) and a config carries the value.
TOLERANCE_FRAMES = 3


def sliding_windowed_max(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Each frame takes the largest value within ``radius`` frames of it.

    Stride 1, so the time axis is unchanged. The padding is ``-inf`` rather
    than a repeat or a wrap, which is what we want at the clip edges — a crop
    boundary is not a beat, and
    :func:`~musicality.trainers.beat_phase_module.align_time` has already
    trimmed the clip by the time a loss sees it.

    :param x: Tensor whose last axis is time, any leading shape — ``(B, T)``
        for a single channel, ``(B, C, T)`` for a block of them.
    :param radius: Half-width of the window in frames. ``0`` is the identity,
        which is how every caller's default reduces to its previous behaviour.
    :returns: Tensor of the same shape as ``x``.
    """

    if radius == 0:
        return x

    flat = x.reshape(-1, 1, x.shape[-1])
    pooled = F.max_pool1d(flat, kernel_size=2 * radius + 1, stride=1, padding=radius)

    return pooled.reshape(x.shape)


def shift_tolerant_bce(
    logits: torch.Tensor,
    beat_y: torch.Tensor,
    pos_weight: torch.Tensor | float | str = 6.0,
    pos_weight_alpha: float = AUTO_POS_WEIGHT_ALPHA,
    tolerance_frames: int = 0,
    ignore_frames: int | None = None,
) -> torch.Tensor:
    r"""Weighted BCE on the beat head, optionally forgiving small timing errors.

    With ``tolerance_frames=0`` this is exactly
    :func:`~torch.nn.functional.binary_cross_entropy_with_logits` under
    :func:`~musicality.losses.pos_weight.beat_pos_weight` — bit for bit, so it
    is a drop-in for the beat term of any loss here. Above zero it becomes
    [BT24]_'s shift-tolerant weighted BCE:

    .. math::

        \mathcal{L}_{st} = -\frac{1}{BT} \sum_{i,t}
            w_i \, b_{i,t} \log m_r(\hat{b})_{i,t}
            + \big(1 - m_{\rho}(b)_{i,t}\big) \log\big(1 - m_r(\hat{b})_{i,t}\big)

    with :math:`m_k` the max over a ``±k``-frame window, :math:`r` =
    ``tolerance_frames`` and :math:`\rho` = ``ignore_frames``.

    Two halves that would otherwise contradict each other. The first says "the
    highest prediction within ``±r`` of the annotation should be high", which
    accepts a peak ``r`` frames off the annotation. The second says "every
    other frame should be low" — including that same frame. So the negative
    term is switched off near each annotation: a peak at ``+r``, pooled over
    ``±r``, reaches ``+2r``, which is where [BT24]_'s default
    ``ignore_frames = 2 * tolerance_frames`` comes from.

    .. warning::
       That default is too wide for our fastest corpus. It ignores ``4r + 1``
       frames per beat, and at 43.07 fps jtd's 193 BPM leaves only 13.4 frames
       *between* beats — so ``r = 3`` retains 3.8% of frames as negatives, on
       63.8% of the merged split's tracks. The negative term all but vanishes,
       and a model that fires everywhere scores well. Hence ``ignore_frames``
       being its own knob rather than derived: ``r = 3`` (to keep ±70 ms) with
       ``ignore_frames = 4`` (to keep a third of jtd's frames) accepts a mild
       contradiction at the edge of the window, which biases peaks towards its
       centre — no bad thing. Frames surviving the band, at ``ρ = 6`` against
       ``ρ = 4``: jtd 3.8% against 33.4%, ballroom 37.7% against 56.9%,
       rwc_classical's 10th percentile 71.7% against 80.4%.

    :param logits: Raw per-frame model output, shape ``(B, T)`` — unactivated.
    :param beat_y: Beat target channel, shape ``(B, T)``, values in ``[0, 1]``.
        Shift tolerance assumes this is *sharp* (``sigma_frames: 0``); see the
        module docstring on why the two do not compose.
    :param pos_weight: Positive-class weight, or ``"auto"`` to derive one per
        sample — see :func:`~musicality.losses.pos_weight.beat_pos_weight`.
        Derived here rather than by the caller because ``"auto"`` has to see
        which negatives survive the ignore band: at ballroom's 125 BPM with
        sharp targets and ``ρ = 4`` the honest ratio is 13.2, against 22.1 from
        the raw target alone.
    :param pos_weight_alpha: Scale on the derived weight. Read only when
        ``pos_weight == "auto"``.
    :param tolerance_frames: Half-width ``r`` of the prediction window, in
        frames. ``0`` disables shift tolerance entirely.
    :param ignore_frames: Half-width ``ρ`` of the band around each annotation
        where the negative term is switched off. ``None`` uses [BT24]_'s
        ``2 * tolerance_frames``. Read only when ``tolerance_frames > 0``.
    :returns: Scalar mean loss, shape ``()``.
    :raises ValueError: If either radius is negative.
    """

    if tolerance_frames < 0 or (ignore_frames is not None and ignore_frames < 0):
        raise ValueError(
            f"tolerance_frames and ignore_frames are half-widths in frames and "
            f"cannot be negative, got {tolerance_frames} and {ignore_frames}"
        )

    if tolerance_frames == 0:
        return F.binary_cross_entropy_with_logits(
            logits,
            beat_y,
            pos_weight=beat_pos_weight(beat_y, pos_weight, pos_weight_alpha),
        )

    if ignore_frames is None:
        ignore_frames = 2 * tolerance_frames

    # Pool the logits rather than the probabilities: sigmoid is monotone
    # increasing, so max(σ(z)) == σ(max(z)) and we never leave logit space.
    pooled = sliding_windowed_max(logits, tolerance_frames)
    keep = 1.0 - sliding_windowed_max(beat_y, ignore_frames)

    weight = beat_pos_weight(beat_y, pos_weight, pos_weight_alpha, neg_weight=keep)

    # softplus(-z) is -log σ(z) and softplus(z) is -log(1 - σ(z)): the same
    # stable formulation binary_cross_entropy_with_logits uses, written out so
    # the negative term can carry `keep` in place of the (1 - y) it hardcodes.
    positive = weight * beat_y * F.softplus(-pooled)
    negative = keep * F.softplus(pooled)

    return (positive + negative).mean()
