"""Which frames the bar-position terms are supervised on.

Both frame-wise position losses here — :func:`~musicality.losses.beat_phase.beat_phase_loss`
and :func:`~musicality.losses.beat_position.beat_position_loss` — average their
position term over a *weighted* subset of frames rather than all of them. This
module holds that weight, so the two losses share one definition and one
explanation of why the choice matters.
"""

import torch

PHASE_CONDITIONING_MODES = ("mask", "beat")


def phase_weight(
    beat_y: torch.Tensor,
    mask: torch.Tensor,
    phase_conditioning: str,
) -> torch.Tensor:
    r"""Per-frame weight for a bar-position loss term.

    .. math::

        w_{i,t} = \begin{cases}
            m_{i,t} & \text{phase\_conditioning} = \text{"mask"} \\
            m_{i,t} \, b_{i,t} & \text{phase\_conditioning} = \text{"beat"}
        \end{cases}

    The ``mask`` factor is always present: not every dataset carries position
    annotations (see :class:`musicality.loaders.beat_dataset.BeatDataset`), so
    unannotated clips must contribute nothing.

    ``"mask"`` optimises the position heads on a distribution they are never
    read at — :func:`musicality.postprocess.label_bar_position` samples them
    *only* at detected beat times, so ~96% of the gradient goes into
    re-learning "is this a beat at all", which the beat head already does at
    0.92 F. ``"beat"`` restricts the term to the frames the decoder actually
    reads, turning 1-vs-3 from a rare-event detection problem (~1 positive
    frame in 23) into a balanced classification problem (~1 beat in 4). See
    ``docs/beat_phase_improvement_review.md`` step 2.

    .. note::
       A hand-set ``pos_weight`` on the position heads must be retuned
       alongside this — the imbalance it compensates for largely disappears.
       Measured neg:pos mass on ballroom is ~20:1 under ``"mask"`` but ~4.7:1
       under ``"beat"`` (not 3:1 — the positive mass is a product of the
       smeared beat weight and the smeared position target, so it decays
       faster than the weight alone). ``configs/beat_train.yaml`` uses ``18``
       and ``4`` respectively.

    :param beat_y: Beat target channel, shape ``(B, T)``. Already
        Gaussian-smeared (peak 1.0 at a beat, exactly 0 more than ~5 frames
        away), so it doubles as a soft "near a beat" weight with no threshold
        or window size to invent.
    :param mask: Target's mask channel, shape ``(B, T)`` — 1 where the clip
        carries bar-position annotations, 0 where it does not.
    :param phase_conditioning: ``"mask"`` or ``"beat"``.
    :returns: Weight tensor, shape ``(B, T)``.
    :raises ValueError: If ``phase_conditioning`` is not a known mode.
    """

    if phase_conditioning not in PHASE_CONDITIONING_MODES:
        raise ValueError(
            f"Unknown phase_conditioning {phase_conditioning!r} — "
            "expected 'mask' or 'beat'"
        )

    return mask * beat_y if phase_conditioning == "beat" else mask
