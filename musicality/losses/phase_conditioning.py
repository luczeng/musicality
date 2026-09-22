"""Which frames the bar-position terms are supervised on.

Both frame-wise position losses here — :func:`~musicality.losses.beat_phase.beat_phase_loss`
and :func:`~musicality.losses.beat_position.beat_position_loss` — average their
position term over a *weighted* subset of frames rather than all of them. This
module holds that weight, so the two losses share one definition and one
explanation of why the choice matters.
"""

import torch

from musicality.losses.shift_tolerance import sliding_windowed_max

PHASE_CONDITIONING_MODES = ("mask", "beat")


def phase_weight(
    beat_y: torch.Tensor,
    mask: torch.Tensor,
    phase_conditioning: str,
    tolerance_frames: int = 0,
) -> torch.Tensor:
    r"""Per-frame weight for a bar-position loss term.

    .. math::

        w_{i,t} = \begin{cases}
            m_{i,t} & \text{phase\_conditioning} = \text{"mask"} \\
            m_{i,t} \, \max_{|s| \le r} b_{i,t+s}
                & \text{phase\_conditioning} = \text{"beat"}
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
       and ``4`` respectively. ``tolerance_frames`` pushes back the other way:
       widening the window raises the positive mass again.

    :param beat_y: Beat target channel, shape ``(B, T)``. Under the default
        ``sigma_frames: 1.5`` it is Gaussian-smeared (peak 1.0 at a beat,
        exactly 0 more than 4 frames away — ``gaussian_smear`` truncates its
        kernel at ``round(3 * sigma)``), so it doubles as a soft "near a beat"
        weight with no threshold or window size to invent. Under sharp
        targets it is a single frame per beat, and ``tolerance_frames``
        supplies the window the smear used to.
    :param mask: Target's mask channel, shape ``(B, T)`` — 1 where the clip
        carries bar-position annotations, 0 where it does not.
    :param phase_conditioning: ``"mask"`` or ``"beat"``.
    :param tolerance_frames: Half-width, in frames, of the window the
        ``"beat"`` gate is widened over — see
        :func:`~musicality.losses.shift_tolerance.sliding_windowed_max`. ``0``
        (the default) leaves the gate exactly as the target came.

        This exists because the decoder does not read the position head at the
        annotated frame; it reads it at the frame the *beat head's* peak
        rounded to, which shift tolerance explicitly allows to sit up to ``r``
        frames away (:func:`musicality.postprocess.label_bar_position_global`).
        The head therefore has to be right across that whole window, so the
        term is supervised across it.

        .. warning::
           Whatever widens this must widen the position *target* too.
           :class:`~musicality.loaders.beat_dataset.BeatDataset` gives frames
           away from a beat a *uniform* row, meaning "no information here", so
           widening the gate alone would train the head towards maximum
           uncertainty at precisely the frames the decoder reads.
           :func:`~musicality.losses.beat_position.beat_position_loss` does
           both together; nothing else should call this with a non-zero radius.
    :returns: Weight tensor, shape ``(B, T)``.
    :raises ValueError: If ``phase_conditioning`` is not a known mode.
    """

    if phase_conditioning not in PHASE_CONDITIONING_MODES:
        raise ValueError(
            f"Unknown phase_conditioning {phase_conditioning!r} — "
            "expected 'mask' or 'beat'"
        )

    if phase_conditioning == "mask":
        return mask

    return mask * sliding_windowed_max(beat_y, tolerance_frames)
