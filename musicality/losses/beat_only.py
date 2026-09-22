"""Beat detection alone, with no bar-position term.

The objective behind ``configs/beat_only_train.yaml`` and
:class:`~musicality.trainers.beat_module.BeatModule`: a single frame-wise
sigmoid over "is there a beat here?". It is the ``beat`` term of
:func:`~musicality.losses.beat_position.beat_position_loss` on its own, and
shares that loss's positive-class weighting and shift tolerance.
"""

import torch

from musicality.losses.pos_weight import AUTO_POS_WEIGHT_ALPHA
from musicality.losses.shift_tolerance import shift_tolerant_bce


def beat_only_loss(
    logits: torch.Tensor,
    beat_y: torch.Tensor,
    pos_weight: torch.Tensor | float | str = 6.0,
    pos_weight_alpha: float = AUTO_POS_WEIGHT_ALPHA,
    tolerance_frames: int = 0,
    ignore_frames: int | None = None,
) -> torch.Tensor:
    r"""Frame-wise weighted BCE against the beat target.

    .. math::

        \mathcal{L} = \frac{1}{BT} \sum_{i,t} \ell(\hat{b}_{i,t}, b_{i,t})

    where :math:`\ell` is binary cross-entropy weighted by ``pos_weight`` on
    the positive class.

    :param logits: Raw per-frame model output, shape ``(B, T)`` — unactivated
        (see :class:`musicality.models.tcn.TCNTempoNet` with
        ``frame_level=True`` and ``n_outputs=1``).
    :param beat_y: Beat target channel, shape ``(B, T)``, values in ``[0, 1]``.
    :param pos_weight: Positive-class weight, compensating for beat frames
        being a small fraction of all frames. A number, or ``"auto"`` to
        derive one per sample from the target — see
        :func:`~musicality.losses.pos_weight.beat_pos_weight`.
    :param pos_weight_alpha: Scale on the derived ``pos_weight``. Read only
        when ``pos_weight == "auto"``.
    :param tolerance_frames: Half-width, in frames, of the window the model's
        peak may sit anywhere in without penalty. ``0`` (the default) is a
        plain BCE, bit for bit. See :mod:`musicality.losses.shift_tolerance`,
        and pair a non-zero value with ``sigma_frames: 0``.
    :param ignore_frames: Half-width of the band around each beat where the
        negative term is switched off. ``None`` derives it as
        ``2 * tolerance_frames``. Read only when ``tolerance_frames > 0``.
    :returns: Scalar mean loss, shape ``()``.
    """

    return shift_tolerant_bce(
        logits,
        beat_y,
        pos_weight=pos_weight,
        pos_weight_alpha=pos_weight_alpha,
        tolerance_frames=tolerance_frames,
        ignore_frames=ignore_frames,
    )
