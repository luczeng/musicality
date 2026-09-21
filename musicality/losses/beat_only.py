"""Beat detection alone, with no bar-position term.

The objective behind ``configs/beat_only_train.yaml`` and
:class:`~musicality.trainers.beat_module.BeatModule`: a single frame-wise
sigmoid over "is there a beat here?". It is the ``beat`` term of
:func:`~musicality.losses.beat_position.beat_position_loss` on its own, and
shares that loss's positive-class weighting.
"""

import torch
import torch.nn.functional as F

from musicality.losses.pos_weight import AUTO_POS_WEIGHT_ALPHA, beat_pos_weight


def beat_only_loss(
    logits: torch.Tensor,
    beat_y: torch.Tensor,
    pos_weight: torch.Tensor | float | str = 6.0,
    pos_weight_alpha: float = AUTO_POS_WEIGHT_ALPHA,
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
    :returns: Scalar mean loss, shape ``()``.
    """

    return F.binary_cross_entropy_with_logits(
        logits,
        beat_y,
        pos_weight=beat_pos_weight(beat_y, pos_weight, pos_weight_alpha),
    )
