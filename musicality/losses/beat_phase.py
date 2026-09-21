"""Bar position as two independent binary detectors (``one`` and ``last``).

The original beat-phase objective, superseded by
:func:`~musicality.losses.beat_position.beat_position_loss`. Kept because
checkpoints trained under it are still evaluated and compared; new runs should
use the position-softmax loss instead.
"""

import torch
import torch.nn.functional as F

from musicality.losses.phase_conditioning import phase_weight


def beat_phase_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | float = 8.0,
    phase_conditioning: str = "mask",
) -> torch.Tensor:
    r"""Masked multi-head frame-wise BCE loss for beat-phase detection.

    Sums three per-frame binary-cross-entropy terms (beat, one, last). ``beat``
    is supervised on every frame. ``one``/``last`` are supervised on a weighted
    subset, ``w``, controlled by ``phase_conditioning`` — see
    :func:`~musicality.losses.phase_conditioning.phase_weight`.

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
    ``pos_weight``) and :math:`w` is the phase weight.

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
        averaged over — ``"mask"`` (default) or ``"beat"``. See
        :func:`~musicality.losses.phase_conditioning.phase_weight`, including
        why ``pos_weight`` must be retuned alongside it.
    :returns: Scalar mean loss, shape ``()``.
    """

    beat_logits, one_logits, last_logits = logits[:, 0], logits[:, 1], logits[:, 2]
    beat_y, one_y, last_y, mask = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    phase_w = phase_weight(beat_y, mask, phase_conditioning)

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

    # Normalising by the weight *sum* rather than the frame count makes each
    # term a weighted mean, so it stays on the same scale as `beat_term`
    # however many frames carry weight — otherwise a fast track, having more
    # beats, would contribute a proportionally larger phase loss.
    n_weighted = phase_w.sum().clamp(min=1.0)

    beat_term = beat_loss.mean()
    one_term = (one_loss * phase_w).sum() / n_weighted
    last_term = (last_loss * phase_w).sum() / n_weighted

    return beat_term + one_term + last_term
