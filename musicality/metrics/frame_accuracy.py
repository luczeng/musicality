"""Frame-level accuracy: a cheap per-epoch training signal on raw frame probabilities."""

import torch


def frame_accuracy(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Coarse frame-level binary accuracy, for training-time monitoring only.

    Thresholds both the predicted probability and the (Gaussian-smeared) target
    at ``threshold`` and checks agreement. This is not the F-measure evaluation
    the project ultimately cares about (see :func:`musicality.metrics.f_measure.beat_f_measure`
    and :func:`musicality.metrics.f_measure.downbeat_f_measures`) — it's a cheap
    per-epoch signal, dominated by the true-negative rate since beat/one/last
    frames are a small minority.

    :param probs: Predicted probabilities (post-sigmoid), shape ``(B, T)``.
    :param target: Ground-truth (possibly smeared) target, shape ``(B, T)``.
    :param mask: Optional per-frame mask; masked-out frames are excluded from
        the average. Shape ``(B, T)``.
    :param threshold: Decision threshold applied to both ``probs`` and ``target``.
    :returns: Scalar accuracy in ``[0, 1]``, shape ``()``.
    """

    agree = ((probs > threshold) == (target > threshold)).float()

    if mask is None:
        return agree.mean()

    denom = mask.sum().clamp(min=1.0)

    return (agree * mask).sum() / denom
