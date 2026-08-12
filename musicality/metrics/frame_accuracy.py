"""Frame-level accuracy: a cheap per-epoch training signal on raw frame probabilities."""

import torch


def frame_accuracy(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    balanced: bool = False,
) -> torch.Tensor:
    """Coarse frame-level binary accuracy, for training-time monitoring only.

    Thresholds both the predicted probability and the (Gaussian-smeared) target
    at ``threshold`` and checks agreement. This is not the F-measure evaluation
    the project ultimately cares about (see :func:`musicality.metrics.f_measure.beat_f_measure`
    and :func:`musicality.metrics.f_measure.downbeat_f_measures`) — it's a cheap
    per-epoch signal.

    By default (``balanced=False``) all frames are pooled into one mean, which
    is dominated by the true-negative rate since beat/one/last frames are a
    small minority: a model that never fires still scores close to 1.0. Pass
    ``balanced=True`` to instead average the true-positive rate and true-negative
    rate, so a model that never fires floors at 0.5 rather than ~1.0.

    :param probs: Predicted probabilities (post-sigmoid), shape ``(B, T)``.
    :param target: Ground-truth (possibly smeared) target, shape ``(B, T)``.
    :param mask: Optional per-frame mask; masked-out frames are excluded from
        the average. Shape ``(B, T)``.
    :param threshold: Decision threshold applied to both ``probs`` and ``target``.
    :param balanced: If ``True``, return ``0.5 * (TPR + TNR)`` instead of the
        pooled mean.
    :returns: Scalar accuracy in ``[0, 1]``, shape ``()``.
    """

    pred_pos = probs > threshold
    targ_pos = target > threshold
    valid = torch.ones_like(probs, dtype=torch.bool) if mask is None else mask > 0

    if not balanced:
        agree = (pred_pos == targ_pos).float()
        denom = valid.float().sum().clamp(min=1.0)

        return (agree * valid).sum() / denom

    pos, neg = targ_pos & valid, ~targ_pos & valid

    tpr = (pred_pos & pos).float().sum() / pos.float().sum().clamp(min=1.0)
    tnr = (~pred_pos & neg).float().sum() / neg.float().sum().clamp(min=1.0)

    return 0.5 * (tpr + tnr)
