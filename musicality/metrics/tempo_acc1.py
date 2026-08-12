"""MIREX Accuracy 1: octave-tolerant tempo accuracy."""

import torch


def tempo_acc1(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: float = 0.08,
    factors: tuple = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """MIREX Accuracy 1: fraction of predictions within ``tolerance`` of any
    octave-equivalent tempo.

    A prediction is correct if it is within ``tolerance × factor × target``
    for any factor in ``factors``. The default 8% tolerance matches the MIREX
    evaluation standard.

    :param pred: Predicted BPM values, shape ``(B,)``.
    :param target: Ground-truth BPM values, shape ``(B,)``.
    :param tolerance: Relative tolerance (default: 0.08 → ±8%).
    :param factors: Metrical multiples to consider (default: 0.5×, 1×, 2×).
    :returns: Fraction of correct predictions in ``[0, 1]``, shape ``()``.
    """

    correct = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)

    for f in factors:
        correct |= (pred - f * target).abs() < tolerance * f * target

    return correct.float().mean()
