"""Regression losses on a single BPM value per clip.

Both are mean absolute error; they differ only in whether predicting a
metrical multiple of the annotated tempo is treated as an error. Selected by
``loss: absolute`` / ``loss: relative`` in ``configs/train.yaml`` and
dispatched by :class:`~musicality.trainers.tempo_module.TempoModule`.
"""

import torch


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
