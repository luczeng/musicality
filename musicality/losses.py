"""Tempo loss functions."""

import torch
import torch.nn.functional as F


def relative_tempo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    factors: tuple = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """MAE loss invariant to metrical octave errors.

    For each sample, computes the absolute error between the prediction and
    each factor × target, then takes the minimum. Predicting double or half
    the annotated tempo incurs zero penalty — both are musically valid
    metrical interpretations of the same groove.

    :param pred: Predicted BPM values, shape ``(B,)``.
    :param target: Ground-truth BPM values, shape ``(B,)``.
    :param factors: Metrical multiples to consider (default: 0.5×, 1×, 2×).
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
    """Plain MAE between predicted and target BPM values.

    Unlike :func:`relative_tempo_loss`, this penalises octave errors in full,
    so the model is pushed to predict the exact annotated tempo.

    :param pred: Predicted BPM values, shape ``(B,)``.
    :param target: Ground-truth BPM values, shape ``(B,)``.
    """
    return (pred - target).abs().mean()


def gaussian_soft_target(
    tempo: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Soft target distribution over tempo bins.

    For each sample, places a Gaussian centred on the true tempo across the
    discrete bin grid, then normalises to a probability distribution. Bins
    near the true tempo receive non-zero target mass, which gives the model
    a smoother gradient than a one-hot target and bakes in the ordinal
    structure of the bin grid.

    :param tempo: True BPM values, shape ``(B,)``.
    :param bin_centers: BPM at the centre of each bin, shape ``(n_bins,)``.
    :param sigma: Gaussian standard deviation in BPM units.
    :returns: Soft target distribution, shape ``(B, n_bins)``.
    """
    diff = bin_centers.unsqueeze(0) - tempo.unsqueeze(1)  # (B, n_bins)
    return F.softmax(-(diff / sigma) ** 2 / 2, dim=-1)


def classification_tempo_loss(
    logits: torch.Tensor,
    tempo: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Cross-entropy between predicted softmax and Gaussian soft target.

    :param logits: Model logits over BPM bins, shape ``(B, n_bins)``.
    :param tempo: True BPM values, shape ``(B,)``.
    :param bin_centers: BPM at the centre of each bin, shape ``(n_bins,)``.
    :param sigma: Gaussian standard deviation in BPM units.
    """
    target = gaussian_soft_target(tempo, bin_centers, sigma)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()
