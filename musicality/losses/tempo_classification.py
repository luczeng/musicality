"""Tempo-as-classification: a softmax over BPM bins with a Gaussian soft target.

Instead of regressing a number, the model scores a discrete grid of BPM bins.
The target is not one-hot but a Gaussian centred on the true tempo, which
gives the bin grid back its ordinal structure — a neighbouring bin is a near
miss, not an unrelated class. Selected by ``loss: classification`` in
``configs/train.yaml``, which also supplies the bin grid and ``sigma``.
"""

import torch
import torch.nn.functional as F


def gaussian_soft_target(
    tempo: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    r"""Soft target distribution over tempo bins.

    For each sample, places a Gaussian centred on the true tempo across the
    discrete bin grid, then normalises to a probability distribution. Bins
    near the true tempo receive non-zero target mass, which gives the model
    a smoother gradient than a one-hot target and bakes in the ordinal
    structure of the bin grid.

    .. math::

        p_{i,j} = \frac{\exp\left(-\frac{1}{2}\left(\frac{c_j - y_i}{\sigma}\right)^2\right)}
        {\sum_{k=1}^{n_{\text{bins}}} \exp\left(-\frac{1}{2}\left(\frac{c_k - y_i}{\sigma}\right)^2\right)}

    where :math:`c_j` is the centre of bin :math:`j` and :math:`y_i` is the
    true tempo of sample :math:`i`.

    :param tempo: True BPM values, shape ``(B,)``.
    :param bin_centers: BPM at the centre of each bin, shape ``(n_bins,)``.
    :param sigma: Gaussian standard deviation in BPM units.
    :returns: Soft target distribution, shape ``(B, n_bins)``.
    """

    diff = bin_centers.unsqueeze(0) - tempo.unsqueeze(1)  # (B, n_bins)

    return F.softmax(-((diff / sigma) ** 2) / 2, dim=-1)


def classification_tempo_loss(
    logits: torch.Tensor,
    tempo: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    r"""Cross-entropy between predicted softmax and Gaussian soft target.

    .. math::

        \mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{n_{\text{bins}}}
        p_{i,j} \log \hat{p}_{i,j}

    where :math:`p_{i,j}` is the Gaussian soft target from
    :func:`gaussian_soft_target` and :math:`\hat{p}_{i,j}` is the model's
    softmax probability for bin :math:`j`.

    :param logits: Model logits over BPM bins, shape ``(B, n_bins)``.
    :param tempo: True BPM values, shape ``(B,)``.
    :param bin_centers: BPM at the centre of each bin, shape ``(n_bins,)``.
    :param sigma: Gaussian standard deviation in BPM units.
    :returns: Scalar mean loss, shape ``()``.
    """

    target = gaussian_soft_target(tempo, bin_centers, sigma)
    log_probs = F.log_softmax(logits, dim=-1)

    return -(target * log_probs).sum(dim=-1).mean()
