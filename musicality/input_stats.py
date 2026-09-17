"""Per-band log-mel statistics, measured once over the training data and frozen
into the model.

``TCNTempoNet``'s original normalisation reduced over *both* axes of whatever
tensor it was handed::

    mean = x.mean(dim=(1, 2), keepdim=True)   # (B, n_mels, T) -> one scalar
    std = x.std(dim=(1, 2), keepdim=True)
    x = (x - mean) / (std + 1e-6)

Training passes a 16 s crop; :func:`musicality.inference.run_inference` passes
an entire track in one forward call. The same eight bars therefore reach the
network at a different scale depending on what surrounds them, and one quiet
intro re-scales every frame of the track.

Measured over 100 tracks at three crop positions each (mean absolute difference
between a 16 s window normalised as a clip and the same window normalised inside
its parent track, in units of the clip's own sigma):

==============  =============  ==========  ========================
corpus          median shift   worst       clip/track sigma ratio
==============  =============  ==========  ========================
gtzan           0.040          0.137       0.96-1.03
jtd             0.143          0.621       0.93-1.12
ballroom        0.214          0.396       0.52-1.29
rwc_classical   0.331          1.613       0.77-2.51
==============  =============  ==========  ========================

The ordering is the inverse of our accuracy ranking, and rwc_classical — where
the mismatch is worst, unsurprisingly given its dynamic range — is the corpus
where every tracker in ``plans/08_rethinking_the_approach.md`` §1.3 collapses.
gtzan's flat number is partly an artefact of its being uniform 30 s excerpts, so
read the table as a motivating correlation, not a proof.

Freezing one mean and one standard deviation *per mel band* removes the window
dependence by construction rather than by approximation: the same audio produces
the same normalised input whatever it is batched with. The statistics live in
the model as buffers, so they travel inside the checkpoint and inference
reconstructs them exactly — see ``plans/08_rethinking_the_approach.md`` §2.1
item 4 and ``docs/source/configuration.rst``.
"""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn as nn


#: Batches drawn from the training loader to estimate the statistics. At the
#: shipped ``batch_size: 16`` and 16 s crops this is ~700 k frames per band,
#: far past what a mean and a variance need, and it keeps the startup pass to
#: about a minute rather than a full epoch.
DEFAULT_STATS_BATCHES = 64

#: Floor on the per-band standard deviation, guarding a band that is silent
#: throughout the sample (division by ~0 would turn its noise floor into signal).
MIN_STD = 1e-3


@torch.no_grad()
def compute_band_stats(
    mel: nn.Module,
    batches: Iterable[torch.Tensor],
    max_batches: int = DEFAULT_STATS_BATCHES,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of each mel band over a sample of the data.

    Accumulates in float64 sums rather than averaging per-batch means, so the
    result does not depend on how the frames were divided into batches (the last
    batch of an epoch is usually short).

    :param mel: The model's own mel transform, so the statistics describe
        exactly the tensor ``forward`` will normalise — same ``n_fft``, same
        hop, same dB conversion.
    :param batches: Iterable of waveform batches, shape ``(B, 1, N)`` or
        ``(B, N)``. A training dataloader yields ``(wav, target)`` pairs; pass
        :func:`waveform_batches` to unwrap them.
    :param max_batches: Stop after this many batches.
    :param device: Device to run the mel transform on.
    :returns: ``(mean, std)``, both shape ``(n_mels,)``, on the CPU.
    :raises ValueError: If *batches* yields nothing.
    """

    mel = mel.to(device)

    total = torch.zeros((), dtype=torch.float64)
    sum_x = sum_x2 = None

    # islice rather than a break: a plain loop pulls one batch past the limit
    # before stopping, and with a real dataloader that is a batch of audio
    # decoded and augmented for nothing.
    for wav in islice(batches, max_batches):
        x = mel(wav.to(device)).double()  # (B, [1,] n_mels, T)

        # MelSpectrogram keeps whatever channel axis the waveform had, so a
        # (B, 1, N) input comes back as (B, 1, n_mels, T) and a (B, N) input as
        # (B, n_mels, T). Reduce everything that is not the band axis.
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # (B', n_mels, T)

        batch_sum = x.sum(dim=(0, 2)).cpu()
        batch_sum2 = (x * x).sum(dim=(0, 2)).cpu()

        if sum_x is None:
            sum_x, sum_x2 = batch_sum, batch_sum2
        else:
            sum_x += batch_sum
            sum_x2 += batch_sum2

        total += x.shape[0] * x.shape[2]

    if sum_x is None:
        raise ValueError("no batches to compute input statistics from")

    mean = sum_x / total
    var = (sum_x2 / total) - mean * mean

    # Numerical safety: the two-pass identity can go slightly negative when the
    # variance is tiny next to the mean, which is exactly the silent-band case.
    std = var.clamp_min(0.0).sqrt().clamp_min(MIN_STD)

    return mean.float(), std.float()


def waveform_batches(loader: Iterable) -> Iterable[torch.Tensor]:
    """Yield just the waveform from a loader that yields ``(wav, target)``.

    :param loader: Any iterable of batches; a bare tensor is passed through, so
        this is safe to wrap around either shape.
    """

    for batch in loader:
        yield batch[0] if isinstance(batch, (tuple, list)) else batch


def fit_input_stats(
    model: nn.Module,
    loader: Iterable,
    max_batches: int = DEFAULT_STATS_BATCHES,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Measure and install per-band statistics on *model*, if it wants them.

    A no-op for a model whose ``input_norm`` is not ``"fixed"``, and for one
    whose statistics are already fitted — so resuming a run, or loading a
    checkpoint and continuing, does not re-measure or overwrite them.

    Note that the statistics are measured through the *training* loader, with
    augmentation live. That is deliberate: they should describe the distribution
    the model is actually trained on, and gain augmentation is a constant offset
    in dB. It does make them depend on the seed, which
    :func:`lightning.seed_everything` pins.

    :param model: A :class:`~musicality.models.tcn.TCNTempoNet`, or anything
        exposing ``input_norm``, ``input_stats_fitted`` and ``set_input_stats``.
    :param loader: Training dataloader.
    :param max_batches: Batches to measure over.
    :param device: Device to run the mel transform on.
    :returns: The ``(mean, std)`` that were installed, or ``None`` if nothing
        was done.
    """

    if getattr(model, "input_norm", "global") != "fixed":
        return None

    if model.input_stats_fitted:
        return None

    mean, std = compute_band_stats(
        model.mel, waveform_batches(loader), max_batches=max_batches, device=device
    )
    model.set_input_stats(mean, std)

    return mean, std
