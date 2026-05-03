"""Waveform augmentations for tempo estimation training.

All augmentations operate on (1, T) float32 tensors.  The composed
:class:`TempoAugmenter` is the public entry point; use
:func:`build_augmenter` to construct one from a Hydra config section.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

class TimeStretch:
    """Randomly stretch or compress audio and scale the tempo label accordingly.

    Implemented via resampling: treating the waveform as if it were recorded
    at ``sr * rate`` Hz and resampling back to ``sr`` changes its duration by
    ``1 / rate``, which is equivalent to speeding up (rate > 1) or slowing
    down (rate < 1).  Pitch also shifts as a side-effect, which is acceptable
    for tempo estimation.

    :param min_rate: Minimum speed multiplier (< 1 = slower, lower tempo).
    :param max_rate: Maximum speed multiplier (> 1 = faster, higher tempo).
    """

    def __init__(self, min_rate: float = 0.85, max_rate: float = 1.15) -> None:
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(
        self, wav: torch.Tensor, tempo: float, sr: int
    ) -> tuple[torch.Tensor, float]:
        rate = random.uniform(self.min_rate, self.max_rate)
        new_len = int(wav.shape[-1] / rate)
        stretched = F.interpolate(
            wav.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)
        return stretched, tempo * rate


class RandomGain:
    """Scale amplitude by a random gain drawn uniformly from a dB range.

    :param min_db: Lower bound of the gain range in dB (negative = quieter).
    :param max_db: Upper bound of the gain range in dB (positive = louder).
    """

    def __init__(self, min_db: float = -6.0, max_db: float = 6.0) -> None:
        self.min_db = min_db
        self.max_db = max_db

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(self.min_db, self.max_db)
        return wav * (10 ** (gain_db / 20.0))


class AddNoise:
    """Add white Gaussian noise at a fixed standard deviation.

    :param std: Noise standard deviation (relative to full-scale ±1 audio).
    """

    def __init__(self, std: float = 0.005) -> None:
        self.std = std

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        return wav + torch.randn_like(wav) * self.std


# ---------------------------------------------------------------------------
# Composed augmenter
# ---------------------------------------------------------------------------

class TempoAugmenter:
    """Composes waveform augmentations and returns an updated (wav, tempo) pair.

    Applied in order:
    1. :class:`TimeStretch` — changes length; re-crops/pads back to ``n_samples``.
    2. :class:`RandomGain`
    3. :class:`AddNoise`
    """

    def __init__(
        self,
        time_stretch: TimeStretch | None = None,
        gain: RandomGain | None = None,
        noise: AddNoise | None = None,
    ) -> None:
        self.time_stretch = time_stretch
        self.gain = gain
        self.noise = noise

    def __call__(
        self, wav: torch.Tensor, tempo: float, sr: int, n_samples: int
    ) -> tuple[torch.Tensor, float]:
        if self.time_stretch is not None:
            wav, tempo = self.time_stretch(wav, tempo, sr)
            # Re-crop or zero-pad back to the expected fixed length
            if wav.shape[-1] >= n_samples:
                wav = wav[..., :n_samples]
            else:
                wav = F.pad(wav, (0, n_samples - wav.shape[-1]))

        if self.gain is not None:
            wav = self.gain(wav)

        if self.noise is not None:
            wav = self.noise(wav)

        return wav, tempo


# ---------------------------------------------------------------------------
# Dataset wrapper (train-only)
# ---------------------------------------------------------------------------

class AugmentedDataset(Dataset):
    """Wraps a Subset and applies :class:`TempoAugmenter` at item-access time.

    Use this to augment only the training split while leaving the validation
    split unchanged, since both splits share the same underlying dataset object.

    :param subset: A ``torch.utils.data.Subset`` of a :class:`TempoDataset`.
    :param augmenter: The augmenter to apply.
    :param sample_rate: Sample rate of the audio (passed to the augmenter).
    :param n_samples: Fixed clip length in samples (passed to the augmenter).
    """

    def __init__(
        self,
        subset,
        augmenter: TempoAugmenter,
        sample_rate: int,
        n_samples: int,
    ) -> None:
        self.subset = subset
        self.augmenter = augmenter
        self.sample_rate = sample_rate
        self.n_samples = n_samples

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        wav, label = self.subset[idx]
        wav, new_tempo = self.augmenter(wav, label.item(), self.sample_rate, self.n_samples)
        return wav, torch.tensor(new_tempo, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_augmenter(cfg: DictConfig) -> TempoAugmenter | None:
    """Build a :class:`TempoAugmenter` from a Hydra augmentations config section.

    Returns ``None`` if ``cfg.enabled`` is false or no individual augmentation
    is enabled, so callers can skip wrapping altogether.
    """
    if not cfg.get("enabled", True):
        return None

    time_stretch = None
    if cfg.time_stretch.get("enabled", False):
        time_stretch = TimeStretch(
            min_rate=cfg.time_stretch.min_rate,
            max_rate=cfg.time_stretch.max_rate,
        )

    gain = None
    if cfg.gain.get("enabled", False):
        gain = RandomGain(
            min_db=cfg.gain.min_db,
            max_db=cfg.gain.max_db,
        )

    noise = None
    if cfg.noise.get("enabled", False):
        noise = AddNoise(std=cfg.noise.std)

    if time_stretch is None and gain is None and noise is None:
        return None

    return TempoAugmenter(time_stretch=time_stretch, gain=gain, noise=noise)
