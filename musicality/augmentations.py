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

from musicality.loaders.beat_dataset import PAD_TIME


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
    """Wraps a dataset and applies :class:`TempoAugmenter` at item-access time.

    Use this to augment only the training split while leaving the validation
    split unchanged — the two are independent dataset instances (see
    :func:`~musicality.trainers.train.build_dataloaders`), so wrapping one
    has no effect on the other.

    :param subset: A :class:`TempoDataset` (or ``torch.utils.data.Subset`` of one).
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
        wav, new_tempo = self.augmenter(
            wav, label.item(), self.sample_rate, self.n_samples
        )
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


# ---------------------------------------------------------------------------
# Beat-phase augmentations (frame-level targets, not a scalar label)
# ---------------------------------------------------------------------------


class FrameTimeStretch:
    """Randomly stretch or compress audio and resample the frame-level target to match.

    Same resampling mechanism as :class:`TimeStretch` (treating the waveform as
    recorded at ``sr * rate`` and resampling back to ``sr``), but instead of
    rescaling a scalar label, resamples the target's time axis by the same
    ``rate`` so beat/one/last/mask events stay aligned with the stretched
    audio's frames. Frame count and sample count scale identically under a
    constant hop length, so using one shared ``rate`` for both is what keeps
    them in sync.

    Optional ``beats`` (see :class:`~musicality.loaders.beat_dataset.BeatDataset`'s
    ``return_beats``) are moved by the same mapping as the target. Output frame
    ``i`` reads input position ``(i + 0.5) * T / T' - 0.5``, so a beat at ``b``
    lands at ``(b + 0.5) * T' / T - 0.5`` — using the actual ratio ``T' / T``
    rather than ``rate``, so the beats follow the target's peaks exactly. A beat
    at frame 0 lands just below 0 and is clamped back, since ``time < 0`` means
    padding.

    :param min_rate: Minimum speed multiplier (< 1 = slower, lower tempo).
    :param max_rate: Maximum speed multiplier (> 1 = faster, higher tempo).
    """

    def __init__(self, min_rate: float = 0.85, max_rate: float = 1.15) -> None:
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
        sr: int,
        beats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        rate = random.uniform(self.min_rate, self.max_rate)

        new_wav_len = int(wav.shape[-1] / rate)
        stretched_wav = F.interpolate(
            wav.unsqueeze(0), size=new_wav_len, mode="linear", align_corners=False
        ).squeeze(0)

        new_n_frames = int(target.shape[-1] / rate)
        stretched_target = F.interpolate(
            target.unsqueeze(0), size=new_n_frames, mode="linear", align_corners=False
        ).squeeze(0)

        if beats is not None:
            valid = beats[0] >= 0
            moved = (beats[0] + 0.5) * new_n_frames / target.shape[-1] - 0.5

            beats = beats.clone()
            beats[0] = torch.where(valid, moved.clamp(min=0.0), beats[0])

        return stretched_wav, stretched_target, beats


def drop_beats_past(beats: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Turn beats whose rounded frame is past the clip's end into padding.

    Beats are in time order, so the ones dropped are the last valid slots and
    the padding stays contiguous at the end.
    """

    past = beats[0].round() >= n_frames

    beats = beats.clone()
    beats[0, past] = PAD_TIME
    beats[1, past] = 0.0

    return beats


class BeatPhaseAugmenter:
    """Composes waveform+target augmentations for beat-phase training.

    Applied in order:

    1. :class:`FrameTimeStretch` — changes length; re-crops/pads both the
       waveform and the target back to their expected fixed lengths, and
       turns optional ``beats`` stretched past the end into padding.
    2. :class:`RandomGain`
    3. :class:`AddNoise`
    """

    def __init__(
        self,
        time_stretch: FrameTimeStretch | None = None,
        gain: RandomGain | None = None,
        noise: AddNoise | None = None,
    ) -> None:
        self.time_stretch = time_stretch
        self.gain = gain
        self.noise = noise

    def __call__(
        self,
        wav: torch.Tensor,
        target: torch.Tensor,
        sr: int,
        n_samples: int,
        n_frames: int,
        beats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.time_stretch is not None:
            wav, target, beats = self.time_stretch(wav, target, sr, beats)

            if wav.shape[-1] >= n_samples:
                wav = wav[..., :n_samples]
            else:
                wav = F.pad(wav, (0, n_samples - wav.shape[-1]))

            if target.shape[-1] >= n_frames:
                target = target[..., :n_frames]
            else:
                target = F.pad(target, (0, n_frames - target.shape[-1]))

            if beats is not None:
                beats = drop_beats_past(beats, n_frames)

        if self.gain is not None:
            wav = self.gain(wav)

        if self.noise is not None:
            wav = self.noise(wav)

        return wav, target, beats


class AugmentedBeatDataset(Dataset):
    """Wraps a :class:`~musicality.loaders.beat_dataset.BeatDataset` and
    applies :class:`BeatPhaseAugmenter` at item-access time.

    Use this to augment only the training split while leaving the validation
    split unchanged — the two are independent dataset instances (see
    :func:`~musicality.trainers.common.build_beat_dataloaders`), so wrapping
    one has no effect on the other.

    :param subset: A ``BeatDataset`` (or ``torch.utils.data.Subset`` of one).
        If it yields ``(wav, target, beats)``, the beats are moved with the
        target and forwarded as a third element.
    :param augmenter: The augmenter to apply.
    :param sample_rate: Sample rate of the audio (passed to the augmenter).
    :param n_samples: Fixed clip length in samples (passed to the augmenter).
    :param n_frames: Fixed target length in frames (passed to the augmenter).
    """

    def __init__(
        self,
        subset,
        augmenter: BeatPhaseAugmenter,
        sample_rate: int,
        n_samples: int,
        n_frames: int,
    ) -> None:
        self.subset = subset
        self.augmenter = augmenter
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_frames = n_frames

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        wav, target, *rest = self.subset[idx]
        beats = rest[0] if rest else None

        wav, target, beats = self.augmenter(
            wav, target, self.sample_rate, self.n_samples, self.n_frames, beats
        )

        return (wav, target) if beats is None else (wav, target, beats)


def build_beat_phase_augmenter(cfg: DictConfig) -> BeatPhaseAugmenter | None:
    """Build a :class:`BeatPhaseAugmenter` from a Hydra augmentations config section.

    Mirrors :func:`build_augmenter`, using the frame-target-aware
    :class:`FrameTimeStretch` in place of :class:`TimeStretch`.

    Returns ``None`` if ``cfg.enabled`` is false or no individual augmentation
    is enabled, so callers can skip wrapping altogether.
    """
    if not cfg.get("enabled", True):
        return None

    time_stretch = None
    if cfg.time_stretch.get("enabled", False):
        time_stretch = FrameTimeStretch(
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

    return BeatPhaseAugmenter(time_stretch=time_stretch, gain=gain, noise=noise)
