"""Fixed-length audio crop loading, shared by this project's dataset loaders.

Crops are read straight off disk with :mod:`soundfile`'s seek-based read, so
the cost is O(crop length) rather than O(track length) — only the crop's bytes
are decoded, whether the track runs 30 seconds or 7 minutes.

That distinction dominates training throughput here, because the corpus mixes
short tracks (ballroom, ~30 s) with long ones (jtd ~2 min, rwc ~4 min): reading
a whole track only to keep 15 seconds of it left the GPU idle waiting on the
dataloader. ``torchaudio.load`` is deliberately not used, not even with
``frame_offset``/``num_frames`` — as of torchaudio 2.10 it dispatches to
torchcodec, whose per-call decoder setup costs ~60 ms regardless of how few
frames are asked for, against ~3 ms for the equivalent libsndfile seek+read.
"""

from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T


CropMode = Literal["start", "middle", "random"]

# One resampler per (source rate, target rate) pair. Build cost depends
# entirely on gcd(orig, target): it is negligible for the corpus's own
# 44100 -> 22050 (a 2-row kernel, ~0.07 ms) but 1-2 ms for awkward ratios such
# as 48000 or 32000 -> 22050 (320- and 640-row kernels). So this is cheap
# insurance against a non-44.1 kHz source entering the corpus, not a speedup
# on today's data. Module-level rather than per-dataset because DataLoader
# workers are separate processes: each fills its own cache lazily, and a
# dataset instance holding one would only make itself harder to pickle.
_RESAMPLERS: dict[tuple[int, int], T.Resample] = {}


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample *wav*, reusing a cached kernel for this rate pair.

    :param wav: Waveform, shape ``(channels, n)``.
    :param orig_sr: The waveform's current sample rate.
    :param target_sr: Wanted sample rate. Equal to *orig_sr* returns *wav* untouched.
    :returns: Waveform at *target_sr*, shape ``(channels, ~n * target_sr / orig_sr)``.
    """

    if orig_sr == target_sr:
        return wav

    key = (orig_sr, target_sr)
    resampler = _RESAMPLERS.get(key)

    if resampler is None:
        resampler = T.Resample(orig_sr, target_sr)
        _RESAMPLERS[key] = resampler

    return resampler(wav)


def _to_mono(block: np.ndarray) -> np.ndarray:
    """Mix an interleaved ``(frames, channels)`` block down to ``(frames,)``.

    Sums the channels column by column rather than calling ``block.mean(axis=1)``:
    numpy's reduction over a length-2 axis is an order of magnitude slower than
    the explicit adds (~6 ms vs ~0.5 ms for a 16 s stereo crop), and stereo is
    by far the common case here — enough to matter per dataloader item.
    """

    if block.shape[1] == 1:
        return block[:, 0]

    mono = block[:, 0].copy()

    for channel in range(1, block.shape[1]):
        mono += block[:, channel]

    mono /= block.shape[1]

    return mono


def crop_start_frame(n_frames: int, crop_frames: int, mode: CropMode) -> int:
    """Offset, in frames, at which a *crop_frames*-long window starts.

    :param n_frames: Length of the track being cropped, in frames.
    :param crop_frames: Length of the wanted window, in frames.
    :param mode: ``"start"`` (offset 0), ``"middle"``, or ``"random"``.
    :returns: The start offset, ``0`` whenever the track is no longer than the crop.
    """

    max_start = n_frames - crop_frames

    if max_start <= 0:
        return 0

    if mode == "random":
        return random.randint(0, max_start)

    if mode == "middle":
        return max_start // 2

    return 0


def load_crop(
    path: str,
    sample_rate: int,
    n_samples: int,
    crop: CropMode = "middle",
) -> tuple[torch.Tensor, float]:
    """Read one fixed-length mono crop from an audio file.

    Only the crop's own bytes are read: the header gives the track's native
    rate and length, the window is picked at that native rate, and resampling
    runs on the crop alone.

    :param path: Path to an audio file (anything libsndfile can open).
    :param sample_rate: Target sample rate. The crop is resampled if the file differs.
    :param n_samples: Wanted crop length, in samples at *sample_rate*. The
        returned waveform always has exactly this length — zero-padded on the
        right if the track is shorter.
    :param crop: Which window to take (see :func:`crop_start_frame`). Use
        ``"random"`` for training and ``"middle"`` for evaluation.
    :returns: ``(wav, start_seconds)`` — the waveform, shape ``(1, n_samples)``,
        and the crop's offset into the track in seconds, for shifting
        time-based annotations onto the cropped window.
    :raises RuntimeError: If the file can't be opened or decoded
        (``soundfile.LibsndfileError`` subclasses ``RuntimeError``).
    """

    info = sf.info(path)

    native_n = math.ceil(n_samples * info.samplerate / sample_rate)
    start = crop_start_frame(info.frames, native_n, crop)

    # stop past the end of the file is fine — soundfile returns what it has.
    block, _ = sf.read(
        path, start=start, stop=start + native_n, dtype="float32", always_2d=True
    )  # (frames, channels)

    wav = torch.from_numpy(np.ascontiguousarray(_to_mono(block))).unsqueeze(0)  # (1, n)

    if wav.shape[1] > 0:
        wav = resample(wav, info.samplerate, sample_rate)

    # Rounding in the native-rate conversion above — and tracks shorter than
    # the crop — can leave the result a sample or two off n_samples.
    if wav.shape[1] >= n_samples:
        wav = wav[:, :n_samples]
    else:
        wav = torch.nn.functional.pad(wav, (0, n_samples - wav.shape[1]))

    return wav, start / info.samplerate
