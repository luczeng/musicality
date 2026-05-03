"""PyTorch Dataset for mirdata beat estimation datasets."""

from pathlib import Path

import mirdata
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

import musicality.dataformats as dataformats


_fmt = dataformats.load()
DATA_DIR = Path(__file__).parent.parent / _fmt.data_dir


class BeatDataset(Dataset):
    """Generic mirdata dataset returning waveforms and frame-level beat activations.

    Loads any mirdata dataset that exposes a ``beats`` attribute per track.
    Tracks without beat annotations or missing audio are silently skipped.

    The beat target is a 1-D binary tensor of shape ``(n_frames,)`` where
    ``n_frames = n_samples // hop_length``.  Each frame is 1.0 if a beat
    falls within that frame window, 0.0 otherwise.

    :param name: mirdata dataset name (e.g. ``"ballroom"``).
    :param data_home: Path to the dataset directory. Defaults to ``data/<name>``.
    :param sample_rate: Target sample rate. Audio is resampled if needed.
    :param duration: Clip duration in seconds. Longer clips are truncated,
        shorter clips are zero-padded.
    :param hop_length: Frame hop size in samples used to build the beat target.
    """

    def __init__(
        self,
        name: str,
        data_home: Path | None = None,
        sample_rate: int = 22050,
        duration: float = 10.0,
        hop_length: int = 512,
    ):
        if data_home is None:
            data_home = DATA_DIR / name

        self.sample_rate = sample_rate
        self.n_samples = int(duration * sample_rate)
        self.hop_length = hop_length
        self.n_frames = self.n_samples // hop_length

        ds = mirdata.initialize(name, data_home=str(data_home))

        self.samples = []
        n_skipped = 0
        for tid in ds.track_ids:
            track = ds.track(tid)
            if track.beats is None or not Path(track.audio_path).exists():
                n_skipped += 1
            else:
                self.samples.append((track.audio_path, track.beats.times))

        if n_skipped:
            print(f"[BeatDataset] {name}: skipped {n_skipped} track(s) with no beat annotation or missing audio")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):

        audio_path, beat_times = self.samples[idx]

        wav, sr = torchaudio.load(audio_path)  # (C, N)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = T.Resample(sr, self.sample_rate)(wav)

        if wav.shape[1] >= self.n_samples:
            wav = wav[:, : self.n_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.n_samples - wav.shape[1]))

        # Build frame-level binary beat activation
        target = torch.zeros(self.n_frames)
        beat_frames = np.round(
            beat_times * self.sample_rate / self.hop_length
        ).astype(int)
        valid = beat_frames[(beat_frames >= 0) & (beat_frames < self.n_frames)]
        target[valid] = 1.0

        return wav, target  # (1, T), (n_frames,)
