"""PyTorch Dataset and DataLoader for mirdata tempo datasets."""

from pathlib import Path

import mirdata
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

import musicality.dataformats as dataformats

_fmt = dataformats.load()
DATA_DIR = Path(__file__).parent.parent / _fmt.data_dir


class TempoDataset(Dataset):
    """Generic mirdata dataset returning mel spectrograms and tempo labels.

    Loads any mirdata dataset that exposes a ``tempo`` attribute per track.
    Tracks without a tempo annotation or missing audio are silently skipped.

    :param name: mirdata dataset name (e.g. ``"brid"``, ``"ballroom"``).
    :param data_home: Path to the dataset directory. Defaults to ``data/<name>``.
    :param sample_rate: Target sample rate. Audio is resampled if needed.
    :param n_mels: Number of mel filterbanks.
    :param duration: Clip duration in seconds. Longer clips are truncated,
        shorter clips are zero-padded.
    """

    def __init__(
        self,
        name: str,
        data_home: Path | None = None,
        sample_rate: int = 22050,
        n_mels: int = 128,
        duration: float = 10.0,
    ):
        if data_home is None:
            data_home = DATA_DIR / name

        self.sample_rate = sample_rate
        self.n_samples = int(duration * sample_rate)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=2048
        )
        self.log_transform = T.AmplitudeToDB()

        ds = mirdata.initialize(name, data_home=str(data_home))

        # Store only (audio_path, tempo) to keep the dataset picklable for multiprocessing
        self.samples = [
            (ds.track(tid).audio_path, ds.track(tid).tempo)
            for tid in ds.track_ids
            if ds.track(tid).tempo is not None
            and Path(ds.track(tid).audio_path).exists()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):

        audio_path, tempo = self.samples[idx]

        wav, sr = torchaudio.load(audio_path)  # (C, N)

        # Mix down to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            wav = T.Resample(sr, self.sample_rate)(wav)

        # Truncate or zero-pad to fixed length
        if wav.shape[1] >= self.n_samples:
            wav = wav[:, : self.n_samples]
        else:
            wav = torch.nn.functional.pad(
                wav, (0, self.n_samples - wav.shape[1])
            )

        mel = self.log_transform(self.mel_transform(wav))  # (1, n_mels, T)

        label = torch.tensor(tempo, dtype=torch.float32)

        return mel, label


def get_loader(
    name: str,
    data_home: Path | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """Return a DataLoader for a mirdata tempo dataset.

    Each batch is a tuple of:

    - ``mel``   — ``(B, 1, n_mels, T)``  log-mel spectrogram
    - ``tempo`` — ``(B,)``               BPM label

    :param name: mirdata dataset name.
    :param data_home: Path to the dataset directory. Defaults to ``data/<name>``.
    :param batch_size: Number of samples per batch.
    :param shuffle: Whether to shuffle the data each epoch.
    :param num_workers: Number of worker processes for loading.
    :param dataset_kwargs: Forwarded to TempoDataset (sample_rate, n_mels, duration).
    :returns: Configured DataLoader.
    :rtype: DataLoader
    """

    dataset = TempoDataset(name=name, data_home=data_home, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
