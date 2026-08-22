"""PyTorch Dataset and DataLoader for mirdata tempo datasets."""

from pathlib import Path

import mirdata
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

import musicality.dataformats as dataformats
from musicality.loaders.mirdata_audio import index_audio, resolve_audio_path


DATA_DIR = dataformats.DATA_DIR


class TempoDataset(Dataset):
    """Generic mirdata dataset returning raw waveforms and tempo labels.

    Loads any mirdata dataset that exposes a ``tempo`` attribute per track.
    Tracks without a tempo annotation or missing audio are silently skipped.
    Preprocessing (e.g. mel transform) is left to the model.

    :param name: mirdata dataset name (e.g. ``"brid"``, ``"ballroom"``).
    :param data_home: Path to the dataset directory. Defaults to
        ``DATA_DIR/<name>`` (``DATA_DIR`` from :mod:`musicality.dataformats`).
    :param sample_rate: Target sample rate. Audio is resampled if needed.
    :param duration: Clip duration in seconds. Longer clips are truncated,
        shorter clips are zero-padded.
    """

    def __init__(
        self,
        name: str,
        data_home: Path | None = None,
        sample_rate: int = 22050,
        duration: float = 10.0,
    ):
        if data_home is None:
            data_home = DATA_DIR / name

        self.sample_rate = sample_rate
        self.n_samples = int(duration * sample_rate)

        ds = mirdata.initialize(name, data_home=str(data_home))

        audio_index = index_audio(data_home)

        # Store only (audio_path, tempo) to keep the dataset picklable for multiprocessing.
        # Some mirdata datasets' Track class has no `tempo` attribute at all (e.g.
        # rwc_jazz/rwc_classical only carry beat annotations) — getattr treats that
        # the same as an annotated-but-empty track and skips it.
        self.samples = []

        for tid in ds.track_ids:
            track = ds.track(tid)
            tempo = getattr(track, "tempo", None)

            if tempo is None:
                continue

            audio_path = resolve_audio_path(track, name, audio_index)
            if audio_path is not None:
                self.samples.append((str(audio_path), tempo))

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
            wav = torch.nn.functional.pad(wav, (0, self.n_samples - wav.shape[1]))

        label = torch.tensor(tempo, dtype=torch.float32)

        return wav, label  # (1, T), scalar
