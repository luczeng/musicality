"""PyTorch Dataset and DataLoader for this project's own tempo-annotated
datasets (see docs/source/data.rst's "Data format" section)."""

import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import (
    TrackRef,
    list_track_refs,
    load_metadata,
    resolve_track_audio,
)


class TempoDataset(Dataset):
    """Dataset returning raw waveforms and tempo labels, read entirely from
    this project's own tracks/+annotations/ format.

    Loads every track with a migrated ``.beats`` annotation whose
    ``.meta.json`` carries a ``bpm_median`` (see
    :func:`musicality.dataformats.track_io.bpm_stats`, computed by the
    migration tools from the beat annotation itself — there's no separate
    ground-truth tempo field in this format). Tracks missing either are
    silently skipped. Preprocessing (e.g. mel transform) is left to the
    model.

    :param name: Dataset name (e.g. ``"rwc_popular"``, ``"swing"``) — must
        already be migrated to this project's own format (see
        ``tools/migrate_mirdata_dataset.py`` / ``tools/migrate_rwc_genre.py``).
        Mutually exclusive with *refs*.
    :param data_home: Dataset directory. Defaults to ``DATA_DIR/<name>``
        (``DATA_DIR`` from :mod:`musicality.dataformats`). Ignored if *refs*
        is given.
    :param refs: Explicit list of tracks to load, bypassing *name*/*data_home*
        resolution entirely — e.g. tracks pulled from several source
        datasets (see :func:`~musicality.splits.splitter.Splitter.load_refs`).
        Mutually exclusive with *name*.
    :param sample_rate: Target sample rate. Audio is resampled if needed.
    :param duration: Clip duration in seconds. Longer clips are truncated,
        shorter clips are zero-padded.
    """

    def __init__(
        self,
        name: str | None = None,
        data_home: Path | None = None,
        *,
        refs: list[TrackRef] | None = None,
        sample_rate: int = 22050,
        duration: float = 10.0,
    ):
        if refs is None:
            if name is None:
                raise ValueError("Must provide either `name` or `refs`.")
            if data_home is None:
                data_home = dataformats.DATA_DIR / name
            refs = list_track_refs(name, data_home)

        self.sample_rate = sample_rate
        self.n_samples = int(duration * sample_rate)

        # Resample transforms are expensive to build (they compute a sinc
        # filter kernel) — cache one per source sample rate instead of
        # rebuilding it on every __getitem__ call.
        self._resamplers: dict[int, T.Resample] = {}

        # Store only (audio_path, tempo) to keep the dataset picklable for multiprocessing.
        self.samples = []
        self.refs = []

        for ref in refs:
            metadata = load_metadata(
                ref.dataset_name, ref.track_id, data_home=ref.data_home
            )
            if metadata is None or metadata.bpm_median is None:
                continue

            audio_path = resolve_track_audio(
                ref.dataset_name, ref.track_id, ref.data_home
            )
            if audio_path is not None:
                self.samples.append((str(audio_path), metadata.bpm_median))
                self.refs.append(ref)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):

        audio_path, tempo = self.samples[idx]

        try:
            # Read just the header to learn the track's native length/rate,
            # then decode only the `duration`-second window we actually need
            # — tracks can run several minutes, so decoding the whole file
            # just to crop it down afterwards wastes most of the work.
            info = sf.info(audio_path)
            native_n_samples = int(round(self.n_samples / self.sample_rate * info.samplerate))
            num_frames = min(native_n_samples, info.frames)
            wav, sr = torchaudio.load(audio_path, frame_offset=0, num_frames=num_frames)
        except RuntimeError as e:
            print(f"[TempoDataset] failed to decode {audio_path!r} ({e}); skipping")
            return self.__getitem__(random.randrange(len(self)))

        # Mix down to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = self._resamplers.get(sr)
            if resampler is None:
                resampler = T.Resample(sr, self.sample_rate)
                self._resamplers[sr] = resampler
            wav = resampler(wav)

        # Truncate or zero-pad to fixed length
        if wav.shape[1] >= self.n_samples:
            wav = wav[:, : self.n_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.n_samples - wav.shape[1]))

        label = torch.tensor(tempo, dtype=torch.float32)

        return wav, label  # (1, T), scalar
