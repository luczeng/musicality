"""PyTorch Dataset for mirdata beat/bar-position estimation datasets."""

import random
from pathlib import Path

import mirdata
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

import musicality.dataformats as dataformats
from musicality.splits.splitter import Splitter


DATA_DIR = dataformats.DATA_DIR

TARGET_CHANNELS = ("beat", "one", "last", "mask")


def gaussian_smear(spike: np.ndarray, sigma: float) -> np.ndarray:
    """Smear a 0/1 spike train into a soft target with a Gaussian bump per event.

    The kernel is left unnormalized (peak value 1.0 at its center), so an
    isolated spike keeps peak 1.0 after convolution. Overlapping bumps are
    clipped back to 1.0 rather than allowed to sum above it.

    :param spike: Binary spike train, shape ``(n_frames,)``.
    :param sigma: Gaussian standard deviation, in frames. ``0`` returns ``spike`` unchanged.
    :returns: Smeared target, shape ``(n_frames,)``, values in ``[0, 1]``.
    """

    if sigma <= 0 or not spike.any():
        return spike.astype(np.float32)

    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)

    smeared = np.convolve(spike, kernel, mode="same")

    return np.clip(smeared, 0.0, 1.0).astype(np.float32)


class BeatDataset(Dataset):
    """Generic mirdata dataset returning waveforms and frame-level beat-phase targets.

    Loads any mirdata dataset that exposes a ``beats`` attribute per track.
    Tracks without beat annotations or missing audio are silently skipped.

    The target is a 4-channel tensor of shape ``(4, n_frames)`` where
    ``n_frames = n_samples // hop_length``:

    - ``beat`` — any beat.
    - ``one``  — position 1 of the group (the downbeat, for the default
      ``group_size=4`` bar-position case).
    - ``last`` — the last beat of the group (bar position 4 for
      ``group_size=4``; e.g. phrase position 8 for a phrase-annotated dataset
      with ``group_size=8``).
    - ``mask`` — constant 1.0 across all frames if this track carries position
      annotations (``mirdata``'s ``bar_index`` unit, or an equivalent field for
      a custom phrase-annotated dataset), else constant 0.0. Datasets without
      position annotations (e.g. ``rwc_popular``) still contribute their
      ``beat`` channel; the ``one``/``last`` channels should be excluded from
      the loss for those tracks via this mask.

    Each channel is Gaussian-smeared (see :func:`gaussian_smear`) rather than a
    hard 0/1 spike.

    :param name: mirdata dataset name (e.g. ``"ballroom"``).
    :param data_home: Path to the dataset directory. Defaults to ``data/<name>``.
    :param sample_rate: Target sample rate. Audio is resampled if needed.
    :param duration: Clip duration in seconds. Longer clips are truncated,
        shorter clips are zero-padded.
    :param hop_length: Frame hop size in samples used to build the frame targets.
    :param sigma_frames: Gaussian smearing width, in frames, applied to each target channel.
    :param group_size: Number of beats per group that ``positions`` counts across —
        ``4`` (default) for bar-position (1-4) datasets, ``8`` for a phrase-position
        (1-8) dataset. Only affects which position value is read out as ``last``
        (``positions == group_size``); ``one`` is always ``positions == 1``.
    :param binary_only: If ``True``, drop tracks whose beats-per-bar (the
        annotated position cycle length) isn't a multiple of 2 — e.g.
        ballroom's waltz/Viennese waltz tracks, which cycle ``1, 2, 3`` in
        triple meter — as well as tracks with no position annotation at all,
        since their meter can't be confirmed. Independent of ``group_size``:
        a track only needs an even beats-per-bar count, not one equal to
        ``group_size``.
    :param random_crop: If ``True``, draw the ``duration``-second window at a
        random offset into the track on every access. Use for training (so
        the model doesn't just memorize one fixed window per track across
        epochs). If ``False``, always take a fixed window from the middle of
        the track — deterministic/reproducible for validation, and more
        representative than the start, which is often a sparse intro.
    """

    def __init__(
        self,
        name: str,
        data_home: Path | None = None,
        sample_rate: int = 22050,
        duration: float = 10.0,
        hop_length: int = 512,
        sigma_frames: float = 1.5,
        group_size: int = 4,
        binary_only: bool = False,
        random_crop: bool = False,
    ):
        if data_home is None:
            data_home = DATA_DIR / name

        self.sample_rate = sample_rate
        self.n_samples = int(duration * sample_rate)
        self.hop_length = hop_length
        self.n_frames = self.n_samples // hop_length
        self.sigma_frames = sigma_frames
        self.group_size = group_size
        self.random_crop = random_crop

        ds = mirdata.initialize(name, data_home=str(data_home))

        self.samples = []
        n_skipped = 0
        n_no_positions = 0
        n_non_binary = 0
        for tid in ds.track_ids:
            track = ds.track(tid)
            if track.beats is None or not Path(track.audio_path).exists():
                n_skipped += 1
                continue

            positions = track.beats.positions
            has_positions = positions is not None and np.any(np.asarray(positions) > 0)

            if binary_only and (
                not has_positions or int(np.max(np.asarray(positions))) % 2 != 0
            ):
                n_non_binary += 1
                continue

            if not has_positions:
                n_no_positions += 1

            self.samples.append(
                (track.audio_path, track.beats.times, positions, has_positions)
            )

        if n_skipped:
            print(
                f"[BeatDataset] {name}: skipped {n_skipped} track(s) with no beat annotation or missing audio"
            )
        if n_non_binary:
            print(
                f"[BeatDataset] {name}: skipped {n_non_binary} non-binary-meter track(s) (binary_only=True)"
            )
        if n_no_positions:
            print(
                f"[BeatDataset] {name}: {n_no_positions} track(s) have no position annotation (one/last masked)"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _times_to_spike(self, times: np.ndarray) -> np.ndarray:
        """Convert annotation timestamps into a 0/1 per-frame spike train."""

        spike = np.zeros(self.n_frames, dtype=np.float32)
        if len(times) == 0:
            return spike

        frames = np.round(times * self.sample_rate / self.hop_length).astype(int)
        valid = frames[(frames >= 0) & (frames < self.n_frames)]
        spike[valid] = 1.0

        return spike

    def __getitem__(self, idx: int):

        audio_path, beat_times, positions, has_positions = self.samples[idx]

        wav, sr = torchaudio.load(audio_path)  # (C, N)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = T.Resample(sr, self.sample_rate)(wav)

        if wav.shape[1] >= self.n_samples:
            max_start = wav.shape[1] - self.n_samples
            # Fixed (non-random) crops are taken from the middle rather than
            # the start, since intros are often sparse/atypical (e.g. no
            # beat yet) and a less representative eval window than the rest
            # of the track.
            start = random.randint(0, max_start) if self.random_crop else max_start // 2
            wav = wav[:, start : start + self.n_samples]
        else:
            start = 0
            wav = torch.nn.functional.pad(wav, (0, self.n_samples - wav.shape[1]))

        # Shift annotation times to be relative to the cropped window's start,
        # so frame indices computed below line up with the cropped audio.
        beat_times = beat_times - start / self.sample_rate

        if has_positions:
            positions = np.asarray(positions)
            one_times = beat_times[positions == 1]
            last_times = beat_times[positions == self.group_size]
        else:
            one_times = np.array([])
            last_times = np.array([])

        beat = gaussian_smear(self._times_to_spike(beat_times), self.sigma_frames)
        one = gaussian_smear(self._times_to_spike(one_times), self.sigma_frames)
        last = gaussian_smear(self._times_to_spike(last_times), self.sigma_frames)
        mask = np.full(self.n_frames, 1.0 if has_positions else 0.0, dtype=np.float32)

        target = torch.from_numpy(np.stack([beat, one, last, mask]))

        return wav, target  # (1, T), (4, n_frames)


def beat_split_name(name: str, binary_only: bool = False) -> str:
    """Split directory name for a BeatDataset build: ``beat_phase-<name>[-binary]``.

    Namespaced by dataset name and ``binary_only`` only — not by which heads a
    trainer/eval script actually uses, since ``BeatDataset``'s filtering (and
    therefore its length) only depends on those two things. So beat-phase and
    beat-only runs over the same dataset share the exact same held-out split,
    keeping their eval numbers directly comparable.
    """

    return f"beat_phase-{name}" + ("-binary" if binary_only else "")


def select_indices(
    dataset: "BeatDataset",
    name: str,
    split: str,
    val_split: float,
    binary_only: bool = False,
) -> list[int]:
    """Return ``dataset`` indices for ``split``, reusing a training run's
    cached train/val split (see :class:`~musicality.splits.splitter.Splitter`)
    so ``"val"`` means genuinely held-out tracks.

    :param dataset: The ``BeatDataset`` to select indices from.
    :param name: mirdata dataset name (e.g. ``"ballroom"``) used to look up the split.
    :param split: ``"train"``, ``"val"``, or ``"all"`` (every index — no split
        file is read in this case).
    :param val_split: Fraction of the dataset held out for validation. Must
        match how the split was created.
    :param binary_only: Must match how the split was created — see
        :func:`beat_split_name`.
    """

    if split == "all":
        return list(range(len(dataset)))

    _fmt = dataformats.load()
    splits_dir = dataformats.ROOT / _fmt.splits_dir

    train_ds, val_ds = Splitter(
        dataset, splits_dir, beat_split_name(name, binary_only), val_split
    ).run()

    return list((val_ds if split == "val" else train_ds).indices)
