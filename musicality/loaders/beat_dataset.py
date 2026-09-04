"""PyTorch Dataset for this project's own beat/bar-position-annotated
datasets (see docs/source/data.rst's "Data format" section)."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import (
    TrackRef,
    list_track_refs,
    read_beats_file,
    resolve_track_audio,
)
from musicality.loaders.audio_io import load_crop
from musicality.splits.splitter import Splitter


DATA_DIR = dataformats.DATA_DIR

TARGET_CHANNELS = ("beat", "one", "last", "mask")


def position_target_channels(group_size: int) -> tuple[str, ...]:
    """Channel names for the ``target_layout="positions"`` target.

    ``("beat", "pos_1", ..., "pos_<group_size>", "mask")`` — ``beat`` first and
    ``mask`` last, matching :data:`TARGET_CHANNELS`, so ``target[0]`` and
    ``target[-1]`` mean the same thing under both layouts.
    """

    return ("beat", *(f"pos_{p}" for p in range(1, group_size + 1)), "mask")


def fold_positions(positions: np.ndarray, group_size: int) -> np.ndarray | None:
    """Fold an annotated bar-position cycle onto ``1..group_size``.

    Annotations count positions across whatever the annotator treated as one
    bar, and that is not always ``group_size`` beats. A track annotated
    ``1..8`` against ``group_size=4`` is two bars of four, so its beats 5-8
    are the 1-4 of the next bar and belong in the same four channels.

    Without folding, positions above ``group_size`` match no channel at all:
    the position block is all-zero on those frames, so they hit the
    uniform-target fallback in :meth:`BeatDataset.__getitem__` and the loss
    teaches "every bar position is equally likely" at full beat weight — on
    beats that are perfectly well annotated.

    :param positions: 1-indexed bar positions, shape ``(n_beats,)``.
    :param group_size: Beats per group the position head predicts over.
    :returns: Folded positions, or ``None`` when the annotated cycle is not a
        multiple of *group_size* (e.g. a 6-beat bar against a 4-way head).
        No consistent folding exists there — 1,2,3,4,1,2 puts the downbeat 4
        beats after one bar and 2 after the next — so the caller must drop
        that track's position supervision rather than fold it wrongly.
    """

    positions = np.asarray(positions, dtype=int)
    cycle = int(positions.max())

    # A cycle shorter than the group (a 2-beat bar against a 4-way head)
    # already lands in real channels and needs no folding — it describes a
    # valid, if shorter, period. Only overflow past group_size is the problem.
    if cycle <= group_size:
        return positions

    if cycle % group_size != 0:
        return None

    return ((positions - 1) % group_size) + 1


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
    """Dataset returning waveforms and frame-level beat-phase targets, read
    entirely from this project's own tracks/+annotations/ format.

    Loads every track with a migrated ``.beats`` annotation and a
    resolvable ``tracks/<id>.wav``.

    The target is a 4-channel tensor of shape ``(4, n_frames)`` where
    ``n_frames = n_samples // hop_length``:

    - ``beat`` — any beat.
    - ``one``  — position 1 of the group (the downbeat, for the default
      ``group_size=4`` bar-position case).
    - ``last`` — the last beat of the group (bar position 4 for
      ``group_size=4``; e.g. phrase position 8 for a phrase-annotated dataset
      with ``group_size=8``).
    - ``mask`` — constant 1.0 across all frames if this track carries usable
      position annotations, else constant 0.0. Two cases give 0.0: no position
      annotation at all (e.g. ``rwc_popular``), and a bar length that cannot be
      folded onto ``group_size`` (see :func:`fold_positions`). Both still
      contribute their ``beat`` channel; the position channels should be
      excluded from the loss for those tracks via this mask.

    Each channel is Gaussian-smeared (see :func:`gaussian_smear`) rather than a
    hard 0/1 spike.

    :param name: Dataset name (e.g. ``"ballroom"``, ``"swing"``) — must
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
    :param hop_length: Frame hop size in samples used to build the frame targets.
    :param sigma_frames: Gaussian smearing width, in frames, applied to each target channel.
    :param group_size: Number of beats per group that the target counts across —
        ``4`` (default) for bar-position (1-4) datasets, ``8`` for a phrase-position
        (1-8) dataset. Annotations counting a *longer* bar are folded onto
        ``1..group_size`` at load time (see :func:`fold_positions`), so a track
        annotated 1-8 trains a ``group_size=4`` head as two bars of four rather
        than falling off the end of the channel list. Tracks whose bar length is
        not a multiple of ``group_size`` keep their beats but have their position
        supervision masked off.
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
    :param target_layout: Which phase target to build.

        - ``"one_last"`` (default): the 4-channel target described above —
          two independent binary detectors for positions ``1`` and
          ``group_size``, with positions in between carrying no supervision
          at all.
        - ``"positions"``: a ``(2 + group_size, n_frames)`` target —
          ``beat``, then one channel per bar position, then ``mask`` (see
          :func:`position_target_channels`). The position block is normalized
          to a per-frame probability distribution, so it pairs with a softmax
          head and a cross-entropy loss rather than per-channel sigmoids.
          Every position gets its own supervised channel, which makes
          "is this a 1 or a 3?" a question the model is actually asked. See
          docs/beat_phase_improvement_review.md section 3.
    """

    def __init__(
        self,
        name: str | None = None,
        data_home: Path | None = None,
        *,
        refs: list[TrackRef] | None = None,
        sample_rate: int = 22050,
        duration: float = 10.0,
        hop_length: int = 512,
        sigma_frames: float = 1.5,
        group_size: int = 4,
        binary_only: bool = False,
        random_crop: bool = False,
        target_layout: str = "one_last",
    ):
        if target_layout not in ("one_last", "positions"):
            raise ValueError(
                f"Unknown target_layout {target_layout!r} — "
                "expected 'one_last' or 'positions'"
            )
        if refs is None:
            if name is None:
                raise ValueError("Must provide either `name` or `refs`.")
            if data_home is None:
                data_home = dataformats.DATA_DIR / name
            refs = list_track_refs(name, data_home)

        self.sample_rate = sample_rate
        self.n_samples = int(duration * sample_rate)
        self.hop_length = hop_length
        self.n_frames = self.n_samples // hop_length
        self.sigma_frames = sigma_frames
        self.group_size = group_size
        self.random_crop = random_crop
        self.target_layout = target_layout

        self.samples = []
        self.refs = []
        n_skipped = 0
        n_no_positions = 0
        n_non_binary = 0
        n_folded = 0
        n_unfoldable = 0

        for ref in refs:
            audio_path = resolve_track_audio(
                ref.dataset_name, ref.track_id, ref.data_home
            )
            if audio_path is None:
                n_skipped += 1
                continue

            beats_path = (
                ref.data_home
                / dataformats.FORMAT.annotations_dirname
                / f"{ref.track_id}{dataformats.FORMAT.beats_suffix}"
            )
            beat_times, positions = read_beats_file(beats_path)

            has_positions = positions is not None and np.any(np.asarray(positions) > 0)

            if binary_only and (
                not has_positions or int(np.max(np.asarray(positions))) % 2 != 0
            ):
                n_non_binary += 1
                continue

            if not has_positions:
                n_no_positions += 1
            else:
                folded = fold_positions(positions, group_size)

                if folded is None:
                    # A meter this head cannot represent. Keep the track for
                    # its beat channel and switch the position supervision
                    # off, which is exactly what has_positions=False already
                    # means everywhere downstream.
                    n_unfoldable += 1
                    has_positions = False
                    positions = None
                else:
                    n_folded += int(not np.array_equal(folded, positions))
                    positions = folded

            self.samples.append((str(audio_path), beat_times, positions, has_positions))
            self.refs.append(ref)

        label = name if name is not None else "<refs>"

        if n_skipped:
            print(
                f"[BeatDataset] {label}: skipped {n_skipped} track(s) with missing audio"
            )
        if n_non_binary:
            print(
                f"[BeatDataset] {label}: skipped {n_non_binary} non-binary-meter track(s) (binary_only=True)"
            )
        if n_no_positions:
            print(
                f"[BeatDataset] {label}: {n_no_positions} track(s) have no position annotation (one/last masked)"
            )
        if n_folded:
            print(
                f"[BeatDataset] {label}: folded {n_folded} track(s) with a longer "
                f"annotated bar onto positions 1-{group_size}"
            )
        if n_unfoldable:
            print(
                f"[BeatDataset] {label}: {n_unfoldable} track(s) have a bar length that "
                f"is not a multiple of group_size={group_size} "
                "(position supervision masked off, beats kept)"
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

        try:
            # Only the crop's bytes are read off disk, so cost is independent
            # of track length (see musicality.loaders.audio_io). Fixed
            # (non-random) crops are taken from the middle rather than the
            # start, since intros are often sparse/atypical (e.g. no beat
            # yet) and a less representative eval window than the rest of
            # the track.
            wav, start_seconds = load_crop(
                audio_path,
                self.sample_rate,
                self.n_samples,
                crop="random" if self.random_crop else "middle",
            )
        except RuntimeError as e:
            print(f"[BeatDataset] failed to decode {audio_path!r} ({e}); skipping")
            return self.__getitem__(random.randrange(len(self)))

        # Shift annotation times to be relative to the cropped window's start,
        # so frame indices computed below line up with the cropped audio.
        beat_times = beat_times - start_seconds

        positions = np.asarray(positions) if has_positions else None

        beat = gaussian_smear(self._times_to_spike(beat_times), self.sigma_frames)
        mask = np.full(self.n_frames, 1.0 if has_positions else 0.0, dtype=np.float32)

        def _smeared_at(position: int) -> np.ndarray:
            times = beat_times[positions == position] if has_positions else np.array([])
            return gaussian_smear(self._times_to_spike(times), self.sigma_frames)

        if self.target_layout == "one_last":
            channels = [beat, _smeared_at(1), _smeared_at(self.group_size), mask]
        else:
            block = np.stack([_smeared_at(p) for p in range(1, self.group_size + 1)])

            # Normalize the position block into a per-frame distribution, so it
            # can be the target of a softmax + cross-entropy. Frames with no
            # beat nearby carry no position information at all — they get a
            # uniform row, and the loss masks them out by beat weight anyway.
            total = block.sum(axis=0, keepdims=True)
            block = np.divide(
                block,
                total,
                out=np.full_like(block, 1.0 / self.group_size),
                where=total > 1e-6,
            )
            channels = [beat, *block, mask]

        target = torch.from_numpy(np.stack(channels).astype(np.float32))

        return wav, target  # (1, T), (4 | 2 + group_size, n_frames)


def beat_split_name(name: str, binary_only: bool = False) -> str:
    """Split directory name for a BeatDataset build: ``beat_phase-<name>[-binary]``.

    Namespaced by dataset name and ``binary_only`` only — not by which heads a
    trainer/eval script actually uses, since ``BeatDataset``'s filtering (and
    therefore its length) only depends on those two things. So beat-phase and
    beat-only runs over the same dataset share the exact same held-out split,
    keeping their eval numbers directly comparable.
    """

    return f"beat_phase-{name}" + ("-binary" if binary_only else "")


def indices_for_split(
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
    :param name: Dataset name (e.g. ``"ballroom"``) used to look up the split.
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
