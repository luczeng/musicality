"""Training-pipeline plumbing shared across the tempo, beat-phase, and beat-only trainers."""

import random
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

import musicality.dataformats as dataformats
from musicality.augmentations import AugmentedBeatDataset, build_beat_phase_augmenter
from musicality.dataformats.track_io import TrackRef
from musicality.loaders.beat_dataset import BeatDataset, beat_split_name
from musicality.splits.splitter import Splitter


def resolve_split_refs(
    cfg: DictConfig, splits_dir: Path, split_name: str
) -> tuple[list[TrackRef], list[TrackRef]]:
    """Return ``(train_refs, val_refs)`` for a training config's ``data`` section.

    ``cfg.data.input`` is one field serving two purposes, told apart by
    whether it contains a ``/``:

    - A bare name (e.g. ``"ballroom"``, ``"ballroom_brid"``) — looked up
      under the canonical ``splits_dir`` via *split_name* (which may already
      have a trainer-specific naming convention applied, e.g.
      ``beat_split_name``): ``Splitter.load_refs(splits_dir, split_name)``.
    - A path (e.g. ``"../musicality_db/splits/ballroom"``, or anywhere else
      on disk) — used directly as the split folder via
      :meth:`~musicality.splits.splitter.Splitter.load_refs_from_dir`,
      bypassing ``splits_dir`` and any naming convention entirely. Lets a
      split be trained on without being "registered" under the canonical
      ``splits_dir`` first.
    """

    input_ = cfg.data.input
    if "/" in input_:
        return Splitter.load_refs_from_dir(Path(input_))

    return Splitter.load_refs(splits_dir, split_name)


def build_beat_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, int, int]:
    """Build train/val DataLoaders over a :class:`~musicality.loaders.beat_dataset.BeatDataset`.

    Shared by ``train_beat_phase.py`` and ``train_beat_only.py`` — both read
    the same ``(beat, one, last, mask)`` target, just different subsets of
    it, so loader construction itself doesn't need to know which heads the
    calling trainer will actually use.

    :param cfg: Hydra config with ``data.*``, ``hop_length``, ``sigma_frames``,
        ``batch_size``, and (optionally) ``group_size`` / ``binary_only`` /
        ``augmentations`` / ``train_subsample`` fields.
    :returns: ``(train_loader, val_loader, n_train, n_val)``.
    """

    binary_only = cfg.get("binary_only", False)

    dataset_kwargs = dict(
        sample_rate=cfg.data.sample_rate,
        duration=cfg.data.duration,
        hop_length=cfg.hop_length,
        sigma_frames=cfg.sigma_frames,
        group_size=cfg.get("group_size", 4),
        target_layout=cfg.get("target_layout", "one_last"),
        binary_only=binary_only,
    )

    _fmt = dataformats.load()
    splits_dir = dataformats.ROOT / _fmt.splits_dir
    split_name = beat_split_name(cfg.data.input, binary_only)

    train_refs, val_refs = resolve_split_refs(cfg, splits_dir, split_name)

    # Separate dataset instances for train/val so only the train split draws
    # a random crop window per track per epoch — val stays fixed at the
    # middle of the track (see BeatDataset) for reproducible eval.
    train_ds = BeatDataset(
        refs=train_refs, random_crop=cfg.data.get("random_crop", True), **dataset_kwargs
    )
    val_ds = BeatDataset(refs=val_refs, random_crop=False, **dataset_kwargs)

    augmenter = (
        build_beat_phase_augmenter(cfg.augmentations)
        if cfg.get("augmentations")
        else None
    )
    if augmenter is not None:
        n_samples = int(cfg.data.duration * cfg.data.sample_rate)
        n_frames = n_samples // cfg.hop_length
        train_ds = AugmentedBeatDataset(
            train_ds, augmenter, cfg.data.sample_rate, n_samples, n_frames
        )

    subsample = cfg.get("train_subsample", None)
    if subsample is not None:
        n_before = len(train_ds)
        n = max(1, int(n_before * subsample))
        indices = random.sample(range(n_before), n)
        train_ds = Subset(train_ds, indices)
        print(
            f"[build_beat_dataloaders] Subsampled train set: {n}/{n_before} ({subsample:.0%})"
        )

    n_train, n_val = len(train_ds), len(val_ds)

    persistent_workers = cfg.data.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, n_train, n_val


def build_checkpoint_callback(cfg: DictConfig, prefix: str) -> ModelCheckpoint:
    """One directory per run, holding that run's ``save_top_k`` best checkpoints.

    Two things this gets right that a bare :class:`ModelCheckpoint` does not:

    **The metric name must not reach the filename.** Lightning's
    ``auto_insert_metric_name`` splices the monitored key into the filename, and
    the key here is ``val/loss`` — the slash is a path separator, so
    ``"{epoch:02d}-{val/loss:.4f}"`` produced a *directory* per checkpoint
    (``beat-phase-epoch=139-val/loss=1.9543.ckpt``). Pruning a checkpoint then
    removed the file but left its directory behind, so a run accumulated one
    empty directory per superseded epoch. ``auto_insert_metric_name=False``
    substitutes the value only, keeping everything on one path segment.

    **Runs must not share a directory.** ``checkpoint_dir`` is a fixed path in
    the config, so consecutive runs wrote into the same place and their
    checkpoints interleaved — ``save_top_k`` is per-run bookkeeping and cannot
    prune another run's files. Each run gets its own subdirectory instead.

    :param cfg: Needs ``checkpoint_dir``, ``trainer.save_top_k`` (optional,
        defaults to 3) and ``wandb.run_name`` (optional).
    :param prefix: Filename stem for this task, e.g. ``"beat-phase"``.
    """

    # Prefer the W&B run name so the checkpoint directory matches the run in
    # the dashboard. It is usually null (W&B generates one only once the logger
    # connects, which is after callbacks are built), so fall back to a
    # timestamp — which also works offline.
    run_name = cfg.get("wandb", {}).get("run_name") or datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    return ModelCheckpoint(
        dirpath=Path(cfg.checkpoint_dir) / run_name,
        monitor="val/loss",
        mode="min",
        save_top_k=cfg.trainer.get("save_top_k", 3),
        filename=f"{prefix}-epoch{{epoch:02d}}-valloss{{val/loss:.4f}}",
        auto_insert_metric_name=False,
        save_weights_only=True,
    )


def build_trainer(cfg: DictConfig, callbacks: list) -> L.Trainer:
    """Construct the Lightning ``Trainer`` + W&B logger shared by every training entry point.

    Pure infrastructure — reads only ``cfg.trainer.*`` / ``cfg.wandb.*`` and
    has no task-specific behavior, so it's identical across the tempo,
    beat-phase, and beat-only pipelines.
    """

    return L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        ),
        enable_progress_bar=True,
    )
