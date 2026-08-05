"""Core training routine for frame-level beat-phase detection (beat/one/last)."""

import logging
import random

import lightning as L

# Suppress Lightning's promotional tip about LitLogger (INFO-level noise)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

import musicality.dataformats as dataformats
from musicality.augmentations import AugmentedBeatDataset, build_beat_phase_augmenter
from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.loaders.beat_dataset import BeatDataset
from musicality.splits.splitter import Splitter
from musicality.trainers.beat_phase_module import BeatPhaseModule


_TRACKED_KEYS = (
    "train/loss",
    "train/acc_beat",
    "train/acc_one",
    "train/acc_last",
    "val/loss",
    "val/acc_beat",
    "val/acc_one",
    "val/acc_last",
)


def train(cfg: DictConfig) -> None:

    L.seed_everything(42)

    train_loader, val_loader, n_train, n_val = build_dataloaders(cfg)

    module = build_module(cfg)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks)

    trainer.logger.experiment.config.update(
        {
            "data/n_train": n_train,
            "data/n_val": n_val,
            "model/arch": cfg.model.get("arch"),
        }
    )

    trainer.fit(module, train_loader, val_loader)


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, int, int]:

    binary_only = cfg.get("binary_only", False)

    dataset = BeatDataset(
        name=cfg.data.name,
        data_home=cfg.data.data_home,
        sample_rate=cfg.data.sample_rate,
        duration=cfg.data.duration,
        hop_length=cfg.hop_length,
        sigma_frames=cfg.sigma_frames,
        group_size=cfg.get("group_size", 4),
        binary_only=binary_only,
    )

    _fmt = dataformats.load()
    splits_dir = dataformats.ROOT / _fmt.splits_dir
    # Namespaced separately from TempoDataset's splits/<name>/ cache: BeatDataset
    # filters tracks differently (by beat annotation, not tempo) and can have a
    # different length for the same mirdata dataset name, so the two must never
    # share a split-index cache. The -binary suffix keeps the same guarantee
    # against the unfiltered beat-phase split, since binary_only changes the
    # dataset length too.
    dataset_name = f"beat_phase-{cfg.data.name}" + ("-binary" if binary_only else "")

    train_ds, val_ds = Splitter(
        dataset, splits_dir, dataset_name, cfg.data.val_split
    ).run()

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
            f"[train_beat_phase] Subsampled train set: {n}/{n_before} ({subsample:.0%})"
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


def build_module(cfg: DictConfig) -> BeatPhaseModule:

    return BeatPhaseModule(
        model=cfg.model,
        pos_weight=cfg.pos_weight,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def build_callbacks(cfg: DictConfig) -> list:

    return [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="beat-phase-{epoch:02d}-{val/loss:.4f}",
            save_weights_only=True,
        ),
        BestMetricsPrinter(keys=_TRACKED_KEYS),
    ]


def build_trainer(cfg: DictConfig, callbacks: list) -> L.Trainer:

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
