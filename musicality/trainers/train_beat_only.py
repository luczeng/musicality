"""Core training routine for frame-level, beat-only detection (no bar-position heads)."""

import logging

import lightning as L

# Suppress Lightning's promotional tip about LitLogger (INFO-level noise)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
from omegaconf import DictConfig

from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.trainers.beat_module import BeatModule
from musicality.trainers.common import (
    build_beat_dataloaders,
    build_checkpoint_callback,
    build_trainer,
)


_TRACKED_KEYS = (
    "train/loss",
    "train/f_beat",
    "val/loss",
    "val/f_beat",
)


def train(cfg: DictConfig) -> None:

    L.seed_everything(42)

    train_loader, val_loader, n_train, n_val = build_beat_dataloaders(cfg)

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


def build_module(cfg: DictConfig) -> BeatModule:

    return BeatModule(
        model=cfg.model,
        pos_weight=cfg.pos_weight,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        balanced=cfg.balanced,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        task=cfg.task,
    )


def build_callbacks(cfg: DictConfig) -> list:

    return [
        build_checkpoint_callback(cfg, "beat-only"),
        BestMetricsPrinter(keys=_TRACKED_KEYS),
    ]
