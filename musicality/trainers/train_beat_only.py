"""Core training routine for frame-level, beat-only detection (no bar-position heads)."""

import logging

import lightning as L

# Suppress Lightning's promotional tip about LitLogger (INFO-level noise)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.trainers.beat_module import BeatModule
from musicality.trainers.common import build_beat_dataloaders, build_trainer


_TRACKED_KEYS = (
    "train/loss",
    "train/acc_beat",
    "val/loss",
    "val/acc_beat",
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
        task=cfg.task,
    )


def build_callbacks(cfg: DictConfig) -> list:

    return [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="beat-only-{epoch:02d}-{val/loss:.4f}",
            save_weights_only=True,
        ),
        BestMetricsPrinter(keys=_TRACKED_KEYS),
    ]
