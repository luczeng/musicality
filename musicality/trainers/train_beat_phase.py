"""Core training routine for frame-level beat-phase detection (beat/one/last)."""

import logging

import lightning as L

# Suppress Lightning's promotional tip about LitLogger (INFO-level noise)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
from omegaconf import DictConfig

from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.trainers.beat_phase_module import BeatPhaseModule
from musicality.trainers.common import (
    build_beat_dataloaders,
    build_checkpoint_callback,
    build_trainer,
)


_TRACKED_KEYS = (
    "train/loss",
    "train/acc_beat",
    "train/acc_one",
    "train/acc_last",
    "train/acc_position",
    "val/loss",
    "val/acc_beat",
    "val/acc_one",
    "val/acc_last",
    "val/acc_position",
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


def build_module(cfg: DictConfig) -> BeatPhaseModule:

    return BeatPhaseModule(
        model=cfg.model,
        pos_weight=cfg.pos_weight,
        phase_conditioning=cfg.get("phase_conditioning", "mask"),
        # Only set when the config asks for the softmax head, so a plain
        # one/last run keeps the original three-channel parameterization.
        group_size=(
            cfg.get("group_size", 4)
            if cfg.get("target_layout", "one_last") == "positions"
            else None
        ),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        balanced=cfg.balanced,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        task=cfg.task,
    )


def build_callbacks(cfg: DictConfig) -> list:

    return [
        build_checkpoint_callback(cfg, "beat-phase"),
        BestMetricsPrinter(keys=_TRACKED_KEYS),
    ]
