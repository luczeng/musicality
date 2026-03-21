"""Core training routine for tempo estimation."""

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import musicality.dataformats as dataformats
from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.loaders.loader import TempoDataset
from musicality.splits.splitter import Splitter
from musicality.trainers.tempo_module import TempoModule


def train(cfg: DictConfig) -> None:

    L.seed_everything(42)

    train_loader, val_loader = build_dataloaders(cfg)

    module = build_module(cfg)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks)

    trainer.fit(module, train_loader, val_loader)


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:

    dataset = TempoDataset(
        name=cfg.data.name,
        data_home=cfg.data.data_home,
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.data.n_mels,
        duration=cfg.data.duration,
    )

    _fmt = dataformats.load()
    splits_dir = dataformats.ROOT / _fmt.splits_dir
    dataset_name = cfg.data.name

    train_ds, val_ds = Splitter(dataset, splits_dir, dataset_name, cfg.data.val_split).run()

    persistent_workers = cfg.data.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader


def build_module(cfg: DictConfig) -> TempoModule:

    return TempoModule(
        model=cfg.model,
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
            filename="tempo-{epoch:02d}-{val/loss:.4f}",
        ),
        EarlyStopping(monitor="val/loss", patience=10, mode="min"),
        BestMetricsPrinter(),
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
            config=dict(cfg),
            anonymous=None,
        ),
        enable_progress_bar=True,
    )
