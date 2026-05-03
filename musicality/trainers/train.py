"""Core training routine for tempo estimation."""

import logging
import random

import lightning as L

# Suppress Lightning's promotional tip about LitLogger (INFO-level noise)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

import musicality.dataformats as dataformats
from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.loaders.tempo_dataset import TempoDataset
from musicality.splits.splitter import Splitter
from musicality.trainers.tempo_module import TempoModule


def train(cfg: DictConfig) -> None:

    L.seed_everything(42)

    train_loader, val_loader, n_train, n_val = build_dataloaders(cfg)

    module = build_module(cfg)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks)

    trainer.logger.experiment.config.update({
        "data/n_train": n_train,
        "data/n_val": n_val,
        "model/arch": cfg.model.get("arch"),
    })

    trainer.fit(module, train_loader, val_loader)


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, int, int]:

    dataset = TempoDataset(
        name=cfg.data.name,
        data_home=cfg.data.data_home,
        sample_rate=cfg.data.sample_rate,
        duration=cfg.data.duration,
    )

    _fmt = dataformats.load()
    splits_dir = dataformats.ROOT / _fmt.splits_dir
    dataset_name = cfg.data.name

    train_ds, val_ds = Splitter(
        dataset, splits_dir, dataset_name, cfg.data.val_split
    ).run()

    subsample = cfg.get("train_subsample", None)
    if subsample is not None:
        n_before = len(train_ds)
        n = max(1, int(n_before * subsample))
        indices = random.sample(range(n_before), n)
        train_ds = Subset(train_ds, indices)
        print(f"[train] Subsampled train set: {n}/{n_before} ({subsample:.0%})")

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


def build_module(cfg: DictConfig) -> TempoModule:

    return TempoModule(
        model=cfg.model,
        loss=cfg.loss,
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
            save_weights_only=True,
        ),
        # EarlyStopping(monitor="val/loss", patience=10, mode="min"),
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
            config=OmegaConf.to_container(cfg, resolve=True),
        ),
        enable_progress_bar=True,
    )
