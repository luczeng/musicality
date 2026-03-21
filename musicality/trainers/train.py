"""Core training routine for tempo estimation."""

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split

from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.loaders.loader import BRIDDataset
from musicality.trainers.tempo_module import TempoModule


def train(cfg: DictConfig) -> None:

    L.seed_everything(42)

    train_loader, val_loader = build_dataloaders(cfg)

    module = build_module(cfg)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks)
    trainer.fit(module, train_loader, val_loader)


def _load_split(splits_dir: Path, key: str) -> tuple[list, list] | None:
    train_file = splits_dir / f"{key}_train.txt"
    val_file = splits_dir / f"{key}_val.txt"
    if train_file.exists() and val_file.exists():
        train_indices = list(map(int, train_file.read_text().splitlines()))
        val_indices = list(map(int, val_file.read_text().splitlines()))
        return train_indices, val_indices
    return None


def _save_split(splits_dir: Path, key: str, train_indices: list, val_indices: list) -> None:
    splits_dir.mkdir(exist_ok=True)
    (splits_dir / f"{key}_train.txt").write_text("\n".join(map(str, train_indices)))
    (splits_dir / f"{key}_val.txt").write_text("\n".join(map(str, val_indices)))


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    dataset = BRIDDataset(
        data_home=cfg.data.data_home,
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.data.n_mels,
        duration=cfg.data.duration,
    )

    splits_dir = Path(cfg.data.data_home) / "splits"
    key = f"{Path(cfg.data.data_home).name}_{len(dataset)}_{cfg.data.val_split}"
    existing = _load_split(splits_dir, key)

    if existing is not None:
        train_indices, val_indices = existing
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
    else:
        n_val = int(len(dataset) * cfg.data.val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        _save_split(splits_dir, key, list(train_ds.indices), list(val_ds.indices))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
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
        ),
        enable_progress_bar=True,
    )
