#!/usr/bin/env python3
"""Launch tempo estimation training with Hydra config."""

import sys
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.loaders.loader import BRIDDataset
from musicality.trainers.tempo_module import TempoModule


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    L.seed_everything(42)

    # --- Data ---
    dataset = BRIDDataset(
        data_home=cfg.data.data_home,
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.data.n_mels,
        duration=cfg.data.duration,
    )
    n_val = int(len(dataset) * cfg.data.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

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

    # --- Model ---
    module = TempoModule(
        model=cfg.model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # --- Callbacks ---
    callbacks = [
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

    # --- Trainer ---
    trainer = L.Trainer(
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

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
