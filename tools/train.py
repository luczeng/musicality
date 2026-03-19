#!/usr/bin/env python3
"""Launch tempo estimation training with Hydra config."""

import os
import sys

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split

from musicality.loader import BRIDDataset
from musicality.trainers.tempo_module import TempoModule


class BestMetricsPrinter(L.Callback):
    """Prints a summary of the best validation metrics after training."""

    def __init__(self):
        self.best = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in ("val/loss", "val/mae_bpm"):
            val = metrics.get(key)
            if val is None:
                continue
            if key not in self.best or val < self.best[key]:
                self.best[key] = val.item()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        parts = [f"epoch {epoch:>3}"]
        for key in ("train/loss", "train/mae_bpm", "val/loss", "val/mae_bpm"):
            val = metrics.get(key)
            if val is not None:
                parts.append(f"{key}: {val:.4f}")
        print("  |  ".join(parts))

    def on_fit_end(self, trainer, pl_module):
        if not self.best:
            return
        print("\n── Best validation metrics ──────────────────")
        for key, val in self.best.items():
            print(f"  {key}: {val:.4f}")
        print("─────────────────────────────────────────────")


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
        callbacks=callbacks,
        logger=CSVLogger("logs", name="tempo"),
        enable_progress_bar=True,
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
