#!/usr/bin/env python3
"""Batch-train the beat-phase model across a list of learning rates and
compare their best validation metrics side by side.

Builds the dataloaders once (learning rate doesn't affect dataset/split/
augmentation config) and re-runs model/trainer construction + `fit()` per
learning rate, so the sweep only pays the dataset-loading cost once. Every
run reuses the same seed before model init, so the learning rate is the only
thing that varies between runs. Each run gets its own checkpoint directory
and W&B run name/tag so they don't collide or get lost in one run.

Usage
-----
    # sweep three learning rates on the default (ballroom) config
    uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3

    # pass through any other Hydra override to every run in the sweep
    uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3 \\
        --overrides trainer.max_epochs=5 train_subsample=0.2 data.name=ballroom
"""

import argparse

import lightning as L
import wandb
from hydra import compose, initialize
from omegaconf import DictConfig

import musicality.dataformats as dataformats
from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.trainers.train_beat_phase import (
    build_callbacks,
    build_dataloaders,
    build_module,
    build_trainer,
)

SEED = 42


def _compose(overrides: list[str]) -> DictConfig:
    """Compose ``beat_train`` with ``overrides``, then patch ``data.data_home``.

    The yaml default for ``data.data_home`` is a ``${hydra:runtime.cwd}``
    interpolation, which only resolves inside a ``@hydra.main``-managed run
    (it needs `HydraConfig` to be set). This script uses the Compose API
    directly instead, so replace it with an equivalent plain path — unless
    the caller already overrode ``data.data_home`` explicitly.
    """

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="beat_train", overrides=overrides)

    if not any(o.startswith("data.data_home=") for o in overrides):
        cfg.data.data_home = str(
            dataformats.ROOT / dataformats.load().data_dir / cfg.data.name
        )

    return cfg


def run_one(
    cfg: DictConfig, train_loader, val_loader, n_train: int, n_val: int
) -> dict:
    """Train one lr's worth of the sweep and return its best validation metrics."""

    L.seed_everything(SEED)

    module = build_module(cfg)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks)

    trainer.logger.experiment.config.update(
        {
            "data/n_train": n_train,
            "data/n_val": n_val,
            "model/arch": cfg.model.get("arch"),
            "lr": cfg.lr,
        }
    )

    trainer.fit(module, train_loader, val_loader)
    wandb.finish()

    printer = next(c for c in callbacks if isinstance(c, BestMetricsPrinter))

    return dict(printer.best)


def print_summary(results: dict[float, dict]) -> None:

    keys = sorted({k for best in results.values() for k in best})
    if not keys:
        print("[sweep_lr] no metrics recorded — nothing to summarize")
        return

    print(f"\n{'lr':>10}  " + "  ".join(f"{k:>14}" for k in keys))
    for lr in sorted(results):
        row = "  ".join(f"{results[lr].get(k, float('nan')):>14.4f}" for k in keys)
        print(f"{lr:>10.2e}  {row}")

    if "val/loss" in keys:
        best_lr = min(results, key=lambda lr: results[lr].get("val/loss", float("inf")))
        print(
            f"\nBest lr by val/loss: {best_lr:.2e} (val/loss={results[best_lr]['val/loss']:.4f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Batch-train the beat-phase model over a list of learning rates."
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        required=True,
        help="Learning rates to try, e.g. --lrs 1e-4 5e-4 1e-3",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides applied to every run, e.g. trainer.max_epochs=5 data.name=ballroom",
    )
    args = parser.parse_args()

    base_cfg = _compose(args.overrides)

    print(
        f"[sweep_lr] dataset={base_cfg.data.name}  group_size={base_cfg.group_size}  lrs={args.lrs}"
    )

    L.seed_everything(SEED)
    train_loader, val_loader, n_train, n_val = build_dataloaders(base_cfg)

    results = {}
    for lr in args.lrs:
        print(f"\n{'=' * 60}\n[sweep_lr] lr={lr}\n{'=' * 60}")

        cfg = _compose(
            args.overrides
            + [
                f"lr={lr}",
                f"checkpoint_dir=checkpoints_beat/lr_sweep/lr_{lr}/",
                f"wandb.run_name=lr_sweep-{lr}",
                "wandb.tags=[lr_sweep]",
            ]
        )

        results[lr] = run_one(cfg, train_loader, val_loader, n_train, n_val)

    print(f"\n{'=' * 60}\n[sweep_lr] summary\n{'=' * 60}")
    print_summary(results)


if __name__ == "__main__":
    main()
