#!/usr/bin/env python3
"""Batch-train the beat-phase model across a list of learning rates and
compare their best validation metrics side by side.

Builds the dataloaders once (learning rate doesn't affect dataset/split/
augmentation config) and re-runs model/trainer construction + `fit()` per
learning rate, so the sweep only pays the dataset-loading cost once. Every
run reuses the same seed before model init, so the learning rate is the only
thing that varies between runs.

Layout
------
Every sweep writes into its own directory, stamped with the moment it started::

    checkpoints_beat/lr_sweep-20260918-141530/
        lr_0.0008/
            beat-phase-epoch91-valloss1.4794.ckpt
            training_report.json
        lr_0.002/
            beat-phase-epoch95-valloss1.4757.ckpt
            training_report.json
        sweep_results.csv

The stamp is what lets two sweeps run on the same day. Without it the path was
a constant (`checkpoints_beat/lr_sweep/lr_<lr>/`), so a second sweep wrote its
checkpoints on top of the first's: `save_top_k` is per-run bookkeeping and
cannot prune another run's files, so the directory accumulated one group of
`save_top_k` checkpoints per sweep and kept only the last sweep's
`training_report.json` — or none at all, when that last sweep was interrupted
before `on_fit_end`.

Usage
-----
    # sweep three learning rates on the default config
    uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3

    # pass through any other Hydra override to every run in the sweep
    uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3 \\
        --overrides trainer.max_epochs=5 train_subsample=0.2 data.input=ballroom

    # name the sweep yourself instead of taking the timestamp
    uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 --sweep-id deeper-trunk

    # write the comparison table somewhere other than inside the sweep directory
    uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3 --output ~/sweeps/lr.csv
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import lightning as L
import wandb
from hydra import compose, initialize
from omegaconf import DictConfig

from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.trainers.train_beat_phase import (
    build_callbacks,
    build_module,
    build_trainer,
)
from musicality.trainers.common import build_beat_dataloaders

SEED = 42


def _compose(overrides: list[str]) -> DictConfig:
    """Compose ``beat_train`` with ``overrides``."""

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="beat_train", overrides=overrides)

    return cfg


def sweep_directory(cfg: DictConfig, sweep_id: str) -> Path:
    """Directory holding every run of one sweep, below the config's ``checkpoint_dir``.

    Derived from the config rather than hardcoded, so ``--overrides
    checkpoint_dir=...`` still decides where a sweep lands.
    """

    return Path(cfg.checkpoint_dir) / f"lr_sweep-{sweep_id}"


def run_overrides(sweep_dir: Path, sweep_id: str, lr: float) -> list[str]:
    """Hydra overrides placing one learning rate's run inside *sweep_dir*.

    `build_checkpoint_callback` names a run's subdirectory after
    ``wandb.run_name``, so that key sets both the leaf directory and the name
    the run carries in W&B. The sweep's identity rides along as a tag instead
    of in the name: it keeps the paths free of a stamp repeated at two levels,
    and a tag is what the W&B UI filters on.
    """

    return [
        f"lr={lr}",
        f"checkpoint_dir={sweep_dir}/",
        f"wandb.run_name=lr_{lr}",
        f"wandb.tags=[lr_sweep,'sweep-{sweep_id}']",
    ]


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


def _summary_lines(results: dict[float, dict], keys: list[str]) -> list[str]:
    """Render the lr-vs-metric comparison table as plain-text lines (shared by
    `print_summary` and `write_summary`'s non-CSV branch, so the two never drift)."""

    lines = [f"{'lr':>10}  " + "  ".join(f"{k:>14}" for k in keys)]
    for lr in sorted(results):
        row = "  ".join(f"{results[lr].get(k, float('nan')):>14.4f}" for k in keys)
        lines.append(f"{lr:>10.2e}  {row}")

    if "val/loss" in keys:
        best_lr = min(results, key=lambda lr: results[lr].get("val/loss", float("inf")))
        lines.append("")
        lines.append(
            f"Best lr by val/loss: {best_lr:.2e} (val/loss={results[best_lr]['val/loss']:.4f})"
        )

    return lines


def print_summary(results: dict[float, dict]) -> None:

    keys = sorted({k for best in results.values() for k in best})
    if not keys:
        print("[sweep_lr] no metrics recorded — nothing to summarize")
        return

    print("\n" + "\n".join(_summary_lines(results, keys)))


def write_summary(results: dict[float, dict], path: Path) -> None:
    """Write the comparison table to disk: CSV if `path` ends in `.csv`
    (one row per lr, one column per metric — easiest to load elsewhere for
    plotting), otherwise the same plain-text table `print_summary` prints."""

    keys = sorted({k for best in results.values() for k in best})
    if not keys:
        print("[sweep_lr] no metrics recorded — nothing to write")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lr", *keys])
            for lr in sorted(results):
                writer.writerow([lr, *(results[lr].get(k, "") for k in keys)])
    else:
        path.write_text("\n".join(_summary_lines(results, keys)) + "\n")

    print(f"[sweep_lr] summary written to {path}")


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
        help="Extra Hydra overrides applied to every run, e.g. trainer.max_epochs=5 data.input=ballroom",
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Name for this sweep's directory (default: a %%Y%%m%%d-%%H%%M%%S "
        "timestamp, so two sweeps on the same day don't overwrite each other)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the comparison table to this path (.csv for a CSV file, any "
        "other extension for plain text). Defaults to sweep_results.csv inside "
        "the sweep directory.",
    )
    args = parser.parse_args()

    sweep_id = args.sweep_id or datetime.now().strftime("%Y%m%d-%H%M%S")

    base_cfg = _compose(args.overrides)
    sweep_dir = sweep_directory(base_cfg, sweep_id)

    print(
        f"[sweep_lr] dataset={base_cfg.data.input}  group_size={base_cfg.group_size}  lrs={args.lrs}"
    )
    print(f"[sweep_lr] sweep_dir={sweep_dir}")

    L.seed_everything(SEED)
    train_loader, val_loader, n_train, n_val = build_beat_dataloaders(base_cfg)

    results = {}
    for lr in args.lrs:
        print(f"\n{'=' * 60}\n[sweep_lr] lr={lr}\n{'=' * 60}")

        cfg = _compose(args.overrides + run_overrides(sweep_dir, sweep_id, lr))

        results[lr] = run_one(cfg, train_loader, val_loader, n_train, n_val)

    print(f"\n{'=' * 60}\n[sweep_lr] summary\n{'=' * 60}")
    print_summary(results)

    # Unconditional, and inside the sweep directory: a sweep's own comparison
    # table belongs beside the checkpoints it ranks, not only in a terminal.
    write_summary(results, args.output or sweep_dir / "sweep_results.csv")


if __name__ == "__main__":
    main()
