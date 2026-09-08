"""Core training routine for frame-level beat-phase detection (beat/one/last)."""

import logging

import lightning as L

# Suppress Lightning's promotional tip about LitLogger (INFO-level noise)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
from omegaconf import DictConfig

from musicality.callbacks.event_metrics import (
    LOGGED_KEYS,
    PREFIX,
    EventMetricsLogger,
)
from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.losses import AUTO_POS_WEIGHT_ALPHA
from musicality.trainers.beat_phase_module import BeatPhaseModule
from musicality.trainers.common import (
    build_beat_dataloaders,
    build_checkpoint_callback,
    build_trainer,
    resolve_beat_split_refs,
)


# The `val_event/*` keys are logged only on scoring epochs (see
# EventMetricsLogger.should_run). Lightning's `callback_metrics` keeps the last
# value it saw rather than clearing it, so the per-epoch line BestMetricsPrinter
# prints repeats the most recent event metrics on the epochs in between. What
# reaches W&B does not: the results collection is reset each validation loop, so
# the logged series has one point per scoring epoch.
_TRACKED_KEYS = (
    "train/loss",
    "train/f_beat",
    "train/acc_one",
    "train/acc_last",
    "train/acc_position",
    "val/loss",
    "val/f_beat",
    "val/acc_one",
    "val/acc_last",
    "val/acc_position",
    *(f"{PREFIX}/{key}" for key in LOGGED_KEYS),
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
        pos_weight_alpha=cfg.get("pos_weight_alpha", AUTO_POS_WEIGHT_ALPHA),
        position_norm=cfg.get("position_norm", "global"),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        balanced=cfg.balanced,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        task=cfg.task,
    )


def build_callbacks(cfg: DictConfig) -> list:

    callbacks = [build_checkpoint_callback(cfg, "beat-phase")]

    # Ahead of BestMetricsPrinter, and that is load-bearing: both act in
    # `on_validation_epoch_end`, one writing metrics and the other reading
    # them. `trainer.callback_metrics` is recomputed from the live results
    # collection on every read, so a metric logged by a *later* callback in the
    # same hook is simply not there yet — with the order reversed the printer
    # tracks a best for every key except these, and the end-of-run summary is
    # missing exactly the numbers this callback exists to produce.
    event_metrics = build_event_metrics_callback(cfg)
    if event_metrics is not None:
        callbacks.append(event_metrics)

    callbacks.append(BestMetricsPrinter(keys=_TRACKED_KEYS))

    return callbacks


def build_event_metrics_callback(cfg: DictConfig) -> EventMetricsLogger | None:
    """Build the event-level validation metrics callback, or ``None`` when the
    config switches it off.

    Off by omission as well as by ``enabled: false``, so a config written
    before this block existed (or a sweep config trimmed down to essentials)
    trains exactly as it did before.

    Reads the validation refs straight from the split rather than from the
    validation dataloader: the loader hands out fixed 16-second crops, and
    these metrics are defined on full tracks.
    """

    settings = cfg.get("event_metrics") or {}
    if not settings.get("enabled", False):
        return None

    _train_refs, val_refs = resolve_beat_split_refs(cfg)

    return EventMetricsLogger(
        val_refs,
        n_tracks=settings.get("n_tracks", 50),
        every_n_epochs=settings.get("every_n_epochs", 5),
        sample_rate=cfg.data.sample_rate,
        hop_length=cfg.hop_length,
        group_size=cfg.get("group_size", 4),
        binary_only=cfg.get("binary_only", False),
        name=cfg.data.input,
    )
