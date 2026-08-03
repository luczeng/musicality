"""Tests for musicality.callbacks.metrics_logger.BestMetricsPrinter."""

import torch
import pytest

from musicality.callbacks.metrics_logger import BestMetricsPrinter


class _FakeTrainer:
    def __init__(
        self, metrics: dict, current_epoch: int = 0, sanity_checking: bool = False
    ):
        self.callback_metrics = {k: torch.tensor(v) for k, v in metrics.items()}
        self.current_epoch = current_epoch
        self.sanity_checking = sanity_checking


class TestBestMetricsPrinter:
    def test_default_keys_used_when_omitted(self):
        cb = BestMetricsPrinter()
        assert "val/loss" in cb.keys
        assert "val/mae" in cb.keys

    def test_custom_keys_tracked_instead_of_default(self):
        cb = BestMetricsPrinter(keys=("val/acc_beat", "val/acc_one"))
        trainer = _FakeTrainer({"val/acc_beat": 0.9, "val/acc_one": 0.5})
        cb.on_validation_epoch_end(trainer, None)
        assert cb.best == pytest.approx({"val/acc_beat": 0.9, "val/acc_one": 0.5})

    def test_lower_is_better_for_loss(self):
        cb = BestMetricsPrinter(keys=("val/loss",))
        cb.on_validation_epoch_end(_FakeTrainer({"val/loss": 1.0}), None)
        cb.on_validation_epoch_end(_FakeTrainer({"val/loss": 0.5}), None)
        assert cb.best["val/loss"] == 0.5

    def test_higher_is_better_for_accuracy(self):
        cb = BestMetricsPrinter(keys=("val/acc_beat",))
        cb.on_validation_epoch_end(_FakeTrainer({"val/acc_beat": 0.5}), None)
        cb.on_validation_epoch_end(_FakeTrainer({"val/acc_beat": 0.9}), None)
        assert cb.best["val/acc_beat"] == pytest.approx(0.9)

    def test_missing_keys_are_skipped(self):
        cb = BestMetricsPrinter(keys=("val/loss", "val/does_not_exist"))
        cb.on_validation_epoch_end(_FakeTrainer({"val/loss": 1.0}), None)
        assert "val/does_not_exist" not in cb.best

    def test_sanity_check_pass_is_ignored(self):
        """The pre-training sanity-check validation pass must not seed `best`
        with untrained-model metrics that real training may never beat."""

        cb = BestMetricsPrinter(keys=("val/loss",))
        cb.on_validation_epoch_end(
            _FakeTrainer({"val/loss": 0.01}, sanity_checking=True), None
        )
        assert cb.best == {}

        cb.on_validation_epoch_end(_FakeTrainer({"val/loss": 5.0}), None)
        assert cb.best["val/loss"] == pytest.approx(5.0)
