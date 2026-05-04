"""Terminal metrics logging callback."""

import lightning as L


_TRACKED_KEYS = (
    "train/loss", "train/mae", "train/acc1",
    "train/mae_argmax", "train/mae_expected",
    "train/acc1_argmax", "train/acc1_expected",
    "val/loss", "val/mae", "val/acc1",
    "val/mae_argmax", "val/mae_expected",
    "val/acc1_argmax", "val/acc1_expected",
)
_LOWER_BETTER = ("loss", "mae")  # substring match against the metric key


def _is_better(key: str, val: float, current_best: float) -> bool:
    if any(s in key for s in _LOWER_BETTER):
        return val < current_best
    return val > current_best


class BestMetricsPrinter(L.Callback):
    """Prints metrics to the terminal each epoch and summarises the best at the end."""

    def __init__(self):
        self.best = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in _TRACKED_KEYS:
            if not key.startswith("val/"):
                continue
            val = metrics.get(key)
            if val is None:
                continue
            if key not in self.best:
                self.best[key] = val.item()
                continue
            if _is_better(key, val.item(), self.best[key]):
                self.best[key] = val.item()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        parts = [f"epoch {epoch:>3}"]
        for key in _TRACKED_KEYS:
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
