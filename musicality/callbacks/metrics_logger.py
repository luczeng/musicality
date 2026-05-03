"""Terminal metrics logging callback."""

import lightning as L


class BestMetricsPrinter(L.Callback):
    """Prints metrics to the terminal each epoch and summarises the best at the end."""

    def __init__(self):
        self.best = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in ("val/loss", "val/mae", "val/acc1"):
            val = metrics.get(key)
            if val is None:
                continue
            if key not in self.best:
                self.best[key] = val.item()
                continue
            better = val > self.best[key] if key == "val/acc1" else val < self.best[key]
            if better:
                self.best[key] = val.item()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        parts = [f"epoch {epoch:>3}"]
        for key in ("train/loss", "train/mae", "train/acc1", "val/loss", "val/mae", "val/acc1"):
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
