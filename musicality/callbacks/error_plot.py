"""Callback that plots prediction error vs tempo and logs it to W&B."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import lightning as L


class ErrorVsTempoPlot(L.Callback):
    """Scatter-plots |pred - target| vs target tempo at the end of each validation epoch.

    Points are coloured green (pass MIREX acc1) or red (fail). The dashed curve
    shows the 8% tolerance boundary so you can immediately see which tempo
    regions are hardest for the model.
    """

    def on_validation_epoch_start(self, trainer, pl_module):
        self._preds = []
        self._targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None:
            return
        self._preds.append(outputs["pred"])
        self._targets.append(outputs["target"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._preds or trainer.sanity_checking:
            return

        preds = torch.cat(self._preds).numpy()
        targets = torch.cat(self._targets).numpy()
        errors = np.abs(preds - targets)

        # MIREX acc1: correct if within 8% of any factor × target
        correct = np.zeros(len(preds), dtype=bool)
        for f in (0.5, 1.0, 2.0):
            correct |= errors < 0.08 * f * targets

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.scatter(targets[correct], errors[correct], s=20, alpha=0.7,
                   color="#2ecc71", label="pass acc1")
        ax.scatter(targets[~correct], errors[~correct], s=20, alpha=0.7,
                   color="#e74c3c", label="fail acc1")

        t_range = np.linspace(targets.min() * 0.9, targets.max() * 1.1, 200)
        ax.plot(t_range, 0.08 * t_range, "k--", linewidth=1, label="8% tolerance")

        ax.set_xlabel("Target tempo (BPM)")
        ax.set_ylabel("Absolute error (BPM)")
        ax.set_title(f"Error vs tempo — epoch {trainer.current_epoch}")
        ax.legend(fontsize=8)
        fig.tight_layout()

        # Write to summary so the image is overwritten each epoch rather than
        # accumulating a new entry per step in the media panel.
        trainer.logger.experiment.summary["val/error_vs_tempo"] = wandb.Image(fig)
        plt.close(fig)
