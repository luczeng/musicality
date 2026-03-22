"""PyTorch Lightning module for tempo estimation."""

import torch
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from hydra.utils import instantiate


def relative_tempo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    factors: tuple = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """MAE loss that is invariant to metrical octave errors.

    For each sample, computes the absolute error between the prediction and
    each factor × target, then takes the minimum. This means predicting double
    or half the annotated tempo incurs zero penalty — both are musically valid
    metrical interpretations of the same groove.

    :param pred: Predicted BPM values, shape ``(B,)``.
    :param target: Ground-truth BPM values, shape ``(B,)``.
    :param factors: Metrical multiples to consider (default: 0.5×, 1×, 2×).
    :returns: Scalar mean relative tempo loss.
    :rtype: torch.Tensor
    """

    errors = torch.stack(
        [(pred - f * target).abs() for f in factors],
        dim=1,
    )  # (B, n_factors)

    return errors.min(dim=1).values.mean()


class TempoModule(L.LightningModule):
    """LightningModule wrapping a tempo regression model.

    :param model: DictConfig for instantiating the backbone (via hydra.utils.instantiate).
    :param lr: Learning rate.
    :param weight_decay: L2 regularisation.
    """

    def __init__(self, model: DictConfig, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch, stage: str):

        x, tempo = batch
        pred = self(x)

        loss = relative_tempo_loss(pred, tempo)
        mae = (pred - tempo).abs().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/mae", mae, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):

        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):

        self._step(batch, "val")

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
