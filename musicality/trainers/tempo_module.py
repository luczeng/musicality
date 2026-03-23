"""PyTorch Lightning module for tempo estimation."""

import torch
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


def tempo_acc1(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: float = 0.08,
    factors: tuple = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """MIREX Accuracy 1: fraction of predictions within ``tolerance`` of any
    octave-equivalent tempo.

    A prediction is correct if it is within ``tolerance × factor × target``
    for any factor in ``factors``. The default 8% tolerance matches the MIREX
    evaluation standard.

    :param pred: Predicted BPM values, shape ``(B,)``.
    :param target: Ground-truth BPM values, shape ``(B,)``.
    :param tolerance: Relative tolerance (default: 0.08 → ±8%).
    :param factors: Metrical multiples to consider (default: 0.5×, 1×, 2×).
    :returns: Fraction of correct predictions in ``[0, 1]``.
    :rtype: torch.Tensor
    """

    correct = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)

    for f in factors:
        correct |= (pred - f * target).abs() < tolerance * f * target

    return correct.float().mean()


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
        acc1 = tempo_acc1(pred, tempo)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc1", acc1, prog_bar=True, on_step=False, on_epoch=True)

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
