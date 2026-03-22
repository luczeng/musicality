"""PyTorch Lightning module for tempo estimation."""

import torch
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from hydra.utils import instantiate


class TempoModule(L.LightningModule):
    """LightningModule wrapping a tempo regression model.

    Args:
        model: DictConfig for instantiating the backbone (via hydra.utils.instantiate).
        lr: Learning rate.
        weight_decay: L2 regularisation.
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
        loss = F.mse_loss(pred, tempo)

        mae = (pred - tempo).abs().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/mse", mae, prog_bar=True, on_step=False, on_epoch=True)

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
