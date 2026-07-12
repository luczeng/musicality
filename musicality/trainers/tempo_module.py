"""PyTorch Lightning module for tempo estimation."""

import torch
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from musicality.losses import (
    absolute_tempo_loss,
    classification_tempo_loss,
    relative_tempo_loss,
)


_REGRESSION_LOSSES = {
    "relative": relative_tempo_loss,
    "absolute": absolute_tempo_loss,
}
_VALID_LOSSES = set(_REGRESSION_LOSSES) | {"classification"}


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
    :returns: Fraction of correct predictions in ``[0, 1]``, shape ``()``.
    """

    correct = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)

    for f in factors:
        correct |= (pred - f * target).abs() < tolerance * f * target

    return correct.float().mean()


class TempoModule(L.LightningModule):
    """LightningModule wrapping a tempo regression or classification model.

    Three loss modes are supported:
      * ``"absolute"`` — plain MAE between predicted and true BPM.
      * ``"relative"`` — octave-invariant MAE.
      * ``"classification"`` — softmax over BPM bins with a Gaussian soft
        target. Requires the ``classification`` config section.

    For classification mode the model's ``n_outputs`` is overridden to
    ``classification.n_bins`` automatically.

    :param model: DictConfig for instantiating the backbone.
    :param loss: Loss name — ``"absolute"``, ``"relative"``, or ``"classification"``.
    :param classification: Required when ``loss == "classification"``. Must have
        ``bpm_min``, ``bpm_max``, ``n_bins``, ``sigma``.
    :param lr: Learning rate.
    :param weight_decay: L2 regularisation.
    """

    def __init__(
        self,
        model: DictConfig,
        loss: str = "absolute",
        classification: DictConfig | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        if loss not in _VALID_LOSSES:
            raise ValueError(
                f"Unknown loss '{loss}'. Choose from: {sorted(_VALID_LOSSES)}"
            )
        self.loss_name = loss

        model_cfg = OmegaConf.to_container(model, resolve=True)
        model_cfg.pop("arch", None)

        if loss == "classification":
            if classification is None:
                raise ValueError(
                    "loss='classification' requires a 'classification' config section"
                )
            self.bpm_min = float(classification.bpm_min)
            self.bpm_max = float(classification.bpm_max)
            self.n_bins = int(classification.n_bins)
            self.sigma = float(classification.sigma)
            bin_centers = torch.linspace(self.bpm_min, self.bpm_max, self.n_bins)
            self.register_buffer("bin_centers", bin_centers)
            model_cfg["n_outputs"] = self.n_bins

        self.model = instantiate(model_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _decode(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode classification logits into BPM predictions.

        :param logits: Raw model output, shape ``(B, n_bins)``.
        :returns: Tuple of ``(argmax_bpm, expected_bpm)``, each shape ``(B,)``.
        """
        probs = F.softmax(logits, dim=-1)
        pred_argmax = self.bin_centers[probs.argmax(dim=-1)]
        pred_expected = (probs * self.bin_centers).sum(dim=-1)
        return pred_argmax, pred_expected

    def _step_classification(self, batch, stage: str):
        x, tempo = batch
        logits = self(x)
        loss = classification_tempo_loss(logits, tempo, self.bin_centers, self.sigma)
        pred_argmax, pred_expected = self._decode(logits)

        mae_argmax = (pred_argmax - tempo).abs().mean()
        mae_expected = (pred_expected - tempo).abs().mean()
        acc1_argmax = tempo_acc1(pred_argmax, tempo)
        acc1_expected = tempo_acc1(pred_expected, tempo)

        log_kw = dict(on_step=False, on_epoch=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        self.log(f"{stage}/mae_argmax", mae_argmax, **log_kw)
        self.log(f"{stage}/mae_expected", mae_expected, **log_kw)
        self.log(f"{stage}/acc1_argmax", acc1_argmax, prog_bar=True, **log_kw)
        self.log(f"{stage}/acc1_expected", acc1_expected, **log_kw)

        return loss, pred_expected

    def _step_regression(self, batch, stage: str):
        x, tempo = batch
        pred = self(x)
        loss_fn = _REGRESSION_LOSSES[self.loss_name]
        loss = loss_fn(pred, tempo)
        mae = (pred - tempo).abs().mean()
        acc1 = tempo_acc1(pred, tempo)

        log_kw = dict(on_step=False, on_epoch=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        self.log(f"{stage}/mae", mae, **log_kw)
        self.log(f"{stage}/acc1", acc1, prog_bar=True, **log_kw)

        return loss, pred

    def _step(self, batch, stage: str):
        if self.loss_name == "classification":
            return self._step_classification(batch, stage)
        return self._step_regression(batch, stage)

    def training_step(self, batch, batch_idx):

        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):

        _, tempo = batch
        loss, pred = self._step(batch, "val")
        return {"pred": pred.detach().cpu(), "target": tempo.detach().cpu()}

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
