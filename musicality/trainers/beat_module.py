"""PyTorch Lightning module for frame-level beat-only detection (no bar-position heads)."""

import torch
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from musicality.metrics.frame_accuracy import frame_accuracy
from musicality.trainers.beat_phase_module import align_time


class BeatModule(L.LightningModule):
    """LightningModule wrapping a frame-level, single-head beat detector.

    Sibling of :class:`~musicality.trainers.beat_phase_module.BeatPhaseModule`
    for the case where only the ``beat`` head is wanted — e.g. to establish a
    beat-only baseline, or to train on datasets without bar-position
    annotations. The backbone is instantiated with ``frame_level=True`` and
    ``n_outputs=1`` regardless of what the config says, mirroring how
    :class:`~musicality.trainers.tempo_module.TempoModule` overrides
    ``n_outputs`` for classification mode.

    Trains against :class:`~musicality.loaders.beat_dataset.BeatDataset`'s
    ``beat`` channel only — the ``one``/``last``/``mask`` channels are read
    but ignored, so this works on any beat-annotated dataset regardless of
    whether it carries bar-position annotations.

    :param model: DictConfig for instantiating the backbone (e.g. ``TCNTempoNet``).
    :param pos_weight: Positive-class weight passed to ``BCEWithLogitsLoss``,
        compensating for beat frames being a small fraction of all frames.
    :param lr: Learning rate.
    :param weight_decay: L2 regularisation.
    :param threshold: Sigmoid/target threshold used only for the logged accuracy
        metric, not for the loss itself.
    :param balanced: Passed through to :func:`~musicality.metrics.frame_accuracy.frame_accuracy`
        for the logged accuracy metric. Defaults to ``True`` since beat frames
        are a small minority and a pooled mean is dominated by the
        true-negative rate.
    """

    def __init__(
        self,
        model: DictConfig,
        pos_weight: float = 6.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        threshold: float = 0.5,
        balanced: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        model_cfg = OmegaConf.to_container(model, resolve=True)
        model_cfg.pop("arch", None)
        model_cfg["frame_level"] = True
        model_cfg["n_outputs"] = 1

        self.model = instantiate(model_cfg)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(wav)

    def _step(self, batch, stage: str):

        wav, target = batch
        logits = self(wav)
        beat_y = target[:, 0]
        logits, beat_y = align_time(logits, beat_y)

        pos_weight = torch.as_tensor(
            self.hparams.pos_weight, device=logits.device, dtype=logits.dtype
        )
        loss = F.binary_cross_entropy_with_logits(logits, beat_y, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)

        log_kw = dict(on_step=False, on_epoch=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        self.log(
            f"{stage}/acc_beat",
            frame_accuracy(
                probs,
                beat_y,
                threshold=self.hparams.threshold,
                balanced=self.hparams.balanced,
            ),
            **log_kw,
        )

        return loss, probs

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

        loss, probs = self._step(batch, "val")
        return {"probs": probs.detach().cpu()}

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
