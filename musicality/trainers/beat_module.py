"""PyTorch Lightning module for frame-level beat-only detection (no bar-position heads)."""

import torch
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from musicality.metrics.frame_accuracy import peak_f_measure
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
    :param threshold: Peak-picking threshold for the logged ``{stage}/f_beat``
        metric, not used by the loss itself. Note the beat-only task's tuned
        evaluation threshold is ``0.8`` (``configs/eval_beat.yaml``), so unless
        this is set to match, the training-time number is a close proxy for the
        reported ``f_beat`` rather than the identical computation.
    :param balanced: Unused since the beat head moved from
        :func:`~musicality.metrics.frame_accuracy.frame_accuracy` to
        :func:`~musicality.metrics.frame_accuracy.peak_f_measure`, which has no
        true-negative term to balance. Kept as a constructor parameter because
        it is saved in every existing checkpoint's hyperparameters and passed
        back by :func:`~musicality.inference.load_module` on load.
    :param task: Saved into the checkpoint's hyperparameters for
        :func:`~musicality.inference.detect_task` to read back at eval/inference
        time. Always ``"beat_only"`` for this class; exists as a parameter
        (rather than hardcoded) so ``configs/beat_only_train.yaml``'s ``task:``
        field is the visible, single source of truth for what a checkpoint is.
    :param check_val_every_n_epoch: How often the trainer actually runs
        validation (``cfg.trainer.check_val_every_n_epoch``). The
        ``ReduceLROnPlateau`` scheduler needs this as its ``frequency`` —
        Lightning otherwise tries to step it (and read ``val/loss``) every
        epoch regardless of how often validation runs, raising
        ``MisconfigurationException`` on any epoch without a fresh value.
    """

    def __init__(
        self,
        model: DictConfig,
        pos_weight: float = 6.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        threshold: float = 0.5,
        balanced: bool = True,
        check_val_every_n_epoch: int = 1,
        task: str = "beat_only",
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
            f"{stage}/f_beat",
            peak_f_measure(probs, beat_y, threshold=self.hparams.threshold),
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": self.hparams.check_val_every_n_epoch,
            },
        }
