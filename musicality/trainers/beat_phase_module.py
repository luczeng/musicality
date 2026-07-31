"""PyTorch Lightning module for frame-level beat-phase detection (beat/one/four)."""

import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from musicality.losses import beat_phase_loss


def align_time(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop the longer of ``(logits, target)`` along the time axis to match the shorter.

    ``TCNTempoNet``'s frame-level output length (from ``torchaudio.MelSpectrogram``)
    isn't guaranteed to exactly equal :class:`~musicality.loaders.beat_dataset.BeatDataset`'s
    ``n_frames`` — off-by-one in practice. Cropping (rather than padding) keeps both
    sides comparing only real, non-fabricated frames.

    :param logits: Model output, shape ``(B, 3, T_logits)``.
    :param target: Ground-truth target, shape ``(B, 4, T_target)``.
    :returns: ``(logits, target)`` both cropped to ``(..., min(T_logits, T_target))``.
    """

    t = min(logits.shape[-1], target.shape[-1])

    return logits[..., :t], target[..., :t]


def frame_accuracy(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Coarse frame-level binary accuracy, for training-time monitoring only.

    Thresholds both the predicted probability and the (Gaussian-smeared) target
    at ``threshold`` and checks agreement. This is not the F-measure evaluation
    the project ultimately cares about (see the beat-phase plan's evaluation
    step) — it's a cheap per-epoch signal, dominated by the true-negative rate
    since beat/one/four frames are a small minority.

    :param probs: Predicted probabilities (post-sigmoid), shape ``(B, T)``.
    :param target: Ground-truth (possibly smeared) target, shape ``(B, T)``.
    :param mask: Optional per-frame mask; masked-out frames are excluded from
        the average. Shape ``(B, T)``.
    :param threshold: Decision threshold applied to both ``probs`` and ``target``.
    :returns: Scalar accuracy in ``[0, 1]``, shape ``()``.
    """

    agree = ((probs > threshold) == (target > threshold)).float()

    if mask is None:
        return agree.mean()

    denom = mask.sum().clamp(min=1.0)

    return (agree * mask).sum() / denom


class BeatPhaseModule(L.LightningModule):
    """LightningModule wrapping a frame-level beat-phase model (beat/one/four).

    The backbone is instantiated with ``frame_level=True`` and ``n_outputs=3``
    regardless of what the config says, mirroring how :class:`~musicality.trainers.tempo_module.TempoModule`
    overrides ``n_outputs`` for classification mode.

    :param model: DictConfig for instantiating the backbone (e.g. ``TCNTempoNet``).
    :param pos_weight: Positive-class weight passed to :func:`~musicality.losses.beat_phase_loss`.
        Scalar (shared across heads) or a 3-element sequence (per-head).
    :param lr: Learning rate.
    :param weight_decay: L2 regularisation.
    :param threshold: Sigmoid/target threshold used only for the logged accuracy
        metrics, not for the loss itself.
    """

    def __init__(
        self,
        model: DictConfig,
        pos_weight: float | list[float] = 8.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        model_cfg = OmegaConf.to_container(model, resolve=True)
        model_cfg.pop("arch", None)
        model_cfg["frame_level"] = True
        model_cfg["n_outputs"] = 3

        self.model = instantiate(model_cfg)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(wav)

    def _step(self, batch, stage: str):

        wav, target = batch
        logits = self(wav)
        logits, target = align_time(logits, target)

        loss = beat_phase_loss(logits, target, pos_weight=self.hparams.pos_weight)
        probs = torch.sigmoid(logits)

        beat_p, one_p, four_p = probs[:, 0], probs[:, 1], probs[:, 2]
        beat_y, one_y, four_y, mask = (
            target[:, 0],
            target[:, 1],
            target[:, 2],
            target[:, 3],
        )

        log_kw = dict(on_step=False, on_epoch=True)
        threshold = self.hparams.threshold
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        self.log(
            f"{stage}/acc_beat",
            frame_accuracy(beat_p, beat_y, threshold=threshold),
            **log_kw,
        )
        self.log(
            f"{stage}/acc_one",
            frame_accuracy(one_p, one_y, mask=mask, threshold=threshold),
            **log_kw,
        )
        self.log(
            f"{stage}/acc_four",
            frame_accuracy(four_p, four_y, mask=mask, threshold=threshold),
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
