"""PyTorch Lightning module for frame-level beat-phase detection (beat/one/last)."""

import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from musicality.losses import beat_phase_loss, beat_position_loss
from musicality.metrics.frame_accuracy import frame_accuracy


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


class BeatPhaseModule(L.LightningModule):
    """LightningModule wrapping a frame-level beat-phase model (beat/one/last).

    Two output parameterizations, selected by ``group_size``:

    - ``group_size=None`` (default): the original three-channel head —
      ``beat``/``one``/``last`` as independent sigmoids, trained with
      :func:`~musicality.losses.beat_phase_loss`. Pairs with
      :class:`~musicality.loaders.beat_dataset.BeatDataset`'s
      ``target_layout="one_last"``.
    - ``group_size=G``: ``1 + G`` channels — ``beat`` as a sigmoid, then a
      softmax over the ``G`` bar positions, trained with
      :func:`~musicality.losses.beat_position_loss`. Pairs with
      ``target_layout="positions"``. Positions 2..G-1 gain their own
      supervised logits, so "is this a 1 or a 3?" becomes a question the model
      is actually asked — see docs/beat_phase_improvement_review.md section 3.

    Either way ``n_outputs`` is forced regardless of what the config says,
    mirroring how :class:`~musicality.trainers.tempo_module.TempoModule`
    overrides it for classification mode.

    :param model: DictConfig for instantiating the backbone (e.g. ``TCNTempoNet``).
    :param pos_weight: Positive-class weight passed to :func:`~musicality.losses.beat_phase_loss`.
        Scalar (shared across heads) or a 3-element sequence (per-head).
    :param phase_conditioning: Passed to :func:`~musicality.losses.beat_phase_loss`
        — ``"mask"`` supervises the one/last heads on every frame,
        ``"beat"`` only on frames at or near a beat, which is where
        :func:`musicality.postprocess.label_bar_position` actually reads them.
        Must be retuned together with ``pos_weight``: the class imbalance the
        latter compensates for largely disappears under ``"beat"``.
    :param lr: Learning rate.
    :param weight_decay: L2 regularisation.
    :param threshold: Sigmoid/target threshold used only for the logged accuracy
        metrics, not for the loss itself.
    :param balanced: Passed through to :func:`~musicality.metrics.frame_accuracy.frame_accuracy`
        for the logged accuracy metrics. Defaults to ``True`` since beat/one/last
        frames are a small minority and a pooled mean is dominated by the
        true-negative rate.
    :param group_size: Number of bar positions. ``None`` keeps the original
        one/last sigmoid head; an integer switches to a ``group_size``-way
        softmax over positions. Saved to the checkpoint's hyperparameters, so
        :func:`~musicality.inference.load_module` reconstructs the right head
        and downstream code can tell the two apart.
    :param check_val_every_n_epoch: How often the trainer actually runs
        validation (``cfg.trainer.check_val_every_n_epoch``). The
        ``ReduceLROnPlateau`` scheduler needs this as its ``frequency`` —
        Lightning otherwise tries to step it (and read ``val/loss``) every
        epoch regardless of how often validation runs, raising
        ``MisconfigurationException`` on any epoch without a fresh value.
    :param task: Saved into the checkpoint's hyperparameters for
        :func:`~musicality.inference.detect_task` to read back at eval/inference
        time. Always ``"beat_phase"`` for this class; exists as a parameter
        (rather than hardcoded) so ``configs/beat_train.yaml``'s ``task:`` field
        is the visible, single source of truth for what a checkpoint is.
    """

    def __init__(
        self,
        model: DictConfig,
        pos_weight: float | list[float] = 8.0,
        phase_conditioning: str = "mask",
        group_size: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        threshold: float = 0.5,
        balanced: bool = True,
        check_val_every_n_epoch: int = 1,
        task: str = "beat_phase",
    ):
        super().__init__()
        self.save_hyperparameters()

        model_cfg = OmegaConf.to_container(model, resolve=True)
        model_cfg.pop("arch", None)
        if group_size is not None and group_size < 2:
            raise ValueError(f"group_size must be >= 2 or None, got {group_size}")

        model_cfg["frame_level"] = True
        model_cfg["n_outputs"] = 3 if group_size is None else 1 + group_size

        self.model = instantiate(model_cfg)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(wav)

    def _position_step(self, logits, target, stage: str):
        """Loss + logged metrics for the ``group_size``-way softmax head."""

        loss = beat_position_loss(
            logits,
            target,
            pos_weight=self.hparams.pos_weight,
            phase_conditioning=self.hparams.phase_conditioning,
        )

        beat_p = torch.sigmoid(logits[:, 0])
        beat_y, position_y, mask = target[:, 0], target[:, 1:-1], target[:, -1]

        log_kw = dict(on_step=False, on_epoch=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        self.log(
            f"{stage}/acc_beat",
            frame_accuracy(
                beat_p,
                beat_y,
                threshold=self.hparams.threshold,
                balanced=self.hparams.balanced,
            ),
            **log_kw,
        )

        # Position accuracy at the frames the decoder actually reads: argmax
        # over the softmax against argmax of the target, weighted by how much
        # beat there is. Unlike the one/last frame accuracies this replaces,
        # a model that never fires cannot score well on it — it has to pick a
        # position, and picking the wrong one is a miss.
        correct = (logits[:, 1:].argmax(dim=1) == position_y.argmax(dim=1)).to(
            logits.dtype
        )
        weight = mask * beat_y
        self.log(
            f"{stage}/acc_position",
            (correct * weight).sum() / weight.sum().clamp(min=1.0),
            **log_kw,
        )

        return loss, torch.softmax(logits[:, 1:], dim=1)

    def _step(self, batch, stage: str):

        wav, target = batch
        logits = self(wav)
        logits, target = align_time(logits, target)

        if self.hparams.group_size is not None:
            return self._position_step(logits, target, stage)

        loss = beat_phase_loss(
            logits,
            target,
            pos_weight=self.hparams.pos_weight,
            phase_conditioning=self.hparams.phase_conditioning,
        )
        probs = torch.sigmoid(logits)

        beat_p, one_p, last_p = probs[:, 0], probs[:, 1], probs[:, 2]
        beat_y, one_y, last_y, mask = (
            target[:, 0],
            target[:, 1],
            target[:, 2],
            target[:, 3],
        )

        # `beat_y` is exactly 0 more than ~5 frames from a beat (see
        # beat_phase_loss), so this doubles as a hard "near a beat" gate with
        # no threshold to invent. Restricts acc_one/acc_last to the frames
        # label_bar_position actually reads at inference, independent of
        # `phase_conditioning` (which only controls what the *loss* weights).
        beat_mask = mask * beat_y

        log_kw = dict(on_step=False, on_epoch=True)
        threshold = self.hparams.threshold
        balanced = self.hparams.balanced
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        self.log(
            f"{stage}/acc_beat",
            frame_accuracy(beat_p, beat_y, threshold=threshold, balanced=balanced),
            **log_kw,
        )
        self.log(
            f"{stage}/acc_one",
            frame_accuracy(
                one_p, one_y, mask=beat_mask, threshold=threshold, balanced=balanced
            ),
            **log_kw,
        )
        self.log(
            f"{stage}/acc_last",
            frame_accuracy(
                last_p, last_y, mask=beat_mask, threshold=threshold, balanced=balanced
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": self.hparams.check_val_every_n_epoch,
            },
        }
