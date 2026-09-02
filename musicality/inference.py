"""Shared inference plumbing for beat and beat-phase checkpoints: loading a
Lightning checkpoint (auto-detecting which task it was trained for), and
running a forward pass through :mod:`musicality.postprocess`'s readout
functions.

Task type is identified entirely from a checkpoint's own saved
``hyper_parameters`` — specifically the ``task`` field declared explicitly in
``configs/beat_train.yaml``/``configs/beat_only_train.yaml`` and threaded
through :class:`~musicality.trainers.beat_phase_module.BeatPhaseModule`/
:class:`~musicality.trainers.beat_module.BeatModule`'s ``save_hyperparameters()``
call. No companion Hydra config file is read at inference time.
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from musicality.postprocess import readout, readout_beat_only
from musicality.trainers.beat_module import BeatModule
from musicality.trainers.beat_phase_module import BeatPhaseModule


_MODULE_CLASSES = {"beat_only": BeatModule, "beat_phase": BeatPhaseModule}


def detect_task(hyper_parameters: dict) -> str:
    """Task tag declared explicitly by the checkpoint's training config
    (``configs/beat_train.yaml``'s / ``configs/beat_only_train.yaml``'s
    ``task:`` field).

    :param hyper_parameters: A checkpoint's ``hyper_parameters`` dict, as
        saved by Lightning's ``save_hyperparameters()``.
    :returns: ``"beat_only"`` or ``"beat_phase"``.
    :raises KeyError: If ``task`` is missing — the checkpoint predates this
        field and must be retrained.
    :raises ValueError: If ``task`` isn't a recognized value.
    """

    if "task" not in hyper_parameters:
        raise KeyError(
            "Checkpoint has no 'task' field in its saved hyperparameters — it "
            "predates the explicit task: config field (configs/beat_train.yaml / "
            "configs/beat_only_train.yaml) and must be retrained."
        )

    task = hyper_parameters["task"]
    if task not in _MODULE_CLASSES:
        raise ValueError(
            f"Unknown task {task!r} — expected one of {list(_MODULE_CLASSES)}"
        )

    return task


def load_module(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[BeatModule | BeatPhaseModule, str]:
    """Load a beat-only or beat-phase checkpoint, auto-detecting which.

    Migrates checkpoints saved before ``TCNTempoNet``'s frame head gained a
    dropout layer (``Conv1d`` -> ``Sequential(Dropout, Conv1d)``), which
    shifted its state dict keys from ``frame_head.{weight,bias}`` to
    ``frame_head.1.{weight,bias}``. Safe to delete once no pre-dropout
    checkpoints are still in use.

    :param checkpoint_path: Path to a Lightning ``.ckpt`` file.
    :param device: Torch device the returned module is moved to.
    :returns: ``(module, task)`` — a loaded, eval-mode module already moved to
        *device*, and its task tag (see :func:`detect_task`).
    """

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    for suffix in ("weight", "bias"):
        legacy_key = f"model.frame_head.{suffix}"
        if legacy_key in state_dict:
            state_dict[f"model.frame_head.1.{suffix}"] = state_dict.pop(legacy_key)

    task = detect_task(checkpoint["hyper_parameters"])
    module = _MODULE_CLASSES[task](**checkpoint["hyper_parameters"])
    module.load_state_dict(state_dict)
    module.eval()
    module.to(device)

    return module, task


def load_track_waveform(audio_path: str, sample_rate: int) -> torch.Tensor:
    """Load a full track (no cropping/padding), mono, at ``sample_rate``.

    :param audio_path: Path to an audio file.
    :param sample_rate: Target sample rate — resampled if the file differs.
    :returns: Waveform, shape ``(1, N)``.
    """

    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        wav = T.Resample(sr, sample_rate)(wav)

    return wav


@torch.no_grad()
def run_inference(
    module: BeatModule | BeatPhaseModule,
    task: str,
    wav: torch.Tensor,
    fps: float,
    device: str | torch.device,
    beat_threshold: float = 0.3,
    min_distance_frames: int = 1,
    gate_tolerance: float = 0.2,
    anchor_threshold: float = 0.5,
    group_size: int = 4,
    decoder: str = "greedy",
    switch_penalty: float | None = None,
    advance: str = "index",
) -> list[dict] | np.ndarray:
    """Run *module* on one full-track waveform and decode it via the readout
    function matching *task*.

    :param module: A loaded, eval-mode module (see :func:`load_module`).
    :param task: ``"beat_phase"`` or ``"beat_only"`` (see :func:`detect_task`).
    :param wav: Mono waveform, shape ``(1, N)`` (e.g. from
        :func:`load_track_waveform`) — batched and moved to *device* internally.
    :param fps: Frames per second (``sample_rate / hop_length``).
    :param anchor_threshold, group_size, decoder, switch_penalty: Forwarded to
        :func:`~musicality.postprocess.readout`'s bar-position stage only —
        ignored when ``task="beat_only"``. ``decoder="global"`` uses the
        whole-track maximum-likelihood decode
        (:func:`~musicality.postprocess.label_bar_position_global`) instead of
        the greedy count-forward one, which measurably lowers phase confusion
        on the same probabilities — see docs/beat_phase_improvement_review.md.
    :returns: :func:`~musicality.postprocess.readout`'s ``list[dict]`` for
        beat-phase, or :func:`~musicality.postprocess.readout_beat_only`'s
        ``np.ndarray`` for beat-only.
    :raises ValueError: Unknown *task*.
    """

    logits = module(wav.unsqueeze(0).to(device))[0]  # (n_outputs, T') or (T',)

    # A softmax-head checkpoint (BeatPhaseModule with group_size set) emits
    # `beat` plus a group_size-way distribution over bar positions; the older
    # head emits three independent sigmoids. `group_size` in the checkpoint's
    # hyperparameters is what tells them apart — see BeatPhaseModule.
    module_group_size = getattr(module, "hparams", {}).get("group_size")
    position_probs = one_probs = last_probs = None

    if task == "beat_phase" and module_group_size is not None:
        group_size = module_group_size
        beat_probs = torch.sigmoid(logits[0]).cpu().numpy()
        position_probs = torch.softmax(logits[1:], dim=0).cpu().numpy()
        # Kept for callers and metrics that still speak one/last; the decoder
        # itself reads position_probs.
        one_probs, last_probs = position_probs[0], position_probs[-1]
    else:
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 1:
            beat_probs = probs
        else:
            beat_probs, one_probs, last_probs = probs[0], probs[1], probs[2]

    if task == "beat_phase":
        return readout(
            beat_probs,
            one_probs,
            last_probs,
            fps=fps,
            beat_threshold=beat_threshold,
            min_distance_frames=min_distance_frames,
            gate_tolerance=gate_tolerance,
            anchor_threshold=anchor_threshold,
            group_size=group_size,
            decoder=decoder,
            switch_penalty=switch_penalty,
            advance=advance,
            position_probs=position_probs,
        )

    if task == "beat_only":
        return readout_beat_only(
            beat_probs,
            fps=fps,
            beat_threshold=beat_threshold,
            min_distance_frames=min_distance_frames,
            gate_tolerance=gate_tolerance,
        )

    raise ValueError(f"Unknown task {task!r} — expected one of {list(_MODULE_CLASSES)}")
