"""On-demand beat inference for the annotator, using a trained beat-phase checkpoint.

Loads a :class:`~musicality.trainers.beat_phase_module.BeatPhaseModule` via
:func:`musicality.inference.load_module` and decodes its per-frame
beat/one/last probability curves into beat timestamps + bar positions via
:func:`musicality.inference.run_inference`, at the same sample rate / hop
length the model was trained with.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T
import yaml

import musicality.dataformats as dataformats
from musicality.inference import load_module, run_inference
from musicality.trainers.beat_phase_module import BeatPhaseModule

CHECKPOINT_ROOT = Path(__file__).parent.parent.parent / "checkpoints"
SAMPLE_RATE = 22050
HOP_LENGTH = 512

# Per-task tuned postprocessing defaults — same file tools/eval_beat.py reads.
# Only "beat_phase" is used today (CHECKPOINT_ROOT only ever discovers
# beat-phase checkpoints), but this stays a genuine task lookup rather than a
# hardcoded block so it doesn't need revisiting if checkpoint discovery is
# ever widened to include beat-only checkpoints too.
EVAL_DEFAULTS = yaml.safe_load(
    (dataformats.ROOT / "configs" / "eval_beat.yaml").read_text()
)

_LOSS_RE = re.compile(r"loss=(\d+\.\d+)")


def list_checkpoints(root: Path = CHECKPOINT_ROOT) -> list[Path]:
    """Return all ``.ckpt`` files under *root*, best (lowest val loss) first.

    Sorting relies on the ``loss=<value>.ckpt`` naming convention used by the
    training callbacks; checkpoints that don't match fall to the end.
    """

    def _loss(path: Path) -> float:
        m = _LOSS_RE.search(path.name)
        return float(m.group(1)) if m else float("inf")

    if not root.exists():
        return []

    return sorted(root.rglob("*.ckpt"), key=_loss)


def checkpoint_label(path: Path, root: Path = CHECKPOINT_ROOT) -> str:
    """Short display label for a checkpoint path, e.g. ``lr_sweep/lr_0.001/.../loss=1.6721``."""

    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path

    return str(rel.with_suffix(""))


def _fill_positions(labels: list[int | None], group_size: int) -> np.ndarray:
    """Back-fill leading unresolved (``None``) bar-position labels.

    :func:`~musicality.postprocess.label_bar_position` leaves beats before the
    first confident "one"/"last" anchor unresolved. This counts backward from
    the first resolved label to fill them in, matching the forward-counting
    convention used for the rest of the sequence. Falls back to a plain
    ``1..group_size`` cycle if no anchor was found at all.
    """

    filled: list[int | None] = list(labels)
    first_known = next((i for i, v in enumerate(filled) if v is not None), None)

    if first_known is None:
        return np.array([(i % group_size) + 1 for i in range(len(filled))], dtype=int)

    value = filled[first_known]
    for i in range(first_known - 1, -1, -1):
        value = value - 1 if value > 1 else group_size
        filled[i] = value

    return np.array(filled, dtype=int)


@torch.no_grad()
def infer_beats(
    module: BeatPhaseModule,
    task: str,
    audio: np.ndarray,
    sr: int,
    group_size: int = 4,
    beat_threshold: float = 0.3,
    min_distance_frames: int = 1,
    gate_tolerance: float = 0.2,
    anchor_threshold: float = 0.5,
    decoder: str = "greedy",
    switch_penalty: float | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run *module* on a full track and decode beat times + bar positions.

    :param module: A loaded, eval-mode :class:`BeatPhaseModule` (see
        :func:`musicality.inference.load_module`).
    :param task: Must be ``"beat_phase"`` — the annotator only displays bar
        positions, which a beat-only checkpoint has no concept of.
    :param audio: Mono float32 waveform.
    :param sr: Sample rate of *audio* — resampled to :data:`SAMPLE_RATE` if it differs.
    :param group_size: Beats per group the model's "one"/"last" heads were trained
        against — ``4`` for bar position (the current checkpoints' convention).
    :param decoder: Bar-position stage — ``"greedy"`` or ``"global"``. Forwarded
        to :func:`musicality.inference.run_inference`; callers should pass the
        tuned value from ``configs/eval_beat.yaml``.
    :param switch_penalty: Forwarded to :func:`musicality.inference.run_inference`.
        Only used when ``decoder="global"``.
    :returns: ``(beat_times, beat_positions)`` — seconds and 1-indexed bar
        positions, following the same convention as manually annotated beats.
    :raises ValueError: *task* isn't ``"beat_phase"``.
    """

    if task != "beat_phase":
        raise ValueError(
            f"The annotator's beat inference only supports beat-phase "
            f"checkpoints (it displays bar positions) — got task={task!r}."
        )

    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)  # (1, N)
    if sr != SAMPLE_RATE:
        wav = T.Resample(sr, SAMPLE_RATE)(wav)

    fps = SAMPLE_RATE / HOP_LENGTH
    events = run_inference(
        module,
        task,
        wav,
        fps,
        device,
        beat_threshold=beat_threshold,
        min_distance_frames=min_distance_frames,
        gate_tolerance=gate_tolerance,
        anchor_threshold=anchor_threshold,
        group_size=group_size,
        decoder=decoder,
        switch_penalty=switch_penalty,
    )

    beat_times = np.array([e["time"] for e in events], dtype=float)
    beat_positions = _fill_positions([e["beat_in_bar"] for e in events], group_size)

    return beat_times, beat_positions
