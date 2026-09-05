"""Training-time metrics computed directly from per-frame probability curves.

Two of them, and the difference matters:

- :func:`frame_accuracy` compares the *shape* of the predicted curve against
  the *shape* of the Gaussian-smeared target, frame by frame. Cheap, but it
  penalises a peak that is wider than the target even when it is perfectly
  centred — width that :func:`musicality.postprocess.pick_peaks` discards at
  inference. Kept for the one/last heads; no longer the headline beat metric.
- :func:`peak_f_measure` picks peaks first and then matches events, which is
  what inference actually does, so it moves for the same reasons the reported
  ``f_beat`` moves.

Measured on ``merge_v4`` over 60 merge-val clips, ``frame_accuracy`` reads
0.857 and decomposes into a true-positive rate of 0.915 against a precision of
**0.487** — the model fires above threshold on roughly twice as many frames as
the target marks positive. ``peak_f_measure`` reads 0.900 on the same clips
against a full-track ``f_beat`` of 0.845. See
``plans/06_metric_calibration_and_eval_consolidation.md`` sections 1.1 and 2.
"""

import numpy as np
import torch

from musicality.metrics.f_measure import beat_f_measure
from musicality.postprocess import pick_peaks


def frame_accuracy(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    balanced: bool = False,
) -> torch.Tensor:
    """Coarse frame-level binary accuracy, for training-time monitoring only.

    Thresholds both the predicted probability and the (Gaussian-smeared) target
    at ``threshold`` and checks agreement. This is not the F-measure evaluation
    the project ultimately cares about (see :func:`musicality.metrics.f_measure.beat_f_measure`
    and :func:`musicality.metrics.f_measure.downbeat_f_measures`) — it's a cheap
    per-epoch signal.

    At the default ``threshold`` and the configured ``sigma_frames: 1.5``, a
    frame counts as target-positive only for ``|offset| <= 1`` frame
    (``exp(-d^2 / 2 sigma^2) > 0.5`` gives ``|d| < 1.766``), i.e. a +/-23 ms
    window against ``mir_eval``'s +/-70 ms. Note also that ``balanced=True``
    never observes precision, which is the term that actually moves — see
    :func:`peak_f_measure`.

    By default (``balanced=False``) all frames are pooled into one mean, which
    is dominated by the true-negative rate since beat/one/last frames are a
    small minority: a model that never fires still scores close to 1.0. Pass
    ``balanced=True`` to instead average the true-positive rate and true-negative
    rate, so a model that never fires floors at 0.5 rather than ~1.0.

    :param probs: Predicted probabilities (post-sigmoid), shape ``(B, T)``.
    :param target: Ground-truth (possibly smeared) target, shape ``(B, T)``.
    :param mask: Optional per-frame mask; masked-out frames are excluded from
        the average. Shape ``(B, T)``.
    :param threshold: Decision threshold applied to both ``probs`` and ``target``.
    :param balanced: If ``True``, return ``0.5 * (TPR + TNR)`` instead of the
        pooled mean.
    :returns: Scalar accuracy in ``[0, 1]``, shape ``()``.
    """

    pred_pos = probs > threshold
    targ_pos = target > threshold
    valid = torch.ones_like(probs, dtype=torch.bool) if mask is None else mask > 0

    if not balanced:
        agree = (pred_pos == targ_pos).float()
        denom = valid.float().sum().clamp(min=1.0)

        return (agree * valid).sum() / denom

    pos, neg = targ_pos & valid, ~targ_pos & valid

    tpr = (pred_pos & pos).float().sum() / pos.float().sum().clamp(min=1.0)
    tnr = (~pred_pos & neg).float().sum() / neg.float().sum().clamp(min=1.0)

    return 0.5 * (tpr + tnr)


def peak_f_measure(
    probs: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    min_distance: int = 4,
    tolerance_frames: float = 3.0,
) -> torch.Tensor:
    """Event-level F-measure between the peaks of two per-frame curves.

    Runs :func:`musicality.postprocess.pick_peaks` over the prediction *and*
    the target, then scores the two event lists with
    :func:`musicality.metrics.f_measure.beat_f_measure` — the same matching the
    offline evaluation performs. Because only each bump's summit survives on
    both sides, a correctly centred but over-wide peak costs nothing here,
    exactly as it costs nothing at inference.

    Everything stays in **frame units**: the peaks are frame indices and
    ``tolerance_frames`` is a frame count, so no ``fps`` is needed (neither
    LightningModule carries ``hop_length`` in its hyperparameters). ``mir_eval``
    is unit-agnostic, and ``trim`` is off so its 5-second warm-up convention
    cannot silently drop the first five *frames*.

    The default ``tolerance_frames=3`` is 70 ms at ``sample_rate: 22050`` /
    ``hop_length: 512`` (``0.07 * 43.07 = 3.01``), matching ``mir_eval``'s
    window. ``threshold`` and ``min_distance`` default to the tuned beat-phase
    values in ``configs/eval_beat.yaml``; the beat-only task is tuned to
    ``beat_threshold: 0.8`` instead, so for that head the training-time number
    is a proxy rather than the identical computation.

    :param probs: Predicted probabilities (post-sigmoid), shape ``(B, T)``.
    :param target: Ground-truth Gaussian-smeared target, shape ``(B, T)``.
    :param threshold: Minimum probability for a peak, on both curves.
    :param min_distance: Minimum frame gap between accepted peaks.
    :param tolerance_frames: Matching window, in frames.
    :returns: Mean F-measure over the batch, shape ``()``. Clips whose target
        holds no beat at all are skipped rather than scored 0 — a silent crop
        is not a failed prediction. Returns 0.0 if no clip was scorable.
    """

    p = probs.detach().float().cpu().numpy()
    y = target.detach().float().cpu().numpy()

    scores = []

    for p_i, y_i in zip(p, y):
        ref = pick_peaks(y_i, threshold=threshold, min_distance=min_distance)
        if len(ref) == 0:
            continue

        est = pick_peaks(p_i, threshold=threshold, min_distance=min_distance)
        if len(est) == 0:
            # mir_eval warns on an empty estimate; the answer is 0 either way.
            scores.append(0.0)
            continue

        scores.append(beat_f_measure(ref, est, tolerance=tolerance_frames, trim=False))

    value = float(np.mean(scores)) if scores else 0.0

    return torch.tensor(value, device=probs.device, dtype=torch.float32)
