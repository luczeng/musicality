"""Per-track phase-offset profile: *how* a track's predicted bar grid is
misaligned with the reference, rather than just how often it is.

:func:`musicality.metrics.confusion.confusion_half_cycle_rate` says a
prediction is half a bar out; it can't say whether the model got the phase
wrong for the *whole* track (a consistent offset — an acoustic/model failure)
or kept flipping between phases *within* the track (an unstable decode — a
postprocessing failure). Those two have opposite fixes, so this separates
them.
"""

import numpy as np


def phase_offset_profile(
    ref_times: np.ndarray,
    ref_positions: np.ndarray,
    pred_events: list[dict],
    tolerance: float = 0.07,
    group_size: int = 4,
) -> dict | None:
    """Distribution of per-beat phase offsets between prediction and reference.

    For every reference beat, finds the nearest predicted event within
    ``tolerance`` seconds and records
    ``offset = (predicted_position - true_position) mod group_size``. Unlike
    :func:`~musicality.metrics.confusion.confusion_half_cycle_rate`, *all*
    positions are eligible, not just 1 and the half-cycle opposite — an
    off-by-one error is exactly what this is meant to be able to see.

    The two numbers that matter:

    - ``modal_offset`` — the track's dominant misalignment. ``0`` is correct;
      ``group_size // 2`` is the half-cycle swap; anything else is an
      off-by-one-style error.
    - ``stability`` — the fraction of matched beats that agree with
      ``modal_offset``. Near ``1.0`` means the track has *one* consistent
      phase (right or wrong) from start to finish. Well below ``1.0`` means
      the predicted phase changes partway through the track.

    ``correct_fraction`` and ``stability`` are the project's headline position
    metrics, reported as ``position_acc`` and ``position_acc_best_offset``:
    the first is the accuracy as labelled, the second is the accuracy the model
    would have if allowed to rotate its bar numbering by one constant per
    track. Their difference, ``anchor_error``, is therefore the cost of
    choosing the wrong global anchor — a bar grid that is right in every
    respect except where it starts counting. Measured on ``merge_v4``, that is
    0.090 of a total 0.419 position error; see
    ``plans/06_metric_calibration_and_eval_consolidation.md`` section 1.2.

    Reading them together:

    - high ``stability``, ``modal_offset != 0`` → the model committed to a
      single wrong phase for the whole track. A better decoder can't help;
      the per-beat evidence itself is wrong.
    - low ``stability`` → the predicted phase flips mid-track. The evidence
      may well be fine on average and the decoder is losing it (see
      :func:`musicality.postprocess.label_bar_position`'s resync behaviour).

    :param ref_times: Reference beat times, in seconds, shape ``(n_beats,)``.
    :param ref_positions: Reference group position (1-``group_size``) per
        beat, same shape as ``ref_times``.
    :param pred_events: Predicted beat list, as returned by
        :func:`musicality.postprocess.readout`.
    :param tolerance: Matching window, in seconds.
    :param group_size: Beats per group — ``4`` for bar position (default),
        ``8`` for phrase position.
    :returns: ``None`` if no reference beat found a matching, resolved
        prediction (nothing to profile), else a dict with:

        - ``n_matched`` (int) — reference beats that matched a resolved label.
        - ``histogram`` (list[int], length ``group_size``) — count per offset.
        - ``modal_offset`` (int) — most common offset, in ``0..group_size-1``.
        - ``stability`` (float) — fraction of matched beats at ``modal_offset``.
        - ``correct_fraction`` (float) — fraction of matched beats at offset 0.
        - ``anchor_error`` (float) — ``stability - correct_fraction``, always
          ``>= 0`` since offset 0 is one of the candidates the maximum is taken
          over. Zero exactly when the dominant offset is the correct one.
    """

    ref_times = np.asarray(ref_times, dtype=float)
    ref_positions = np.asarray(ref_positions)

    if len(ref_times) == 0 or len(pred_events) == 0:
        return None

    pred_times = np.array([e["time"] for e in pred_events])
    pred_labels = [e["beat_in_bar"] for e in pred_events]

    histogram = np.zeros(group_size, dtype=int)

    for t_ref, truth in zip(ref_times, ref_positions):
        idx = int(np.argmin(np.abs(pred_times - t_ref)))
        if abs(pred_times[idx] - t_ref) > tolerance:
            continue

        label = pred_labels[idx]
        if label is None:
            continue

        histogram[(int(label) - int(truth)) % group_size] += 1

    n_matched = int(histogram.sum())
    if n_matched == 0:
        return None

    modal_offset = int(np.argmax(histogram))
    stability = float(histogram[modal_offset] / n_matched)
    correct_fraction = float(histogram[0] / n_matched)

    return {
        "n_matched": n_matched,
        "histogram": histogram.tolist(),
        "modal_offset": modal_offset,
        "stability": stability,
        "correct_fraction": correct_fraction,
        "anchor_error": stability - correct_fraction,
    }
