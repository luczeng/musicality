"""Phase-confusion metric: half-cycle "1" <-> opposite-position swap rate."""

import numpy as np


def confusion_half_cycle_rate(
    ref_times: np.ndarray,
    ref_positions: np.ndarray,
    pred_events: list[dict],
    tolerance: float = 0.07,
    group_size: int = 4,
) -> float | None:
    """Rate of half-cycle phase-parity errors: predicted "1" <-> true position ``1 + group_size // 2`` (or vice versa).

    Analogue of the original phrase-tracker plan's "beat-5 confusion" metric:
    the failure mode where the model has locked onto the right beat
    periodicity but the wrong phase — exactly half a group off, so a "1" is
    mistaken for the position diametrically opposite it in the cycle (bar
    position 3 when ``group_size=4``; phrase position 5 when ``group_size=8``)
    and vice versa. This is a different failure from a generic wrong label or
    a missed detection, which is why it's measured separately from
    :func:`musicality.metrics.f_measure.downbeat_f_measures`.

    For each reference beat at position 1 or the opposite position, this finds
    the nearest predicted event within ``tolerance`` seconds. Only beats with
    a resolved (1 or opposite) match are counted as *eligible* — a missed
    detection or an unresolved/off-parity label is a different failure mode
    and is excluded rather than counted as "not confused", so it doesn't
    dilute the signal this metric is meant to isolate.

    :param ref_times: Reference beat times, in seconds, shape ``(n_beats,)``.
    :param ref_positions: Reference group position (1-``group_size``) per beat,
        same shape as ``ref_times``.
    :param pred_events: Predicted beat list, as returned by
        :func:`musicality.postprocess.readout`.
    :param tolerance: Matching window, in seconds.
    :param group_size: Beats per group — ``4`` for bar position (default),
        ``8`` for phrase position. Must be even for "half a cycle" to land on
        an integer position.
    :returns: Fraction of eligible beats that were swapped, in ``[0, 1]``, or
        ``None`` if there were no eligible beats (nothing to measure).
    """

    ref_times = np.asarray(ref_times, dtype=float)
    ref_positions = np.asarray(ref_positions)

    opposite = 1 + group_size // 2

    eligible_mask = (ref_positions == 1) | (ref_positions == opposite)
    eligible_times = ref_times[eligible_mask]
    eligible_truth = ref_positions[eligible_mask]

    if len(eligible_times) == 0 or len(pred_events) == 0:
        return None

    pred_times = np.array([e["time"] for e in pred_events])
    pred_labels = [e["beat_in_bar"] for e in pred_events]

    n_matched = 0
    n_swapped = 0

    for t_ref, truth in zip(eligible_times, eligible_truth):
        idx = int(np.argmin(np.abs(pred_times - t_ref)))
        if abs(pred_times[idx] - t_ref) > tolerance:
            continue

        label = pred_labels[idx]
        if label not in (1, opposite):
            continue

        n_matched += 1
        other = 1 if truth == opposite else opposite
        if label == other:
            n_swapped += 1

    if n_matched == 0:
        return None

    return n_swapped / n_matched
