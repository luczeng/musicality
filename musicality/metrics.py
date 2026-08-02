"""Event-level evaluation for beat-phase detection: beat / "1" / "4" F-measure
and a "1-vs-3" phase-confusion rate.

Consumes the labeled beat list produced by :func:`musicality.postprocess.readout`
(or raw reference annotations) — this is the "does the final output line up with
the ground truth" evaluation, distinct from :func:`musicality.trainers.beat_phase_module.frame_accuracy`,
which is a cheap per-epoch training signal on raw frame probabilities.
"""

import numpy as np
from mir_eval.beat import f_measure as _mir_eval_f_measure
from mir_eval.beat import trim_beats


def beat_f_measure(
    ref_times: np.ndarray,
    est_times: np.ndarray,
    tolerance: float = 0.07,
    trim: bool = True,
) -> float:
    """F-measure between two sets of event times, matched within a tolerance window.

    Thin wrapper around ``mir_eval.beat.f_measure`` — usable for the beat channel
    directly, and for the "1"/"4" channels by passing only the events of that
    position (see :func:`downbeat_f_measures`).

    :param ref_times: Reference event times, in seconds.
    :param est_times: Estimated event times, in seconds.
    :param tolerance: Matching window, in seconds (``mir_eval``'s default is 0.07).
    :param trim: If ``True``, drop events before 5s from both sets first —
        ``mir_eval``'s standard convention, so an algorithm isn't penalised for not
        having locked onto the tempo yet. Meaningful for full-track evaluation;
        pass ``False`` for short clips (e.g. the ~10s training clips), where
        trimming would discard most of the signal.
    :returns: F-measure in ``[0, 1]``.
    """

    ref = np.sort(np.asarray(ref_times, dtype=float))
    est = np.sort(np.asarray(est_times, dtype=float))

    if trim:
        ref = trim_beats(ref)
        est = trim_beats(est)

    return float(_mir_eval_f_measure(ref, est, f_measure_threshold=tolerance))


def downbeat_f_measures(
    ref_times: np.ndarray,
    ref_positions: np.ndarray,
    pred_events: list[dict],
    tolerance: float = 0.07,
    trim: bool = True,
) -> tuple[float, float]:
    """F-measure for the "1" (downbeat) and "4" positions, computed separately.

    :param ref_times: Reference beat times, in seconds, shape ``(n_beats,)``.
    :param ref_positions: Reference bar position (1-4) per beat, same shape as
        ``ref_times``.
    :param pred_events: Predicted beat list, as returned by
        :func:`musicality.postprocess.readout` — a list of
        ``{"time": float, "beat_in_bar": int | None}``.
    :param tolerance: Passed to :func:`beat_f_measure`.
    :param trim: Passed to :func:`beat_f_measure`.
    :returns: ``(f_one, f_four)``.
    """

    ref_times = np.asarray(ref_times, dtype=float)
    ref_positions = np.asarray(ref_positions)

    ref_one = ref_times[ref_positions == 1]
    ref_four = ref_times[ref_positions == 4]

    est_one = np.array([e["time"] for e in pred_events if e["beat_in_bar"] == 1])
    est_four = np.array([e["time"] for e in pred_events if e["beat_in_bar"] == 4])

    f_one = beat_f_measure(ref_one, est_one, tolerance=tolerance, trim=trim)
    f_four = beat_f_measure(ref_four, est_four, tolerance=tolerance, trim=trim)

    return f_one, f_four


def confusion_1_vs_3_rate(
    ref_times: np.ndarray,
    ref_positions: np.ndarray,
    pred_events: list[dict],
    tolerance: float = 0.07,
) -> float | None:
    """Rate of half-bar phase-parity errors: predicted "1" <-> true position 3 (or vice versa).

    Bar-level analogue of the original phrase-tracker plan's "beat-5 confusion"
    metric: the failure mode where the model has locked onto the right beat
    periodicity but the wrong phase — exactly half a bar (2 of 4 beats) off,
    so a "1" is mistaken for a "3" and a "3" for a "1". This is a different
    failure from a generic wrong label or a missed detection, which is why it's
    measured separately from :func:`downbeat_f_measures`.

    For each reference beat at position 1 or 3, this finds the nearest predicted
    event within ``tolerance`` seconds. Only beats with a resolved ("1" or "3")
    match are counted as *eligible* — a missed detection or an unresolved/
    off-parity label (2, 4, or ``None``) is a different failure mode and is
    excluded rather than counted as "not confused", so it doesn't dilute the
    signal this metric is meant to isolate.

    :param ref_times: Reference beat times, in seconds, shape ``(n_beats,)``.
    :param ref_positions: Reference bar position (1-4) per beat, same shape as
        ``ref_times``.
    :param pred_events: Predicted beat list, as returned by
        :func:`musicality.postprocess.readout`.
    :param tolerance: Matching window, in seconds.
    :returns: Fraction of eligible beats that were swapped, in ``[0, 1]``, or
        ``None`` if there were no eligible beats (nothing to measure).
    """

    ref_times = np.asarray(ref_times, dtype=float)
    ref_positions = np.asarray(ref_positions)

    eligible_mask = (ref_positions == 1) | (ref_positions == 3)
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
        if label not in (1, 3):
            continue

        n_matched += 1
        other = 1 if truth == 3 else 3
        if label == other:
            n_swapped += 1

    if n_matched == 0:
        return None

    return n_swapped / n_matched
