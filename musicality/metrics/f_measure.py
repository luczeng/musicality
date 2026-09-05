"""F-measure metrics: overall beat F-measure and per-position (downbeat) F-measure.

Consumes the labeled beat list produced by :func:`musicality.postprocess.readout`
(or raw reference annotations) — this is the "does the final output line up with
the ground truth" evaluation, distinct from :func:`musicality.metrics.frame_accuracy.frame_accuracy`,
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
    group_size: int = 4,
) -> tuple[float, float]:
    """F-measure for the "1" (downbeat) and "last" (position ``group_size``) positions, computed separately.

    .. note::

       Only positions ``1`` and ``group_size`` are ever scored. Positions
       ``2..group_size-1`` are invisible to this metric even under the softmax
       head, which predicts all of them — an error that turns a "2" into a "3"
       cannot register. That is why
       :func:`musicality.metrics.phase_offset.phase_offset_profile`'s
       ``correct_fraction`` is the headline bar-position metric and these two
       are reported alongside it rather than in place of it.

    :param ref_times: Reference beat times, in seconds, shape ``(n_beats,)``.
    :param ref_positions: Reference group position (1-``group_size``) per beat,
        same shape as ``ref_times``.
    :param pred_events: Predicted beat list, as returned by
        :func:`musicality.postprocess.readout` — a list of
        ``{"time": float, "beat_in_bar": int | None}``.
    :param tolerance: Passed to :func:`beat_f_measure`.
    :param trim: Passed to :func:`beat_f_measure`.
    :param group_size: Beats per group — ``4`` for bar position (default),
        ``8`` for phrase position. Determines which reference/predicted
        position counts as "last".
    :returns: ``(f_one, f_last)``.
    """

    ref_times = np.asarray(ref_times, dtype=float)
    ref_positions = np.asarray(ref_positions)

    ref_one = ref_times[ref_positions == 1]
    ref_last = ref_times[ref_positions == group_size]

    est_one = np.array([e["time"] for e in pred_events if e["beat_in_bar"] == 1])
    est_last = np.array(
        [e["time"] for e in pred_events if e["beat_in_bar"] == group_size]
    )

    f_one = beat_f_measure(ref_one, est_one, tolerance=tolerance, trim=trim)
    f_last = beat_f_measure(ref_last, est_last, tolerance=tolerance, trim=trim)

    return f_one, f_last
