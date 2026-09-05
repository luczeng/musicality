"""Continuity metrics: how much of a track is tracked at the *right* metrical
level, versus at any self-consistent one.

:func:`musicality.metrics.f_measure.beat_f_measure` scores each beat
independently against a 70 ms window, so it cannot separate "the model is
mistiming beats" from "the model is confidently tracking half-time". Those have
different fixes, and the gap between ``cmlt`` and ``amlt`` is what tells them
apart: ``amlt`` forgives a whole-track octave or offbeat shift, ``cmlt`` does
not.

Measured on ``merge_v4`` over 60 merge-val tracks: ``cmlt`` 0.665 against
``amlt`` 0.839, so roughly 17 points of that checkpoint's beat error is
metrical-level ambiguity rather than timing failure. See
``plans/06_metric_calibration_and_eval_consolidation.md`` section 1.4.
"""

import numpy as np
from mir_eval.beat import continuity as _mir_eval_continuity
from mir_eval.beat import trim_beats


def beat_continuity(
    ref_times: np.ndarray,
    est_times: np.ndarray,
    trim: bool = True,
) -> dict | None:
    """Continuity-based beat scores — thin wrapper around ``mir_eval.beat.continuity``.

    ``mir_eval`` counts a beat as correct when both its own error and the
    preceding inter-beat interval fall within a relative tolerance (0.175) of
    the reference, so phase *and* local period have to agree — unlike
    :func:`~musicality.metrics.f_measure.beat_f_measure`, which judges each
    beat in isolation.

    :param ref_times: Reference beat times, in seconds.
    :param est_times: Estimated beat times, in seconds.
    :param trim: If ``True``, drop events before 5s from both sets first —
        ``mir_eval``'s standard warm-up convention (see
        :func:`~musicality.metrics.f_measure.beat_f_measure`).
    :returns: ``None`` if either sequence has fewer than 2 beats left, since
        ``mir_eval`` cannot derive a period from one beat and returns zeros
        rather than raising — which would average in as if it were a real
        score. Otherwise a dict with:

        - ``cmlc`` — longest *continuously* correct stretch, as a fraction of
          the track, at the correct metrical level.
        - ``cmlt`` — *total* correct fraction at the correct metrical level,
          continuity not required.
        - ``amlc`` / ``amlt`` — the same two, but also accepting double-time,
          half-time and offbeat interpretations of the reference.

        ``amlt - cmlt`` is the share of the track tracked consistently but at
        the wrong metrical level; ``amlt`` is always >= ``cmlt``.
    """

    ref = np.sort(np.asarray(ref_times, dtype=float))
    est = np.sort(np.asarray(est_times, dtype=float))

    if trim:
        ref = trim_beats(ref)
        est = trim_beats(est)

    if len(ref) < 2 or len(est) < 2:
        return None

    cmlc, cmlt, amlc, amlt = _mir_eval_continuity(ref, est)

    return {
        "cmlc": float(cmlc),
        "cmlt": float(cmlt),
        "amlc": float(amlc),
        "amlt": float(amlt),
    }
