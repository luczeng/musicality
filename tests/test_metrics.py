"""Tests for musicality.metrics: beat_f_measure, downbeat_f_measures,
confusion_1_vs_3_rate.
"""

import numpy as np
import pytest

from musicality.metrics import (
    beat_f_measure,
    confusion_1_vs_3_rate,
    downbeat_f_measures,
)


# ---------------------------------------------------------------------------
# beat_f_measure
# ---------------------------------------------------------------------------


class TestBeatFMeasure:
    def test_perfect_match(self):
        times = np.arange(6.0, 20.0, 0.5)  # well past the 5s trim window
        assert beat_f_measure(times, times) == pytest.approx(1.0)

    def test_empty_estimate_is_zero(self):
        ref = np.arange(6.0, 10.0, 0.5)
        assert beat_f_measure(ref, np.array([])) == 0.0

    def test_empty_reference_is_zero(self):
        est = np.arange(6.0, 10.0, 0.5)
        assert beat_f_measure(np.array([]), est) == 0.0

    def test_partial_match_between_zero_and_one(self):
        ref = np.arange(6.0, 16.0, 0.5)
        est = ref.copy()
        est[::2] += 1.0  # push every other beat far outside the tolerance window
        f = beat_f_measure(ref, est)
        assert 0.0 < f < 1.0

    def test_trim_drops_pre_5s_events(self):
        # Only events before 5s — trimmed away entirely, so nothing to match.
        ref = np.array([1.0, 2.0, 3.0])
        est = np.array([1.0, 2.0, 3.0])
        assert beat_f_measure(ref, est, trim=True) == 0.0
        assert beat_f_measure(ref, est, trim=False) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# downbeat_f_measures
# ---------------------------------------------------------------------------


def _events(times, labels):
    return [{"time": t, "beat_in_bar": l} for t, l in zip(times, labels)]


class TestDownbeatFMeasures:
    def test_perfect_prediction(self):
        ref_times = np.arange(6.0, 6.0 + 0.5 * 8, 0.5)  # 2 bars of 4 beats
        ref_positions = np.tile([1, 2, 3, 4], 2)
        pred = _events(ref_times, ref_positions)

        f_one, f_four = downbeat_f_measures(ref_times, ref_positions, pred)
        assert f_one == pytest.approx(1.0)
        assert f_four == pytest.approx(1.0)

    def test_no_predicted_downbeats_is_zero(self):
        ref_times = np.arange(6.0, 6.0 + 0.5 * 8, 0.5)
        ref_positions = np.tile([1, 2, 3, 4], 2)
        pred = _events(ref_times, [2, 2, 3, 3, 2, 2, 3, 3])  # never predicts 1 or 4

        f_one, f_four = downbeat_f_measures(ref_times, ref_positions, pred)
        assert f_one == 0.0
        assert f_four == 0.0


# ---------------------------------------------------------------------------
# confusion_1_vs_3_rate
# ---------------------------------------------------------------------------


class TestConfusion1v3Rate:
    def test_no_eligible_beats_returns_none(self):
        ref_times = np.array([0.0, 0.5])
        ref_positions = np.array([2, 4])  # no position-1 or -3 beats
        pred = _events(ref_times, [2, 4])
        assert confusion_1_vs_3_rate(ref_times, ref_positions, pred) is None

    def test_no_predictions_returns_none(self):
        ref_times = np.array([0.0, 1.0])
        ref_positions = np.array([1, 3])
        assert confusion_1_vs_3_rate(ref_times, ref_positions, []) is None

    def test_correct_labels_zero_confusion(self):
        ref_times = np.array([0.0, 0.5, 1.0, 1.5])
        ref_positions = np.array([1, 2, 3, 4])
        pred = _events(ref_times, [1, 2, 3, 4])
        assert confusion_1_vs_3_rate(ref_times, ref_positions, pred) == 0.0

    def test_systematic_half_bar_swap_full_confusion(self):
        ref_times = np.array([0.0, 0.5, 1.0, 1.5])
        ref_positions = np.array([1, 2, 3, 4])
        pred = _events(ref_times, [3, 2, 1, 4])  # 1<->3 swapped, 2/4 untouched
        assert confusion_1_vs_3_rate(ref_times, ref_positions, pred) == pytest.approx(
            1.0
        )

    def test_unresolved_or_off_parity_labels_excluded_not_counted_as_correct(self):
        ref_times = np.array([0.0, 1.0])
        ref_positions = np.array([1, 3])
        # neither predicted label is 1 or 3 -> both excluded, nothing eligible left
        pred = _events(ref_times, [2, None])
        assert confusion_1_vs_3_rate(ref_times, ref_positions, pred) is None

    def test_unmatched_beat_excluded(self):
        ref_times = np.array([0.0, 1.0])
        ref_positions = np.array([1, 3])
        pred = _events([0.0, 5.0], [1, 3])  # second prediction far outside tolerance
        # only the first beat is eligible+matched, and it's correct
        assert (
            confusion_1_vs_3_rate(ref_times, ref_positions, pred, tolerance=0.07) == 0.0
        )
