"""Tests for musicality.metrics: beat_f_measure, downbeat_f_measures,
confusion_half_cycle_rate.
"""

import numpy as np
import pytest

from musicality.metrics.confusion import confusion_half_cycle_rate
from musicality.metrics.f_measure import beat_f_measure, downbeat_f_measures


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

        f_one, f_last = downbeat_f_measures(ref_times, ref_positions, pred)
        assert f_one == pytest.approx(1.0)
        assert f_last == pytest.approx(1.0)

    def test_no_predicted_downbeats_is_zero(self):
        ref_times = np.arange(6.0, 6.0 + 0.5 * 8, 0.5)
        ref_positions = np.tile([1, 2, 3, 4], 2)
        pred = _events(ref_times, [2, 2, 3, 3, 2, 2, 3, 3])  # never predicts 1 or 4

        f_one, f_last = downbeat_f_measures(ref_times, ref_positions, pred)
        assert f_one == 0.0
        assert f_last == 0.0

    def test_group_size_8_uses_position_8_as_last(self):
        # 2 phrases of 8 beats; position 4 must NOT count as "last" here.
        ref_times = np.arange(6.0, 6.0 + 0.5 * 16, 0.5)
        ref_positions = np.tile(np.arange(1, 9), 2)
        pred = _events(ref_times, ref_positions)

        f_one, f_last = downbeat_f_measures(
            ref_times, ref_positions, pred, group_size=8
        )
        assert f_one == pytest.approx(1.0)
        assert f_last == pytest.approx(1.0)

        # predicting "4" instead of "8" should score zero on the last-position F-measure
        wrong_pred = _events(ref_times, [4 if p == 8 else p for p in ref_positions])
        _, f_last_wrong = downbeat_f_measures(
            ref_times, ref_positions, wrong_pred, group_size=8
        )
        assert f_last_wrong == 0.0


# ---------------------------------------------------------------------------
# confusion_half_cycle_rate
# ---------------------------------------------------------------------------


class TestConfusionHalfCycleRate:
    def test_no_eligible_beats_returns_none(self):
        ref_times = np.array([0.0, 0.5])
        ref_positions = np.array([2, 4])  # no position-1 or -3 beats
        pred = _events(ref_times, [2, 4])
        assert confusion_half_cycle_rate(ref_times, ref_positions, pred) is None

    def test_no_predictions_returns_none(self):
        ref_times = np.array([0.0, 1.0])
        ref_positions = np.array([1, 3])
        assert confusion_half_cycle_rate(ref_times, ref_positions, []) is None

    def test_correct_labels_zero_confusion(self):
        ref_times = np.array([0.0, 0.5, 1.0, 1.5])
        ref_positions = np.array([1, 2, 3, 4])
        pred = _events(ref_times, [1, 2, 3, 4])
        assert confusion_half_cycle_rate(ref_times, ref_positions, pred) == 0.0

    def test_systematic_half_bar_swap_full_confusion(self):
        ref_times = np.array([0.0, 0.5, 1.0, 1.5])
        ref_positions = np.array([1, 2, 3, 4])
        pred = _events(ref_times, [3, 2, 1, 4])  # 1<->3 swapped, 2/4 untouched
        assert confusion_half_cycle_rate(
            ref_times, ref_positions, pred
        ) == pytest.approx(1.0)

    def test_unresolved_or_off_parity_labels_excluded_not_counted_as_correct(self):
        ref_times = np.array([0.0, 1.0])
        ref_positions = np.array([1, 3])
        # neither predicted label is 1 or 3 -> both excluded, nothing eligible left
        pred = _events(ref_times, [2, None])
        assert confusion_half_cycle_rate(ref_times, ref_positions, pred) is None

    def test_unmatched_beat_excluded(self):
        ref_times = np.array([0.0, 1.0])
        ref_positions = np.array([1, 3])
        pred = _events([0.0, 5.0], [1, 3])  # second prediction far outside tolerance
        # only the first beat is eligible+matched, and it's correct
        assert (
            confusion_half_cycle_rate(ref_times, ref_positions, pred, tolerance=0.07)
            == 0.0
        )

    def test_group_size_8_swaps_1_vs_5(self):
        # half of an 8-beat phrase is 4 beats away: opposite of position 1 is position 5.
        ref_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ref_positions = np.arange(1, 9)
        pred = _events(
            ref_times, [5 if p == 1 else (1 if p == 5 else p) for p in ref_positions]
        )
        assert confusion_half_cycle_rate(
            ref_times, ref_positions, pred, group_size=8
        ) == pytest.approx(1.0)

    def test_group_size_8_no_confusion_when_correct(self):
        ref_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ref_positions = np.arange(1, 9)
        pred = _events(ref_times, ref_positions)
        assert (
            confusion_half_cycle_rate(ref_times, ref_positions, pred, group_size=8)
            == 0.0
        )
