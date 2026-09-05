"""Tests for musicality.metrics: beat_f_measure, downbeat_f_measures,
beat_continuity, confusion_half_cycle_rate, frame_accuracy, peak_f_measure,
tempo_acc1.
"""

import numpy as np
import pytest
import torch

from musicality.metrics.confusion import confusion_half_cycle_rate
from musicality.metrics.continuity import beat_continuity
from musicality.metrics.f_measure import beat_f_measure, downbeat_f_measures
from musicality.metrics.frame_accuracy import frame_accuracy, peak_f_measure
from musicality.metrics.tempo_acc1 import tempo_acc1


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


# ---------------------------------------------------------------------------
# frame_accuracy
# ---------------------------------------------------------------------------


class TestFrameAccuracy:
    def test_perfect_match_is_one(self):
        probs = torch.tensor([0.9, 0.1, 0.9, 0.1])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert frame_accuracy(probs, target).item() == pytest.approx(1.0)

    def test_total_mismatch_is_zero(self):
        probs = torch.tensor([0.9, 0.1])
        target = torch.tensor([0.0, 1.0])
        assert frame_accuracy(probs, target).item() == pytest.approx(0.0)

    def test_mask_excludes_frames(self):
        probs = torch.tensor([[0.9, 0.1, 0.9]])
        target = torch.tensor([[1.0, 1.0, 1.0]])  # frame 1 is wrong
        mask = torch.tensor([[1.0, 0.0, 1.0]])  # frame 1 excluded
        assert frame_accuracy(probs, target, mask=mask).item() == pytest.approx(1.0)

    def test_all_masked_out_no_nan(self):
        probs = torch.tensor([[0.9, 0.1]])
        target = torch.tensor([[1.0, 0.0]])
        mask = torch.zeros(1, 2)
        assert torch.isfinite(frame_accuracy(probs, target, mask=mask)).all()

    def test_balanced_perfect_match_is_one(self):
        probs = torch.tensor([0.9, 0.1, 0.9, 0.1])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert frame_accuracy(probs, target, balanced=True).item() == pytest.approx(1.0)

    def test_balanced_always_negative_floors_at_half(self):
        # 1 positive frame out of 10; a model that never fires still gets 9/10
        # right under the pooled mean, but should floor at 0.5 when balanced.
        target = torch.tensor([[1.0] + [0.0] * 9])
        probs = torch.full((1, 10), 0.1)

        assert frame_accuracy(probs, target).item() == pytest.approx(0.9)
        assert frame_accuracy(probs, target, balanced=True).item() == pytest.approx(0.5)

    def test_balanced_mask_excludes_frames(self):
        probs = torch.tensor([[0.9, 0.1, 0.9]])
        target = torch.tensor([[1.0, 1.0, 1.0]])  # frame 1 is wrong
        mask = torch.tensor([[1.0, 0.0, 1.0]])  # frame 1 excluded

        assert frame_accuracy(
            probs, target, mask=mask, balanced=True
        ).item() == pytest.approx(0.5)

    def test_balanced_all_masked_out_no_nan(self):
        probs = torch.tensor([[0.9, 0.1]])
        target = torch.tensor([[1.0, 0.0]])
        mask = torch.zeros(1, 2)
        assert torch.isfinite(
            frame_accuracy(probs, target, mask=mask, balanced=True)
        ).all()


# ---------------------------------------------------------------------------
# peak_f_measure
# ---------------------------------------------------------------------------


def _curve(n_frames: int, centers, sigma: float = 1.5) -> torch.Tensor:
    """A Gaussian-smeared frame curve, shape ``(1, n_frames)`` — the same shape
    BeatDataset produces, so pick_peaks sees a real bump rather than a spike."""

    t = np.arange(n_frames, dtype=float)
    y = np.zeros(n_frames, dtype=float)

    for c in centers:
        y = np.maximum(y, np.exp(-((t - c) ** 2) / (2 * sigma**2)))

    return torch.tensor(y, dtype=torch.float32).unsqueeze(0)


class TestPeakFMeasure:
    def test_perfect_match_is_one(self):
        beats = [10, 30, 50, 70]
        curve = _curve(100, beats)
        assert peak_f_measure(curve, curve).item() == pytest.approx(1.0)

    def test_over_wide_peaks_cost_nothing(self):
        # The load-bearing property: a correctly centred but twice-as-wide
        # prediction is what drags frame_accuracy down (precision 0.487 on the
        # real checkpoint), and it is exactly what pick_peaks discards.
        beats = [10, 30, 50, 70]
        target = _curve(100, beats, sigma=1.5)
        wide = _curve(100, beats, sigma=3.0)

        assert peak_f_measure(wide, target).item() == pytest.approx(1.0)
        assert frame_accuracy(wide, target, balanced=True).item() < 0.95

    def test_shift_inside_tolerance_still_matches(self):
        target = _curve(100, [10, 30, 50, 70])
        shifted = _curve(100, [12, 32, 52, 72])  # +2 frames, tolerance is 3
        assert peak_f_measure(shifted, target).item() == pytest.approx(1.0)

    def test_shift_outside_tolerance_scores_zero(self):
        target = _curve(100, [10, 30, 50, 70])
        shifted = _curve(100, [16, 36, 56, 76])  # +6 frames, beyond tolerance
        assert peak_f_measure(shifted, target).item() == pytest.approx(0.0)

    def test_silent_prediction_scores_zero(self):
        target = _curve(100, [10, 30, 50, 70])
        silent = torch.zeros(1, 100)
        assert peak_f_measure(silent, target).item() == pytest.approx(0.0)

    def test_beatless_clip_is_skipped_not_scored_zero(self):
        # A crop that happens to contain no annotated beat is not a failed
        # prediction, so it must not drag the batch mean toward 0.
        target = torch.cat([_curve(100, [10, 30, 50, 70]), torch.zeros(1, 100)])
        probs = torch.cat([_curve(100, [10, 30, 50, 70]), torch.zeros(1, 100)])

        assert peak_f_measure(probs, target).item() == pytest.approx(1.0)

    def test_no_scorable_clip_returns_zero_not_nan(self):
        empty = torch.zeros(2, 100)
        value = peak_f_measure(empty, empty)

        assert torch.isfinite(value).all()
        assert value.item() == pytest.approx(0.0)

    def test_batch_mean_over_clips(self):
        target = torch.cat([_curve(100, [10, 30, 50, 70])] * 2)
        probs = torch.cat(
            [_curve(100, [10, 30, 50, 70]), _curve(100, [16, 36, 56, 76])]
        )
        assert peak_f_measure(probs, target).item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# beat_continuity
# ---------------------------------------------------------------------------


class TestBeatContinuity:
    def test_perfect_match_is_one(self):
        times = np.arange(6.0, 20.0, 0.5)
        scores = beat_continuity(times, times)

        assert scores["cmlt"] == pytest.approx(1.0)
        assert scores["amlt"] == pytest.approx(1.0)

    def test_double_time_is_forgiven_only_by_amlt(self):
        # The reason this metric earns its place: a confident double-time
        # tracker is not a mistimed tracker, and f_measure cannot tell them
        # apart.
        ref = np.arange(6.0, 20.0, 0.5)
        est = np.arange(6.0, 20.0, 0.25)
        scores = beat_continuity(ref, est)

        assert scores["amlt"] > 0.9
        assert scores["cmlt"] < scores["amlt"]

    def test_amlt_never_below_cmlt(self):
        rng = np.random.RandomState(0)
        ref = np.arange(6.0, 20.0, 0.5)
        est = ref + rng.normal(0, 0.03, ref.shape)
        scores = beat_continuity(ref, est)

        assert scores["amlt"] >= scores["cmlt"]

    def test_too_few_beats_returns_none(self):
        assert beat_continuity(np.array([6.0]), np.array([6.0, 6.5])) is None
        assert beat_continuity(np.array([]), np.array([6.0, 6.5])) is None

    def test_trim_can_leave_too_few_beats(self):
        # Everything before 5s is dropped by mir_eval's warm-up convention, so
        # a short clip evaluated with trim=True has nothing left to score.
        times = np.arange(0.0, 3.0, 0.5)

        assert beat_continuity(times, times, trim=True) is None
        assert beat_continuity(times, times, trim=False)["cmlt"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# tempo_acc1
# ---------------------------------------------------------------------------


class TestTempoAcc1:
    def test_exact_match_is_one(self):
        target = torch.tensor([80.0, 100.0, 120.0])
        assert tempo_acc1(target, target).item() == pytest.approx(1.0)

    def test_within_tolerance_is_one(self):
        target = torch.tensor([120.0])
        pred = torch.tensor([125.0])  # +4.2%, within the default 8% tolerance
        assert tempo_acc1(pred, target).item() == pytest.approx(1.0)

    def test_outside_tolerance_is_zero(self):
        target = torch.tensor([120.0])
        pred = torch.tensor([140.0])  # +16.7%, outside tolerance and not an octave
        assert tempo_acc1(pred, target).item() == pytest.approx(0.0)

    def test_half_tempo_counts_correct(self):
        target = torch.tensor([120.0])
        pred = torch.tensor([60.0])  # exactly half-tempo
        assert tempo_acc1(pred, target).item() == pytest.approx(1.0)

    def test_double_tempo_counts_correct(self):
        target = torch.tensor([120.0])
        pred = torch.tensor([240.0])  # exactly double-tempo
        assert tempo_acc1(pred, target).item() == pytest.approx(1.0)

    def test_mixed_batch_partial_accuracy(self):
        target = torch.tensor([120.0, 120.0])
        pred = torch.tensor([120.0, 140.0])  # one correct, one not
        assert tempo_acc1(pred, target).item() == pytest.approx(0.5)
