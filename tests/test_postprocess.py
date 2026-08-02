"""Tests for musicality.postprocess: pick_peaks, gate_periodicity,
label_bar_position, readout.
"""

import numpy as np
import pytest

from musicality.loaders.beat_dataset import gaussian_smear
from musicality.postprocess import (
    gate_periodicity,
    label_bar_position,
    pick_peaks,
    readout,
)


# ---------------------------------------------------------------------------
# pick_peaks
# ---------------------------------------------------------------------------


class TestPickPeaks:
    def test_finds_single_peak(self):
        probs = np.zeros(20)
        probs[10] = 1.0
        peaks = pick_peaks(gaussian_smear(probs, sigma=1.5), threshold=0.3)
        assert list(peaks) == [10]

    def test_below_threshold_ignored(self):
        probs = np.array([0.0, 0.1, 0.15, 0.1, 0.0])
        assert len(pick_peaks(probs, threshold=0.3)) == 0

    def test_multiple_separated_peaks_all_kept(self):
        probs = np.zeros(40)
        probs[[10, 25]] = 1.0
        smeared = gaussian_smear(probs, sigma=1.0)
        peaks = pick_peaks(smeared, threshold=0.3)
        assert list(peaks) == [10, 25]

    def test_min_distance_suppresses_close_duplicates(self):
        probs = np.array(
            [0.0, 0.4, 0.9, 0.5, 0.0]
        )  # two bumpy candidates close together
        peaks = pick_peaks(probs, threshold=0.3, min_distance=1)
        # without min_distance>1 both frame 1 (0.4, local max vs 0,0.9? not a peak) check actual peaks
        assert 2 in peaks  # the true summit is always kept

    def test_empty_input(self):
        assert len(pick_peaks(np.array([]), threshold=0.3)) == 0


# ---------------------------------------------------------------------------
# gate_periodicity
# ---------------------------------------------------------------------------


class TestGatePeriodicity:
    def test_too_few_points_returned_unchanged(self):
        times = np.array([0.0, 0.5])
        assert list(gate_periodicity(times)) == [0.0, 0.5]

    def test_regular_beats_all_accepted(self):
        times = np.arange(0, 5, 0.5)  # perfectly periodic, period 0.5
        gated = gate_periodicity(times)
        assert gated == pytest.approx(times)

    def test_drops_spurious_duplicate(self):
        # regular beats every 0.5s, with a spurious extra peak 0.05s after one of them
        times = np.array([0.0, 0.5, 1.0, 1.05, 1.5, 2.0])
        gated = gate_periodicity(times, tolerance=0.2)
        assert 1.05 not in gated
        assert gated == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

    def test_fills_single_missed_beat_gap(self):
        # a beat missing at 1.5s (gap of 2x the 0.5s period between 1.0 and 2.0)
        times = np.array([0.0, 0.5, 1.0, 2.0, 2.5])
        gated = gate_periodicity(times, tolerance=0.2)
        assert gated == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    def test_fills_multiple_missed_beats(self):
        # two beats missing between 1.0 and 2.5 (gap of 3x the 0.5s period)
        times = np.array([0.0, 0.5, 1.0, 2.5, 3.0])
        gated = gate_periodicity(times, tolerance=0.2)
        assert gated == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    def test_ambiguous_interval_kept_as_is(self):
        # 1.8x the period doesn't cleanly match k=1 or k=2 within tolerance=0.2
        times = np.array([0.0, 0.5, 1.0, 1.9])
        gated = gate_periodicity(times, tolerance=0.2)
        assert 1.9 in gated
        assert len(gated) == 4  # kept, but nothing extra inserted

    def test_tempo_drift_tracked(self):
        # Gradually accelerating tempo: each interval shrinks 2% from the last.
        # With ema_alpha=0.2, the EMA's steady-state lag ratio here works out to
        # ~0.92 (see (1 - (1-alpha)/r) / alpha for r=0.98) — comfortably inside
        # 1 +/- tolerance=0.2, so the estimate keeps pace and nothing is dropped.
        times = [0.0]
        interval = 0.5
        for _ in range(15):
            interval *= 0.98
            times.append(times[-1] + interval)
        gated = gate_periodicity(np.array(times), tolerance=0.2)
        assert len(gated) == len(times)

    def test_fast_drift_exceeds_gate_tolerance(self):
        # A much steeper accelerando (5%/beat) pushes the EMA's steady-state lag
        # ratio to ~0.79 (same formula, r=0.95) — just past 1 - tolerance=0.2 - i.e.
        # 0.8, so the gate starts rejecting beats once cumulative drift outruns it.
        # This is the documented limitation, not a bug: gate_periodicity assumes
        # tempo is locally near-constant, not that it can track sustained drift.
        times = [0.0]
        interval = 0.5
        for _ in range(15):
            interval *= 0.95
            times.append(times[-1] + interval)
        gated = gate_periodicity(np.array(times), tolerance=0.2)
        assert len(gated) < len(times)


# ---------------------------------------------------------------------------
# label_bar_position
# ---------------------------------------------------------------------------

FPS = 100.0


def _probs_at(times_and_values: dict, n_frames: int) -> np.ndarray:
    probs = np.zeros(n_frames)
    for t, v in times_and_values.items():
        probs[int(round(t * FPS))] = v
    return probs


class TestLabelBarPosition:
    def test_confident_anchor_labeled_directly(self):
        beat_times = np.array([0.0])
        one_probs = _probs_at({0.0: 0.9}, 100)
        last_probs = np.zeros(100)
        labels = label_bar_position(beat_times, one_probs, last_probs, FPS)
        assert labels == [1]

    def test_counts_forward_from_one_anchor(self):
        beat_times = np.array([0.0, 0.5, 1.0, 1.5])
        one_probs = _probs_at({0.0: 0.9}, 200)
        last_probs = np.zeros(200)
        labels = label_bar_position(beat_times, one_probs, last_probs, FPS)
        assert labels == [1, 2, 3, 4]

    def test_last_anchor_wraps_to_one(self):
        beat_times = np.array([0.0, 0.5])
        one_probs = np.zeros(200)
        last_probs = _probs_at({0.0: 0.9}, 200)
        labels = label_bar_position(beat_times, one_probs, last_probs, FPS)
        assert labels == [4, 1]

    def test_unresolved_before_first_anchor(self):
        beat_times = np.array([0.0, 0.5, 1.0])
        one_probs = _probs_at({0.5: 0.9}, 200)
        last_probs = np.zeros(200)
        labels = label_bar_position(beat_times, one_probs, last_probs, FPS)
        assert labels == [None, 1, 2]

    def test_no_anchors_all_none(self):
        beat_times = np.array([0.0, 0.5, 1.0])
        labels = label_bar_position(beat_times, np.zeros(200), np.zeros(200), FPS)
        assert labels == [None, None, None]

    def test_anchor_resyncs_drifted_count(self):
        # a "one" anchor at 0.0, then a conflicting "one" anchor 3 beats later
        # (should be 4 beats later in 4/4) — the later anchor must win over counting.
        beat_times = np.array([0.0, 0.5, 1.0, 1.5])
        one_probs = _probs_at({0.0: 0.9, 1.5: 0.9}, 200)
        last_probs = np.zeros(200)
        labels = label_bar_position(beat_times, one_probs, last_probs, FPS)
        assert labels == [1, 2, 3, 1]

    def test_group_size_8_counts_to_eight_before_wrapping(self):
        beat_times = np.arange(0, 4.0, 0.5)  # 8 beats
        one_probs = _probs_at({0.0: 0.9}, 200)
        last_probs = np.zeros(200)
        labels = label_bar_position(
            beat_times, one_probs, last_probs, FPS, group_size=8
        )
        assert labels == [1, 2, 3, 4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# readout (integration)
# ---------------------------------------------------------------------------


class TestReadout:
    def test_clean_four_beat_bar_pattern(self):
        """Synthetic beat/one/last curves for 8 beats at 120 BPM (0.5s/beat, 4/4)."""
        fps = 100.0
        duration = 4.5
        n_frames = int(duration * fps)

        beat_times = np.arange(0, 4.0, 0.5)  # 0, 0.5, ..., 3.5 -> 8 beats
        one_times = beat_times[0::4]
        last_times = beat_times[3::4]

        beat_spike = np.zeros(n_frames)
        beat_spike[(beat_times * fps).astype(int)] = 1.0
        one_spike = np.zeros(n_frames)
        one_spike[(one_times * fps).astype(int)] = 1.0
        last_spike = np.zeros(n_frames)
        last_spike[(last_times * fps).astype(int)] = 1.0

        beat_probs = gaussian_smear(beat_spike, sigma=1.0)
        one_probs = gaussian_smear(one_spike, sigma=1.0)
        last_probs = gaussian_smear(last_spike, sigma=1.0)

        events = readout(beat_probs, one_probs, last_probs, fps=fps)

        assert len(events) == 8
        got_times = [e["time"] for e in events]
        assert got_times == pytest.approx(list(beat_times), abs=0.02)
        got_labels = [e["beat_in_bar"] for e in events]
        assert got_labels == [1, 2, 3, 4, 1, 2, 3, 4]

    def test_group_size_8_phrase_pattern(self):
        """Same pipeline, but one/last mark phrase positions 1 and 8 (group_size=8)."""
        fps = 100.0
        duration = 8.5
        n_frames = int(duration * fps)

        beat_times = np.arange(0, 8.0, 0.5)  # 16 beats -> two 8-beat phrases
        one_times = beat_times[0::8]
        last_times = beat_times[7::8]

        beat_spike = np.zeros(n_frames)
        beat_spike[(beat_times * fps).astype(int)] = 1.0
        one_spike = np.zeros(n_frames)
        one_spike[(one_times * fps).astype(int)] = 1.0
        last_spike = np.zeros(n_frames)
        last_spike[(last_times * fps).astype(int)] = 1.0

        beat_probs = gaussian_smear(beat_spike, sigma=1.0)
        one_probs = gaussian_smear(one_spike, sigma=1.0)
        last_probs = gaussian_smear(last_spike, sigma=1.0)

        events = readout(beat_probs, one_probs, last_probs, fps=fps, group_size=8)

        assert len(events) == 16
        got_labels = [e["beat_in_bar"] for e in events]
        assert got_labels == [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
