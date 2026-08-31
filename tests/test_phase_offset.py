"""Tests for musicality.metrics.phase_offset.phase_offset_profile."""

import numpy as np

from musicality.metrics.phase_offset import phase_offset_profile


def _events(times, labels):
    return [{"time": t, "beat_in_bar": p} for t, p in zip(times, labels)]


class TestPhaseOffsetProfile:
    def test_perfect_prediction_is_offset_zero_and_fully_stable(self):
        times = np.arange(16) * 0.5
        positions = np.array([(i % 4) + 1 for i in range(16)])

        profile = phase_offset_profile(times, positions, _events(times, positions))

        assert profile["modal_offset"] == 0
        assert profile["stability"] == 1.0
        assert profile["correct_fraction"] == 1.0
        assert profile["n_matched"] == 16

    def test_whole_track_half_cycle_shift(self):
        times = np.arange(16) * 0.5
        positions = np.array([(i % 4) + 1 for i in range(16)])
        shifted = np.array([((i + 2) % 4) + 1 for i in range(16)])

        profile = phase_offset_profile(times, positions, _events(times, shifted))

        assert profile["modal_offset"] == 2  # group_size // 2
        assert profile["stability"] == 1.0  # stable, just wrong
        assert profile["correct_fraction"] == 0.0

    def test_off_by_one_is_visible_here(self):
        """confusion_half_cycle_rate is blind to this; the profile is not."""

        times = np.arange(16) * 0.5
        positions = np.array([(i % 4) + 1 for i in range(16)])
        shifted = np.array([((i + 1) % 4) + 1 for i in range(16)])

        profile = phase_offset_profile(times, positions, _events(times, shifted))

        assert profile["modal_offset"] == 1
        assert profile["stability"] == 1.0

    def test_mid_track_flip_lowers_stability(self):
        times = np.arange(16) * 0.5
        positions = np.array([(i % 4) + 1 for i in range(16)])
        # Correct for the first half, half-cycle out for the second.
        flipped = np.array(
            [(i % 4) + 1 if i < 8 else ((i + 2) % 4) + 1 for i in range(16)]
        )

        profile = phase_offset_profile(times, positions, _events(times, flipped))

        assert profile["stability"] == 0.5
        assert profile["histogram"][0] == 8
        assert profile["histogram"][2] == 8

    def test_unmatched_and_unresolved_beats_are_skipped(self):
        times = np.array([0.0, 0.5, 1.0, 1.5])
        positions = np.array([1, 2, 3, 4])
        events = _events([0.0, 0.5, 1.0, 9.9], [1, None, 3, 4])

        profile = phase_offset_profile(times, positions, events)

        assert profile["n_matched"] == 2  # None label and the far-away beat dropped

    def test_returns_none_when_nothing_matches(self):
        times = np.array([0.0, 0.5])
        positions = np.array([1, 2])

        assert phase_offset_profile(times, positions, []) is None
        assert (
            phase_offset_profile(times, positions, _events([50.0, 51.0], [1, 2]))
            is None
        )

    def test_group_size_eight(self):
        times = np.arange(16) * 0.5
        positions = np.array([(i % 8) + 1 for i in range(16)])
        shifted = np.array([((i + 4) % 8) + 1 for i in range(16)])

        profile = phase_offset_profile(
            times, positions, _events(times, shifted), group_size=8
        )

        assert len(profile["histogram"]) == 8
        assert profile["modal_offset"] == 4
