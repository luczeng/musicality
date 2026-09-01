"""Tests for time-based (rather than index-based) bar-position advances.

Covers :func:`musicality.postprocess.phase_advances` and the ``advance="time"``
path of :func:`~musicality.postprocess.label_bar_position_global`, which
advances the bar count by elapsed time so a missed or spurious beat detection
no longer shifts the grid for the rest of the track.

Background: docs/switch_penalty_explained.md, docs/beat_phase_improvement_review.md.
"""

import numpy as np
import pytest

from musicality.postprocess import (
    label_bar_position_global,
    phase_advances,
    readout,
)


FPS = 10.0
PERIOD = 0.5  # seconds per beat — 120 BPM


def _grid(n_beats: int, period: float = PERIOD) -> np.ndarray:
    """A perfectly regular beat grid."""

    return np.arange(n_beats) * period


def _truth(beat_times: np.ndarray, group_size: int = 4) -> list[int]:
    """True bar position of each beat, from where it sits on the ideal grid."""

    return [(int(round(t / PERIOD)) % group_size) + 1 for t in np.asarray(beat_times)]


def _curves(
    grid: np.ndarray,
    group_size: int = 4,
    p: float = 0.92,
    floor: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """``one``/``last`` probability curves for a full, ideal grid.

    Built from the *grid*, not from whatever subset the peak-picker found —
    the audio still contains every beat even when the detector misses one.
    """

    n_frames = int(grid[-1] * FPS) + 20
    one = np.full(n_frames, floor)
    last = np.full(n_frames, floor)

    idx = np.round(grid / PERIOD).astype(int)
    frames = np.round(grid * FPS).astype(int)

    one[frames[idx % group_size == 0]] = p
    last[frames[idx % group_size == group_size - 1]] = p

    return one, last


class TestPhaseAdvances:
    """musicality.postprocess.phase_advances"""

    def test_regular_beats_all_advance_one(self):
        advances = phase_advances(_grid(24))

        assert advances.tolist() == [1] * 23

    def test_length_is_one_less_than_beat_count(self):
        assert len(phase_advances(_grid(24))) == 23
        assert len(phase_advances(_grid(3))) == 2

    def test_empty_and_single_beat_return_empty(self):
        assert len(phase_advances(np.array([]))) == 0
        assert len(phase_advances(np.array([1.5]))) == 0

    def test_returns_non_negative_integers(self):
        advances = phase_advances(_grid(24))

        assert np.issubdtype(advances.dtype, np.integer)
        assert (advances >= 0).all()

    def test_missing_beat_advances_two(self):
        """The detector dropped one beat: the gap is two periods wide."""

        detected = np.delete(_grid(24), 10)  # 4.5 -> 5.5

        assert phase_advances(detected)[9] == 2

    def test_two_consecutive_missing_beats_advance_three(self):
        detected = np.delete(_grid(24), [10, 11])  # 4.5 -> 6.0

        assert phase_advances(detected)[9] == 3

    def test_spurious_beat_advances_zero(self):
        """A near-duplicate detection must consume no bar position."""

        beats = np.sort(np.append(_grid(12), 5.12))
        advances = phase_advances(beats)

        assert advances[10] == 0  # 5.00 -> 5.12, a fifth of a period
        assert advances[11] == 1  # 5.12 -> 5.50, the rest of the beat

    def test_long_gap_advances_by_the_number_of_periods(self):
        beats = np.append(_grid(10), 7.0)  # 4.5 -> 7.0 is five periods

        assert phase_advances(beats)[-1] == 5

    def test_tracks_a_steep_tempo_ramp(self):
        """The period estimate must be local, not one constant for the track.

        Deliberately extreme — the period triples across the sequence, so a
        single global median sits ~2x away from both ends and rounds their
        ratios to 0 and 2 instead of 1. A running/local estimate keeps every
        advance at 1, which is the truth: no beat is missing here.
        """

        periods = np.linspace(0.30, 0.90, 23)
        beats = np.concatenate([[0.0], np.cumsum(periods)])

        assert phase_advances(beats).tolist() == [1] * 23


class TestTimeBasedAdvance:
    """label_bar_position_global(advance="time")"""

    def test_matches_index_advance_on_regular_beats(self):
        beats = _grid(24)
        one, last = _curves(beats)

        assert label_bar_position_global(
            beats, one, last, FPS, advance="time"
        ) == label_bar_position_global(beats, one, last, FPS, advance="index")

    def test_recovers_from_a_missing_beat(self):
        """The headline case, and the one docs/switch_penalty_explained.md draws.

        The music is plain 4/4 throughout; the detector simply misses one
        beat. Counting per detected beat slips by one from there to the end of
        the track. Counting elapsed time does not.
        """

        grid = _grid(24)
        detected = np.delete(grid, 10)
        one, last = _curves(grid)

        by_time = label_bar_position_global(detected, one, last, FPS, advance="time")
        by_index = label_bar_position_global(detected, one, last, FPS, advance="index")

        assert by_time == _truth(detected)
        assert by_index != _truth(detected)

    def test_recovers_from_a_spurious_beat(self):
        grid = _grid(24)
        detected = np.sort(np.append(grid, 5.12))
        one, last = _curves(grid)

        labels = label_bar_position_global(detected, one, last, FPS, advance="time")

        real = [label for label, t in zip(labels, detected) if not np.isclose(t, 5.12)]
        assert real == _truth(grid)

    def test_works_on_the_viterbi_path_too(self):
        grid = _grid(32)
        detected = np.delete(grid, 12)
        one, last = _curves(grid)

        labels = label_bar_position_global(
            detected, one, last, FPS, switch_penalty=2.0, advance="time"
        )

        assert labels == _truth(detected)

    def test_group_size_eight(self):
        grid = _grid(24)
        detected = np.delete(grid, 12)
        one, last = _curves(grid, group_size=8)

        labels = label_bar_position_global(
            detected, one, last, FPS, group_size=8, advance="time"
        )

        assert labels == _truth(detected, group_size=8)

    def test_rejects_unknown_advance(self):
        beats = _grid(8)
        one, last = _curves(beats)

        with pytest.raises(ValueError, match="advance"):
            label_bar_position_global(beats, one, last, FPS, advance="bogus")


class TestReadoutAdvance:
    """readout(decoder="global", advance=...) — parameter threading only."""

    @staticmethod
    def _probs(n_beats: int = 24):
        grid = _grid(n_beats)
        one, last = _curves(grid)

        beat = np.full(len(one), 0.02)
        beat[np.round(grid * FPS).astype(int)] = 0.95

        return beat, one, last

    def test_threads_advance_through(self):
        beat, one, last = self._probs()

        events = readout(beat, one, last, fps=FPS, decoder="global", advance="time")

        assert events
        assert all(e["beat_in_bar"] is not None for e in events)

    def test_rejects_unknown_advance(self):
        beat, one, last = self._probs()

        with pytest.raises(ValueError, match="advance"):
            readout(beat, one, last, fps=FPS, decoder="global", advance="bogus")
