"""Tests for tools.annotator.data — pure functions only, no I/O or mirdata."""

import numpy as np
import pytest

from tools.annotator.data import (
    TrackData,
    active_beat_position,
    add_beat,
    beats_per_bar,
    load_annotations,
    remove_beat,
    save_annotations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track(beat_times, beat_positions=None, tempo=120.0):
    return TrackData(
        dataset_name="test",
        track_id="t1",
        audio_path="/fake/t1.wav",
        tempo=tempo,
        beat_times=np.array(beat_times, dtype=float),
        beat_positions=(
            np.array(beat_positions, dtype=int) if beat_positions is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# beats_per_bar
# ---------------------------------------------------------------------------

class TestBeatsPerBar:
    def test_none_returns_4(self):
        assert beats_per_bar(None) == 4

    def test_empty_returns_4(self):
        assert beats_per_bar(np.array([])) == 4

    def test_waltz_returns_3(self):
        assert beats_per_bar(np.array([1, 2, 3, 1, 2, 3])) == 3

    def test_four_four(self):
        assert beats_per_bar(np.array([1, 2, 3, 4, 1, 2, 3, 4])) == 4


# ---------------------------------------------------------------------------
# active_beat_position
# ---------------------------------------------------------------------------

class TestActiveBeatPosition:
    def test_before_first_beat_returns_none(self):
        times = np.array([1.0, 2.0, 3.0])
        assert active_beat_position(times, None, 0.5) is None

    def test_exactly_at_first_beat(self):
        times = np.array([1.0, 2.0, 3.0])
        positions = np.array([1, 2, 3])
        assert active_beat_position(times, positions, 1.0) == 1

    def test_between_beats_returns_previous(self):
        times = np.array([1.0, 2.0, 3.0])
        positions = np.array([1, 2, 3])
        assert active_beat_position(times, positions, 1.7) == 1

    def test_downbeat_returns_1(self):
        times = np.array([0.0, 0.5, 1.0, 1.5])
        positions = np.array([1, 2, 1, 2])
        assert active_beat_position(times, positions, 1.0) == 1

    def test_no_positions_returns_sequential(self):
        times = np.array([0.0, 0.5, 1.0, 1.5])
        pos = active_beat_position(times, None, 1.0)
        assert 1 <= pos <= 4

    def test_empty_beats_returns_none(self):
        assert active_beat_position(np.array([]), None, 1.0) is None


# ---------------------------------------------------------------------------
# add_beat
# ---------------------------------------------------------------------------

class TestAddBeat:
    def test_beat_is_inserted_sorted(self):
        track = _track([1.0, 3.0], [1, 1])
        result = add_beat(track, 2.0)
        assert list(result.beat_times) == [1.0, 2.0, 3.0]

    def test_positions_recomputed(self):
        track = _track([1.0, 3.0], [1, 2])
        result = add_beat(track, 2.0)
        assert len(result.beat_positions) == 3

    def test_original_track_unchanged(self):
        track = _track([1.0, 3.0], [1, 2])
        _ = add_beat(track, 2.0)
        assert len(track.beat_times) == 2

    def test_beats_per_bar_preserved(self):
        track = _track([0.0, 0.33, 0.67], [1, 2, 3])
        result = add_beat(track, 1.0)
        assert beats_per_bar(result.beat_positions) == 3


# ---------------------------------------------------------------------------
# remove_beat
# ---------------------------------------------------------------------------

class TestRemoveBeat:
    def test_removes_nearest_within_tolerance(self):
        track = _track([1.0, 2.0, 3.0], [1, 2, 3])
        result = remove_beat(track, 2.05, tolerance=0.1)
        assert len(result.beat_times) == 2
        assert 2.0 not in result.beat_times

    def test_no_removal_outside_tolerance(self):
        track = _track([1.0, 3.0], [1, 2])
        result = remove_beat(track, 2.0, tolerance=0.1)
        assert len(result.beat_times) == 2

    def test_empty_beats_returns_unchanged(self):
        track = _track([])
        result = remove_beat(track, 1.0)
        assert len(result.beat_times) == 0

    def test_original_track_unchanged(self):
        track = _track([1.0, 2.0, 3.0], [1, 2, 3])
        _ = remove_beat(track, 1.0)
        assert len(track.beat_times) == 3


# ---------------------------------------------------------------------------
# save / load annotations
# ---------------------------------------------------------------------------

class TestSaveLoadAnnotations:
    def test_round_trip_with_positions(self, tmp_path):
        track = _track([1.0, 2.0, 3.0], [1, 2, 3])
        path = tmp_path / "ann.json"
        save_annotations(track, path)
        loaded = load_annotations(path)
        np.testing.assert_array_almost_equal(loaded["beat_times"], track.beat_times)
        np.testing.assert_array_equal(loaded["beat_positions"], track.beat_positions)

    def test_round_trip_no_positions(self, tmp_path):
        track = _track([1.0, 2.0])
        path = tmp_path / "ann.json"
        save_annotations(track, path)
        loaded = load_annotations(path)
        assert loaded["beat_positions"] is None

    def test_creates_parent_dirs(self, tmp_path):
        track = _track([1.0])
        path = tmp_path / "deep" / "dir" / "ann.json"
        save_annotations(track, path)
        assert path.exists()

    def test_tempo_preserved(self, tmp_path):
        track = _track([1.0], tempo=98.6)
        path = tmp_path / "ann.json"
        save_annotations(track, path)
        loaded = load_annotations(path)
        assert loaded["tempo"] == pytest.approx(98.6)
