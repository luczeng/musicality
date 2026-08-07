"""Tests for tools.annotator.data — pure functions only, no I/O or mirdata."""

import numpy as np
import pytest

import musicality.dataformats as dataformats
import tools.annotator.data as annotator_data
from tools.annotator.data import (
    DEFAULT_N_BEATS,
    TrackData,
    _read_beats_file,
    active_beat_position,
    add_beat,
    annotation_path,
    beats_per_bar,
    cycle_positions,
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
    def test_none_returns_default(self):
        assert beats_per_bar(None) == DEFAULT_N_BEATS

    def test_empty_returns_default(self):
        assert beats_per_bar(np.array([])) == DEFAULT_N_BEATS

    def test_custom_default(self):
        assert beats_per_bar(None, default=4) == 4

    def test_waltz_returns_3(self):
        assert beats_per_bar(np.array([1, 2, 3, 1, 2, 3])) == 3

    def test_four_four(self):
        assert beats_per_bar(np.array([1, 2, 3, 4, 1, 2, 3, 4])) == 4


# ---------------------------------------------------------------------------
# cycle_positions
# ---------------------------------------------------------------------------


class TestCyclePositions:
    def test_starts_at_1(self):
        positions = cycle_positions(3, 8)
        assert positions[0] == 1

    def test_cycles(self):
        positions = cycle_positions(10, 4)
        assert list(positions) == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]

    def test_empty(self):
        assert len(cycle_positions(0, 8)) == 0


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
        pos = active_beat_position(times, None, 1.0, default_n_beats=4)
        assert 1 <= pos <= 4

    def test_no_positions_uses_default_n_beats(self):
        times = np.array([0.0, 0.5, 1.0, 1.5])
        pos = active_beat_position(times, None, 1.0)
        assert 1 <= pos <= DEFAULT_N_BEATS

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

    def test_caller_supplied_n_beats_applied(self):
        """add_beat trusts the caller's n_beats rather than inferring it from
        the (possibly still-partial) existing positions."""
        track = _track([0.0, 0.33, 0.67], [1, 2, 3])
        result = add_beat(track, 1.0, n_beats=3)
        assert beats_per_bar(result.beat_positions) == 3

    def test_first_tap_starts_at_position_1(self):
        track = _track([])
        result = add_beat(track, 0.5)
        assert list(result.beat_positions) == [1]

    def test_fresh_track_cycles_with_n_beats(self):
        track = _track([])
        for t in range(10):
            track = add_beat(track, float(t), n_beats=4)
        assert list(track.beat_positions) == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]

    def test_default_n_beats_is_8(self):
        track = _track([])
        for t in range(9):
            track = add_beat(track, float(t))
        assert list(track.beat_positions) == [1, 2, 3, 4, 5, 6, 7, 8, 1]


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

    def test_remaining_positions_recycled_from_1(self):
        track = _track([1.0, 2.0, 3.0], [1, 2, 3])
        result = remove_beat(track, 1.0, tolerance=0.1, n_beats=3)
        assert list(result.beat_positions) == [1, 2]


# ---------------------------------------------------------------------------
# save / load annotations (mirdata's "<time> <position>" format)
# ---------------------------------------------------------------------------


class TestSaveLoadAnnotations:
    def test_round_trip_with_positions(self, tmp_path):
        track = _track([1.0, 2.0, 3.0], [1, 2, 3])
        path = tmp_path / "t1.beats"
        save_annotations(track, path)
        times, positions = _read_beats_file(path)
        np.testing.assert_array_almost_equal(times, track.beat_times)
        np.testing.assert_array_equal(positions, track.beat_positions)

    def test_round_trip_no_positions(self, tmp_path):
        track = _track([1.0, 2.0])
        path = tmp_path / "t1.beats"
        save_annotations(track, path)
        times, positions = _read_beats_file(path)
        np.testing.assert_array_almost_equal(times, track.beat_times)
        assert positions is None

    def test_creates_parent_dirs(self, tmp_path):
        track = _track([1.0])
        path = tmp_path / "deep" / "dir" / "t1.beats"
        save_annotations(track, path)
        assert path.exists()

    def test_file_format_matches_mirdata(self, tmp_path):
        """One '<time> <position>' pair per line, e.g. ballroom's raw .beats files."""
        track = _track([0.5, 1.25], [1, 2])
        path = tmp_path / "t1.beats"
        save_annotations(track, path)
        lines = path.read_text().splitlines()
        assert lines == ["0.500000 1", "1.250000 2"]

    def test_reads_legacy_timestamp_only_file(self, tmp_path):
        """Files saved before position tracking (bare timestamps) still load."""
        path = tmp_path / "t1.beats"
        path.write_text("1.000000\n2.000000\n3.000000")
        times, positions = _read_beats_file(path)
        np.testing.assert_array_almost_equal(times, [1.0, 2.0, 3.0])
        assert positions is None


# ---------------------------------------------------------------------------
# annotation_path
# ---------------------------------------------------------------------------


class TestAnnotationPathUsesConfig:
    def test_path_uses_configured_dirname_and_suffix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        monkeypatch.setattr(dataformats.FORMAT, "annotations_dirname", "notes")
        monkeypatch.setattr(dataformats.FORMAT, "beats_suffix", ".taps")
        track = _track([1.0])
        path = annotation_path(track)
        assert path == tmp_path / "test" / "notes" / "t1.taps"
