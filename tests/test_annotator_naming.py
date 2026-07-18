"""Tests for tools.annotator.naming — pure functions only, no I/O."""

import re

from tools.annotator.naming import generate_track_id, sanitize_track_name


class TestSanitizeTrackName:
    def test_passes_through_safe_name(self):
        assert sanitize_track_name("track1") == "track1"

    def test_replaces_spaces(self):
        assert sanitize_track_name("my recording") == "my_recording"

    def test_replaces_special_characters(self):
        assert sanitize_track_name("song #1 (live)!") == "song__1__live__"

    def test_keeps_hyphens_and_underscores(self):
        assert sanitize_track_name("field-take_2") == "field-take_2"

    def test_strips_surrounding_whitespace(self):
        assert sanitize_track_name("  padded  ") == "padded"

    def test_empty_string_falls_back(self):
        assert sanitize_track_name("") == "recording"

    def test_whitespace_only_falls_back(self):
        assert sanitize_track_name("   ") == "recording"


class TestGenerateTrackId:
    def test_matches_expected_pattern(self):
        assert re.fullmatch(r"field_\d{8}_\d{6}", generate_track_id())

    def test_is_a_valid_sanitized_name(self):
        track_id = generate_track_id()
        assert sanitize_track_name(track_id) == track_id
