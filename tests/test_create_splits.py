"""Tests for tools.create_splits' naming of a ``--contains`` (subset) split.

The filtering itself lives in ``track_io.list_track_refs`` (covered by
tests/test_track_io.py); what's specific to this tool is that a narrowed
run writes to its own split name rather than over the dataset's full split.
"""

from tools.create_splits import split_base_name


class TestSplitBaseName:
    def test_no_filter_keeps_the_plain_dataset_name(self):
        assert split_base_name("gtzan", None) == "gtzan"

    def test_empty_filter_keeps_the_plain_dataset_name(self):
        assert split_base_name("gtzan", "") == "gtzan"

    def test_filter_gets_its_own_split_name(self):
        assert split_base_name("gtzan", "blues") == "gtzan-blues"

    def test_filter_is_sanitized_into_a_valid_directory_name(self):
        """A filter is free-form shell input; the split name it produces has
        to stay a legal single directory."""
        assert split_base_name("gtzan", "blues/rock 01") == "gtzan-blues_rock_01"
