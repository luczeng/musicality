"""Tests for musicality.dataformats."""

import musicality.dataformats as dataformats


class TestLoad:
    def test_returns_dataformat_with_all_fields_populated(self):
        fmt = dataformats.load()

        assert fmt.data_dir
        assert fmt.splits_dir
        assert fmt.tracks_dirname
        assert fmt.annotations_dirname
        assert fmt.beats_suffix
        assert fmt.metadata_suffix
        assert fmt.manifest_filename

    def test_annotation_layout_matches_existing_convention(self):
        fmt = dataformats.load()

        assert fmt.tracks_dirname == "tracks"
        assert fmt.annotations_dirname == "annotations"
        assert fmt.beats_suffix == ".beats"
        assert fmt.metadata_suffix == ".meta.json"
        assert fmt.manifest_filename == "tracks.txt"
