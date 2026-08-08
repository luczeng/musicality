"""Tests for tools.annotator.data's track metadata persistence."""

import musicality.dataformats as dataformats
import tools.annotator.data as annotator_data
from tools.annotator.data import (
    TrackMetadata,
    load_metadata,
    metadata_path,
    save_metadata,
)


class TestMetadataPath:
    def test_path_matches_annotations_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        path = metadata_path("swing", "take1")
        assert path == tmp_path / "swing" / "annotations" / "take1.meta.json"


class TestSaveLoadMetadata:
    def test_round_trips_all_fields(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        metadata = TrackMetadata(
            location="The Blue Room", device="iPhone 13 mini", structure="blues"
        )
        save_metadata("swing", "take1", metadata)
        assert load_metadata("swing", "take1") == metadata

    def test_round_trips_partial_fields(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        metadata = TrackMetadata(structure="swing")
        save_metadata("swing", "take2", metadata)
        assert load_metadata("swing", "take2") == metadata

    def test_creates_parent_dirs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        save_metadata("new_dataset", "take1", TrackMetadata(device="desktop"))
        assert (tmp_path / "new_dataset" / "annotations" / "take1.meta.json").exists()

    def test_missing_metadata_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        assert load_metadata("swing", "never_saved") is None


class TestMetadataPathUsesConfig:
    def test_path_uses_configured_dirname_and_suffix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        monkeypatch.setattr(dataformats.FORMAT, "annotations_dirname", "notes")
        monkeypatch.setattr(dataformats.FORMAT, "metadata_suffix", ".info.json")
        path = metadata_path("swing", "take1")
        assert path == tmp_path / "swing" / "notes" / "take1.info.json"


class TestSchemaVersion:
    def test_new_metadata_defaults_to_version_1(self):
        assert TrackMetadata().schema_version == 1

    def test_save_stamps_current_version(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        metadata = TrackMetadata(structure="swing")
        save_metadata("swing", "take1", metadata)
        assert metadata.schema_version == annotator_data.METADATA_SCHEMA_VERSION
        assert load_metadata("swing", "take1").schema_version == (
            annotator_data.METADATA_SCHEMA_VERSION
        )

    def test_pre_versioning_file_loads_as_version_1(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        path = metadata_path("swing", "legacy")
        path.parent.mkdir(parents=True)
        path.write_text('{"structure": "blues"}')
        assert load_metadata("swing", "legacy").schema_version == 1


# ---------------------------------------------------------------------------
# metadata_path — per-annotator slots
# ---------------------------------------------------------------------------


class TestMetadataPathAnnotatorSlot:
    def test_default_slot_when_annotator_id_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        path = metadata_path("swing", "take1")
        assert path == tmp_path / "swing" / "annotations" / "take1.meta.json"

    def test_nests_under_annotator_id(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        path = metadata_path("swing", "take1", annotator_id="alice")
        assert path == tmp_path / "swing" / "annotations" / "alice" / "take1.meta.json"


# ---------------------------------------------------------------------------
# save_metadata / load_metadata — per-annotator slots
# ---------------------------------------------------------------------------


class TestMetadataAnnotatorSlot:
    def test_saves_under_metadata_annotator_id(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        metadata = TrackMetadata(structure="swing", annotator_id="alice")
        save_metadata("swing", "take1", metadata)
        assert metadata_path("swing", "take1", "alice").exists()
        assert not metadata_path("swing", "take1").exists()

    def test_default_and_named_slots_dont_collide(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        save_metadata("swing", "take1", TrackMetadata(structure="swing"))
        save_metadata(
            "swing", "take1", TrackMetadata(structure="blues", annotator_id="alice")
        )

        assert load_metadata("swing", "take1").structure == "swing"
        assert load_metadata("swing", "take1", "alice").structure == "blues"

    def test_load_missing_named_slot_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        save_metadata("swing", "take1", TrackMetadata(structure="swing"))
        assert load_metadata("swing", "take1", "alice") is None


# ---------------------------------------------------------------------------
# section_aligned
# ---------------------------------------------------------------------------


class TestSectionAligned:
    def test_defaults_to_none(self):
        assert TrackMetadata().section_aligned is None

    def test_round_trips_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        metadata = TrackMetadata(section_aligned=True)
        save_metadata("swing", "take1", metadata)
        assert load_metadata("swing", "take1").section_aligned is True

    def test_round_trips_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        metadata = TrackMetadata(section_aligned=False)
        save_metadata("swing", "take1", metadata)
        assert load_metadata("swing", "take1").section_aligned is False
