"""Tests for tools.merge_datasets — combining several datasets' existing
splits into one merged split (train tracks from every source's train split,
val tracks from every source's val split), without writing anything under
DATA_DIR.
"""

import pytest

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef
from musicality.splits.splitter import Splitter
from tools.merge_datasets import merge


def _refs(*pairs):
    return [
        TrackRef(name, track_id, dataformats.DATA_DIR / name)
        for name, track_id in pairs
    ]


@pytest.fixture(autouse=True)
def _splits_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
    splits_dir = tmp_path / "splits"
    monkeypatch.setattr("tools.merge_datasets.SPLITS_DIR", splits_dir)
    return splits_dir


class TestMerge:
    def test_concatenates_train_and_val_across_sources(self, _splits_dir):
        Splitter.save_refs(
            _splits_dir,
            "ballroom",
            _refs(("ballroom", "a"), ("ballroom", "b")),
            _refs(("ballroom", "c")),
        )
        Splitter.save_refs(
            _splits_dir, "brid", _refs(("brid", "x")), _refs(("brid", "y"))
        )

        merge(
            ["ballroom", "brid"],
            "ballroom_brid",
            ["tempo"],
            binary_only=False,
            force=False,
        )

        train_refs, val_refs = Splitter.load_refs(_splits_dir, "ballroom_brid")
        assert {(r.dataset_name, r.track_id) for r in train_refs} == {
            ("ballroom", "a"),
            ("ballroom", "b"),
            ("brid", "x"),
        }
        assert {(r.dataset_name, r.track_id) for r in val_refs} == {
            ("ballroom", "c"),
            ("brid", "y"),
        }

    def test_raises_before_writing_if_a_source_is_missing_a_split(self, _splits_dir):
        Splitter.save_refs(_splits_dir, "ballroom", _refs(("ballroom", "a")), [])
        # "brid" has no split at all.

        with pytest.raises(RuntimeError):
            merge(
                ["ballroom", "brid"],
                "ballroom_brid",
                ["tempo"],
                binary_only=False,
                force=False,
            )

        assert not (_splits_dir / "ballroom_brid").exists()

    def test_raises_if_output_exists_without_force(self, _splits_dir):
        Splitter.save_refs(_splits_dir, "ballroom", _refs(("ballroom", "a")), [])
        Splitter.save_refs(_splits_dir, "brid", _refs(("brid", "b")), [])
        Splitter.save_refs(_splits_dir, "ballroom_brid", _refs(("ballroom", "a")), [])

        with pytest.raises(FileExistsError):
            merge(
                ["ballroom", "brid"],
                "ballroom_brid",
                ["tempo"],
                binary_only=False,
                force=False,
            )

    def test_force_overwrites_existing_output(self, _splits_dir):
        Splitter.save_refs(_splits_dir, "ballroom", _refs(("ballroom", "a")), [])
        Splitter.save_refs(_splits_dir, "brid", _refs(("brid", "b")), [])
        Splitter.save_refs(_splits_dir, "ballroom_brid", _refs(("stale", "z")), [])

        merge(
            ["ballroom", "brid"],
            "ballroom_brid",
            ["tempo"],
            binary_only=False,
            force=True,
        )

        train_refs, _ = Splitter.load_refs(_splits_dir, "ballroom_brid")
        assert {(r.dataset_name, r.track_id) for r in train_refs} == {
            ("ballroom", "a"),
            ("brid", "b"),
        }

    def test_beat_kind_uses_beat_phase_split_names(self, _splits_dir):
        Splitter.save_refs(
            _splits_dir, "beat_phase-ballroom", _refs(("ballroom", "a")), []
        )
        Splitter.save_refs(_splits_dir, "beat_phase-brid", _refs(("brid", "b")), [])

        merge(
            ["ballroom", "brid"],
            "ballroom_brid",
            ["beat"],
            binary_only=False,
            force=False,
        )

        train_refs, _ = Splitter.load_refs(_splits_dir, "beat_phase-ballroom_brid")
        assert {(r.dataset_name, r.track_id) for r in train_refs} == {
            ("ballroom", "a"),
            ("brid", "b"),
        }
        assert not (_splits_dir / "ballroom_brid").exists()

    def test_no_dataset_directory_is_written(self, _splits_dir, monkeypatch, tmp_path):
        Splitter.save_refs(_splits_dir, "ballroom", _refs(("ballroom", "a")), [])
        Splitter.save_refs(_splits_dir, "brid", _refs(("brid", "b")), [])

        merge(
            ["ballroom", "brid"],
            "ballroom_brid",
            ["tempo"],
            binary_only=False,
            force=False,
        )

        assert not (dataformats.DATA_DIR / "ballroom_brid").exists()
