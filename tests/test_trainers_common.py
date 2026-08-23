"""Tests for musicality.trainers.common's shared plumbing."""

import pytest
from omegaconf import OmegaConf

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef
from musicality.splits.splitter import Splitter
from musicality.trainers.common import resolve_split_refs


def _refs(*pairs):
    return [
        TrackRef(name, track_id, dataformats.DATA_DIR / name)
        for name, track_id in pairs
    ]


class TestResolveSplitRefs:
    def test_falls_back_to_splits_dir_lookup_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        train_refs = _refs(("ballroom", "a"))
        val_refs = _refs(("ballroom", "b"))
        Splitter.save_refs(splits_dir, "ballroom", train_refs, val_refs)

        cfg = OmegaConf.create({"data": {"name": "ballroom"}})
        loaded_train, loaded_val = resolve_split_refs(cfg, splits_dir, "ballroom")

        assert loaded_train == train_refs
        assert loaded_val == val_refs

    def test_split_dir_overrides_name_lookup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        # A split registered under the canonical name — should be ignored.
        Splitter.save_refs(splits_dir, "ballroom", _refs(("ballroom", "wrong")), [])

        # An ad hoc split living elsewhere entirely.
        adhoc_dir = tmp_path / "experiments" / "my_split"
        adhoc_train = _refs(("ballroom", "a"), ("brid", "b"))
        Splitter.save_refs(adhoc_dir.parent, "my_split", adhoc_train, [])

        cfg = OmegaConf.create(
            {"data": {"name": "ballroom", "split_dir": str(adhoc_dir)}}
        )
        loaded_train, loaded_val = resolve_split_refs(cfg, splits_dir, "ballroom")

        assert loaded_train == adhoc_train
        assert loaded_val == []

    def test_null_split_dir_falls_back_to_name_lookup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        train_refs = _refs(("ballroom", "a"))
        Splitter.save_refs(splits_dir, "ballroom", train_refs, [])

        cfg = OmegaConf.create({"data": {"name": "ballroom", "split_dir": None}})
        loaded_train, _ = resolve_split_refs(cfg, splits_dir, "ballroom")

        assert loaded_train == train_refs

    def test_missing_split_dir_raises(self, tmp_path):
        cfg = OmegaConf.create(
            {"data": {"name": "ballroom", "split_dir": str(tmp_path / "nope")}}
        )

        with pytest.raises(FileNotFoundError):
            resolve_split_refs(cfg, tmp_path / "splits", "ballroom")
