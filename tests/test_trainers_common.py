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
    def test_bare_name_resolves_under_splits_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        train_refs = _refs(("ballroom", "a"))
        val_refs = _refs(("ballroom", "b"))
        Splitter.save_refs(splits_dir, "ballroom", train_refs, val_refs)

        cfg = OmegaConf.create({"data": {"input": "ballroom"}})
        loaded_train, loaded_val = resolve_split_refs(cfg, splits_dir, "ballroom")

        assert loaded_train == train_refs
        assert loaded_val == val_refs

    def test_path_input_bypasses_splits_dir_entirely(self, monkeypatch, tmp_path):
        """Anything containing a "/" is read as a literal folder path, not a
        name — even if a split of the same bare name exists under splits_dir."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        # A split registered under the canonical name — should be ignored.
        Splitter.save_refs(splits_dir, "ballroom", _refs(("ballroom", "wrong")), [])

        # An ad hoc split living elsewhere entirely.
        adhoc_dir = tmp_path / "experiments" / "my_split"
        adhoc_train = _refs(("ballroom", "a"), ("brid", "b"))
        Splitter.save_refs(adhoc_dir.parent, "my_split", adhoc_train, [])

        cfg = OmegaConf.create({"data": {"input": str(adhoc_dir)}})
        # split_name (the "as-a-name" resolution) is passed through but must
        # be ignored once input is recognized as a path.
        loaded_train, loaded_val = resolve_split_refs(cfg, splits_dir, "ballroom")

        assert loaded_train == adhoc_train
        assert loaded_val == []

    def test_relative_path_input_is_also_recognized(self, monkeypatch, tmp_path):
        """The "/" check doesn't care whether the path is absolute or
        relative — only that it contains a separator at all."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        adhoc_train = _refs(("ballroom", "a"))
        Splitter.save_refs(tmp_path, "my_split", adhoc_train, [])

        monkeypatch.chdir(tmp_path)
        cfg = OmegaConf.create({"data": {"input": "./my_split"}})
        loaded_train, _ = resolve_split_refs(cfg, splits_dir, "irrelevant")

        assert loaded_train == adhoc_train

    def test_missing_path_input_raises(self, tmp_path):
        cfg = OmegaConf.create({"data": {"input": str(tmp_path / "nope")}})

        with pytest.raises(FileNotFoundError):
            resolve_split_refs(cfg, tmp_path / "splits", "ballroom")
