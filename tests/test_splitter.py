"""Tests for musicality.splits.splitter — the on-disk train/val split format
(dataset_name/track_id lines, not positional indices) that lets a split be
read back, concatenated, or merged across datasets independently of any one
dataset instance's ordering.
"""

import pytest
from torch.utils.data import Dataset

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef
from musicality.splits.splitter import Splitter


class _FakeDataset(Dataset):
    """Minimal Dataset exposing .refs, as TempoDataset/BeatDataset do."""

    def __init__(self, refs):
        self.refs = list(refs)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        return self.refs[idx]


def _refs(*pairs):
    return [
        TrackRef(name, track_id, dataformats.DATA_DIR / name)
        for name, track_id in pairs
    ]


# ---------------------------------------------------------------------------
# save_refs / load_refs
# ---------------------------------------------------------------------------


class TestSaveLoadRefs:
    def test_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        train_refs = _refs(("ballroom", "a"), ("ballroom", "b"))
        val_refs = _refs(("ballroom", "c"))

        Splitter.save_refs(splits_dir, "ballroom", train_refs, val_refs)
        loaded_train, loaded_val = Splitter.load_refs(splits_dir, "ballroom")

        assert loaded_train == train_refs
        assert loaded_val == val_refs

    def test_round_trip_across_multiple_source_datasets(self, monkeypatch, tmp_path):
        """A merged split's lines span several dataset_names — load_refs must
        resolve each one against its own name, not a single shared root."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        train_refs = _refs(("ballroom", "a"), ("brid", "b"))
        Splitter.save_refs(splits_dir, "ballroom_brid", train_refs, [])

        loaded_train, loaded_val = Splitter.load_refs(splits_dir, "ballroom_brid")

        assert loaded_train == train_refs
        assert loaded_val == []

    def test_load_refs_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Splitter.load_refs(tmp_path / "splits", "nope")


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


class TestCreate:
    def test_saves_and_returns_matching_split(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        refs = _refs(*[("ballroom", f"t{i}") for i in range(10)])
        dataset = _FakeDataset(refs)

        train_ds, val_ds = Splitter(dataset, splits_dir, "ballroom", 0.3).create()

        assert len(train_ds) + len(val_ds) == 10
        assert len(val_ds) == 3

        loaded_train, loaded_val = Splitter.load_refs(splits_dir, "ballroom")
        assert {r.track_id for r in loaded_train} == {
            refs[i].track_id for i in train_ds.indices
        }
        assert {r.track_id for r in loaded_val} == {
            refs[i].track_id for i in val_ds.indices
        }


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_loads_existing_split_as_subsets(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        refs = _refs(("ballroom", "a"), ("ballroom", "b"), ("ballroom", "c"))
        Splitter.save_refs(splits_dir, "ballroom", refs[:2], refs[2:])

        dataset = _FakeDataset(refs)
        train_ds, val_ds = Splitter(dataset, splits_dir, "ballroom", 0.5).run()

        assert [dataset.refs[i].track_id for i in train_ds.indices] == ["a", "b"]
        assert [dataset.refs[i].track_id for i in val_ds.indices] == ["c"]

    def test_raises_if_no_split(self, tmp_path):
        dataset = _FakeDataset([])
        with pytest.raises(FileNotFoundError):
            Splitter(dataset, tmp_path / "splits", "ballroom", 0.2).run()

    def test_drops_refs_missing_from_current_dataset(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        saved_refs = _refs(("ballroom", "a"), ("ballroom", "b"))
        Splitter.save_refs(splits_dir, "ballroom", saved_refs, [])

        # Current dataset only has "a" — "b" has since been removed.
        dataset = _FakeDataset(_refs(("ballroom", "a")))
        train_ds, val_ds = Splitter(dataset, splits_dir, "ballroom", 0.2).run()

        assert len(train_ds) == 1
        assert len(val_ds) == 0
        assert "dropped" in capsys.readouterr().out
