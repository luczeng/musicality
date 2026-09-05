"""Tests for musicality.splits.splitter — the on-disk train/val split format
(dataset_name/track_id lines, not positional indices) that lets a split be
read back, concatenated, or merged across datasets independently of any one
dataset instance's ordering.
"""

import pytest
from torch.utils.data import Dataset

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackMetadata, TrackRef, save_metadata
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


def _flag(dataset_name, track_id, *, warning=False, needs_review=False):
    """Write a metadata file marking a track as flagged (or not)."""

    save_metadata(
        dataset_name,
        track_id,
        TrackMetadata(warning=warning, needs_review=needs_review),
    )


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


class TestLoadRefsFromDir:
    def test_reads_train_val_from_arbitrary_directory(self, monkeypatch, tmp_path):
        """No splits_dir/name involved at all — any folder with train.txt/
        val.txt works, e.g. one built ad hoc outside the canonical splits_dir."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        split_path = tmp_path / "somewhere" / "my_split"

        train_refs = _refs(("ballroom", "a"), ("brid", "b"))
        val_refs = _refs(("ballroom", "c"))
        Splitter.save_refs(split_path.parent, "my_split", train_refs, val_refs)

        loaded_train, loaded_val = Splitter.load_refs_from_dir(split_path)

        assert loaded_train == train_refs
        assert loaded_val == val_refs

    def test_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Splitter.load_refs_from_dir(tmp_path / "nope")


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


# ---------------------------------------------------------------------------
# Flagged tracks (metadata warning / needs_review)
# ---------------------------------------------------------------------------


class TestFlaggedTracks:
    def test_create_excludes_flagged_tracks(self, monkeypatch, tmp_path):
        """A track flagged in the annotator never enters a newly created
        split — on either side."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        refs = _refs(*[("ballroom", f"t{i}") for i in range(10)])
        _flag("ballroom", "t3", warning=True)
        _flag("ballroom", "t7", needs_review=True)

        train_ds, val_ds = Splitter(
            _FakeDataset(refs), splits_dir, "ballroom", 0.25
        ).create()

        train_refs, val_refs = Splitter.load_refs(splits_dir, "ballroom")
        track_ids = {r.track_id for r in train_refs + val_refs}

        assert track_ids == {f"t{i}" for i in range(10)} - {"t3", "t7"}
        # 8 eligible tracks, 25% held out.
        assert (len(train_ds), len(val_ds)) == (6, 2)

    def test_create_returned_indices_index_into_the_full_dataset(
        self, monkeypatch, tmp_path
    ):
        """Filtering shifts positions in the eligible pool, but the returned
        Subsets must still resolve against the unfiltered dataset."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        refs = _refs(*[("ballroom", f"t{i}") for i in range(6)])
        _flag("ballroom", "t0", warning=True)

        dataset = _FakeDataset(refs)
        train_ds, val_ds = Splitter(dataset, splits_dir, "ballroom", 0.5).create()

        selected = {dataset.refs[i].track_id for i in train_ds.indices + val_ds.indices}
        assert selected == {f"t{i}" for i in range(1, 6)}

    def test_load_refs_drops_tracks_flagged_after_the_split_was_saved(
        self, monkeypatch, tmp_path, capsys
    ):
        """The split file is unchanged — flagging a track in the annotator
        takes it out of training without regenerating any split."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        train_refs = _refs(("ballroom", "a"), ("ballroom", "b"))
        val_refs = _refs(("ballroom", "c"))
        Splitter.save_refs(splits_dir, "ballroom", train_refs, val_refs)

        _flag("ballroom", "b", warning=True)
        _flag("ballroom", "c", needs_review=True)

        loaded_train, loaded_val = Splitter.load_refs(splits_dir, "ballroom")

        assert [r.track_id for r in loaded_train] == ["a"]
        assert loaded_val == []
        assert "flagged" in capsys.readouterr().out

    def test_unflagged_metadata_is_kept(self, monkeypatch, tmp_path):
        """Only a True flag excludes a track — an ordinary metadata file, or
        none at all, doesn't."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        refs = _refs(("ballroom", "a"), ("ballroom", "b"))
        Splitter.save_refs(splits_dir, "ballroom", refs, [])

        _flag("ballroom", "a", warning=False, needs_review=False)  # "b" has none

        loaded_train, _ = Splitter.load_refs(splits_dir, "ballroom")

        assert [r.track_id for r in loaded_train] == ["a", "b"]

    def test_run_skips_flagged_tracks(self, monkeypatch, tmp_path):
        """Same filtering through the Subset-returning path used at train time."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        refs = _refs(("ballroom", "a"), ("ballroom", "b"), ("ballroom", "c"))
        Splitter.save_refs(splits_dir, "ballroom", refs[:2], refs[2:])

        _flag("ballroom", "a", needs_review=True)

        dataset = _FakeDataset(refs)
        train_ds, val_ds = Splitter(dataset, splits_dir, "ballroom", 0.5).run()

        assert [dataset.refs[i].track_id for i in train_ds.indices] == ["b"]
        assert [dataset.refs[i].track_id for i in val_ds.indices] == ["c"]

    def test_load_refs_from_dir_skips_flagged_tracks(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        split_path = tmp_path / "somewhere" / "my_split"

        Splitter.save_refs(
            split_path.parent,
            "my_split",
            _refs(("ballroom", "a"), ("brid", "b")),
            _refs(("ballroom", "c")),
        )

        _flag("brid", "b", warning=True)

        loaded_train, loaded_val = Splitter.load_refs_from_dir(split_path)

        assert [r.track_id for r in loaded_train] == ["a"]
        assert [r.track_id for r in loaded_val] == ["c"]
