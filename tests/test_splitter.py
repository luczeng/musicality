"""Tests for musicality.splits.splitter — the on-disk train/val split format
(dataset_name/track_id lines, not positional indices) that lets a split be
read back, concatenated, or merged across datasets independently of any one
dataset instance's ordering.
"""

import pytest
from torch.utils.data import Dataset

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackMetadata, TrackRef, save_metadata
from musicality.splits.splitter import MissingTrackDataError, Splitter, split_name


class _FakeDataset(Dataset):
    """Minimal Dataset exposing .refs, as TempoDataset/BeatDataset do."""

    def __init__(self, refs):
        self.refs = list(refs)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        return self.refs[idx]


def _write_track_files(ref):
    """Create the files a split entry must resolve to — audio plus its
    default-slot annotation. Split reads verify both exist (see
    ``musicality.splits.splitter.verify_refs_present``), so a ref that isn't
    backed by files is a *missing data* ref, not a generic one.
    """

    fmt = dataformats.FORMAT
    paths = (
        ref.data_home / fmt.tracks_dirname / f"{ref.track_id}.wav",
        ref.data_home / fmt.annotations_dirname / f"{ref.track_id}{fmt.beats_suffix}",
    )

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _refs(*pairs, on_disk=True):
    """Build refs, materializing their files unless ``on_disk=False`` — which
    is how a test stages the partial-data-pull case."""

    refs = [
        TrackRef(name, track_id, dataformats.DATA_DIR / name)
        for name, track_id in pairs
    ]

    if on_disk:
        for ref in refs:
            _write_track_files(ref)

    return refs


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


class TestSplitName:
    """One split per dataset serves every task; only ``binary_only`` changes
    which tracks a split holds, so only it is folded into the name."""

    def test_plain_name_is_the_dataset_name(self):
        assert split_name("merge") == "merge"

    def test_binary_only_gets_its_own_split(self):
        assert split_name("merge", binary_only=True) == "merge-binary"

    def test_the_task_is_not_part_of_the_name(self):
        """A tempo run, a beat-only run and a beat-phase run over one dataset
        must hold out the same tracks, or their numbers aren't comparable."""

        assert split_name("ballroom", binary_only=False) == "ballroom"

    def test_composes_with_a_contains_subset_name(self):
        assert split_name("gtzan-blues", binary_only=True) == "gtzan-blues-binary"


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


# ---------------------------------------------------------------------------
# Missing data (a partial dvc pull)
# ---------------------------------------------------------------------------


class TestMissingTrackData:
    """A split names the tracks a run uses. If some of them aren't on this
    machine — the usual cause being a `dvc pull` that fetched some corpora
    and not others — every read path fails instead of quietly training on
    whatever happens to be there.
    """

    def test_load_refs_raises_when_audio_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        Splitter.save_refs(
            splits_dir,
            "ballroom",
            _refs(("ballroom", "a")) + _refs(("ballroom", "b"), on_disk=False),
            [],
        )

        with pytest.raises(MissingTrackDataError):
            Splitter.load_refs(splits_dir, "ballroom")

    def test_load_refs_raises_when_only_the_annotation_is_missing(
        self, monkeypatch, tmp_path
    ):
        """Audio alone isn't enough — a track with no .beats file carries no
        supervision, and the loader would crash on it mid-epoch."""

        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        (ref,) = _refs(("ballroom", "a"))
        beats = (
            ref.data_home
            / dataformats.FORMAT.annotations_dirname
            / f"a{dataformats.FORMAT.beats_suffix}"
        )
        beats.unlink()

        Splitter.save_refs(splits_dir, "ballroom", [ref], [])

        with pytest.raises(MissingTrackDataError):
            Splitter.load_refs(splits_dir, "ballroom")

    def test_a_missing_val_side_also_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        Splitter.save_refs(
            splits_dir,
            "ballroom",
            _refs(("ballroom", "a")),
            _refs(("ballroom", "c"), on_disk=False),
        )

        with pytest.raises(MissingTrackDataError):
            Splitter.load_refs(splits_dir, "ballroom")

    def test_error_names_the_corpus_and_its_share(self, monkeypatch, tmp_path):
        """A partial pull takes out whole corpora, so the message has to say
        which one and how much of it — that's what turns "some tracks are
        missing" into "gtzan never arrived"."""

        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        present = _refs(("ballroom", "a"), ("ballroom", "b"))
        absent = _refs(
            ("gtzan", "blues_00001"), ("gtzan", "blues_00002"), on_disk=False
        )
        Splitter.save_refs(splits_dir, "merged", present + absent, [])

        with pytest.raises(MissingTrackDataError) as excinfo:
            Splitter.load_refs(splits_dir, "merged")

        message = str(excinfo.value)
        assert "2 of 4 track(s)" in message
        assert "gtzan: 2 of 2 track(s)" in message
        assert "blues_00001.wav" in message
        assert "merged/train.txt" in message

    def test_run_raises_too(self, monkeypatch, tmp_path):
        """The Subset-returning path used by eval — it used to drop the
        missing tracks with a printed warning."""

        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        refs = _refs(("ballroom", "a")) + _refs(("ballroom", "b"), on_disk=False)
        Splitter.save_refs(splits_dir, "ballroom", refs, [])

        with pytest.raises(MissingTrackDataError):
            Splitter(_FakeDataset(refs), splits_dir, "ballroom", 0.5).run()

    def test_verify_false_reads_the_split_anyway(self, monkeypatch, tmp_path):
        """The annotator reads splits for their train/val badges only, and
        must keep working on a machine holding part of the data."""

        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        split_path = tmp_path / "splits" / "ballroom"

        train_refs = _refs(("ballroom", "a"), ("ballroom", "b"), on_disk=False)
        Splitter.save_refs(split_path.parent, "ballroom", train_refs, [])

        loaded_train, loaded_val = Splitter.load_refs_from_dir(split_path, verify=False)

        assert loaded_train == train_refs
        assert loaded_val == []

    def test_flagged_tracks_are_not_required_on_disk(self, monkeypatch, tmp_path):
        """Flagged tracks are dropped before the check, so a track that is
        both flagged and absent is simply out of the split — not a failure."""

        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path / "data")
        splits_dir = tmp_path / "splits"

        refs = _refs(("ballroom", "a")) + _refs(("ballroom", "gone"), on_disk=False)
        Splitter.save_refs(splits_dir, "ballroom", refs, [])
        _flag("ballroom", "gone", warning=True)

        loaded_train, _ = Splitter.load_refs(splits_dir, "ballroom")

        assert [r.track_id for r in loaded_train] == ["a"]
