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
            binary_only=False,
            force=True,
        )

        train_refs, _ = Splitter.load_refs(_splits_dir, "ballroom_brid")
        assert {(r.dataset_name, r.track_id) for r in train_refs} == {
            ("ballroom", "a"),
            ("brid", "b"),
        }

    def test_binary_only_reads_and_writes_the_binary_variant(self, _splits_dir):
        """`binary_only` selects a different split on both ends: the sources
        it reads and the merged split it writes. Mixing the two variants would
        put meter-mixed tracks into a split a binary-only run holds out."""

        Splitter.save_refs(_splits_dir, "ballroom-binary", _refs(("ballroom", "a")), [])
        Splitter.save_refs(_splits_dir, "brid-binary", _refs(("brid", "b")), [])

        merge(["ballroom", "brid"], "ballroom_brid", binary_only=True, force=False)

        train_refs, _ = Splitter.load_refs(_splits_dir, "ballroom_brid-binary")
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
            binary_only=False,
            force=False,
        )

        assert not (dataformats.DATA_DIR / "ballroom_brid").exists()
