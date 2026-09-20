"""Tests for tools.merge_datasets — combining several datasets' existing
splits into one merged split (train tracks from every source's train split,
val tracks from every source's val split), without writing anything under
DATA_DIR.
"""

import pytest

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef
from musicality.splits.splitter import Splitter
from tools.merge_datasets import dedupe_refs, merge


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


class TestOverlappingSources:
    """Two sources can hold the same track — a dataset and its '-binary'
    variant, a dataset and a '--contains' subset of it, or the same name
    given twice. The merged split must still list each track once, on one
    side."""

    def test_repeated_track_is_written_once(self, _splits_dir):
        Splitter.save_refs(
            _splits_dir,
            "swing",
            _refs(("swing", "a"), ("swing", "b")),
            _refs(("swing", "c")),
        )
        # Same tracks, same partition — nothing was dropped by the meter filter.
        Splitter.save_refs(
            _splits_dir,
            "swing-binary",
            _refs(("swing", "a"), ("swing", "b")),
            _refs(("swing", "c")),
        )

        merge(["swing", "swing-binary"], "all", binary_only=False, force=False)

        train_refs, val_refs = Splitter.load_refs(_splits_dir, "all")
        assert [(r.dataset_name, r.track_id) for r in train_refs] == [
            ("swing", "a"),
            ("swing", "b"),
        ]
        assert [(r.dataset_name, r.track_id) for r in val_refs] == [("swing", "c")]

    def test_same_source_twice_is_merged_once(self, _splits_dir):
        Splitter.save_refs(
            _splits_dir, "swing", _refs(("swing", "a")), _refs(("swing", "b"))
        )

        merge(["swing", "swing"], "all", binary_only=False, force=False)

        train_refs, val_refs = Splitter.load_refs(_splits_dir, "all")
        assert len(train_refs) == 1
        assert len(val_refs) == 1

    def test_sources_disagreeing_about_a_track_abort_the_merge(self, _splits_dir):
        """Independently drawn partitions of the same pool put some tracks in
        train on one side and val on the other. Merging them would train on a
        held-out track, so it must fail rather than dedupe to an arbitrary
        side."""

        Splitter.save_refs(
            _splits_dir,
            "swing",
            _refs(("swing", "a"), ("swing", "b")),
            _refs(("swing", "c")),
        )
        Splitter.save_refs(
            _splits_dir,
            "swing-binary",
            _refs(("swing", "a"), ("swing", "c")),
            _refs(("swing", "b")),
        )

        with pytest.raises(RuntimeError, match="train split of one source"):
            merge(["swing", "swing-binary"], "all", binary_only=False, force=False)

        assert not (_splits_dir / "all").exists()


class TestDedupeRefs:
    def test_keeps_first_occurrence_and_counts_the_rest(self):
        refs = _refs(("swing", "a"), ("swing", "b"), ("swing", "a"), on_disk=False)

        unique, n_dropped = dedupe_refs(refs)

        assert [r.track_id for r in unique] == ["a", "b"]
        assert n_dropped == 1

    def test_same_track_id_in_two_datasets_is_not_a_duplicate(self):
        refs = _refs(("swing", "a"), ("ballroom", "a"), on_disk=False)

        unique, n_dropped = dedupe_refs(refs)

        assert len(unique) == 2
        assert n_dropped == 0
