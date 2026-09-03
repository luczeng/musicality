"""Tests for musicality.trainers.common's shared plumbing."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef
from musicality.splits.splitter import Splitter
from musicality.trainers.common import (
    build_checkpoint_callback,
    resolve_split_refs,
)


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


def _cfg(**overrides):
    base = {
        "checkpoint_dir": "ckpts/",
        "trainer": {},
        "wandb": {"run_name": None},
    }
    base.update(overrides)
    return OmegaConf.create(base)


class TestBuildCheckpointCallback:
    """Checkpoint filenames must stay on one path segment, and each run needs
    its own directory."""

    @staticmethod
    def _name(callback, epoch=139, loss=1.9543):
        return callback.format_checkpoint_name({"epoch": epoch, "val/loss": loss})

    def test_filename_has_no_path_separator(self):
        """The regression this guards: `val/loss` spliced into the filename
        made the slash a directory separator, so every checkpoint landed in
        its own folder and pruning left the folder behind."""

        callback = build_checkpoint_callback(_cfg(), "beat-phase")

        rendered = Path(self._name(callback))

        assert rendered.name == "beat-phase-epoch139-valloss1.9543.ckpt"
        # The whole point: the checkpoint sits directly in the run directory,
        # with no extra segment carved out of the metric name. `val/loss` is
        # still in the *template* as a placeholder — that is fine and needed;
        # what must not survive is the slash in the rendered path.
        assert rendered.parent == Path(callback.dirpath)
        assert callback.auto_insert_metric_name is False

    def test_epoch_and_loss_are_both_in_the_name(self):
        callback = build_checkpoint_callback(_cfg(), "tempo")

        stem = Path(self._name(callback, epoch=7, loss=0.5)).name
        assert stem.startswith("tempo-")
        assert "epoch07" in stem and "0.5000" in stem

    def test_each_run_gets_its_own_directory(self):
        callback = build_checkpoint_callback(
            _cfg(wandb={"run_name": "sweep-lr-3e4"}), "beat-phase"
        )

        assert Path(callback.dirpath).name == "sweep-lr-3e4"
        assert Path(callback.dirpath).parent.name == "ckpts"

    def test_falls_back_to_a_timestamp_when_the_run_is_unnamed(self):
        callback = build_checkpoint_callback(_cfg(), "beat-phase")

        run_dir = Path(callback.dirpath).name
        assert run_dir != "ckpts"
        assert len(run_dir) == 15 and run_dir[8] == "-"  # YYYYmmdd-HHMMSS

    def test_keeps_three_checkpoints_by_default(self):
        assert build_checkpoint_callback(_cfg(), "tempo").save_top_k == 3

    def test_save_top_k_is_configurable(self):
        callback = build_checkpoint_callback(_cfg(trainer={"save_top_k": 1}), "tempo")

        assert callback.save_top_k == 1

    def test_monitors_val_loss_and_keeps_the_lowest(self):
        callback = build_checkpoint_callback(_cfg(), "tempo")

        assert callback.monitor == "val/loss"
        assert callback.mode == "min"

    def test_works_without_a_wandb_section(self):
        cfg = OmegaConf.create({"checkpoint_dir": "ckpts/", "trainer": {}})

        assert build_checkpoint_callback(cfg, "tempo").save_top_k == 3
