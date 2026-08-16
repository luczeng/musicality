"""Tests for musicality.inference: detect_task, load_module."""

import pytest
import torch
from omegaconf import OmegaConf

from musicality.inference import detect_task, load_module
from musicality.trainers.beat_module import BeatModule
from musicality.trainers.beat_phase_module import BeatPhaseModule


MODEL_CFG = OmegaConf.create(
    {
        "_target_": "musicality.models.tcn.TCNTempoNet",
        "n_mels": 16,
        "channels": 8,
        "n_layers": 3,
        "dropout": 0.0,
    }
)


class TestDetectTask:
    def test_beat_only(self):
        assert detect_task({"task": "beat_only"}) == "beat_only"

    def test_beat_phase(self):
        assert detect_task({"task": "beat_phase"}) == "beat_phase"

    def test_missing_task_field_raises(self):
        """Every checkpoint currently in the repo predates the explicit
        `task:` config field — this must raise, not silently guess."""

        with pytest.raises(KeyError):
            detect_task({"model": MODEL_CFG, "pos_weight": 6.0})

    def test_unrecognized_task_raises(self):
        with pytest.raises(ValueError):
            detect_task({"task": "bogus"})

    def test_tempo_task_raises(self):
        """A pooled tempo-regression/classification checkpoint's task must
        not be treated as beat-phase just because it exists."""

        with pytest.raises(ValueError):
            detect_task({"task": "tempo"})


class TestLoadModule:
    def _save_legacy_checkpoint(self, module, tmp_path, filename):
        """Rename frame_head.1.{weight,bias} back to frame_head.{weight,bias}
        to simulate a checkpoint saved before TCNTempoNet's frame head gained
        a dropout layer — the exact legacy format load_module must migrate."""

        state_dict = module.state_dict()
        state_dict["model.frame_head.weight"] = state_dict.pop(
            "model.frame_head.1.weight"
        )
        state_dict["model.frame_head.bias"] = state_dict.pop("model.frame_head.1.bias")

        path = tmp_path / filename
        torch.save(
            {
                "state_dict": state_dict,
                "hyper_parameters": dict(module.hparams),
                "pytorch-lightning_version": "2.0.0",
                "epoch": 0,
                "global_step": 0,
            },
            path,
        )
        return path

    def test_legacy_beat_only_checkpoint_loads(self, tmp_path):
        module = BeatModule(model=MODEL_CFG, task="beat_only")
        path = self._save_legacy_checkpoint(module, tmp_path, "legacy_beat_only.ckpt")

        loaded, task = load_module(path, device="cpu")

        assert task == "beat_only"
        assert isinstance(loaded, BeatModule)
        assert not loaded.training

    def test_legacy_beat_phase_checkpoint_loads(self, tmp_path):
        module = BeatPhaseModule(model=MODEL_CFG, task="beat_phase")
        path = self._save_legacy_checkpoint(module, tmp_path, "legacy_beat_phase.ckpt")

        loaded, task = load_module(path, device="cpu")

        assert task == "beat_phase"
        assert isinstance(loaded, BeatPhaseModule)
        assert not loaded.training

    def test_checkpoint_missing_task_field_raises(self, tmp_path):
        """Simulates every checkpoint currently in the repo: predates the
        explicit task field entirely, so load_module must refuse to guess."""

        module = BeatModule(model=MODEL_CFG, task="beat_only")
        state_dict = module.state_dict()
        hyper_parameters = dict(module.hparams)
        del hyper_parameters["task"]

        path = tmp_path / "no_task_field.ckpt"
        torch.save(
            {
                "state_dict": state_dict,
                "hyper_parameters": hyper_parameters,
                "pytorch-lightning_version": "2.0.0",
                "epoch": 0,
                "global_step": 0,
            },
            path,
        )

        with pytest.raises(KeyError):
            load_module(path, device="cpu")
