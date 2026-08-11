"""Tests for musicality.trainers.beat_module.BeatModule."""

import torch
import pytest
from omegaconf import OmegaConf

from musicality.trainers.beat_module import BeatModule

N_SAMPLES = 4096  # short but > n_fft (2048) so STFT doesn't error
B, T = 4, 6

MODEL_CFG = OmegaConf.create(
    {
        "_target_": "musicality.models.tcn.TCNTempoNet",
        "n_mels": 16,
        "channels": 8,
        "n_layers": 3,
        "dropout": 0.0,
    }
)


def _target(n_frames: int) -> torch.Tensor:
    target = torch.zeros(B, 4, n_frames)
    target[:, 0, ::2] = 1.0  # beat every other frame
    target[:, 1, 0] = 1.0  # one at frame 0 (ignored by BeatModule)
    target[:, 3] = 1.0  # positions available (ignored by BeatModule)
    return target


@pytest.fixture
def module():
    return BeatModule(model=MODEL_CFG, lr=1e-3, weight_decay=0.0)


class TestBeatModule:
    def test_forward_shape(self, module):
        wav = torch.randn(B, 1, N_SAMPLES)
        out = module(wav)
        assert out.shape[0] == B
        assert out.dim() == 2  # (B, T'), squeezed — no channel dim for n_outputs=1

    def test_forces_frame_level_and_single_output(self, module):
        assert module.model.frame_level is True
        assert module.model.n_outputs == 1

    def test_training_step_matched_lengths(self, module):
        wav = torch.randn(B, 1, N_SAMPLES)
        with torch.no_grad():
            n_frames = module(wav).shape[-1]
        batch = (wav, _target(n_frames))
        loss, probs = module._step(batch, "train")
        assert loss.shape == ()
        assert loss.item() > 0
        assert probs.shape == (B, n_frames)

    def test_training_step_mismatched_lengths_does_not_raise(self, module):
        """Target frame count off by a few from the model's own T' must not crash."""
        wav = torch.randn(B, 1, N_SAMPLES)
        with torch.no_grad():
            n_frames = module(wav).shape[-1]
        batch = (wav, _target(n_frames + 3))
        loss, _ = module._step(batch, "train")
        assert torch.isfinite(loss).all()

    def test_validation_step(self, module):
        wav = torch.randn(B, 1, N_SAMPLES)
        with torch.no_grad():
            n_frames = module(wav).shape[-1]
        module.validation_step((wav, _target(n_frames)), 0)

    def test_configure_optimizers(self, module):
        out = module.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_hparams_saved(self, module):
        assert module.hparams.lr == 1e-3
        assert module.hparams.pos_weight == 6.0

    def test_backward(self, module):
        wav = torch.randn(B, 1, N_SAMPLES)
        with torch.no_grad():
            n_frames = module(wav).shape[-1]
        loss, _ = module._step((wav, _target(n_frames)), "train")
        loss.backward()
        for p in module.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
