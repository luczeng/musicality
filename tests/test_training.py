"""Fast unit tests for the tempo estimation training components."""

import torch
import pytest
from omegaconf import OmegaConf

from musicality.models.tempo_net import TempoNet
from musicality.trainers.tempo_module import TempoModule

MODEL_CFG = OmegaConf.create(
    {"_target_": "musicality.models.tempo_net.TempoNet", "n_mels": 16, "dropout": 0.0}
)
BATCH = (torch.randn(4, 1, 16, 32), torch.tensor([80.0, 100.0, 120.0, 140.0]))


# ---------------------------------------------------------------------------
# TempoNet
# ---------------------------------------------------------------------------

class TestTempoNet:
    def test_output_shape(self):
        model = TempoNet(n_mels=16)
        out = model(torch.randn(4, 1, 16, 32))
        assert out.shape == (4,)

    def test_single_sample(self):
        model = TempoNet(n_mels=16)
        out = model(torch.randn(1, 1, 16, 32))
        assert out.shape == (1,)

    def test_output_is_finite(self):
        model = TempoNet(n_mels=16)
        out = model(torch.randn(4, 1, 16, 32))
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# TempoModule
# ---------------------------------------------------------------------------

class TestTempoModule:
    @pytest.fixture
    def module(self):
        return TempoModule(model=MODEL_CFG, lr=1e-3, weight_decay=0.0)

    def test_forward(self, module):
        mel, _ = BATCH
        out = module(mel)
        assert out.shape == (4,)

    def test_training_step(self, module):
        loss = module.training_step(BATCH, 0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_validation_step(self, module):
        # should not raise
        module.validation_step(BATCH, 0)

    def test_configure_optimizers(self, module):
        out = module.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_hparams_saved(self, module):
        assert module.hparams.lr == 1e-3

    def test_backward(self, module):
        loss = module.training_step(BATCH, 0)
        loss.backward()
        for p in module.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
