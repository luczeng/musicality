"""Fast unit tests for the tempo estimation training components."""

import torch
import pytest
from omegaconf import OmegaConf

from musicality.models.tempo_net import TempoNet
from musicality.trainers.tempo_module import TempoModule

CLASSIFICATION_CFG = OmegaConf.create(
    {"bpm_min": 30, "bpm_max": 286, "n_bins": 64, "sigma": 1.5}
)

N_SAMPLES = 4096  # short but > n_fft (2048) so STFT doesn't error

MODEL_CFG = OmegaConf.create(
    {"_target_": "musicality.models.tempo_net.TempoNet", "n_mels": 16, "dropout": 0.0}
)
BATCH = (torch.randn(4, 1, N_SAMPLES), torch.tensor([80.0, 100.0, 120.0, 140.0]))


# ---------------------------------------------------------------------------
# TempoNet
# ---------------------------------------------------------------------------

class TestTempoNet:
    def test_output_shape(self):
        model = TempoNet(n_mels=16)
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert out.shape == (4,)

    def test_single_sample(self):
        model = TempoNet(n_mels=16)
        out = model(torch.randn(1, 1, N_SAMPLES))
        assert out.shape == (1,)

    def test_output_is_finite(self):
        model = TempoNet(n_mels=16)
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# TempoModule
# ---------------------------------------------------------------------------

class TestTempoModule:
    @pytest.fixture
    def module(self):
        return TempoModule(model=MODEL_CFG, lr=1e-3, weight_decay=0.0)

    def test_forward(self, module):
        wav, _ = BATCH
        out = module(wav)
        assert out.shape == (4,)

    def test_training_step(self, module):
        loss, pred = module._step(BATCH, "train")
        assert loss.shape == ()
        assert loss.item() > 0
        assert pred.shape == (4,)

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
        loss, _ = module._step(BATCH, "train")
        loss.backward()
        for p in module.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# TempoModule (classification)
# ---------------------------------------------------------------------------

class TestTempoModuleClassification:
    @pytest.fixture
    def module(self):
        return TempoModule(
            model=MODEL_CFG,
            loss="classification",
            classification=CLASSIFICATION_CFG,
            lr=1e-3,
            weight_decay=0.0,
        )

    def test_model_outputs_logits(self, module):
        wav, _ = BATCH
        out = module(wav)
        assert out.shape == (4, CLASSIFICATION_CFG.n_bins)

    def test_step_returns_decoded_pred(self, module):
        loss, pred = module._step(BATCH, "train")
        assert loss.shape == ()
        assert loss.item() > 0
        assert pred.shape == (4,)

    def test_predictions_within_bin_range(self, module):
        _, pred = module._step(BATCH, "train")
        assert (pred >= CLASSIFICATION_CFG.bpm_min - 1).all()
        assert (pred <= CLASSIFICATION_CFG.bpm_max + 1).all()

    def test_backward(self, module):
        loss, _ = module._step(BATCH, "train")
        loss.backward()
        for p in module.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_missing_classification_raises(self):
        with pytest.raises(ValueError, match="classification"):
            TempoModule(model=MODEL_CFG, loss="classification", classification=None)
