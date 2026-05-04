"""Fast unit tests for the tempo estimation training components."""

import torch
import pytest
from omegaconf import OmegaConf

from musicality.models.tempo_net import TempoNet
from musicality.trainers.tempo_module import TempoModule
from musicality.losses import gaussian_soft_target, classification_tempo_loss

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


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

N_BINS = 64
BPM_MIN, BPM_MAX = 30.0, 286.0
BIN_CENTERS = torch.linspace(BPM_MIN, BPM_MAX, N_BINS)
SIGMA = 1.5


class TestGaussianSoftTarget:
    def test_output_shape(self):
        tempo = torch.tensor([120.0, 90.0])
        out = gaussian_soft_target(tempo, BIN_CENTERS, SIGMA)
        assert out.shape == (2, N_BINS)

    def test_sums_to_one(self):
        # softmax output must sum to 1 per sample
        tempo = torch.tensor([60.0, 120.0, 180.0])
        out = gaussian_soft_target(tempo, BIN_CENTERS, SIGMA)
        assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_peak_near_true_tempo(self):
        # argmax of the soft target should be the bin closest to the true tempo
        tempo = torch.tensor([120.0])
        out = gaussian_soft_target(tempo, BIN_CENTERS, SIGMA)
        peak_bpm = BIN_CENTERS[out.argmax(dim=-1).item()].item()
        assert abs(peak_bpm - 120.0) < (BPM_MAX - BPM_MIN) / N_BINS + 1e-3

    def test_smaller_sigma_sharpens_peak(self):
        # a narrower Gaussian should concentrate more mass on the peak bin
        tempo = torch.tensor([120.0])
        wide = gaussian_soft_target(tempo, BIN_CENTERS, sigma=5.0)
        narrow = gaussian_soft_target(tempo, BIN_CENTERS, sigma=0.5)
        assert narrow.max() > wide.max()


class TestClassificationTempoLoss:
    def test_output_is_scalar(self):
        logits = torch.randn(4, N_BINS)
        tempo = torch.tensor([80.0, 100.0, 120.0, 140.0])
        loss = classification_tempo_loss(logits, tempo, BIN_CENTERS, SIGMA)
        assert loss.shape == ()

    def test_loss_is_positive(self):
        logits = torch.randn(4, N_BINS)
        tempo = torch.tensor([80.0, 100.0, 120.0, 140.0])
        loss = classification_tempo_loss(logits, tempo, BIN_CENTERS, SIGMA)
        assert loss.item() > 0

    def test_perfect_prediction_lower_than_random(self):
        # logits sharply peaked at the true bin should give lower loss than
        # random logits. Using a spike (not target.log()) to avoid -inf.
        tempo = torch.tensor([120.0])
        peak_bin = (BIN_CENTERS - 120.0).abs().argmin()
        peaked_logits = torch.full((1, N_BINS), -10.0)
        peaked_logits[0, peak_bin] = 10.0
        random_logits = torch.randn(1, N_BINS)
        peaked_loss = classification_tempo_loss(peaked_logits, tempo, BIN_CENTERS, SIGMA)
        random_loss = classification_tempo_loss(random_logits, tempo, BIN_CENTERS, SIGMA)
        assert peaked_loss.item() < random_loss.item()

    def test_gradients_flow(self):
        logits = torch.randn(4, N_BINS, requires_grad=True)
        tempo = torch.tensor([80.0, 100.0, 120.0, 140.0])
        loss = classification_tempo_loss(logits, tempo, BIN_CENTERS, SIGMA)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
