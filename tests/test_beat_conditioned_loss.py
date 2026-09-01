"""Tests for ``phase_conditioning="beat"`` — the beat-conditioned phase loss.

Covers :func:`musicality.losses.beat_phase_loss`'s weighting and normalisation
under both conditioning modes, and that
:class:`~musicality.trainers.beat_phase_module.BeatPhaseModule` records the
setting in its hyperparameters and passes it through.

Background: docs/beat_phase_improvement_review.md, section 2.
"""

import torch
import torch.nn.functional as F
import pytest
from omegaconf import OmegaConf

from musicality.losses import beat_phase_loss
from musicality.trainers.beat_phase_module import BeatPhaseModule


B, T = 2, 60
BEAT_EVERY = 12  # frames between beats
FAR = 6  # frames further than the smearing radius (~5) from any beat


def _target(
    beat_every: int = BEAT_EVERY,
    mask: float = 1.0,
    n_frames: int = T,
    sigma: float = 1.5,
) -> torch.Tensor:
    """A (B, 4, T) target with Gaussian-smeared beats every `beat_every` frames.

    ``one`` fires on every 4th beat, ``last`` on the beat before it.
    """

    from musicality.loaders.beat_dataset import gaussian_smear
    import numpy as np

    beat_spikes = np.zeros(n_frames, dtype=np.float32)
    one_spikes = np.zeros(n_frames, dtype=np.float32)
    last_spikes = np.zeros(n_frames, dtype=np.float32)

    for n, frame in enumerate(range(0, n_frames, beat_every)):
        beat_spikes[frame] = 1.0
        if n % 4 == 0:
            one_spikes[frame] = 1.0
        elif n % 4 == 3:
            last_spikes[frame] = 1.0

    target = torch.zeros(B, 4, n_frames)
    target[:, 0] = torch.from_numpy(gaussian_smear(beat_spikes, sigma))
    target[:, 1] = torch.from_numpy(gaussian_smear(one_spikes, sigma))
    target[:, 2] = torch.from_numpy(gaussian_smear(last_spikes, sigma))
    target[:, 3] = mask

    return target


def _logits(value: float = 0.0, n_frames: int = T) -> torch.Tensor:
    return torch.full((B, 3, n_frames), value, requires_grad=True)


def _beat_term(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> float:
    """The beat channel's contribution, computed independently of the function
    under test, so the phase terms can be isolated."""

    return (
        F.binary_cross_entropy_with_logits(
            logits[:, 0],
            target[:, 0],
            pos_weight=torch.tensor(pos_weight),
            reduction="none",
        )
        .mean()
        .item()
    )


class TestBeatConditionedPhaseLoss:
    """beat_phase_loss(phase_conditioning="beat")"""

    def test_rejects_unknown_conditioning(self):
        with pytest.raises(ValueError, match="phase_conditioning"):
            beat_phase_loss(_logits(), _target(), phase_conditioning="bogus")

    def test_default_stays_mask(self):
        """Existing behaviour must be untouched by the new parameter."""

        assert beat_phase_loss(_logits(), _target()) == beat_phase_loss(
            _logits(), _target(), phase_conditioning="mask"
        )

    def test_non_beat_frames_do_not_affect_the_phase_terms(self):
        """The headline property.

        Frames far from any beat carry zero weight, so perturbing the
        one/last logits there must not move the loss at all — while under
        "mask" it does.
        """

        target = _target()
        base = _logits()

        perturbed = base.detach().clone()
        perturbed[:, 1:, FAR::BEAT_EVERY] = 9.0  # away from every beat

        by_beat = beat_phase_loss(base, target, phase_conditioning="beat")
        by_beat_perturbed = beat_phase_loss(
            perturbed, target, phase_conditioning="beat"
        )
        by_mask = beat_phase_loss(base, target, phase_conditioning="mask")
        by_mask_perturbed = beat_phase_loss(
            perturbed, target, phase_conditioning="mask"
        )

        assert torch.isclose(by_beat, by_beat_perturbed, atol=1e-6)
        assert not torch.isclose(by_mask, by_mask_perturbed, atol=1e-6)

    def test_beat_frames_still_affect_the_phase_terms(self):
        """Guards against a weight that is simply all zeros."""

        target = _target()
        base = _logits()

        perturbed = base.detach().clone()
        perturbed[:, 1:, ::BEAT_EVERY] = 9.0  # exactly on the beats

        assert not torch.isclose(
            beat_phase_loss(base, target, phase_conditioning="beat"),
            beat_phase_loss(perturbed, target, phase_conditioning="beat"),
            atol=1e-6,
        )

    def test_beat_term_is_unaffected_by_conditioning(self):
        """Only the one/last terms are conditioned; the beat channel keeps its
        plain mean over every frame."""

        target = _target()
        logits = _logits()

        perturbed = logits.detach().clone()
        perturbed[:, 0, FAR::BEAT_EVERY] = 9.0  # beat channel, off-beat frames

        assert not torch.isclose(
            beat_phase_loss(logits, target, phase_conditioning="beat"),
            beat_phase_loss(perturbed, target, phase_conditioning="beat"),
            atol=1e-6,
        )

    def test_mask_still_gates_tracks_without_positions(self):
        """A track with mask=0 contributes nothing to the phase terms, even
        though its beat channel is non-zero."""

        target = _target(mask=0.0)
        logits = _logits()

        perturbed = logits.detach().clone()
        perturbed[:, 1:] = 9.0  # every one/last frame, including on-beat ones

        assert torch.isclose(
            beat_phase_loss(logits, target, phase_conditioning="beat"),
            beat_phase_loss(perturbed, target, phase_conditioning="beat"),
            atol=1e-6,
        )

    def test_zero_weight_batch_is_finite(self):
        """No division by zero when nothing carries weight."""

        loss = beat_phase_loss(_logits(), _target(mask=0.0), phase_conditioning="beat")

        assert torch.isfinite(loss)

    def test_normalises_by_weight_sum_not_frame_count(self):
        """Two clips differing only in beat density must give the same phase
        terms when the per-frame phase loss is constant.

        This is what makes the term a weighted *mean*: dividing by the total
        frame count instead would scale the phase terms with tempo, so fast
        tracks would silently dominate the gradient.
        """

        pos_weight = 3.0
        sparse = _target(beat_every=20)
        dense = _target(beat_every=10)

        # Zero the one/last targets in both, so beat *density* is the only
        # difference between the two clips. Left as built, the fixtures would
        # also carry different numbers of downbeats, which legitimately
        # changes the phase terms and would swamp the property under test.
        sparse[:, 1:3] = 0.0
        dense[:, 1:3] = 0.0

        logits = _logits()  # all zeros + all-zero targets -> constant per-frame loss

        loss_sparse = beat_phase_loss(
            logits, sparse, pos_weight=pos_weight, phase_conditioning="beat"
        )
        loss_dense = beat_phase_loss(
            logits, dense, pos_weight=pos_weight, phase_conditioning="beat"
        )

        phase_sparse = loss_sparse.item() - _beat_term(logits, sparse, pos_weight)
        phase_dense = loss_dense.item() - _beat_term(logits, dense, pos_weight)

        assert phase_sparse == pytest.approx(phase_dense, abs=1e-5)

    def test_gradient_reaches_only_beat_frames(self):
        target = _target()
        logits = _logits()

        beat_phase_loss(logits, target, phase_conditioning="beat").backward()

        grad = logits.grad[:, 1:]  # one/last channels
        assert grad[..., ::BEAT_EVERY].abs().sum() > 0  # on the beats
        assert grad[..., FAR::BEAT_EVERY].abs().sum() == pytest.approx(0.0, abs=1e-9)


class TestModuleThreading:
    """BeatPhaseModule must expose phase_conditioning and pass it to the loss."""

    MODEL_CFG = OmegaConf.create(
        {
            "_target_": "musicality.models.tcn.TCNTempoNet",
            "n_mels": 16,
            "channels": 8,
            "n_layers": 3,
            "dropout": 0.0,
        }
    )

    def test_defaults_to_mask(self):
        module = BeatPhaseModule(model=self.MODEL_CFG)

        assert module.hparams.phase_conditioning == "mask"

    def test_is_saved_to_hparams(self):
        """Checkpoints must record it — inference reconstructs the module from
        hyper_parameters alone (see musicality.inference.load_module)."""

        module = BeatPhaseModule(model=self.MODEL_CFG, phase_conditioning="beat")

        assert module.hparams.phase_conditioning == "beat"

    def test_is_passed_through_to_the_loss(self, monkeypatch):
        seen = {}

        def _spy(logits, target, pos_weight=8.0, phase_conditioning="mask"):
            seen["phase_conditioning"] = phase_conditioning
            return logits.sum() * 0.0

        monkeypatch.setattr(
            "musicality.trainers.beat_phase_module.beat_phase_loss", _spy
        )

        module = BeatPhaseModule(model=self.MODEL_CFG, phase_conditioning="beat")
        wav = torch.randn(B, 1, 4096)
        target = _target(n_frames=9)

        # `_step`, not `training_step` — the latter also reads self.optimizers(),
        # which needs an attached Trainer. Matches tests/test_beat_phase_module.py.
        module._step((wav, target), "train")

        assert seen["phase_conditioning"] == "beat"
