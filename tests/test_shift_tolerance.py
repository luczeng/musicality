"""Tests for :mod:`musicality.losses.shift_tolerance` — the shift-tolerant
weighted BCE from Beat This! (ISMIR 2024), and the ``sliding_windowed_max``
primitive it is built on.

Covers the primitives, both beat losses, both LightningModules, and the config
guard. The properties worth holding onto:

- ``tolerance_frames=0`` is bit-identical to the plain weighted BCE it
  replaces, so every number measured before this change stays reproducible.
- Inside the tolerance window, *where* the peak sits costs nothing.
- A blurred peak costs more than a sharp one of the same height — the paper's
  central claim, and the reason this exists rather than more target smearing.
- ``pos_weight="auto"`` counts the negatives that survive the ignore band,
  not every non-beat frame.
- The position term is *widened* rather than forgiven, and its target widens
  with its gate — a confidently correct prediction across the window has to be
  the cheap one, or the head is being trained towards uncertainty at exactly
  the frames the decoder reads.

Background: plans/09_lessons_from_literature.md §2.1.
"""

import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from musicality.loaders.beat_dataset import gaussian_smear
from musicality.losses.beat_position import beat_position_loss
from musicality.losses.pos_weight import AUTO_POS_WEIGHT_ALPHA, beat_pos_weight
from musicality.losses.shift_tolerance import (
    TOLERANCE_FRAMES,
    shift_tolerant_bce,
    sliding_windowed_max,
)
from musicality.trainers.beat_module import BeatModule
from musicality.trainers.beat_phase_module import BeatPhaseModule
from musicality.trainers.common import warn_if_tolerance_stacks_on_smearing

FPS = 22050 / 512  # the training front-end's frame rate, 43.07
N_FRAMES = int(16.0 * FPS)  # a 16 s crop, configs/beat_train.yaml data.duration

B, T = 2, 64
G = 4  # configs/beat_train.yaml group_size
CENTRE = T // 2  # one beat here, far enough from both edges for any window
R = TOLERANCE_FRAMES  # 3 frames, ±69.7 ms
IGNORE = 2 * R  # the paper's default band

HIGH, LOW = 4.0, -4.0


def _sharp_target(centre: int = CENTRE, n_frames: int = T) -> torch.Tensor:
    """A ``(B, T)`` beat target with a single one-frame spike — what
    ``sigma_frames: 0`` produces, and what shift tolerance assumes."""

    target = torch.zeros(B, n_frames)
    target[:, centre] = 1.0

    return target


def _logits(*peaks: int, n_frames: int = T) -> torch.Tensor:
    """``LOW`` everywhere except ``HIGH`` at each frame in ``peaks``."""

    logits = torch.full((B, n_frames), LOW)
    for frame in peaks:
        logits[:, frame] = HIGH

    return logits


def _beat_channel(bpm: float, sigma: float, n_frames: int = N_FRAMES) -> torch.Tensor:
    """A ``(1, T)`` metronomic beat grid, built through the real smearing.

    Offset by half a period so no bump is truncated by the window edge — the
    same reasoning as ``tests/test_loss_calibration.py::_beat_frames``.
    """

    period = 60.0 * FPS / bpm
    frames = np.round(np.arange(period / 2, n_frames, period)).astype(int)

    spikes = np.zeros(n_frames, dtype=np.float32)
    spikes[frames[frames < n_frames]] = 1.0

    return torch.from_numpy(gaussian_smear(spikes, sigma))[None]


class TestSlidingWindowedMax:
    """sliding_windowed_max(x, radius)"""

    def test_radius_zero_is_the_identity(self):
        """The whole backward-compatibility story rests on this."""

        x = torch.randn(B, T)

        assert sliding_windowed_max(x, 0) is x

    def test_spreads_a_spike_to_exactly_the_window_width(self):
        x = torch.zeros(1, T)
        x[0, CENTRE] = 1.0

        spread = sliding_windowed_max(x, R)[0]

        assert spread[CENTRE - R : CENTRE + R + 1].eq(1.0).all()
        assert spread[: CENTRE - R].eq(0.0).all()
        assert spread[CENTRE + R + 1 :].eq(0.0).all()

    def test_does_not_wrap_around_the_edges(self):
        """A crop boundary is not a beat: max_pool1d pads with -inf, so a spike
        at frame 0 must not reach the last frame."""

        x = torch.zeros(1, T)
        x[0, 0] = 1.0

        spread = sliding_windowed_max(x, R)[0]

        assert spread[: R + 1].eq(1.0).all()
        assert spread[-1] == 0.0

    def test_pools_each_channel_of_a_block_independently(self):
        """beat_position_loss will hand it a (B, G, T) position block."""

        x = torch.zeros(B, 4, T)
        x[:, 2, CENTRE] = 1.0

        spread = sliding_windowed_max(x, R)

        assert spread.shape == x.shape
        assert spread[:, 2, CENTRE - R : CENTRE + R + 1].eq(1.0).all()
        assert spread[:, [0, 1, 3]].eq(0.0).all()


class TestZeroToleranceIsUnchanged:
    """tolerance_frames=0 must reproduce the plain weighted BCE bit for bit."""

    @staticmethod
    def _reference(logits, beat_y, pos_weight, alpha=AUTO_POS_WEIGHT_ALPHA):
        return F.binary_cross_entropy_with_logits(
            logits, beat_y, pos_weight=beat_pos_weight(beat_y, pos_weight, alpha)
        )

    @pytest.mark.parametrize("pos_weight", [5.0, 1.0, "auto"])
    def test_matches_binary_cross_entropy_with_logits(self, pos_weight):
        torch.manual_seed(0)
        beat_y = _beat_channel(125.0, sigma=1.5).expand(B, -1)
        logits = torch.randn(B, beat_y.shape[-1])

        assert torch.equal(
            shift_tolerant_bce(logits, beat_y, pos_weight=pos_weight),
            self._reference(logits, beat_y, pos_weight),
        )

    def test_ignore_frames_is_not_read_when_tolerance_is_off(self):
        torch.manual_seed(0)
        beat_y = _sharp_target()
        logits = torch.randn(B, T)

        assert torch.equal(
            shift_tolerant_bce(logits, beat_y, 5.0, ignore_frames=0),
            shift_tolerant_bce(logits, beat_y, 5.0, ignore_frames=99),
        )


class TestShiftTolerance:
    """The headline property: timing error inside the window is free."""

    @pytest.mark.parametrize("offset", range(-R, R + 1))
    def test_a_peak_anywhere_in_the_window_costs_the_same(self, offset):
        """A peak ``offset`` frames off the annotation is exactly as cheap as
        one dead on it.

        Exactly, not approximately: at the default ``ignore_frames = 2r`` the
        peak's own pooled span stays inside the ignored band wherever it sits
        in the window, so the negative term cannot see it move either.
        """

        beat_y = _sharp_target()

        assert torch.equal(
            shift_tolerant_bce(
                _logits(CENTRE + offset), beat_y, 5.0, tolerance_frames=R
            ),
            shift_tolerant_bce(_logits(CENTRE), beat_y, 5.0, tolerance_frames=R),
        )

    def test_a_peak_past_the_window_costs_more(self):
        """Guards against a tolerance that simply forgives everything."""

        beat_y = _sharp_target()

        assert shift_tolerant_bce(
            _logits(CENTRE + R + 1), beat_y, 5.0, tolerance_frames=R
        ) > shift_tolerant_bce(_logits(CENTRE), beat_y, 5.0, tolerance_frames=R)

    def test_a_blurred_peak_costs_more_than_a_sharp_one(self):
        """Beat This!'s central claim, as a test.

        Both predictions reach ``HIGH`` at the annotation, so the positive term
        is identical; only the flanks of the blurred one, which reach past the
        ignored band, are charged for. Under a plain BCE against a smeared
        target the blurred peak would be the *cheaper* of the two.
        """

        beat_y = _sharp_target()
        blurred = range(CENTRE - IGNORE - 2, CENTRE + IGNORE + 3)

        assert shift_tolerant_bce(
            _logits(*blurred), beat_y, 5.0, tolerance_frames=R
        ) > shift_tolerant_bce(_logits(CENTRE), beat_y, 5.0, tolerance_frames=R)

    def test_gradient_reaches_only_the_window_maximum(self):
        """Inside the tolerance window only the largest frame is read, so its
        neighbours must come away with nothing to learn from."""

        logits = _logits(CENTRE).requires_grad_(True)

        shift_tolerant_bce(logits, _sharp_target(), 5.0, tolerance_frames=R).backward()

        grad = logits.grad[0]
        assert grad[CENTRE] != 0.0  # the peak the positive term reads
        assert grad[CENTRE - 1] == 0.0
        assert grad[CENTRE + 1] == 0.0
        assert grad[CENTRE + IGNORE + R + 1] != 0.0  # a live negative


class TestIgnoreBand:
    """Negatives are switched off near the annotation, and only there."""

    def test_a_spurious_peak_inside_the_band_is_free(self):
        """A second peak whose entire pooled span stays within ``±ignore``
        costs nothing — which is the contradiction the band exists to remove,
        since that peak is a legitimate answer under the tolerance."""

        beat_y = _sharp_target()

        assert torch.equal(
            shift_tolerant_bce(
                _logits(CENTRE, CENTRE + R), beat_y, 5.0, tolerance_frames=R
            ),
            shift_tolerant_bce(_logits(CENTRE), beat_y, 5.0, tolerance_frames=R),
        )

    def test_a_spurious_peak_one_frame_further_out_is_charged(self):
        """Guards against a band that simply swallows everything: at ``r + 1``
        the peak's pooled span reaches past ``±2r`` and is billed."""

        beat_y = _sharp_target()

        assert shift_tolerant_bce(
            _logits(CENTRE, CENTRE + R + 1), beat_y, 5.0, tolerance_frames=R
        ) > shift_tolerant_bce(_logits(CENTRE), beat_y, 5.0, tolerance_frames=R)

    def test_defaults_to_twice_the_tolerance(self):
        torch.manual_seed(0)
        beat_y = _sharp_target()
        logits = torch.randn(B, T)

        assert torch.equal(
            shift_tolerant_bce(logits, beat_y, 5.0, tolerance_frames=R),
            shift_tolerant_bce(
                logits, beat_y, 5.0, tolerance_frames=R, ignore_frames=2 * R
            ),
        )

    def test_a_narrower_band_charges_more(self):
        """The knob we ship narrower than the paper does: fewer ignored frames
        means more surviving negatives, hence a larger negative term."""

        torch.manual_seed(0)
        beat_y = _sharp_target()
        logits = torch.randn(B, T)

        assert shift_tolerant_bce(
            logits, beat_y, 5.0, tolerance_frames=R, ignore_frames=4
        ) > shift_tolerant_bce(
            logits, beat_y, 5.0, tolerance_frames=R, ignore_frames=IGNORE
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(tolerance_frames=-1),
            dict(tolerance_frames=R, ignore_frames=-1),
        ],
    )
    def test_rejects_negative_radii(self, kwargs):
        with pytest.raises(ValueError, match="cannot be negative"):
            shift_tolerant_bce(_logits(CENTRE), _sharp_target(), 5.0, **kwargs)


class TestAutoWeightCountsSurvivingNegatives:
    """pos_weight="auto" has to see the ignore band, or it over-weights the
    positive class by the fraction of negatives that were switched off."""

    def test_no_neg_weight_is_the_previous_behaviour(self):
        beat_y = _beat_channel(125.0, sigma=1.5)

        assert torch.equal(
            beat_pos_weight(beat_y, "auto"),
            beat_pos_weight(beat_y, "auto", AUTO_POS_WEIGHT_ALPHA, neg_weight=None),
        )

    def test_a_neg_weight_of_one_minus_the_target_agrees_with_the_default(self):
        """The default is the special case, not a different formula."""

        beat_y = _beat_channel(125.0, sigma=1.5)

        assert beat_pos_weight(
            beat_y, "auto", neg_weight=1.0 - beat_y
        ).item() == pytest.approx(beat_pos_weight(beat_y, "auto").item(), rel=1e-5)

    @pytest.mark.parametrize(
        "bpm, raw, ignore_aware",
        [
            (56.0, 49.9, 41.0),  # rwc_classical, 10th percentile
            (125.0, 22.1, 13.2),  # ballroom, median
            (193.0, 13.9, 5.0),  # jtd, median
        ],
    )
    def test_reproduces_the_measured_sharp_target_weights(self, bpm, raw, ignore_aware):
        """The numbers docs/source/configuration.rst will quote.

        Both columns matter: ``raw`` is why AUTO_POS_WEIGHT_RANGE's ceiling had
        to rise past 20, and the gap to ``ignore_aware`` is the miscalibration
        this parameter removes.
        """

        beat_y = _beat_channel(bpm, sigma=0.0)
        keep = 1.0 - sliding_windowed_max(beat_y, 4)

        assert beat_pos_weight(beat_y, "auto").item() == pytest.approx(raw, rel=0.02)
        assert beat_pos_weight(beat_y, "auto", neg_weight=keep).item() == pytest.approx(
            ignore_aware, rel=0.02
        )

    def test_the_weight_reaches_the_loss(self):
        """A derived weight moves the loss; a spy would not catch a dropped
        argument here, since the weight is computed inside the function."""

        beat_y = _sharp_target()
        logits = _logits(CENTRE)

        assert shift_tolerant_bce(
            logits, beat_y, "auto", tolerance_frames=R
        ) != shift_tolerant_bce(logits, beat_y, 1.0, tolerance_frames=R)


class TestPositionWidening:
    """beat_position_loss widens the position term rather than forgiving it.

    The position head is never scanned over time — the decoder rounds a beat
    time to a frame and reads that one column — so tolerance there means being
    right across the window the lookup might land in.
    """

    BEAT_EVERY = 12  # > 2R + 1, so neighbouring windows do not overlap

    def _target(self, mask: float = 1.0, n_frames: int = T) -> torch.Tensor:
        """A sharp ``(B, 2 + G, T)`` target, built the way BeatDataset builds
        it — including the uniform 1/G rows away from a beat."""

        beat = np.zeros(n_frames, dtype=np.float32)
        block = np.zeros((G, n_frames), dtype=np.float32)

        for n, frame in enumerate(
            range(self.BEAT_EVERY // 2, n_frames, self.BEAT_EVERY)
        ):
            beat[frame] = 1.0
            block[n % G, frame] = 1.0

        total = block.sum(axis=0, keepdims=True)
        block = np.divide(
            block, total, out=np.full_like(block, 1.0 / G), where=total > 1e-6
        )
        annotated = np.full(n_frames, mask, dtype=np.float32)

        rows = np.stack([beat, *block, annotated])

        return torch.from_numpy(np.stack([rows] * B).astype(np.float32))

    def _confident_logits(self, target: torch.Tensor, spread: int) -> torch.Tensor:
        """Position logits that confidently name each beat's own position, at
        the beat frame and ``spread`` frames either side of it."""

        logits = torch.zeros(B, 1 + G, target.shape[-1])
        beats = target[0, 0].nonzero().flatten().tolist()

        for frame in beats:
            position = int(target[0, 1:-1, frame].argmax())
            lo, hi = frame - spread, frame + spread + 1
            logits[:, 1 + position, max(lo, 0) : hi] = HIGH

        return logits

    def _position_term(self, logits, target, **kwargs) -> torch.Tensor:
        return beat_position_loss(logits, target, 5.0, return_terms=True, **kwargs)[1]

    def test_the_window_is_supervised_against_the_beats_own_position(self):
        """The trap this exists to avoid, as a test.

        Both predictions name the correct position at the annotated frame and
        differ only either side of it: one stays confident, the other falls
        back to uniform. Widening the gate *and* the target together makes
        confidence the cheaper of the two. Widening the gate alone would invert
        this — those frames carry a uniform 1/G row meaning "no information
        here", and cross-entropy against uniform is minimised by predicting
        uniform, so the head would be trained towards maximum uncertainty at
        exactly the frames the decoder reads.
        """

        target = self._target()

        confident = self._position_term(
            self._confident_logits(target, spread=R), target, tolerance_frames=R
        )
        uncertain = self._position_term(
            self._confident_logits(target, spread=0), target, tolerance_frames=R
        )

        assert confident < uncertain

    def test_without_tolerance_the_window_is_not_supervised_at_all(self):
        """The companion to the above: the difference is created by the
        widening, not by the fixtures."""

        target = self._target()

        assert torch.equal(
            self._position_term(self._confident_logits(target, spread=R), target),
            self._position_term(self._confident_logits(target, spread=0), target),
        )

    @pytest.mark.parametrize("distance, supervised", [(R, True), (R + 1, False)])
    def test_the_gate_ends_at_the_tolerance_radius(self, distance, supervised):
        """A frame ``R`` from a beat carries weight; one frame further does
        not, so the window has an edge rather than leaking."""

        target = self._target()
        base = self._confident_logits(target, spread=0)

        # Offset from the beat grid, not from frame 0 — and one channel, not
        # all of them, since raising every position equally leaves the softmax
        # exactly where it was.
        first = self.BEAT_EVERY // 2 + distance
        perturbed = base.clone()
        perturbed[:, 1, first :: self.BEAT_EVERY] = HIGH

        moved = not torch.equal(
            self._position_term(base, target, tolerance_frames=R),
            self._position_term(perturbed, target, tolerance_frames=R),
        )

        assert moved is supervised

    def test_an_unannotated_clip_still_contributes_nothing(self):
        """mask=0 must survive the widening — it gates on annotation, not on
        proximity to a beat."""

        target = self._target(mask=0.0)
        logits = self._confident_logits(target, spread=R)

        assert self._position_term(logits, target, tolerance_frames=R) == 0.0

    def test_defaults_reproduce_the_untouched_loss(self):
        """tolerance_frames=0 leaves beat_position_loss bit-identical."""

        torch.manual_seed(0)
        target = self._target()
        logits = torch.randn(B, 1 + G, T)

        assert torch.equal(
            beat_position_loss(logits, target, 5.0),
            beat_position_loss(logits, target, 5.0, tolerance_frames=0),
        )


class TestModuleWiring:
    """Both LightningModules expose the knobs and pass them to the loss.

    Checkpoints are reconstructed from ``hyper_parameters`` alone
    (:func:`musicality.inference.load_module`), so anything the loss reads has
    to be saved.
    """

    MODEL_CFG = OmegaConf.create(
        {
            "_target_": "musicality.models.tcn.TCNTempoNet",
            "n_mels": 16,
            "channels": 8,
            "n_layers": 3,
            "dropout": 0.0,
        }
    )

    def test_beat_module_defaults_to_off(self):
        module = BeatModule(model=self.MODEL_CFG)

        assert module.hparams.tolerance_frames == 0
        assert module.hparams.ignore_frames is None

    def test_beat_phase_module_defaults_to_off(self):
        module = BeatPhaseModule(model=self.MODEL_CFG)

        assert module.hparams.tolerance_frames == 0
        assert module.hparams.ignore_frames is None

    def test_beat_module_saves_them_to_hparams(self):
        module = BeatModule(model=self.MODEL_CFG, tolerance_frames=R, ignore_frames=4)

        assert module.hparams.tolerance_frames == R
        assert module.hparams.ignore_frames == 4

    def test_beat_phase_module_saves_them_to_hparams(self):
        module = BeatPhaseModule(
            model=self.MODEL_CFG, group_size=G, tolerance_frames=R, ignore_frames=4
        )

        assert module.hparams.tolerance_frames == R
        assert module.hparams.ignore_frames == 4

    @pytest.mark.parametrize("cls", [BeatModule, BeatPhaseModule])
    def test_a_checkpoint_predating_the_knobs_still_loads(self, cls):
        """:func:`musicality.inference.load_module` reconstructs a module by
        splatting the checkpoint's saved hyper_parameters as kwargs, so every
        checkpoint written before this change arrives without the new keys and
        has to fall back to the defaults that reproduce its training."""

        old_hparams = {
            k: v
            for k, v in cls(model=self.MODEL_CFG).hparams.items()
            if k not in ("tolerance_frames", "ignore_frames", "pos_weight_alpha")
        }

        module = cls(**old_hparams)

        assert module.hparams.tolerance_frames == 0
        assert module.hparams.ignore_frames is None

    def test_beat_module_passes_them_to_the_loss(self, monkeypatch):
        seen = {}

        def _spy(logits, beat_y, **kwargs):
            seen.update(kwargs)
            return logits.sum() * 0.0

        monkeypatch.setattr("musicality.trainers.beat_module.beat_only_loss", _spy)

        module = BeatModule(
            model=self.MODEL_CFG, tolerance_frames=R, ignore_frames=4, pos_weight="auto"
        )
        module._step((torch.randn(B, 1, 4096), torch.zeros(B, 4, 9)), "train")

        assert seen["tolerance_frames"] == R
        assert seen["ignore_frames"] == 4
        assert seen["pos_weight_alpha"] == AUTO_POS_WEIGHT_ALPHA

    def test_beat_phase_module_passes_them_to_the_loss(self, monkeypatch):
        seen = {}

        def _spy(logits, target, **kwargs):
            seen.update(kwargs)
            return logits.sum() * 0.0, logits.sum() * 0.0

        monkeypatch.setattr(
            "musicality.trainers.beat_phase_module.beat_position_loss", _spy
        )

        module = BeatPhaseModule(
            model=self.MODEL_CFG, group_size=G, tolerance_frames=R, ignore_frames=4
        )
        target = torch.zeros(B, 2 + G, 9)
        module._position_step(torch.randn(B, 1 + G, 9), target, "train")

        assert seen["tolerance_frames"] == R
        assert seen["ignore_frames"] == 4


class TestSmearingGuard:
    """A config that both smears and forgives is a silent mistake, so it warns."""

    @staticmethod
    def _cfg(sigma_frames: float, tolerance_frames: int) -> OmegaConf:
        return OmegaConf.create(
            {"sigma_frames": sigma_frames, "tolerance_frames": tolerance_frames}
        )

    def test_warns_when_both_are_set(self):
        with pytest.warns(UserWarning, match="meant to replace smearing"):
            warn_if_tolerance_stacks_on_smearing(self._cfg(1.5, R))

    def test_reports_the_combined_radius(self):
        """±4 from the smear plus ±3 from the window, against a ±3 metric."""

        with pytest.warns(UserWarning, match=r"±7 frames"):
            warn_if_tolerance_stacks_on_smearing(self._cfg(1.5, R))

    @pytest.mark.parametrize(
        "sigma_frames, tolerance_frames",
        [(1.5, 0), (0.0, R), (0.0, 0)],
    )
    def test_stays_quiet_otherwise(self, sigma_frames, tolerance_frames):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_tolerance_stacks_on_smearing(
                self._cfg(sigma_frames, tolerance_frames)
            )

    def test_a_config_without_the_key_is_fine(self):
        """Every checkpoint and config predating this change lacks it."""

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_tolerance_stacks_on_smearing(
                OmegaConf.create({"sigma_frames": 1.5})
            )
