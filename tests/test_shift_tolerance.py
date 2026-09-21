"""Tests for :mod:`musicality.losses.shift_tolerance` — the shift-tolerant
weighted BCE from Beat This! (ISMIR 2024), and the ``sliding_windowed_max``
primitive it is built on.

Nothing is wired into a loss or a trainer yet; these cover the primitives
alone. The properties worth holding onto:

- ``tolerance_frames=0`` is bit-identical to the plain weighted BCE it
  replaces, so every number measured before this change stays reproducible.
- Inside the tolerance window, *where* the peak sits costs nothing.
- A blurred peak costs more than a sharp one of the same height — the paper's
  central claim, and the reason this exists rather than more target smearing.
- ``pos_weight="auto"`` counts the negatives that survive the ignore band,
  not every non-beat frame.

Background: plans/09_lessons_from_literature.md §2.1.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from musicality.loaders.beat_dataset import gaussian_smear
from musicality.losses.pos_weight import AUTO_POS_WEIGHT_ALPHA, beat_pos_weight
from musicality.losses.shift_tolerance import (
    TOLERANCE_FRAMES,
    sliding_windowed_max,
    shift_tolerant_bce,
)

FPS = 22050 / 512  # the training front-end's frame rate, 43.07
N_FRAMES = int(16.0 * FPS)  # a 16 s crop, configs/beat_train.yaml data.duration

B, T = 2, 64
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
