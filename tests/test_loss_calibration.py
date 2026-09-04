"""Tests for the phase-2 loss calibration of :func:`musicality.losses.beat_position_loss`.

Two independent changes, both defaulting off:

- ``pos_weight="auto"`` derives the beat head's positive-class weight per
  sample from the target, instead of using one constant tuned at one tempo.
- ``position_norm="per_item"`` averages the bar-position term over clips
  rather than over beats, so a clip's influence stops being proportional to
  its tempo.

Background: plans/04_beat_phase_generalization_and_data_prep.md §2.6a and §3 #2.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from musicality.loaders.beat_dataset import gaussian_smear
from musicality.losses import (
    AUTO_POS_WEIGHT_ALPHA,
    AUTO_POS_WEIGHT_RANGE,
    beat_pos_weight,
    beat_position_loss,
)
from musicality.trainers.beat_phase_module import BeatPhaseModule

FPS = 22050 / 512  # the training front-end's frame rate, 43.07
SIGMA = 1.5  # configs/beat_train.yaml sigma_frames
G = 4
N_FRAMES = int(16.0 * FPS)  # a 16 s crop, configs/beat_train.yaml data.duration


def _spikes(frames: np.ndarray, n_frames: int) -> np.ndarray:
    out = np.zeros(n_frames, dtype=np.float32)
    out[frames] = 1.0

    return out


def _beat_frames(bpm: float, n_frames: int = N_FRAMES) -> np.ndarray:
    """Frame indices of a metronomic beat grid at ``bpm``.

    Offset by half a period so that no bump is truncated by the window edge:
    :func:`~musicality.loaders.beat_dataset.gaussian_smear` convolves in
    ``"same"`` mode, so a beat sitting on frame 0 keeps only half its mass and
    skews the derived ratio by several percent on slow music.
    """

    period = 60.0 * FPS / bpm
    radius = int(round(3 * SIGMA))

    return np.round(np.arange(period / 2, n_frames - radius, period)).astype(int)


def _beat_channel(bpm: float, n_frames: int = N_FRAMES) -> np.ndarray:
    return gaussian_smear(_spikes(_beat_frames(bpm, n_frames), n_frames), SIGMA)


def _target(
    bpms: list[float],
    mask: bool = True,
    uniform_positions: bool = False,
    n_frames: int = N_FRAMES,
) -> torch.Tensor:
    """A ``(B, 2 + G, T)`` target holding one metronomic clip per entry in
    ``bpms``, built the way ``BeatDataset`` builds it.

    ``uniform_positions`` flattens the position block, which makes the
    per-frame cross-entropy constant within a clip — see
    :meth:`TestPerItemNormalization._clip_shares`.
    """

    rows = []
    for bpm in bpms:
        frames = _beat_frames(bpm, n_frames)
        beat = gaussian_smear(_spikes(frames, n_frames), SIGMA)

        if uniform_positions:
            block = np.full((G, n_frames), 1.0 / G, dtype=np.float32)
        else:
            block = np.stack(
                [
                    gaussian_smear(_spikes(frames[p::G], n_frames), SIGMA)
                    for p in range(G)
                ]
            )
            total = block.sum(axis=0, keepdims=True)
            block = np.divide(
                block,
                total,
                out=np.full_like(block, 1.0 / G),
                where=total > 1e-6,
            )

        annotated = np.full(n_frames, 1.0 if mask else 0.0, dtype=np.float32)
        rows.append(np.stack([beat, *block, annotated]))

    return torch.from_numpy(np.stack(rows).astype(np.float32))


def _beat_term(logits: torch.Tensor, target: torch.Tensor, pos_weight=5.0) -> float:
    """The loss's beat half, so the position half can be read by subtraction.

    Same pattern as ``tests/test_beat_conditioned_loss.py``. Only valid for a
    numeric ``pos_weight``.
    """

    return F.binary_cross_entropy_with_logits(
        logits[:, 0],
        target[:, 0],
        pos_weight=torch.as_tensor(pos_weight),
    ).item()


def _position_term(logits: torch.Tensor, target: torch.Tensor, **kwargs) -> float:
    return beat_position_loss(logits, target, **kwargs).item() - _beat_term(
        logits, target, kwargs.get("pos_weight", 5.0)
    )


class TestDefaultsAreUnchanged:
    """Both switches default to exactly the pre-phase-2 behaviour, so every
    number measured before this change stays reproducible after it."""

    @staticmethod
    def _reference(logits, target, pos_weight=5.0, phase_conditioning="beat"):
        """``beat_position_loss`` as it stood before phase 2, inline."""

        beat_logits, position_logits = logits[:, 0], logits[:, 1:]
        beat_y, position_y, mask = target[:, 0], target[:, 1:-1], target[:, -1]

        pw = torch.as_tensor(pos_weight, dtype=logits.dtype)
        beat_term = F.binary_cross_entropy_with_logits(
            beat_logits, beat_y, pos_weight=pw
        )

        log_q = F.log_softmax(position_logits, dim=1)
        position_ce = -(position_y * log_q).sum(dim=1)

        phase_w = mask * beat_y if phase_conditioning == "beat" else mask
        n_weighted = phase_w.sum().clamp(min=1.0)

        return beat_term + (position_ce * phase_w).sum() / n_weighted

    def test_matches_the_pre_phase_two_loss(self):
        torch.manual_seed(0)
        target = _target([125.0, 193.0])
        logits = torch.randn(2, 1 + G, N_FRAMES)

        assert torch.equal(
            beat_position_loss(logits, target), self._reference(logits, target)
        )

    def test_mask_conditioning_matches_too(self):
        torch.manual_seed(0)
        target = _target([125.0, 56.0])
        logits = torch.randn(2, 1 + G, N_FRAMES)

        assert torch.equal(
            beat_position_loss(logits, target, phase_conditioning="mask"),
            self._reference(logits, target, phase_conditioning="mask"),
        )


class TestSelfCalibratingPosWeight:
    def test_scalars_pass_straight_through(self):
        beat_y = torch.from_numpy(_beat_channel(125.0))[None]

        assert beat_pos_weight(beat_y, 5.0).item() == 5.0
        assert beat_pos_weight(beat_y, torch.tensor(3.0)).item() == 3.0

    def test_ballroom_median_reproduces_the_hand_tuned_five(self):
        """Why alpha defaults to 1.11: it anchors on the one corpus every
        earlier measurement was taken on, so switching to ``auto`` is a pure
        cross-tempo change rather than also a change of operating point."""

        beat_y = torch.from_numpy(_beat_channel(125.0))[None]

        assert beat_pos_weight(beat_y, "auto").item() == pytest.approx(5.0, rel=0.02)

    @pytest.mark.parametrize(
        "bpm, ratio",
        [(193.0, 2.6), (125.0, 4.5), (105.0, 5.5), (56.0, 11.3)],
        ids=["jtd", "ballroom", "rwc_classical_median", "rwc_classical_p10"],
    )
    def test_reproduces_the_measured_neg_pos_ratios(self, bpm, ratio):
        """The table in plans/04_...md §2.6a, now derived instead of measured."""

        beat_y = torch.from_numpy(_beat_channel(bpm))[None]

        assert beat_pos_weight(beat_y, "auto", alpha=1.0).item() == pytest.approx(
            ratio, abs=0.1
        )

    def test_slower_music_gets_a_heavier_weight(self):
        weights = [
            beat_pos_weight(torch.from_numpy(_beat_channel(bpm))[None], "auto").item()
            for bpm in (193.0, 125.0, 105.0, 56.0)
        ]

        assert weights == sorted(weights)

    def test_the_weight_is_per_sample_not_per_batch(self):
        beat_y = torch.from_numpy(np.stack([_beat_channel(193.0), _beat_channel(56.0)]))

        pw = beat_pos_weight(beat_y, "auto")

        assert pw.shape == (2, 1)
        assert pw[1].item() > 3 * pw[0].item()

    def test_a_beatless_crop_is_clamped_not_infinite(self):
        pw = beat_pos_weight(torch.zeros(1, N_FRAMES), "auto")

        assert torch.isfinite(pw).all()
        assert pw.item() == AUTO_POS_WEIGHT_RANGE[1]

    def test_a_single_beat_crop_is_clamped(self):
        beat_y = torch.from_numpy(_beat_channel(4.0))[None]  # one beat in 16 s

        assert beat_pos_weight(beat_y, "auto").item() == AUTO_POS_WEIGHT_RANGE[1]

    def test_the_slowest_real_music_stays_off_the_clamp(self):
        """20 guards degenerate crops; it must not cap slow music. 56 BPM is
        rwc_classical's 10th percentile, and 0.85 is the augmenter's floor."""

        beat_y = torch.from_numpy(_beat_channel(56.0 * 0.85))[None]

        assert beat_pos_weight(beat_y, "auto").item() < AUTO_POS_WEIGHT_RANGE[1]

    def test_rejects_an_unknown_string(self):
        with pytest.raises(ValueError, match="pos_weight"):
            beat_pos_weight(torch.zeros(1, 8), "adaptive")

    def test_shifts_the_loss_on_classical_but_barely_on_ballroom(self):
        """The anchor, seen at the loss level rather than the weight level."""

        torch.manual_seed(0)
        logits = torch.randn(1, 1 + G, N_FRAMES)

        def _gap(bpm):
            target = _target([bpm])
            fixed = beat_position_loss(logits, target, pos_weight=5.0).item()
            auto = beat_position_loss(logits, target, pos_weight="auto").item()

            return abs(auto - fixed) / fixed

        assert _gap(125.0) < 0.02
        assert _gap(56.0) > 5 * _gap(125.0)


class TestPerItemNormalization:
    def test_a_homogeneous_batch_is_unchanged(self):
        """Per-item only re-weights *across* clips; with equal beat counts
        there is nothing to re-weight."""

        torch.manual_seed(0)
        target = _target([125.0, 125.0])
        logits = torch.randn(2, 1 + G, N_FRAMES)

        assert torch.isclose(
            beat_position_loss(logits, target, position_norm="global"),
            beat_position_loss(logits, target, position_norm="per_item"),
            atol=1e-6,
        )

    def test_equals_the_mean_of_the_clips_scored_alone(self):
        torch.manual_seed(0)
        target = _target([56.0, 193.0])
        logits = torch.randn(2, 1 + G, N_FRAMES)

        batched = _position_term(logits, target, position_norm="per_item")
        alone = [_position_term(logits[i : i + 1], target[i : i + 1]) for i in (0, 1)]

        assert batched == pytest.approx(sum(alone) / 2, abs=1e-5)

    @staticmethod
    def _clip_shares(bpms: list[float], norm: str) -> list[float]:
        """The share of the position term each clip carries.

        Uses a flat position block and time-constant logits, so each clip's
        cross-entropy is a single number and the position term collapses to a
        plain weighted mean of those numbers. Bumping one clip at a time and
        differencing then reads off that clip's weight directly.
        """

        target = _target(bpms, uniform_positions=True)
        flat = torch.zeros(len(bpms), 1 + G, N_FRAMES)
        base = _position_term(flat, target, position_norm=norm)

        shares = []
        for i in range(len(bpms)):
            bumped = flat.clone()
            bumped[i, 1] = 4.0  # only clip i departs from a flat softmax
            shares.append(_position_term(bumped, target, position_norm=norm) - base)

        return [s / sum(shares) for s in shares]

    def test_global_weights_clips_by_tempo(self):
        """The defect itself: under ``global`` two clips' shares of the
        position gradient stand in exactly the ratio of their tempos."""

        slow, fast = self._clip_shares([125.0, 193.0], "global")

        assert fast / slow == pytest.approx(193.0 / 125.0, rel=0.02)

    def test_per_item_weights_clips_equally(self):
        slow, fast = self._clip_shares([125.0, 193.0], "per_item")

        assert fast / slow == pytest.approx(1.0, rel=1e-4)

    def test_a_fully_unannotated_batch_reduces_to_the_beat_term(self):
        """``global`` clamps its denominator and ``per_item`` clamps its clip
        count — neither may produce a NaN when nothing is annotated."""

        torch.manual_seed(0)
        target = _target([125.0, 193.0], mask=False)
        logits = torch.randn(2, 1 + G, N_FRAMES)

        for norm in ("global", "per_item"):
            loss = beat_position_loss(logits, target, position_norm=norm)

            assert torch.isfinite(loss)
            assert loss.item() == pytest.approx(_beat_term(logits, target), abs=1e-6)

    def test_an_unannotated_batch_still_backpropagates(self):
        torch.manual_seed(0)
        target = _target([125.0], mask=False)
        logits = torch.randn(1, 1 + G, N_FRAMES, requires_grad=True)

        beat_position_loss(logits, target, position_norm="per_item").backward()

        assert torch.isfinite(logits.grad).all()

    def test_unannotated_clips_do_not_dilute_the_annotated_ones(self):
        """A ``mask=0`` clip must contribute nothing at all, not a zero that
        drags the mean of the rest down."""

        torch.manual_seed(0)
        target = _target([125.0, 193.0])
        target[1, -1] = 0.0  # the second clip loses its position annotations
        logits = torch.randn(2, 1 + G, N_FRAMES)

        batched = _position_term(logits, target, position_norm="per_item")
        alone = _position_term(logits[:1], target[:1], position_norm="per_item")

        assert batched == pytest.approx(alone, abs=1e-5)

    def test_rejects_an_unknown_norm(self):
        with pytest.raises(ValueError, match="position_norm"):
            beat_position_loss(
                torch.zeros(1, 1 + G, 8),
                _target([125.0], n_frames=8),
                position_norm="macro",
            )


class TestModuleWiring:
    MODEL_CFG = OmegaConf.create(
        {
            "_target_": "musicality.models.tcn.TCNTempoNet",
            "n_mels": 16,
            "channels": 8,
            "n_layers": 3,
            "dropout": 0.0,
        }
    )

    def _module(self, **kwargs):
        return BeatPhaseModule(model=self.MODEL_CFG, group_size=G, **kwargs)

    def test_defaults_preserve_the_old_behaviour(self):
        module = self._module()

        assert module.hparams.position_norm == "global"
        assert module.hparams.pos_weight_alpha == AUTO_POS_WEIGHT_ALPHA

    @pytest.mark.parametrize(
        "pos_weight",
        [[5, 4, 4], OmegaConf.create([5, 4, 4])],
        ids=["list", "ListConfig"],
    )
    def test_a_per_head_pos_weight_is_rejected_for_the_softmax_head(self, pos_weight):
        """``configs/beat_train.yaml`` ships ``[5, 4, 4]``. Against the single
        BCE head of ``beat_position_loss`` that dies as a broadcast error on
        the first batch. OmegaConf's ``ListConfig`` is not a ``list``, hence
        both cases."""

        with pytest.raises(ValueError, match="scalar pos_weight"):
            self._module(pos_weight=pos_weight)

    def test_the_one_last_head_still_takes_a_per_head_list(self):
        module = BeatPhaseModule(model=self.MODEL_CFG, pos_weight=[5, 4, 4])

        assert module.hparams.group_size is None

    def test_auto_and_per_item_reach_the_loss(self):
        module = self._module(pos_weight="auto", position_norm="per_item")
        wav = torch.randn(2, 1, 4096)
        logits = module(wav)

        target = torch.zeros(2, 2 + G, logits.shape[-1])
        target[:, 0, ::8] = 1.0
        target[:, 1:-1] = 1.0 / G
        target[:, -1] = 1.0

        loss, _ = module._step((wav, target), "train")

        assert torch.isfinite(loss)
