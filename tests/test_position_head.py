"""Tests for the group_size-way softmax head over bar position.

Covers the ``target_layout="positions"`` dataset target, the
:func:`musicality.losses.beat_position_loss` objective,
:class:`~musicality.trainers.beat_phase_module.BeatPhaseModule` with
``group_size`` set, and the decoder reading a position distribution directly
instead of inferring one from two independent sigmoids.

Background: docs/beat_phase_improvement_review.md section 3.
"""

import numpy as np
import soundfile as sf
import torch
import pytest
from omegaconf import OmegaConf

from musicality.loaders.beat_dataset import BeatDataset, position_target_channels
from musicality.losses import beat_position_loss
from musicality.postprocess import label_bar_position_global, readout
from musicality.trainers.beat_phase_module import BeatPhaseModule

SAMPLE_RATE = 22050
DURATION = 5.0
HOP_LENGTH = 512
N_FRAMES = int(SAMPLE_RATE * DURATION) // HOP_LENGTH
G = 4


def _write_track(dataset_dir, track_id, beat_times, positions=None):
    tracks_dir = dataset_dir / "tracks"
    ann_dir = dataset_dir / "annotations"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    noise = (np.random.randn(int(DURATION * SAMPLE_RATE), 1) * 0.1).astype(np.float32)
    sf.write(tracks_dir / f"{track_id}.wav", noise, SAMPLE_RATE, subtype="PCM_16")

    if positions is not None:
        lines = (f"{t:.6f} {p}" for t, p in zip(beat_times, positions))
    else:
        lines = (f"{t:.6f}" for t in beat_times)
    (ann_dir / f"{track_id}.beats").write_text("\n".join(lines))


def _dataset(tmp_path, layout="positions", positions=True):
    beats = [0.5 * i for i in range(10)]
    pos = [(i % G) + 1 for i in range(10)] if positions else None
    _write_track(tmp_path / "ballroom", "t0", beats, pos)

    return BeatDataset(
        name="ballroom",
        data_home=tmp_path / "ballroom",
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        hop_length=HOP_LENGTH,
        group_size=G,
        target_layout=layout,
    )


class TestPositionTarget:
    def test_channel_names(self):
        assert position_target_channels(4) == (
            "beat",
            "pos_1",
            "pos_2",
            "pos_3",
            "pos_4",
            "mask",
        )
        assert len(position_target_channels(8)) == 10

    def test_shape_is_two_plus_group_size(self, tmp_path):
        _, target = _dataset(tmp_path)[0]

        assert target.shape == (2 + G, N_FRAMES)

    def test_beat_first_and_mask_last_match_the_old_layout(self, tmp_path):
        """target[0] and target[-1] mean the same thing under both layouts."""

        _, positions_t = _dataset(tmp_path, layout="positions")[0]
        _, one_last_t = _dataset(tmp_path, layout="one_last")[0]

        assert torch.allclose(positions_t[0], one_last_t[0])  # beat
        assert torch.allclose(positions_t[-1], one_last_t[-1])  # mask

    def test_position_block_is_a_distribution(self, tmp_path):
        _, target = _dataset(tmp_path)[0]

        assert torch.allclose(target[1:-1].sum(dim=0), torch.ones(N_FRAMES), atol=1e-5)

    def test_downbeat_frames_favour_position_one(self, tmp_path):
        ds = _dataset(tmp_path)
        _, target = ds[0]

        frame = int(round(0.0 * SAMPLE_RATE / HOP_LENGTH))  # beat 0 -> position 1
        assert target[1:-1, frame].argmax().item() == 0

        frame = int(round(1.0 * SAMPLE_RATE / HOP_LENGTH))  # beat 2 -> position 3
        assert target[1:-1, frame].argmax().item() == 2

    def test_middle_positions_are_supervised(self, tmp_path):
        """The whole point of section 3: positions 2 and 3 stop being identical."""

        _, target = _dataset(tmp_path)[0]

        assert target[2].max() > 0.5  # pos_2 fires somewhere
        assert target[3].max() > 0.5  # pos_3 fires somewhere
        assert not torch.allclose(target[2], target[3])

    def test_tracks_without_positions_get_a_uniform_block(self, tmp_path):
        _, target = _dataset(tmp_path, positions=False)[0]

        assert target[-1].max() == 0.0  # mask off
        assert torch.allclose(target[1:-1], torch.full((G, N_FRAMES), 1.0 / G))

    def test_rejects_unknown_layout(self, tmp_path):
        with pytest.raises(ValueError, match="target_layout"):
            _dataset(tmp_path, layout="bogus")


class TestBeatPositionLoss:
    @staticmethod
    def _batch(b=2, t=32, g=G):
        logits = torch.zeros(b, 1 + g, t, requires_grad=True)
        target = torch.zeros(b, 2 + g, t)
        target[:, 0, ::8] = 1.0  # beats
        target[:, 1:-1] = 1.0 / g  # uniform positions
        target[:, 1, ::8] = 1.0  # ...except downbeats are position 1
        target[:, 2:-1, ::8] = 0.0
        target[:, -1] = 1.0  # mask on
        return logits, target

    def test_output_is_scalar_and_finite(self):
        loss = beat_position_loss(*self._batch())

        assert loss.shape == () and torch.isfinite(loss)

    def test_positions_compete(self):
        """A softmax makes 1-vs-3 a single decision: raising one lowers the other."""

        logits, target = self._batch()

        right = logits.detach().clone()
        right[:, 1, ::8] = 6.0  # confident position 1 on the downbeats
        wrong = logits.detach().clone()
        wrong[:, 3, ::8] = 6.0  # equally confident position 3

        assert beat_position_loss(right, target) < beat_position_loss(wrong, target)

    def test_rejects_channel_mismatch(self):
        logits, target = self._batch()

        with pytest.raises(ValueError, match="position channels"):
            beat_position_loss(logits[:, :-1], target)

    def test_rejects_unknown_conditioning(self):
        with pytest.raises(ValueError, match="phase_conditioning"):
            beat_position_loss(*self._batch(), phase_conditioning="bogus")

    def test_beat_conditioning_ignores_non_beat_frames(self):
        logits, target = self._batch()

        perturbed = logits.detach().clone()
        # One channel, not all of them — a softmax is shift-invariant, so
        # raising every position logit equally would change nothing.
        perturbed[:, 1, 4::8] = 9.0  # halfway between beats

        assert torch.isclose(
            beat_position_loss(logits, target, phase_conditioning="beat"),
            beat_position_loss(perturbed, target, phase_conditioning="beat"),
            atol=1e-6,
        )
        assert not torch.isclose(
            beat_position_loss(logits, target, phase_conditioning="mask"),
            beat_position_loss(perturbed, target, phase_conditioning="mask"),
            atol=1e-6,
        )

    def test_group_size_eight(self):
        loss = beat_position_loss(*self._batch(g=8))

        assert torch.isfinite(loss)


class TestModuleWithPositionHead:
    MODEL_CFG = OmegaConf.create(
        {
            "_target_": "musicality.models.tcn.TCNTempoNet",
            "n_mels": 16,
            "channels": 8,
            "n_layers": 3,
            "dropout": 0.0,
        }
    )

    def test_defaults_to_the_one_last_head(self):
        module = BeatPhaseModule(model=self.MODEL_CFG)

        assert module.hparams.group_size is None
        assert module.model.n_outputs == 3

    def test_group_size_widens_the_head(self):
        module = BeatPhaseModule(model=self.MODEL_CFG, group_size=G)

        assert module.model.n_outputs == 1 + G

    def test_forward_and_step(self):
        module = BeatPhaseModule(model=self.MODEL_CFG, group_size=G)
        wav = torch.randn(2, 1, 4096)
        logits = module(wav)

        target = torch.zeros(2, 2 + G, logits.shape[-1])
        target[:, 1:-1] = 1.0 / G
        target[:, -1] = 1.0

        loss, probs = module._step((wav, target), "train")

        assert logits.shape[1] == 1 + G
        assert torch.isfinite(loss)
        assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-5)

    def test_rejects_group_size_below_two(self):
        with pytest.raises(ValueError, match="group_size"):
            BeatPhaseModule(model=self.MODEL_CFG, group_size=1)


class TestDecodeFromPositionProbs:
    FPS = 10.0
    PERIOD = 0.5

    def _curves(self, n_beats=24, p=0.9):
        beat_times = np.arange(n_beats) * self.PERIOD
        n_frames = int(beat_times[-1] * self.FPS) + 20

        position_probs = np.full((G, n_frames), (1 - p) / (G - 1))
        for i, t in enumerate(beat_times):
            frame = int(round(t * self.FPS))
            position_probs[:, frame] = (1 - p) / (G - 1)
            position_probs[i % G, frame] = p

        return beat_times, position_probs

    def test_recovers_phase_from_the_distribution(self):
        beat_times, position_probs = self._curves()
        blind = np.full(position_probs.shape[1], 0.5)  # useless one/last

        labels = label_bar_position_global(
            beat_times, blind, blind, self.FPS, position_probs=position_probs
        )

        assert labels == [(i % G) + 1 for i in range(len(beat_times))]

    def test_middle_positions_are_distinguishable(self):
        """Two sigmoids cannot express 'this is a 2, not a 3'; a softmax can."""

        beat_times, position_probs = self._curves()
        blind = np.full(position_probs.shape[1], 0.5)

        labels = label_bar_position_global(
            beat_times, blind, blind, self.FPS, position_probs=position_probs
        )

        assert 2 in labels and 3 in labels

    def test_rejects_wrong_row_count(self):
        beat_times, position_probs = self._curves()
        blind = np.full(position_probs.shape[1], 0.5)

        with pytest.raises(ValueError, match="group_size"):
            label_bar_position_global(
                beat_times, blind, blind, self.FPS, position_probs=position_probs[:3]
            )

    def test_readout_threads_position_probs(self):
        beat_times, position_probs = self._curves()
        beat_probs = np.full(position_probs.shape[1], 0.02)
        beat_probs[np.round(beat_times * self.FPS).astype(int)] = 0.95
        blind = np.full(position_probs.shape[1], 0.5)

        events = readout(
            beat_probs,
            blind,
            blind,
            fps=self.FPS,
            decoder="global",
            position_probs=position_probs,
        )

        assert events
        assert all(e["beat_in_bar"] is not None for e in events)
