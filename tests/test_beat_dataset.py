"""Tests for musicality.loaders.beat_dataset — real tracks/+annotations/
fixtures on disk, audio included. Audio *content* is irrelevant to this
loader's own logic (only the .beats file's times/positions and file
presence/absence matter here), but the files have to be real and decodable,
since the loader reads crops straight off disk through soundfile.

Fixture tracks are exactly DURATION long by default, so the crop window
starts at 0 and annotation times land on the frames the assertions expect.
"""

import numpy as np
import soundfile as sf
import torch
import pytest

SAMPLE_RATE = 22050
DURATION = 5.0
HOP_LENGTH = 512
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_FRAMES = N_SAMPLES // HOP_LENGTH

BEAT, ONE, LAST, MASK = range(4)


def _write_track(
    dataset_dir,
    track_id,
    beat_times=None,
    positions=None,
    duration=DURATION,
    sr=SAMPLE_RATE,
    channels=1,
):
    """Create tracks/<id>.wav + annotations/<id>.beats for one track.

    If beat_times is None, no audio or .beats file is written at all
    (simulates a track that was never migrated — the dataset should never
    see it, since it only iterates tracks with a .beats file). If positions
    is None but beat_times is given, the .beats file carries bare timestamps
    (no position column).
    """
    if beat_times is None:
        return

    tracks_dir = dataset_dir / "tracks"
    ann_dir = dataset_dir / "annotations"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    noise = (np.random.randn(int(duration * sr), channels) * 0.1).astype(np.float32)
    sf.write(tracks_dir / f"{track_id}.wav", noise, sr, subtype="PCM_16")

    if positions is not None:
        lines = (f"{t:.6f} {p}" for t, p in zip(beat_times, positions))
    else:
        lines = (f"{t:.6f}" for t in beat_times)
    (ann_dir / f"{track_id}.beats").write_text("\n".join(lines))


class TestBeatDataset:
    def test_len(self, tmp_path):
        """Dataset length equals the number of tracks that have beat annotations."""
        dataset_dir = tmp_path / "ballroom"
        for i in range(4):
            _write_track(dataset_dir, f"t{i}", beat_times=[0.5 * j for j in range(8)])
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
        )
        assert len(ds) == 4

    def test_skips_missing_beats(self, tmp_path):
        """A track with no .beats file at all (never migrated) is excluded."""
        dataset_dir = tmp_path / "ballroom"
        _write_track(dataset_dir, "t1", beat_times=[0.5, 1.0])
        _write_track(dataset_dir, "t2", beat_times=None)
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
        )
        assert len(ds) == 1

    def test_output_shapes(self, tmp_path):
        """Each item is a (waveform, target) pair with the expected fixed shapes.

        The waveform is (1, N_SAMPLES) — mono, fixed length.
        The target is (4, N_FRAMES) — beat/one/last/mask channels, one value
        per hop-length frame.
        """
        dataset_dir = tmp_path / "ballroom"
        _write_track(dataset_dir, "t1", beat_times=[0.5, 1.0, 1.5], positions=[1, 2, 3])
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            hop_length=HOP_LENGTH,
        )
        wav, target = ds[0]

        assert wav.shape == (1, N_SAMPLES)
        assert target.shape == (4, N_FRAMES)

    def test_beat_frames_are_set(self, tmp_path):
        """The beat-channel frame corresponding to each beat timestamp must peak at 1.0.

        Beat time t maps to frame index round(t * sample_rate / hop_length).
        """
        beat_times = [0.5, 1.0]
        dataset_dir = tmp_path / "ballroom"
        _write_track(dataset_dir, "t1", beat_times=beat_times)
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            hop_length=HOP_LENGTH,
        )
        _, target = ds[0]

        for t in beat_times:
            frame = round(t * SAMPLE_RATE / HOP_LENGTH)
            assert target[BEAT, frame].item() == pytest.approx(1.0)

    def test_target_in_unit_range(self, tmp_path):
        """Every value in the smeared target tensor lies in [0, 1]."""
        dataset_dir = tmp_path / "ballroom"
        _write_track(
            dataset_dir,
            "t1",
            beat_times=[0.5, 1.0, 1.5, 2.0],
            positions=[1, 2, 3, 4],
        )
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
        )
        _, target = ds[0]

        assert target.min().item() >= 0.0
        assert target.max().item() <= 1.0

    def test_beat_outside_clip_ignored(self, tmp_path):
        """A beat that falls after the clip duration must not appear in the target."""
        dataset_dir = tmp_path / "ballroom"
        _write_track(dataset_dir, "t1", beat_times=[DURATION + 1.0])
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
        )
        _, target = ds[0]

        assert target[BEAT].sum().item() == pytest.approx(0.0)

    def test_one_and_last_channels_from_positions(self, tmp_path):
        """Bar positions 1 and 4 populate the one/last channels at the right frames."""
        beat_times = [0.5, 1.0, 1.5, 2.0, 2.5]
        positions = [1, 2, 3, 4, 1]
        dataset_dir = tmp_path / "ballroom"
        _write_track(dataset_dir, "t1", beat_times=beat_times, positions=positions)
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            hop_length=HOP_LENGTH,
        )
        _, target = ds[0]

        one_frames = [
            round(t * SAMPLE_RATE / HOP_LENGTH)
            for t, p in zip(beat_times, positions)
            if p == 1
        ]
        last_frames = [
            round(t * SAMPLE_RATE / HOP_LENGTH)
            for t, p in zip(beat_times, positions)
            if p == 4
        ]

        for frame in one_frames:
            assert target[ONE, frame].item() == pytest.approx(1.0)
        for frame in last_frames:
            assert target[LAST, frame].item() == pytest.approx(1.0)

        # Frame at beat position 2 (1.0s) must not register in either channel.
        mid_frame = round(1.0 * SAMPLE_RATE / HOP_LENGTH)
        assert target[ONE, mid_frame].item() == pytest.approx(0.0)
        assert target[LAST, mid_frame].item() == pytest.approx(0.0)

    def test_group_size_8_reads_phrase_position_from_positions(self, tmp_path):
        """With group_size=8, `last` is populated from position==8, not position==4."""
        beat_times = [0.5 * i for i in range(1, 9)]
        positions = list(range(1, 9))  # a single 8-beat phrase, positions 1-8
        dataset_dir = tmp_path / "my_phrase_dataset"
        _write_track(dataset_dir, "t1", beat_times=beat_times, positions=positions)
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="my_phrase_dataset",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            hop_length=HOP_LENGTH,
            group_size=8,
        )
        _, target = ds[0]

        frame_at_pos4 = round(beat_times[3] * SAMPLE_RATE / HOP_LENGTH)
        frame_at_pos8 = round(beat_times[7] * SAMPLE_RATE / HOP_LENGTH)

        assert target[LAST, frame_at_pos4].item() == pytest.approx(0.0)
        assert target[LAST, frame_at_pos8].item() == pytest.approx(1.0)

    def test_mask_set_when_positions_present(self, tmp_path):
        """The mask channel is constant 1.0 when the track has bar-position annotations."""
        dataset_dir = tmp_path / "ballroom"
        _write_track(dataset_dir, "t1", beat_times=[0.5, 1.0], positions=[1, 2])
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
        )
        _, target = ds[0]

        assert torch.all(target[MASK] == 1.0)

    def test_mask_unset_when_positions_missing(self, tmp_path):
        """The mask channel is constant 0.0, and one/last stay empty, without position annotations."""
        dataset_dir = tmp_path / "rwc_popular"
        _write_track(dataset_dir, "t1", beat_times=[0.5, 1.0], positions=None)
        from musicality.loaders.beat_dataset import BeatDataset

        ds = BeatDataset(
            name="rwc_popular",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
        )
        _, target = ds[0]

        assert torch.all(target[MASK] == 0.0)
        assert torch.all(target[ONE] == 0.0)
        assert torch.all(target[LAST] == 0.0)


class TestRefsConstruction:
    def test_pulls_samples_from_multiple_source_datasets(self, tmp_path):
        _write_track(
            tmp_path / "ballroom", "a", beat_times=[0.5, 1.0, 1.5], positions=[1, 2, 3]
        )
        _write_track(tmp_path / "brid", "b", beat_times=[0.5, 1.0])

        from musicality.dataformats.track_io import TrackRef
        from musicality.loaders.beat_dataset import BeatDataset

        refs = [
            TrackRef("ballroom", "a", tmp_path / "ballroom"),
            TrackRef("brid", "b", tmp_path / "brid"),
        ]

        ds = BeatDataset(refs=refs, sample_rate=SAMPLE_RATE, duration=DURATION)

        assert len(ds) == 2
        audio_paths = {sample[0] for sample in ds.samples}
        assert any("ballroom" in p for p in audio_paths)
        assert any("brid" in p for p in audio_paths)

    def test_skips_unresolvable_audio_across_sources(self, tmp_path):
        _write_track(tmp_path / "ballroom", "a", beat_times=[0.5, 1.0])

        from musicality.dataformats.track_io import TrackRef
        from musicality.loaders.beat_dataset import BeatDataset

        # "brid/b" is listed but brid/ was never populated on disk.
        refs = [
            TrackRef("ballroom", "a", tmp_path / "ballroom"),
            TrackRef("brid", "b", tmp_path / "brid"),
        ]

        ds = BeatDataset(refs=refs, sample_rate=SAMPLE_RATE, duration=DURATION)

        assert len(ds) == 1

    def test_refs_populated_in_lockstep_with_samples(self, tmp_path):
        _write_track(tmp_path / "ballroom", "a", beat_times=[0.5, 1.0])

        from musicality.dataformats.track_io import TrackRef
        from musicality.loaders.beat_dataset import BeatDataset

        refs = [
            TrackRef("ballroom", "a", tmp_path / "ballroom"),
            TrackRef("brid", "b", tmp_path / "brid"),
        ]

        ds = BeatDataset(refs=refs, sample_rate=SAMPLE_RATE, duration=DURATION)

        assert len(ds.refs) == len(ds.samples) == 1
        assert ds.refs[0].dataset_name == "ballroom"

    def test_requires_name_or_refs(self):
        from musicality.loaders.beat_dataset import BeatDataset

        with pytest.raises(ValueError):
            BeatDataset()


class TestGaussianSmear:
    def test_isolated_spike_peak_is_one(self):
        from musicality.loaders.beat_dataset import gaussian_smear

        spike = np.zeros(50, dtype=np.float32)
        spike[25] = 1.0
        smeared = gaussian_smear(spike, sigma=1.5)

        assert smeared[25] == pytest.approx(1.0)
        assert smeared.max() == pytest.approx(1.0)

    def test_smear_spreads_to_neighbors(self):
        from musicality.loaders.beat_dataset import gaussian_smear

        spike = np.zeros(50, dtype=np.float32)
        spike[25] = 1.0
        smeared = gaussian_smear(spike, sigma=1.5)

        assert 0.0 < smeared[24] < 1.0
        assert 0.0 < smeared[26] < 1.0

    def test_no_spike_is_all_zero(self):
        from musicality.loaders.beat_dataset import gaussian_smear

        spike = np.zeros(50, dtype=np.float32)
        smeared = gaussian_smear(spike, sigma=1.5)

        assert np.all(smeared == 0.0)

    def test_clips_overlapping_bumps_to_one(self):
        from musicality.loaders.beat_dataset import gaussian_smear

        spike = np.zeros(50, dtype=np.float32)
        spike[25] = 1.0
        spike[26] = 1.0
        smeared = gaussian_smear(spike, sigma=1.5)

        assert smeared.max() == pytest.approx(1.0)
        assert smeared.max() <= 1.0


class TestDataLoader:
    def test_batch_shape(self, tmp_path):
        from torch.utils.data import DataLoader
        from musicality.loaders.beat_dataset import BeatDataset

        dataset_dir = tmp_path / "ballroom"
        for i in range(4):
            _write_track(
                dataset_dir,
                f"t{i}",
                beat_times=[0.5 * j for j in range(4)],
                positions=[1, 2, 3, 4],
            )

        ds = BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        wav, target = next(iter(loader))

        assert wav.shape == (4, 1, N_SAMPLES)
        assert target.shape == (4, 4, N_FRAMES)


class TestCropWindow:
    """The crop is read at an offset into the track, so annotation times have
    to be shifted onto the cropped window (see musicality.loaders.audio_io)."""

    BEAT_TIMES = [0.5 * i for i in range(1, 40)]  # 0.5 s .. 19.5 s

    def _dataset(self, tmp_path, random_crop):
        from musicality.loaders.beat_dataset import BeatDataset

        dataset_dir = tmp_path / "ballroom"
        _write_track(dataset_dir, "t1", beat_times=self.BEAT_TIMES, duration=20.0)

        return BeatDataset(
            name="ballroom",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            hop_length=HOP_LENGTH,
            random_crop=random_crop,
        )

    def test_fixed_crop_shifts_beat_times_to_middle_window(self, tmp_path):
        """A 20 s track cropped to 5 s starts at 7.5 s, so the beat at 8.0 s
        must land on the frame for 0.5 s into the clip."""
        ds = self._dataset(tmp_path, random_crop=False)
        _, target = ds[0]

        start = (20.0 - DURATION) / 2
        for t in self.BEAT_TIMES:
            rel = t - start
            if 0.0 <= rel < DURATION:
                frame = round(rel * SAMPLE_RATE / HOP_LENGTH)
                if frame < N_FRAMES:
                    assert target[BEAT, frame].item() == pytest.approx(1.0)

    def test_fixed_crop_is_deterministic(self, tmp_path):
        ds = self._dataset(tmp_path, random_crop=False)

        assert torch.equal(ds[0][0], ds[0][0])
        assert torch.equal(ds[0][1], ds[0][1])

    def test_random_crop_moves_the_window(self, tmp_path):
        """Repeated access draws different windows, and each keeps its beats
        aligned — a target that stayed identical would mean the offset never
        reached the annotations."""
        ds = self._dataset(tmp_path, random_crop=True)

        targets = [ds[0][1] for _ in range(10)]

        assert any(not torch.equal(targets[0], t) for t in targets[1:])
        # 5 s of beats spaced 0.5 s apart: every window sees ~10 of them.
        for t in targets:
            assert t[BEAT].max().item() == pytest.approx(1.0)


def _frame(t):
    """Frame index for an annotation time — mirrors BeatDataset._times_to_spike."""

    return int(round(t * SAMPLE_RATE / HOP_LENGTH))


class TestFoldPositions:
    """Annotations count across whatever the annotator called a bar. When that
    is longer than the head's group_size, the positions have to fold onto
    1..group_size or they match no channel at all."""

    def test_eight_beat_bar_folds_onto_four(self):
        from musicality.loaders.beat_dataset import fold_positions

        folded = fold_positions(np.array([1, 2, 3, 4, 5, 6, 7, 8]), group_size=4)

        assert list(folded) == [1, 2, 3, 4, 1, 2, 3, 4]

    def test_twelve_beat_bar_folds_as_three_bars_of_four(self):
        from musicality.loaders.beat_dataset import fold_positions

        folded = fold_positions(np.arange(1, 13), group_size=4)

        assert list(folded) == [1, 2, 3, 4] * 3

    def test_downbeat_spacing_stays_uniform(self):
        """The property that makes mod-folding correct: every folded "1" sits
        exactly group_size beats after the previous one. This is what a
        non-multiple cycle cannot give, and why it is rejected instead."""
        from musicality.loaders.beat_dataset import fold_positions

        folded = fold_positions(np.arange(1, 25), group_size=4)

        ones = np.flatnonzero(folded == 1)
        assert list(np.diff(ones)) == [4] * (len(ones) - 1)

    def test_matching_cycle_is_unchanged(self):
        from musicality.loaders.beat_dataset import fold_positions

        positions = np.array([1, 2, 3, 4, 1, 2, 3, 4])

        assert list(fold_positions(positions, group_size=4)) == list(positions)

    def test_shorter_cycle_is_left_alone(self):
        """A 2-beat bar against a 4-way head already lands in real channels —
        it describes a valid, if shorter, period and needs no folding."""
        from musicality.loaders.beat_dataset import fold_positions

        folded = fold_positions(np.array([1, 2, 1, 2]), group_size=4)

        assert list(folded) == [1, 2, 1, 2]

    def test_folding_is_idempotent(self):
        from musicality.loaders.beat_dataset import fold_positions

        once = fold_positions(np.arange(1, 9), group_size=4)
        twice = fold_positions(once, group_size=4)

        assert list(once) == list(twice)

    def test_non_multiple_cycle_is_rejected(self):
        """6 against 4 would give 1,2,3,4,1,2 — the downbeat 4 beats after one
        bar and 2 after the next. No consistent fold exists, so it must be
        refused rather than folded wrongly."""
        from musicality.loaders.beat_dataset import fold_positions

        assert fold_positions(np.array([1, 2, 3, 4, 5, 6]), group_size=4) is None

    def test_phrase_head_keeps_an_eight_beat_phrase_intact(self):
        from musicality.loaders.beat_dataset import fold_positions

        folded = fold_positions(np.arange(1, 9), group_size=8)

        assert list(folded) == list(range(1, 9))


class TestLongerBarsInTheDataset:
    """End-to-end: an 8-beat annotation must supervise all four position
    channels instead of falling into the uniform-target fallback."""

    BEATS = [0.5 * (i + 1) for i in range(8)]

    def _dataset(self, tmp_path, positions, layout="positions", group_size=4):
        from musicality.loaders.beat_dataset import BeatDataset

        dataset_dir = tmp_path / "rwc_classical"
        _write_track(dataset_dir, "t1", beat_times=self.BEATS, positions=positions)

        return BeatDataset(
            name="rwc_classical",
            data_home=dataset_dir,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            hop_length=HOP_LENGTH,
            group_size=group_size,
            target_layout=layout,
        )

    def test_beats_past_four_get_a_confident_target(self, tmp_path):
        """The bug this fixes. Beats 5-8 matched no channel, so the position
        block summed to ~0 there, hit the uniform fallback, and taught "every
        position is equally likely" at full beat weight on real, well
        annotated beats."""
        ds = self._dataset(tmp_path, positions=list(range(1, 9)))

        _, target = ds[0]
        block = target[1:-1]

        for t in self.BEATS:
            column = block[:, _frame(t)]
            assert column.max().item() == pytest.approx(1.0)
            # 1/group_size in every channel is the fallback's signature.
            assert column.min().item() == pytest.approx(0.0)

    def test_folded_positions_are_the_ground_truth_for_eval_too(self, tmp_path):
        """evaluation.py reads reference positions straight out of
        dataset.samples, so folding at load time is what keeps the training
        target and the scoring reference talking about the same thing."""
        ds = self._dataset(tmp_path, positions=list(range(1, 9)))

        _path, _times, positions, has_positions = ds.samples[0]

        assert has_positions
        assert list(positions) == [1, 2, 3, 4, 1, 2, 3, 4]

    def test_one_channel_fires_on_both_downbeats(self, tmp_path):
        """Under one_last, beat 5 of an 8-beat annotation is the downbeat of
        the second bar and must light the `one` channel."""
        ds = self._dataset(tmp_path, positions=list(range(1, 9)), layout="one_last")

        _, target = ds[0]

        assert target[ONE, _frame(0.5)].item() == pytest.approx(1.0)
        assert target[ONE, _frame(2.5)].item() == pytest.approx(1.0)
        assert target[LAST, _frame(2.0)].item() == pytest.approx(1.0)
        assert target[LAST, _frame(4.0)].item() == pytest.approx(1.0)

    def test_six_beat_track_keeps_its_beats_but_masks_positions(self, tmp_path):
        """6 cannot fold onto 4. Masking is strictly better than the uniform
        fallback: the beat head still trains on the track, the position head
        is simply told nothing rather than told something false."""
        ds = self._dataset(tmp_path, positions=[1, 2, 3, 4, 5, 6, 1, 2])

        _path, _times, positions, has_positions = ds.samples[0]
        _, target = ds[0]

        assert len(ds) == 1
        assert has_positions is False
        assert positions is None
        assert target[-1].max().item() == pytest.approx(0.0)
        assert target[0].max().item() == pytest.approx(1.0)

    def test_plain_four_beat_track_is_untouched(self, tmp_path):
        """Regression guard on the common case — ballroom and jtd are 100%
        4-beat, so nothing about them may change."""
        positions = [1, 2, 3, 4, 1, 2, 3, 4]
        ds = self._dataset(tmp_path, positions=positions)

        _path, _times, stored, has_positions = ds.samples[0]

        assert has_positions
        assert list(stored) == positions
