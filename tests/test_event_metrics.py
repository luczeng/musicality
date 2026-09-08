"""Tests for musicality.callbacks.event_metrics.EventMetricsLogger.

The callback exists so that the numbers a training run reports are the same
kind of numbers the evaluation CLI reports — events on full tracks, not frames
in a 16-second clip. Three things have to hold for that to be true, and each
has its own class below:

- the subsample is fixed and spread across corpora, so the logged series is a
  curve over one set of tracks rather than a walk over different ones
  (:class:`TestStratifiedSample`),
- it fires on the epochs it says it does, and never during the sanity check
  (:class:`TestShouldRun`),
- it logs into its own ``val_event/`` namespace, never touching the ``val/loss``
  that ``ModelCheckpoint`` and ``ReduceLROnPlateau`` monitor
  (:class:`TestLogging`),

plus :class:`TestScore`, which runs the real
:meth:`~musicality.evaluation.BeatEvaluator.score` path over a synthetic track
to pin the wiring between the callback and the shared scorer.

Background: plans/06_metric_calibration_and_eval_consolidation.md, phase D.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from musicality.callbacks.event_metrics import (
    LOGGED_KEYS,
    PREFIX,
    EventMetricsLogger,
    stratified_sample,
)
from musicality.callbacks.metrics_logger import _LOWER_BETTER, BestMetricsPrinter
from musicality.dataformats.track_io import TrackRef
from musicality.trainers.train_beat_phase import (
    _TRACKED_KEYS,
    build_callbacks,
    build_event_metrics_callback,
)


# What build_checkpoint_callback needs; irrelevant to the event metrics, but
# build_callbacks builds it too.
_CHECKPOINT_CFG = {
    "checkpoint_dir": "ckpts/",
    "trainer": {},
    "wandb": {"run_name": None},
}


FPS = 10.0
PERIOD = 0.5
G = 4
N_BEATS = 60  # 30s — long enough to survive mir_eval's 5s trim


def _refs(**counts) -> list[TrackRef]:
    """Refs laid out the way a split file is: one corpus after another."""

    return [
        TrackRef(dataset_name=corpus, track_id=f"{corpus}_{i}", data_home=Path("."))
        for corpus, n in counts.items()
        for i in range(n)
    ]


class _FakeTrainer:
    def __init__(
        self,
        current_epoch: int = 0,
        max_epochs: int = 100,
        sanity_checking: bool = False,
        is_global_zero: bool = True,
    ):
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.sanity_checking = sanity_checking
        self.is_global_zero = is_global_zero


class _FakeModule:
    """Stands in for a ``BeatPhaseModule`` mid-training: it answers the two
    things the evaluator asks of it (``hparams``, a forward pass) and records
    what was logged through it."""

    def __init__(self, logits: torch.Tensor | None = None, group_size: int = G):
        self.hparams = {"task": "beat_phase", "group_size": group_size}
        self.device = "cpu"
        self.training = True
        self.logged = {}
        self.n_forward = 0
        self._logits = logits

    def __call__(self, wav):
        self.n_forward += 1

        return self._logits

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def log(self, key, value, **kwargs):
        self.logged[key] = value


class _StubLogger(EventMetricsLogger):
    """An ``EventMetricsLogger`` with the expensive half replaced: fixed rows
    instead of a model pass over real audio."""

    def __init__(self, rows: list[dict], **kwargs):
        super().__init__([], **kwargs)
        self._rows = rows
        self._dataset = list(range(len(rows)))  # only len() is ever taken

    def score(self, pl_module):
        return self._rows


def _row(corpus="ballroom", **overrides):
    row = {
        "corpus": corpus,
        "f_beat": 0.8,
        "cmlt": 0.6,
        "amlt": 0.9,
        "position_acc": 0.5,
        "position_acc_best_offset": 0.7,
        "anchor_error": 0.2,
        "f_one": 0.5,
        "f_last": 0.5,
        "confusion": 0.3,
    }
    row.update(overrides)

    return row


class TestStratifiedSample:
    def test_spreads_evenly_across_corpora(self):
        """The whole point: taking the first N of a corpus-ordered split file
        yields N tracks of one corpus."""

        selected = stratified_sample(_refs(ballroom=40, jtd=15, rwc_classical=5), 9)
        corpora = [ref.dataset_name for ref in selected]

        assert len(selected) == 9
        assert {c: corpora.count(c) for c in set(corpora)} == {
            "ballroom": 3,
            "jtd": 3,
            "rwc_classical": 3,
        }

    def test_naive_head_would_have_taken_one_corpus(self):
        refs = _refs(ballroom=40, jtd=15, rwc_classical=5)

        assert {r.dataset_name for r in refs[:9]} == {"ballroom"}

    def test_exhausted_corpus_drops_out_without_truncating(self):
        """A corpus with 2 tracks contributes 2 and the rest is made up
        elsewhere — the sample is still the requested size."""

        selected = stratified_sample(_refs(ballroom=10, rwc_classical=2), 8)
        corpora = [ref.dataset_name for ref in selected]

        assert len(selected) == 8
        assert corpora.count("rwc_classical") == 2
        assert corpora.count("ballroom") == 6

    def test_is_deterministic(self):
        refs = _refs(ballroom=40, jtd=15)

        assert stratified_sample(refs, 10) == stratified_sample(refs, 10)

    def test_seed_changes_the_selection(self):
        refs = _refs(ballroom=40, jtd=15)

        assert stratified_sample(refs, 10, seed=0) != stratified_sample(
            refs, 10, seed=1
        )

    def test_does_not_take_a_corpus_in_file_order(self):
        """Split files are often ordered *within* a corpus too (ballroom is
        written genre by genre), so the head of one corpus can be one dance
        style. The per-corpus shuffle is what prevents that."""

        refs = _refs(ballroom=40)
        selected = stratified_sample(refs, 5)

        assert selected != refs[:5]
        assert {r.track_id for r in selected} <= {r.track_id for r in refs}

    def test_none_keeps_everything(self):
        refs = _refs(ballroom=4, jtd=3)

        selected = stratified_sample(refs, None)

        assert {r.track_id for r in selected} == {r.track_id for r in refs}

    def test_request_larger_than_the_split_keeps_everything(self):
        refs = _refs(ballroom=4, jtd=3)

        assert len(stratified_sample(refs, 100)) == 7

    def test_empty_refs(self):
        assert stratified_sample([], 10) == []

    def test_single_corpus_is_a_plain_subsample(self):
        selected = stratified_sample(_refs(ballroom=20), 5)

        assert len(selected) == 5
        assert {r.dataset_name for r in selected} == {"ballroom"}


class TestShouldRun:
    @staticmethod
    def _logger(**kwargs):
        return EventMetricsLogger([], **kwargs)

    def test_fires_on_multiples_of_every_n_epochs(self):
        logger = self._logger(every_n_epochs=5)
        fired = [e for e in range(11) if logger.should_run(_FakeTrainer(e))]

        assert fired == [0, 5, 10]

    def test_always_fires_on_the_last_epoch(self):
        """Otherwise the model the run *finishes* with never gets an event
        number, which is the one number anyone wants from the run."""

        logger = self._logger(every_n_epochs=5)

        assert logger.should_run(_FakeTrainer(99, max_epochs=100))

    def test_skips_the_sanity_check(self):
        logger = self._logger(every_n_epochs=1)

        assert not logger.should_run(_FakeTrainer(0, sanity_checking=True))

    def test_skips_non_zero_ranks(self):
        logger = self._logger(every_n_epochs=1)

        assert not logger.should_run(_FakeTrainer(0, is_global_zero=False))

    def test_every_n_epochs_below_one_means_every_epoch(self):
        logger = self._logger(every_n_epochs=0)

        assert logger.every_n_epochs == 1
        assert all(logger.should_run(_FakeTrainer(e)) for e in range(5))

    def test_missing_max_epochs_is_not_a_crash(self):
        """`trainer.max_epochs` is None for a step-limited run."""

        logger = self._logger(every_n_epochs=5)

        assert not logger.should_run(_FakeTrainer(3, max_epochs=None))


class TestLogging:
    def test_logs_every_key_under_the_event_prefix(self):
        logger = _StubLogger([_row(), _row()], every_n_epochs=1)
        module = _FakeModule()

        logger.on_validation_epoch_end(_FakeTrainer(0), module)

        assert set(module.logged) == {f"{PREFIX}/{key}" for key in LOGGED_KEYS}

    def test_logged_values_are_the_summary_means(self):
        rows = [_row(f_beat=0.4), _row(f_beat=0.8)]
        module = _FakeModule()

        _StubLogger(rows, every_n_epochs=1).on_validation_epoch_end(
            _FakeTrainer(0), module
        )

        assert module.logged[f"{PREFIX}/f_beat"] == pytest.approx(0.6)

    def test_never_logs_the_monitored_val_loss(self):
        """`val/loss` drives ReduceLROnPlateau, ModelCheckpoint's top-k, and
        the checkpoint filename. Nothing here may write into that namespace."""

        module = _FakeModule()
        _StubLogger([_row()], every_n_epochs=1).on_validation_epoch_end(
            _FakeTrainer(0), module
        )

        assert not any(key.startswith("val/") for key in module.logged)

    def test_unscorable_metrics_are_skipped_not_logged_as_nan(self):
        """A beat-only split has no bar positions; a NaN point in W&B reads as
        a failed epoch rather than an absent metric."""

        rows = [_row(position_acc=None, position_acc_best_offset=None)]
        module = _FakeModule()

        _StubLogger(rows, every_n_epochs=1).on_validation_epoch_end(
            _FakeTrainer(0), module
        )

        assert f"{PREFIX}/f_beat" in module.logged
        assert f"{PREFIX}/position_acc" not in module.logged

    def test_nothing_is_logged_on_a_skipped_epoch(self):
        module = _FakeModule()
        _StubLogger([_row()], every_n_epochs=5).on_validation_epoch_end(
            _FakeTrainer(3), module
        )

        assert module.logged == {}

    def test_every_logged_key_is_higher_is_better(self):
        """`BestMetricsPrinter` picks the direction by substring match. A
        lower-is-better key added here (`confusion`) would have its *worst*
        value reported as its best unless `_LOWER_BETTER` is updated too."""

        for key in LOGGED_KEYS:
            assert not any(word in key for word in _LOWER_BETTER)

        assert "confusion" not in LOGGED_KEYS


def _spiky_logits(n_beats: int = N_BEATS, group_size: int = G) -> torch.Tensor:
    """Logits for a model that has learned the track perfectly: a confident
    beat spike on the grid and the right bar position at each one."""

    beat_frames = [int(round(i * PERIOD * FPS)) for i in range(n_beats)]
    n_frames = beat_frames[-1] + 10

    logits = torch.zeros(1, 1 + group_size, n_frames)
    logits[0, 0] = -6.0

    for i, frame in enumerate(beat_frames):
        logits[0, 0, frame] = 6.0
        logits[0, 1 + (i % group_size), frame] = 6.0

    return logits


def _fake_dataset(corpora: list[str], n_beats: int = N_BEATS):
    beat_times = np.arange(n_beats) * PERIOD
    positions = np.array([(i % G) + 1 for i in range(n_beats)])

    dataset = MagicMock()
    dataset.samples = [(f"{c}.wav", beat_times, positions, True) for c in corpora]
    dataset.refs = [MagicMock(dataset_name=c) for c in corpora]
    dataset.__len__.return_value = len(corpora)

    return dataset


class TestScore:
    """`score()` goes through the real BeatEvaluator, so these pin the seam
    between an in-training module and the shared scoring path."""

    @staticmethod
    def _logger(corpora):
        logger = EventMetricsLogger(
            [], every_n_epochs=1, sample_rate=int(FPS), hop_length=1, group_size=G
        )
        logger._dataset = _fake_dataset(corpora)

        return logger

    def test_scores_a_clean_track_near_one(self):
        logger = self._logger(["ballroom"])
        module = _FakeModule(_spiky_logits())

        with patch(
            "musicality.evaluation.load_track_waveform",
            return_value=torch.zeros(1, 1000),
        ):
            rows = logger.score(module)

        assert rows[0]["f_beat"] > 0.95
        assert rows[0]["position_acc"] > 0.95

    def test_rows_carry_their_corpus(self):
        logger = self._logger(["ballroom", "jtd"])
        module = _FakeModule(_spiky_logits())

        with patch(
            "musicality.evaluation.load_track_waveform",
            return_value=torch.zeros(1, 1000),
        ):
            rows = logger.score(module)

        assert [r["corpus"] for r in rows] == ["ballroom", "jtd"]
        assert module.n_forward == 2  # one pass per track, not one per metric

    def test_training_mode_is_restored(self):
        """Lightning has already switched to eval for the validation loop, but
        a callback that left the module in eval would silently disable dropout
        for the rest of training."""

        logger = self._logger(["ballroom"])
        module = _FakeModule(_spiky_logits())
        module.training = True

        with patch(
            "musicality.evaluation.load_track_waveform",
            return_value=torch.zeros(1, 1000),
        ):
            logger.score(module)

        assert module.training

    def test_a_fresh_evaluator_per_call_sees_the_new_weights(self):
        """The evaluator caches probabilities to make several *decoder*
        settings share one model pass. Reusing one across epochs would report
        the first epoch's numbers forever."""

        logger = self._logger(["ballroom"])
        module = _FakeModule(_spiky_logits())

        with patch(
            "musicality.evaluation.load_track_waveform",
            return_value=torch.zeros(1, 1000),
        ):
            logger.score(module)
            logger.score(module)

        assert module.n_forward == 2


class TestBuildEventMetricsCallback:
    """The config surface in configs/beat_train.yaml."""

    @staticmethod
    def _cfg(event_metrics=None, **overrides):
        base = {
            "data": {"input": "ballroom", "sample_rate": 22050},
            "hop_length": 512,
            "group_size": 4,
            "binary_only": True,
        }
        if event_metrics is not None:
            base["event_metrics"] = event_metrics
        base.update(overrides)

        return OmegaConf.create(base)

    @contextmanager
    def _split(self, n_refs=20):
        refs = _refs(ballroom=n_refs)
        with patch(
            "musicality.trainers.train_beat_phase.resolve_beat_split_refs",
            return_value=([], refs),
        ):
            yield refs

    def test_disabled_by_omission(self):
        """A config written before this block existed must train exactly as it
        did before, not pick up a 50-track scoring pass."""

        assert build_event_metrics_callback(self._cfg()) is None

    def test_disabled_explicitly(self):
        cfg = self._cfg({"enabled": False, "n_tracks": 50, "every_n_epochs": 5})

        assert build_event_metrics_callback(cfg) is None

    def test_settings_reach_the_callback(self):
        cfg = self._cfg({"enabled": True, "n_tracks": 3, "every_n_epochs": 7})

        with self._split():
            callback = build_event_metrics_callback(cfg)

        assert isinstance(callback, EventMetricsLogger)
        assert len(callback.refs) == 3
        assert callback.every_n_epochs == 7

    def test_audio_settings_are_taken_from_the_training_config(self):
        """Full-track inference has to use the sample rate and hop the model
        was trained at, or the frame grid it decodes is not the one it learned."""

        cfg = self._cfg({"enabled": True}, hop_length=256, group_size=8)

        with self._split():
            callback = build_event_metrics_callback(cfg)

        assert callback.sample_rate == 22050
        assert callback.hop_length == 256
        assert callback.group_size == 8
        assert callback.binary_only is True

    def test_tracked_keys_include_the_event_metrics(self):
        for key in LOGGED_KEYS:
            assert f"{PREFIX}/{key}" in _TRACKED_KEYS

    def test_build_callbacks_appends_it_when_enabled(self):
        cfg = self._cfg({"enabled": True, "n_tracks": 2}, **_CHECKPOINT_CFG)

        with self._split():
            callbacks = build_callbacks(cfg)

        assert any(isinstance(c, EventMetricsLogger) for c in callbacks)

    def test_it_runs_before_the_metrics_printer(self):
        """Both act in `on_validation_epoch_end`, one writing metrics and the
        other reading them, and `trainer.callback_metrics` is recomputed from
        the live results on every read. Reversed, the printer sees every key
        except these and the end-of-run summary omits them."""

        cfg = self._cfg({"enabled": True, "n_tracks": 2}, **_CHECKPOINT_CFG)

        with self._split():
            callbacks = build_callbacks(cfg)

        types = [type(c) for c in callbacks]
        assert types.index(EventMetricsLogger) < types.index(BestMetricsPrinter)

    def test_build_callbacks_omits_it_when_disabled(self):
        cfg = self._cfg({"enabled": False}, **_CHECKPOINT_CFG)
        callbacks = build_callbacks(cfg)

        assert not any(isinstance(c, EventMetricsLogger) for c in callbacks)
