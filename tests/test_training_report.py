"""Tests for musicality.callbacks.training_report.TrainingReportLogger.

The report exists to be handed to someone else — a colleague, a later session,
an analysis — as a single file. That puts three things under test:

- it must **parse strictly**, so `NaN` cannot reach the file
  (:class:`TestJsonable`),
- it must not record a number that was never measured: Lightning echoes the
  previous value of a metric that was not logged this epoch, and a flat line
  through the history reads as a measurement (:class:`TestHistory`),
- it must survive the absence of everything it reports on — no W&B logger, no
  event metrics, no checkpoint callback (:class:`TestDegradedRuns`).

:class:`TestBuild` covers the assembled content.
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from musicality.callbacks.event_metrics import PREFIX, EventMetricsLogger
from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.callbacks.training_report import (
    FILENAME,
    TrainingReportLogger,
    find_callback,
    jsonable,
)


KEYS = ("val/loss", "val/f_beat", f"{PREFIX}/f_beat", f"{PREFIX}/position_acc")


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


class _FakeTrainer:
    def __init__(
        self,
        metrics=None,
        current_epoch=0,
        max_epochs=4,
        callbacks=(),
        sanity_checking=False,
        is_global_zero=True,
        dirpath=None,
        experiment=None,
    ):
        self.callback_metrics = dict(metrics or {})
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.callbacks = list(callbacks)
        self.sanity_checking = sanity_checking
        self.is_global_zero = is_global_zero
        self.checkpoint_callback = MagicMock(
            dirpath=dirpath, best_model_path="", best_model_score=None
        )
        self.logger = MagicMock(experiment=experiment) if experiment else None


def _events(rows=None, scored=(0,)):
    logger = EventMetricsLogger([], {})
    logger.last_rows = list(rows or [_row()])
    logger.scored_epochs = list(scored)

    return logger


class TestJsonable:
    def test_nan_becomes_null(self):
        assert jsonable({"a": float("nan")}) == {"a": None}

    def test_infinity_becomes_null(self):
        assert jsonable([float("inf"), float("-inf")]) == [None, None]

    def test_finite_values_are_untouched(self):
        assert jsonable({"a": 0.5, "b": [1, "x", None]}) == {
            "a": 0.5,
            "b": [1, "x", None],
        }

    def test_nested_structures_are_walked(self):
        value = {"a": {"b": [{"c": float("nan")}]}}

        assert jsonable(value) == {"a": {"b": [{"c": None}]}}

    def test_a_raw_dump_would_not_have_parsed(self):
        """The reason this helper exists: `json.dumps` emits bare `NaN`, which
        Python reads back but which is not valid JSON."""

        raw = json.dumps({"a": float("nan")})

        assert "NaN" in raw
        with pytest.raises(ValueError):
            json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError))

        assert json.loads(json.dumps(jsonable({"a": float("nan")}))) == {"a": None}


class TestHistory:
    def test_records_the_tracked_keys(self):
        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(
            metrics={"val/loss": 1.0, "val/f_beat": 0.5}, callbacks=[_events()]
        )
        logger.on_validation_epoch_end(trainer, None)

        assert logger.history == [{"epoch": 0, "val/loss": 1.0, "val/f_beat": 0.5}]

    def test_skips_the_sanity_check(self):
        logger = TrainingReportLogger(keys=KEYS)
        logger.on_validation_epoch_end(
            _FakeTrainer(metrics={"val/loss": 1.0}, sanity_checking=True), None
        )

        assert logger.history == []

    def test_event_metrics_recorded_on_a_scored_epoch(self):
        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(
            metrics={f"{PREFIX}/f_beat": 0.8},
            current_epoch=2,
            callbacks=[_events(scored=[0, 2])],
        )
        logger.on_validation_epoch_end(trainer, None)

        assert logger.history[0][f"{PREFIX}/f_beat"] == 0.8

    def test_event_metrics_dropped_on_an_unscored_epoch(self):
        """Lightning's `callback_metrics` keeps the last value it saw, so on an
        epoch between scoring passes the event keys are the *previous* pass's
        numbers. Recording them would draw a flat line through the history that
        reads as a measurement."""

        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(
            metrics={"val/loss": 1.0, f"{PREFIX}/f_beat": 0.8},
            current_epoch=1,
            callbacks=[_events(scored=[0, 2])],
        )
        logger.on_validation_epoch_end(trainer, None)

        assert logger.history == [{"epoch": 1, "val/loss": 1.0}]

    def test_frame_metrics_are_kept_on_an_unscored_epoch(self):
        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(
            metrics={"val/f_beat": 0.5, f"{PREFIX}/f_beat": 0.8},
            current_epoch=1,
            callbacks=[_events(scored=[0])],
        )
        logger.on_validation_epoch_end(trainer, None)

        assert logger.history[0] == {"epoch": 1, "val/f_beat": 0.5}


class TestBuild:
    def test_carries_the_per_track_rows_without_rescoring(self):
        rows = [_row("ballroom"), _row("jtd")]
        events = _events(rows)
        logger = TrainingReportLogger(keys=KEYS)

        report = logger.build(_FakeTrainer(callbacks=[events]))

        assert report["event_metrics"]["n_tracks"] == 2
        assert [r["corpus"] for r in report["event_metrics"]["per_track"]] == [
            "ballroom",
            "jtd",
        ]

    def test_per_corpus_separates_the_corpora(self):
        rows = [_row("ballroom", f_beat=0.9), _row("jtd", f_beat=0.1)]
        report = TrainingReportLogger(keys=KEYS).build(
            _FakeTrainer(callbacks=[_events(rows)])
        )
        per_corpus = report["event_metrics"]["per_corpus"]

        assert per_corpus["ballroom"]["f_beat"] == pytest.approx(0.9)
        assert per_corpus["jtd"]["f_beat"] == pytest.approx(0.1)
        assert per_corpus["jtd"]["n_tracks"] == 1

    def test_best_metrics_come_from_the_printer(self):
        printer = BestMetricsPrinter(keys=KEYS)
        printer.best = {"val/loss": 0.25}

        report = TrainingReportLogger(keys=KEYS).build(
            _FakeTrainer(callbacks=[_events(), printer])
        )

        assert report["frame_metrics"]["best"] == {"val/loss": 0.25}

    def test_final_metrics_are_the_last_history_row(self):
        logger = TrainingReportLogger(keys=KEYS)
        logger.history = [{"epoch": 0, "val/loss": 2.0}, {"epoch": 1, "val/loss": 1.0}]

        report = logger.build(_FakeTrainer(callbacks=[_events()]))

        assert report["frame_metrics"]["final"] == {"epoch": 1, "val/loss": 1.0}

    def test_epochs_run_is_not_off_by_one(self):
        """Lightning advances `current_epoch` past the last completed epoch
        before `on_fit_end`, so it is already a count."""

        report = TrainingReportLogger(keys=KEYS).build(
            _FakeTrainer(current_epoch=4, max_epochs=4, callbacks=[_events()])
        )

        assert report["run"]["epochs_run"] == 4

    def test_config_is_embedded_resolved(self):
        cfg = OmegaConf.create({"lr": 5e-4, "data": {"input": "merge"}})
        report = TrainingReportLogger(keys=KEYS, cfg=cfg).build(
            _FakeTrainer(callbacks=[_events()])
        )

        assert report["config"] == {"lr": 5e-4, "data": {"input": "merge"}}

    def test_unmeasurable_metrics_reach_the_file_as_null(self):
        rows = [_row(position_acc=None, position_acc_best_offset=None)]
        report = TrainingReportLogger(keys=KEYS).build(
            _FakeTrainer(callbacks=[_events(rows)])
        )

        assert report["event_metrics"]["summary"]["position_acc"] is None

    def test_the_whole_report_round_trips_through_strict_json(self):
        rows = [_row("ballroom"), _row("jtd", position_acc=None)]
        logger = TrainingReportLogger(keys=KEYS, cfg=OmegaConf.create({"lr": 1e-3}))
        logger.history = [{"epoch": 0, "val/loss": 1.0}]

        report = logger.build(_FakeTrainer(callbacks=[_events(rows)]))
        text = json.dumps(report)

        assert "NaN" not in text and "Infinity" not in text
        assert json.loads(text)["schema"] == report["schema"]


class TestWriteAndUpload:
    def test_writes_beside_the_checkpoints(self, tmp_path):
        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(callbacks=[_events()], dirpath=str(tmp_path / "run"))

        logger.on_fit_end(trainer, None)

        written = tmp_path / "run" / FILENAME
        assert written.is_file()
        assert json.loads(written.read_text())["schema"] == 1

    def test_uploads_to_the_wandb_run(self, tmp_path):
        experiment = MagicMock()
        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(
            callbacks=[_events()], dirpath=str(tmp_path), experiment=experiment
        )

        logger.on_fit_end(trainer, None)

        experiment.save.assert_called_once()
        assert experiment.save.call_args.kwargs["policy"] == "now"

    def test_no_wandb_logger_is_not_an_error(self, tmp_path):
        """`logger=False` is how the test suite and `sweep_lr.py` run."""

        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(callbacks=[_events()], dirpath=str(tmp_path))

        assert logger.upload(trainer, tmp_path / FILENAME) is False

    def test_nothing_is_written_on_a_non_zero_rank(self, tmp_path):
        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(
            callbacks=[_events()], dirpath=str(tmp_path), is_global_zero=False
        )

        logger.on_fit_end(trainer, None)

        assert not (tmp_path / FILENAME).exists()


class TestDegradedRuns:
    def test_no_event_metrics_callback(self, tmp_path):
        """`event_metrics.enabled=false` must still produce a report."""

        logger = TrainingReportLogger(keys=KEYS)
        trainer = _FakeTrainer(dirpath=str(tmp_path))

        logger.on_fit_end(trainer, None)
        report = json.loads((tmp_path / FILENAME).read_text())

        assert report["event_metrics"]["n_tracks"] == 0
        assert report["readable"] == "no event metrics were scored"

    def test_no_checkpoint_dirpath_falls_back_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logger = TrainingReportLogger(keys=KEYS)

        logger.on_fit_end(_FakeTrainer(callbacks=[_events()]), None)

        assert (tmp_path / FILENAME).is_file()

    def test_find_callback_returns_none_when_absent(self):
        assert find_callback(_FakeTrainer(), EventMetricsLogger) is None


class TestJsonableTotality:
    """`jsonable` has to be total: the report is written at the very end of a
    training run, so a `TypeError` from one odd field would discard the whole
    thing at the worst possible moment."""

    def test_unknown_objects_degrade_to_a_string(self):
        assert jsonable({"x": Path("/tmp/a")}) == {"x": "/tmp/a"}

    def test_a_mock_attribute_does_not_break_the_dump(self):
        assert json.dumps(jsonable({"id": MagicMock()}))

    def test_primitives_keep_their_types(self):
        value = jsonable({"a": True, "b": 3, "c": 1.5, "d": "s", "e": None})

        assert value == {"a": True, "b": 3, "c": 1.5, "d": "s", "e": None}
        assert isinstance(value["a"], bool) and isinstance(value["b"], int)

    def test_non_string_dict_keys_are_stringified(self):
        assert jsonable({1: "a"}) == {"1": "a"}
