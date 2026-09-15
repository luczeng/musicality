"""Callback that writes one self-contained report at the end of training and
uploads it to the W&B run's Files tab.

The numbers a run produces are scattered across places that outlive it badly:
the W&B charts (which need the dashboard to read), the terminal scrollback
(which does not survive a closed session), and the checkpoint filename (which
holds one number). Reconstructing "how did that run actually go" afterwards
means opening three of them and remembering which was which.

This writes the whole picture to one JSON file instead — final and best
metrics, the per-epoch history, the per-track and per-corpus event scores, the
resolved config, and the run's own identity — so a run can be handed to someone
else, or to a later session, as a single attachment.

**It is deliberately machine-readable.** JSON, one flat schema, ``NaN`` written
as ``null`` so it parses under a strict reader. A rendered
:func:`~musicality.evaluation.summary_block` is carried in the ``readable``
field so opening the file still shows something a person can skim.

Cheap: it decodes nothing. The event scores are the per-track rows
:class:`~musicality.callbacks.event_metrics.EventMetricsLogger` already
computed on its last scoring pass, which is why that pass always runs on the
final epoch.
"""

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
from omegaconf import DictConfig, OmegaConf

from musicality.callbacks.event_metrics import PREFIX, EventMetricsLogger
from musicality.callbacks.metrics_logger import BestMetricsPrinter
from musicality.evaluation import SCORE_KEYS, group_by_corpus, summarize, summary_block


FILENAME = "training_report.json"
SCHEMA = 1


def jsonable(value):
    """Make *value* survive :func:`json.dumps` and a strict reader afterwards.

    Two conversions, and the report depends on both being total:

    - **Float specials become ``null``.** ``json.dumps`` emits bare
      ``NaN``/``Infinity``, which Python reads back but which is not valid JSON
      and which strict parsers reject. An unmeasurable metric becoming ``null``
      is the same convention :func:`~musicality.evaluation.score_events`
      already uses for a metric that does not apply.
    - **Anything not a JSON primitive becomes its ``repr``-ish string.** The
      report pulls fields off third-party objects (a W&B run's ``id``, a
      checkpoint callback's ``best_model_score``) whose types are not ours to
      guarantee. Degrading one field to a string keeps the report writable;
      raising at ``json.dumps`` would lose the whole thing at the very end of a
      training run, which is the worst possible moment to lose it.
    """

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    # bool before int: `isinstance(True, int)` is True, and `True` is already
    # valid JSON, so the order only matters for readability here.
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return str(value)


def git_commit() -> str | None:
    """The commit the run was launched from, or ``None`` outside a checkout.

    Worth the subprocess: a report that cannot be tied back to the code that
    produced it is very hard to act on six weeks later.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    return result.stdout.strip() or None


def find_callback(trainer, cls):
    """The first callback of type *cls* attached to *trainer*, or ``None``.

    Looked up rather than injected so this callback has no constructor
    dependency on its collaborators — it reports on whatever is present, and a
    run with the event metrics switched off still gets a report.
    """

    return next((c for c in trainer.callbacks if isinstance(c, cls)), None)


class TrainingReportLogger(L.Callback):
    """Accumulates per-epoch metrics and writes the report at ``on_fit_end``.

    Must be registered **after**
    :class:`~musicality.callbacks.event_metrics.EventMetricsLogger`: both act in
    ``on_validation_epoch_end``, and ``trainer.callback_metrics`` is recomputed
    from the live results collection on every read, so metrics logged by a later
    callback are not visible yet.

    :param keys: Metric keys to record each epoch — the trainer's
        ``_TRACKED_KEYS``.
    :param cfg: The resolved Hydra config, embedded verbatim in the report.
        ``None`` omits it.
    :param filename: Report filename, written next to the run's checkpoints.
    """

    def __init__(
        self,
        keys: tuple,
        cfg: DictConfig | None = None,
        filename: str = FILENAME,
    ):
        self.keys = tuple(keys)
        self.cfg = cfg
        self.filename = filename
        self.history: list[dict] = []

    def event_epochs(self, trainer) -> set:
        logger = find_callback(trainer, EventMetricsLogger)

        return set(logger.scored_epochs) if logger else set()

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        scored = self.event_epochs(trainer)
        metrics = trainer.callback_metrics

        row = {"epoch": trainer.current_epoch}
        for key in self.keys:
            value = metrics.get(key)
            if value is None:
                continue

            # Lightning's callback_metrics keeps the last value it saw rather
            # than clearing it, so an event key on a non-scoring epoch is the
            # *previous* pass's number. Recording it would draw a flat line
            # through the history that looks like a measurement and is not one.
            if key.startswith(f"{PREFIX}/") and trainer.current_epoch not in scored:
                continue

            row[key] = float(value)

        self.history.append(row)

    def on_fit_end(self, trainer, pl_module):

        if not trainer.is_global_zero:
            return

        path = self.destination(trainer) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(trainer), indent=2))

        print(f"\n[training report] {path}")
        self.upload(trainer, path)

    def destination(self, trainer) -> Path:
        """Beside the run's checkpoints, so the report and the weights it
        describes stay together. Falls back to the working directory when
        checkpointing is off."""

        dirpath = getattr(trainer.checkpoint_callback, "dirpath", None)

        return Path(dirpath) if dirpath else Path.cwd()

    def upload(self, trainer, path: Path) -> bool:
        """Copy the report into the W&B run's Files tab.

        ``policy="now"`` uploads immediately rather than at the end of the run,
        which has already happened by the time this runs. ``base_path`` pins it
        to the run root so it lands as ``training_report.json`` rather than
        nested under the checkpoint directory's full path.

        Silently does nothing without a W&B logger — ``logger=False``, a bare
        ``CSVLogger``, and the test suite all have to keep working.
        """

        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "save"):
            return False

        experiment.save(str(path), base_path=str(path.parent), policy="now")
        print(f"[training report] uploaded to W&B as {path.name}")

        return True

    def build(self, trainer) -> dict:
        """Assemble the report. Pure — safe to call in a test."""

        rows = self.event_rows(trainer)
        summary = summarize(rows) if rows else {}

        report = {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run": self.run_info(trainer),
            "readable": summary_block(summary)
            if rows
            else "no event metrics were scored",
            "event_metrics": {
                "n_tracks": len(rows),
                "scored_epochs": sorted(self.event_epochs(trainer)),
                "summary": summary,
                "per_corpus": self.per_corpus(rows),
                "per_track": rows,
            },
            "frame_metrics": {
                "final": self.final_metrics(trainer),
                "best": self.best_metrics(trainer),
            },
            "history": self.history,
            "config": OmegaConf.to_container(self.cfg, resolve=True)
            if self.cfg is not None
            else None,
        }

        return jsonable(report)

    def event_rows(self, trainer) -> list[dict]:
        logger = find_callback(trainer, EventMetricsLogger)

        return list(logger.last_rows) if logger else []

    @staticmethod
    def per_corpus(rows: list[dict]) -> dict:
        """Per-corpus means of the canonical metric set.

        The micro mean over a merged split is weighted by however many tracks
        each corpus happens to contribute, so a corpus the model is useless on
        can barely register. This is where that shows.
        """

        return {
            corpus: {
                "n_tracks": len(group),
                **{key: summarize(group)[key] for key in SCORE_KEYS},
            }
            for corpus, group in group_by_corpus(rows).items()
        }

    def final_metrics(self, trainer) -> dict:
        return dict(self.history[-1]) if self.history else {}

    def best_metrics(self, trainer) -> dict:
        printer = find_callback(trainer, BestMetricsPrinter)

        return dict(printer.best) if printer else {}

    def run_info(self, trainer) -> dict:
        experiment = getattr(trainer.logger, "experiment", None)
        checkpoint = trainer.checkpoint_callback
        best_score = getattr(checkpoint, "best_model_score", None)

        return {
            "wandb_id": getattr(experiment, "id", None),
            "wandb_name": getattr(experiment, "name", None),
            "wandb_url": getattr(experiment, "url", None),
            # "offline" means the run is sitting in ./wandb/ unsynced, which is
            # worth knowing before going to look for it in the dashboard.
            "wandb_mode": getattr(getattr(experiment, "settings", None), "mode", None),
            "git_commit": git_commit(),
            # No +1: Lightning advances `current_epoch` past the last completed
            # epoch before `on_fit_end`, so it is already a count rather than
            # the 0-based index it is inside the loop.
            "epochs_run": trainer.current_epoch,
            "max_epochs": trainer.max_epochs,
            "best_checkpoint": getattr(checkpoint, "best_model_path", None) or None,
            "best_checkpoint_score": float(best_score)
            if best_score is not None
            else None,
        }
