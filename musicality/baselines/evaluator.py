"""Scores a third-party tracker on our splits, through our scorer.

The counterpart to :class:`~musicality.evaluation.BeatEvaluator`, with the
model pass replaced by a :class:`~musicality.baselines.base.Baseline` call and
a disk cache. Everything after the prediction is shared code:
:func:`~musicality.evaluation.build_eval_dataset` chooses the tracks,
:func:`~musicality.baselines.base.events_from_beats` builds the event list, and
:func:`~musicality.evaluation.score_events` scores it. A baseline row and a
checkpoint row in the same table therefore differ in exactly one thing — who
produced the beats.
"""

import time
from pathlib import Path

from musicality.baselines.base import CORPUS_EXPOSURE, SEEN, events_from_beats
from musicality.baselines.cache import PredictionCache, track_key
from musicality.evaluation import (
    _fmt,
    build_eval_dataset,
    score_events,
    summarize,
    summary_block,
)


#: Default home for prediction caches. Under ``outputs/``, which is git-ignored.
CACHE_DIR = Path("outputs/baselines")


def default_cache_path(
    baseline: str,
    dataset: str,
    split: str,
    binary_only: bool,
    cache_dir: str | Path = CACHE_DIR,
) -> Path:
    """Where a baseline's predictions land when ``--cache`` isn't given.

    Named after everything that changes *which tracks were predicted*, so two
    runs cannot overwrite each other's work. What changes the *prediction* —
    the checkpoint, the DBN setting — is guarded by the cache header instead
    (:meth:`~musicality.baselines.cache.PredictionCache.open`), because putting
    it in the filename would mean a new file per knob and no reuse at all.
    """

    suffix = "-binary" if binary_only else ""

    return Path(cache_dir) / f"{baseline}-{dataset}-{split}{suffix}.json"


class BaselineEvaluator:
    """Runs a baseline over a split and scores it.

    :param baseline: A constructed :class:`~musicality.baselines.base.Baseline`.
    :param dataset: Dataset or merged-split name, as for
        ``tools/eval_beat.py``.
    :param data_home: Dataset directory; defaults to ``DATA_DIR/<dataset>``.
    :param split: ``"train"``, ``"val"`` or ``"all"``.
    :param val_split: Held-out fraction — must match how the split was created.
    :param sample_rate: Passed through to
        :class:`~musicality.loaders.beat_dataset.BeatDataset`. It does **not**
        reach the tracker: each baseline loads and resamples the audio itself,
        at whatever rate its weights expect.
    :param hop_length: Likewise — frame geometry for the dataset only.
    :param group_size: Beats per bar, for both the reference folding and the
        estimated bar positions.
    :param binary_only: Must match how the split was created.
    :param tolerance: Event matching window, in seconds.
    :param trim: Apply ``mir_eval``'s 5 s warm-up trim.
    :param limit: Score only the first N selected tracks.
    :param cache_path: Prediction cache location; ``None`` uses
        :func:`default_cache_path`. Pass ``False`` to disable caching entirely.
    :param refresh: Re-run the tracker even for tracks already cached.
    :param cache_every: Flush the cache to disk every N newly predicted tracks,
        not only at the end. A full run is tens of minutes of tracker time and
        an interrupted one used to throw all of it away.
    :param verbose: Print progress and the summary block.
    """

    def __init__(
        self,
        baseline,
        dataset: str,
        data_home: str | Path | None = None,
        split: str = "val",
        val_split: float = 0.2,
        sample_rate: int = 22050,
        hop_length: int = 512,
        group_size: int = 4,
        binary_only: bool = False,
        tolerance: float = 0.07,
        trim: bool = True,
        limit: int | None = None,
        cache_path: str | Path | None | bool = None,
        refresh: bool = False,
        cache_every: int = 20,
        verbose: bool = True,
    ):
        self.baseline = baseline
        self.dataset_name = dataset
        self.data_home = Path(data_home) if data_home else None
        self.split = split
        self.val_split = val_split
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.group_size = group_size
        self.binary_only = binary_only
        self.tolerance = tolerance
        self.trim = trim
        self.limit = limit
        self.refresh = refresh
        self.cache_every = max(1, cache_every)
        self.verbose = verbose

        self.cache_path = (
            None
            if cache_path is False
            else Path(cache_path)
            if cache_path is not None
            else default_cache_path(baseline.name, dataset, split, binary_only)
        )

        self._loaded = None
        self._predictions = None

    def load(self) -> tuple:
        """Build the dataset and resolve the split. Memoized.

        :returns: ``(dataset, indices)``.
        """

        if self._loaded is None:
            self._loaded = build_eval_dataset(
                self.dataset_name,
                data_home=self.data_home,
                split=self.split,
                val_split=self.val_split,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                group_size=self.group_size,
                binary_only=self.binary_only,
                limit=self.limit,
            )

        return self._loaded

    def track_corpora(self) -> list[str]:
        """Source corpus per selected track, aligned with :meth:`predict_all`."""

        dataset, indices = self.load()

        return [dataset.refs[i].dataset_name for i in indices]

    def exposure_note(self) -> str | None:
        """A warning naming the selected corpora this baseline was trained on,
        or ``None`` when every corpus is clean or unknown.

        Printed before the numbers rather than after, because the numbers are
        not interpretable without it: on a corpus a baseline was fitted on, its
        score is a memorisation ceiling and our model's is a generalisation
        measurement, and averaging the two into one "baseline beats us by N"
        headline is the single easiest way to draw the wrong conclusion from
        this tool.
        """

        exposure = CORPUS_EXPOSURE.get(self.baseline.name, {})
        seen = sorted(
            {
                corpus
                for corpus in set(self.track_corpora())
                if exposure.get(corpus) == SEEN
            }
        )

        if not seen:
            return None

        return (
            f"[warning] {self.baseline.name} was trained on: {', '.join(seen)}. "
            "Its scores on those corpora are recall, not generalisation — read "
            "the per-genre table, not the mean."
        )

    def predict_all(self) -> list[tuple]:
        """Predict every selected track, reading and updating the cache.

        Memoized in memory as well as on disk, so re-scoring under a different
        ``group_size`` or tolerance costs nothing.

        :returns: One ``(beats, downbeats)`` tuple per selected track, in
            dataset order.
        """

        if self._predictions is not None:
            return self._predictions

        dataset, indices = self.load()

        cache = (
            None
            if self.cache_path is None
            else PredictionCache.open(
                self.cache_path, self.baseline.name, self.baseline.config
            )
        )

        if cache is not None and cache.n_loaded and self.verbose:
            print(f"[baseline] {cache.n_loaded} cached prediction(s) in {cache.path}")

        predictions = []
        n_run = 0

        for position, index in enumerate(indices, start=1):
            ref = dataset.refs[index]
            audio_path = dataset.samples[index][0]
            key = track_key(ref.dataset_name, ref.track_id)

            cached = None if cache is None or self.refresh else cache.get(key)

            if cached is not None:
                predictions.append(cached)
                continue

            started = time.perf_counter()
            beats, downbeats = self.baseline.predict(audio_path)
            elapsed = time.perf_counter() - started
            n_run += 1

            if cache is not None:
                cache.put(key, beats, downbeats)

            if self.verbose:
                # Flushed: a long run is usually redirected to a file, and a
                # progress line that only appears once the run is over is not
                # a progress line.
                print(
                    f"[baseline] {position}/{len(indices)} {key} "
                    f"-> {len(beats)} beat(s) in {elapsed:.1f}s",
                    flush=True,
                )

            predictions.append((beats, downbeats))

            if cache is not None and n_run % self.cache_every == 0:
                cache.save()

        if cache is not None and n_run:
            cache.save()
            if self.verbose:
                print(f"[baseline] cache written to {cache.path}")

        self._predictions = predictions

        return predictions

    def score(self) -> list[dict]:
        """Score every selected track. One row per track, in the same shape
        :meth:`musicality.evaluation.BeatEvaluator.score` returns.
        """

        dataset, indices = self.load()
        predictions = self.predict_all()
        corpora = self.track_corpora()

        rows = []
        for index, (beats, downbeats), corpus in zip(indices, predictions, corpora):
            _path, beat_times, positions, has_positions = dataset.samples[index]

            events = events_from_beats(beats, downbeats, group_size=self.group_size)

            row = score_events(
                beat_times,
                positions,
                has_positions,
                events,
                tolerance=self.tolerance,
                trim=self.trim,
                group_size=self.group_size,
            )
            rows.append({"corpus": corpus, **row})

        return rows

    def run(self) -> list[dict]:
        """Score the split, printing per-track lines and the summary block when
        ``verbose`` — deliberately the same report
        :meth:`musicality.evaluation.BeatEvaluator.run` prints, so the two can
        be read side by side.
        """

        dataset, indices = self.load()

        if self.verbose:
            print(
                f"[baseline] {self.baseline.describe()} on {len(indices)} track(s) "
                f"from '{self.dataset_name}' (split={self.split})"
            )

            note = self.exposure_note()
            if note:
                print(note)

        rows = self.score()

        if self.verbose:
            for index, row in zip(indices, rows):
                label = Path(dataset.samples[index][0]).stem
                line = (
                    f"[{label:30s}] beat={_fmt(row['f_beat'])} cmlt={_fmt(row['cmlt'])}"
                )
                if row["position_acc"] is not None:
                    line += (
                        f"  pos={_fmt(row['position_acc'])} "
                        f"best={_fmt(row['position_acc_best_offset'])}"
                    )
                print(line)

            print("-" * 70)
            print(summary_block(summarize(rows)))

        return rows
