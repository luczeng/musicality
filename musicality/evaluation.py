"""Evaluation engine for beat-only/beat-phase checkpoints on full-length
tracks — not the fixed-duration clips used during training.

:func:`score_events` is the single place every reported metric is computed, and
:meth:`BeatEvaluator.score` is the single path from a checkpoint to a table of
numbers. Both existed in three hand-copied variants before
``plans/06_metric_calibration_and_eval_consolidation.md``; the copies had
already drifted apart, which is the failure mode this module exists to prevent.

The canonical metric set is :data:`SCORE_KEYS`, in report order. Read it as two
pairs plus their leftovers:

- ``f_beat`` says whether the beats landed; ``cmlt``/``amlt`` say whether they
  landed at the *right metrical level*, and ``amlt - cmlt`` is the share of the
  track tracked confidently at the wrong one.
- ``position_acc`` says whether the bar numbering is right;
  ``position_acc_best_offset`` says whether it is at least *consistent*, and
  ``anchor_error`` is the difference — a bar grid correct in every respect
  except where it starts counting.
- ``f_one``/``f_last`` and ``confusion`` are narrower views kept for continuity
  with everything recorded in ``plans/04`` and ``plans/05``. See
  :func:`~musicality.metrics.f_measure.downbeat_f_measures` and
  :func:`~musicality.metrics.confusion.confusion_half_cycle_rate` for what each
  is structurally unable to see.
"""

import math
from pathlib import Path

import numpy as np
import torch
import yaml

import musicality.dataformats as dataformats
from musicality.inference import load_module, load_track_waveform
from musicality.loaders.beat_dataset import (
    BeatDataset,
    beat_split_name,
    indices_for_split,
)
from musicality.metrics.confusion import confusion_half_cycle_rate
from musicality.metrics.continuity import beat_continuity
from musicality.metrics.f_measure import beat_f_measure, downbeat_f_measures
from musicality.metrics.position_accuracy import position_accuracy
from musicality.postprocess import readout, readout_beat_only
from musicality.splits.splitter import Splitter

DATA_DIR = dataformats.ROOT / dataformats.load().data_dir

# Default CLI/postprocessing values — see configs/eval_beat.yaml for what each means.
DEFAULTS = yaml.safe_load((dataformats.ROOT / "configs" / "eval_beat.yaml").read_text())

# The canonical metric set, in report order. Every one is higher-is-better
# except `confusion`. `modal_offset` is deliberately absent: it is categorical,
# so averaging it is meaningless.
SCORE_KEYS = (
    "f_beat",
    "cmlt",
    "amlt",
    "position_acc",
    "position_acc_best_offset",
    "anchor_error",
    "f_one",
    "f_last",
    "confusion",
)

# Postprocessing knobs resolved by :meth:`BeatEvaluator.resolve_postprocess`,
# with the fallback used when the checkpoint's task has no tuned value.
_KNOB_FALLBACKS = {
    "beat_threshold": 0.3,
    "min_distance_frames": 1,
    "gate_tolerance": 0.2,
    "anchor_threshold": 0.5,
    "group_size": 4,
    "decoder": "greedy",
    "switch_penalty": None,
}


def _mean(values: list) -> float:
    """Mean over the non-``None``, non-NaN entries; NaN when there are none."""

    clean = [v for v in values if v is not None and not math.isnan(v)]

    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None or math.isnan(value) else f"{value:.3f}"


def score_events(
    beat_times: np.ndarray,
    positions: np.ndarray | None,
    has_positions: bool,
    events: list[dict],
    *,
    tolerance: float = 0.07,
    trim: bool = True,
    group_size: int = 4,
) -> dict:
    """Score one decoded track against its reference. The single scorer.

    Always returns every key in :data:`SCORE_KEYS` plus ``modal_offset``, so
    callers can build a uniform table without checking which task produced the
    row. Keys that cannot be measured are ``None`` rather than ``0.0`` — a
    beat-only checkpoint has no bar positions to get wrong, and scoring that as
    zero would drag every aggregate down.

    :param beat_times: Reference beat times, in seconds.
    :param positions: Reference bar positions per beat, or ``None``.
    :param has_positions: Whether *positions* is real supervision. ``False``
        for a track whose meter this ``group_size`` cannot represent (see
        :func:`~musicality.loaders.beat_dataset.fold_positions`).
    :param events: Decoded beat list from
        :func:`~musicality.postprocess.readout` — ``{"time", "beat_in_bar"}``
        dicts. Beat-only output is passed in the same shape with every
        ``beat_in_bar`` set to ``None``, which is what makes one scorer serve
        both tasks.
    :param tolerance: Matching window, in seconds.
    :param trim: Drop events before 5s, ``mir_eval``'s warm-up convention.
        Meaningful on full tracks; pass ``False`` for short clips.
    :param group_size: Beats per group.
    :returns: One row of the canonical metric set.
    """

    row = {key: None for key in SCORE_KEYS}
    row["modal_offset"] = None

    est_times = np.array([e["time"] for e in events], dtype=float)

    row["f_beat"] = beat_f_measure(
        beat_times, est_times, tolerance=tolerance, trim=trim
    )

    continuity = beat_continuity(beat_times, est_times, trim=trim)
    if continuity is not None:
        row["cmlt"] = continuity["cmlt"]
        row["amlt"] = continuity["amlt"]

    # Bar-position metrics need reference positions *and* predicted labels.
    # A beat-only readout supplies neither, and a beat-phase decode that
    # resolved nothing supplies only the former.
    labelled = any(e.get("beat_in_bar") is not None for e in events)
    if not (has_positions and positions is not None and labelled):
        return row

    positions = np.asarray(positions)

    row["f_one"], row["f_last"] = downbeat_f_measures(
        beat_times,
        positions,
        events,
        tolerance=tolerance,
        trim=trim,
        group_size=group_size,
    )
    row["confusion"] = confusion_half_cycle_rate(
        beat_times, positions, events, tolerance=tolerance, group_size=group_size
    )

    profile = position_accuracy(
        beat_times, positions, events, tolerance=tolerance, group_size=group_size
    )
    if profile is not None:
        row["position_acc"] = profile["position_acc"]
        row["position_acc_best_offset"] = profile["position_acc_best_offset"]
        row["anchor_error"] = profile["anchor_error"]
        row["modal_offset"] = profile["modal_offset"]

    return row


def group_by_corpus(rows: list[dict]) -> dict[str, list[dict]]:
    """Group scored rows by their source corpus, preserving first-seen order."""

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("corpus", ""), []).append(row)

    return grouped


def summarize(rows: list[dict]) -> dict:
    """Aggregate scored rows into micro *and* macro means.

    - **micro** (the bare :data:`SCORE_KEYS`) averages over *tracks* — the
      historical behaviour.
    - **macro** (the ``macro_*`` keys) averages within each corpus first, then
      averages those means, so every corpus counts once regardless of size.

    They are identical when a single corpus is present, or when every corpus
    contributes the same number of tracks. Where they diverge, micro is being
    carried by whichever corpus happened to contribute the most tracks — an
    accident of what is downloaded, not of anything we care about. For a tool
    meant to work across genres, macro is the number to steer by.

    Also reports the weakest corpus, which even the macro mean averages away
    and which is the real binding constraint on "works everywhere":
    ``worst_position_acc`` (and the corpus that scored it) plus
    ``worst_confusion``, kept because every comparison in ``plans/04`` and
    ``plans/05`` is in those units.
    """

    summary = {"n_tracks": len(rows)}
    for key in SCORE_KEYS:
        summary[key] = _mean([r.get(key) for r in rows])

    per_corpus = group_by_corpus(rows)
    for key in SCORE_KEYS:
        summary[f"macro_{key}"] = _mean(
            [_mean([r.get(key) for r in rs]) for rs in per_corpus.values()]
        )

    def _worst(key: str, pick):
        scored = {
            corpus: _mean([r.get(key) for r in rs]) for corpus, rs in per_corpus.items()
        }
        scored = {c: v for c, v in scored.items() if not math.isnan(v)}

        return (
            pick(scored.items(), key=lambda kv: kv[1])
            if scored
            else (None, float("nan"))
        )

    summary["worst_corpus"], summary["worst_position_acc"] = _worst("position_acc", min)
    _, summary["worst_confusion"] = _worst("confusion", max)

    return summary


class BeatEvaluator:
    """Evaluates a beat-only or beat-phase checkpoint (task auto-detected from
    the checkpoint itself) on full-length tracks.

    Postprocessing knobs left as ``None`` fall back to the tuned defaults for
    the checkpoint's detected task (``configs/eval_beat.yaml``).

    :meth:`score` is the entry point. It runs the model once per track and
    caches the frame probabilities, so scoring several decoder configurations —
    or sweeping postprocessing thresholds — costs one model pass in total
    rather than one per configuration.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        dataset: str,
        data_home: str | Path | None = None,
        split: str = "val",
        val_split: float = 0.2,
        sample_rate: int = 22050,
        hop_length: int = 512,
        group_size: int | None = None,
        binary_only: bool = False,
        tolerance: float = 0.07,
        trim: bool = True,
        beat_threshold: float | None = None,
        min_distance_frames: int | None = None,
        gate_tolerance: float | None = None,
        anchor_threshold: float | None = None,
        decoder: str | None = None,
        switch_penalty: float | None = None,
        limit: int | None = None,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.checkpoint = checkpoint
        self.dataset_name = dataset
        self.data_home = Path(data_home) if data_home else DATA_DIR / dataset
        self.split = split
        self.val_split = val_split
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.group_size = group_size
        self.binary_only = binary_only
        self.tolerance = tolerance
        self.trim = trim
        self.beat_threshold = beat_threshold
        self.min_distance_frames = min_distance_frames
        self.gate_tolerance = gate_tolerance
        self.anchor_threshold = anchor_threshold
        self.decoder = decoder
        self.switch_penalty = switch_penalty
        self.limit = limit
        self.device = device
        self.verbose = verbose
        self._loaded = None
        self._probs = None

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop_length

    def load(self) -> tuple:
        """Load the checkpoint (auto-detecting task) and build the dataset +
        selected split indices. Memoized — safe to call more than once.

        Two ways of building the dataset, chosen by whether ``data_home`` is a
        real directory:

        - **A single dataset** (the usual case): built by ``name`` from its own
          directory, then narrowed to *split* via
          :func:`~musicality.loaders.beat_dataset.indices_for_split`.
        - **A merged split** (e.g. ``dataset="merge"``): ``tools/merge_datasets.py``
          deliberately never creates a merged dataset directory, so there is
          nothing to build by name. The split's ``TrackRef`` entries carry their
          own per-track ``data_home`` and source corpus, so the dataset is built
          straight from them. This is also what makes :meth:`track_corpora`
          meaningful — a merged split is the only case where more than one
          corpus is present.

        :returns: ``(module, task, dataset, indices)``.
        """

        if self._loaded is None:
            module, task = load_module(self.checkpoint, self.device)

            if self.split != "all" and not self.data_home.is_dir():
                splits_dir = dataformats.ROOT / dataformats.load().splits_dir
                split_name = beat_split_name(self.dataset_name, self.binary_only)
                train_refs, val_refs = Splitter.load_refs(splits_dir, split_name)

                dataset = BeatDataset(
                    refs=train_refs if self.split == "train" else val_refs,
                    sample_rate=self.sample_rate,
                    hop_length=self.hop_length,
                    group_size=self.group_size if self.group_size is not None else 4,
                    binary_only=self.binary_only,
                )
                indices = list(range(len(dataset)))
            else:
                dataset = BeatDataset(
                    name=self.dataset_name,
                    data_home=self.data_home,
                    sample_rate=self.sample_rate,
                    hop_length=self.hop_length,
                    group_size=self.group_size if self.group_size is not None else 4,
                    binary_only=self.binary_only,
                )
                indices = indices_for_split(
                    dataset,
                    self.dataset_name,
                    self.split,
                    self.val_split,
                    self.binary_only,
                )

            if self.limit is not None:
                indices = indices[: self.limit]
            self._loaded = (module, task, dataset, indices)

        return self._loaded

    def resolve_postprocess(self, **overrides) -> dict:
        """Resolve every postprocessing knob: explicit *override* beats the
        constructor value, which beats the tuned default for the detected task.

        Overrides are keyed by **presence**, not by value, because ``None`` is
        itself a meaningful ``switch_penalty`` (the exact single-offset decode,
        no mid-track resync allowed). A caller that wants it passes
        ``switch_penalty=None`` explicitly; one that wants the configured value
        omits the key.

        :returns: A dict over :data:`_KNOB_FALLBACKS`' keys.
        """

        _module, task, _dataset, _indices = self.load()
        task_defaults = DEFAULTS[task]

        resolved = {}
        for key, fallback in _KNOB_FALLBACKS.items():
            if key in overrides:
                resolved[key] = overrides[key]
            elif getattr(self, key, None) is not None:
                resolved[key] = getattr(self, key)
            else:
                resolved[key] = task_defaults.get(key, fallback)

        return resolved

    def track_corpora(self) -> list[str]:
        """Source corpus name per selected track, positionally aligned with
        :meth:`compute_track_probs`.

        ``BeatDataset`` appends to ``samples`` and ``refs`` in the same loop
        iteration, and :meth:`compute_track_probs` walks ``indices`` in order
        emitting exactly one entry per index, so the two lists line up
        element-for-element.

        Every entry is the same string for a single-dataset split; a merged
        split is where this becomes useful, as it lets a caller report per
        genre rather than one blended number whose weighting is an accident of
        how many tracks each corpus happens to contribute.
        """

        _module, _task, dataset, indices = self.load()

        return [dataset.refs[i].dataset_name for i in indices]

    @torch.no_grad()
    def compute_track_probs(self) -> list[tuple]:
        """Run the model once per track, returning cached
        ``(beat_times, positions, has_positions, probs)`` tuples — the raw
        per-track model output, activated and unsliced (shape ``(T,)`` for
        beat-only, ``(3, T)`` for the one/last beat-phase head, or
        ``(1 + group_size, T)`` for the softmax bar-position head), plus the
        reference annotations needed to score it.

        **Memoized**, so re-scoring under many decoder or postprocessing
        settings costs one model pass in total. This is what makes
        :meth:`score` cheap to call in a loop.

        Mirrors :func:`~musicality.inference.run_inference`'s activation
        choice: a softmax bar-position checkpoint (``group_size`` set in its
        hyperparameters) gets ``sigmoid`` on the ``beat`` channel and
        ``softmax`` over the position channels, not a blanket ``sigmoid`` —
        the position logits compete against each other, not independently.
        """

        if self._probs is not None:
            return self._probs

        module, _task, dataset, indices = self.load()
        module_group_size = getattr(module, "hparams", {}).get("group_size")

        cached = []
        for i in indices:
            audio_path, beat_times, positions, has_positions = dataset.samples[i]

            wav = (
                load_track_waveform(audio_path, self.sample_rate)
                .unsqueeze(0)
                .to(self.device)
            )
            logits = module(wav)[0]  # (T',) beat-only, or (n_outputs, T') beat-phase

            if module_group_size is not None:
                beat_probs = torch.sigmoid(logits[0]).cpu().numpy()
                position_probs = torch.softmax(logits[1:], dim=0).cpu().numpy()
                probs = np.concatenate([beat_probs[None], position_probs], axis=0)
            else:
                probs = torch.sigmoid(logits).cpu().numpy()  # (T',) or (3, T')

            cached.append((beat_times, positions, has_positions, probs))

        self._probs = cached

        return cached

    def decode(
        self, probs: np.ndarray, knobs: dict, advance: str = "index"
    ) -> list[dict]:
        """Turn one track's cached frame probabilities into a labelled event list.

        Beat-only output is wrapped as ``{"time", "beat_in_bar": None}`` so
        that :func:`score_events` sees one shape for both tasks.
        """

        module, task, _dataset, _indices = self.load()

        if task == "beat_only":
            times = readout_beat_only(
                probs,
                fps=self.fps,
                beat_threshold=knobs["beat_threshold"],
                min_distance_frames=knobs["min_distance_frames"],
                gate_tolerance=knobs["gate_tolerance"],
            )

            return [{"time": float(t), "beat_in_bar": None} for t in times]

        # A softmax head emits `beat` plus a group_size-way distribution; the
        # older head emits three independent sigmoids. one/last are kept for
        # the greedy decoder, which still speaks that language.
        #
        # The checkpoint's own `group_size` hyperparameter is what tells the
        # two apart, not the channel count: at group_size=2 a softmax head is
        # also (3, T), so any shape heuristic silently misreads it as one/last.
        if getattr(module, "hparams", {}).get("group_size") is not None:
            beat_p, position_probs = probs[0], probs[1:]
            one_p, last_p = position_probs[0], position_probs[-1]
        else:
            beat_p, one_p, last_p = probs[0], probs[1], probs[2]
            position_probs = None

        return readout(
            beat_p,
            one_p,
            last_p,
            fps=self.fps,
            beat_threshold=knobs["beat_threshold"],
            min_distance_frames=knobs["min_distance_frames"],
            gate_tolerance=knobs["gate_tolerance"],
            anchor_threshold=knobs["anchor_threshold"],
            group_size=knobs["group_size"],
            decoder=knobs["decoder"],
            switch_penalty=knobs["switch_penalty"],
            advance=advance,
            position_probs=position_probs,
        )

    def score(self, *, advance: str = "index", **overrides) -> list[dict]:
        """Decode and score every selected track. One row per track.

        *overrides* are postprocessing knobs (see :meth:`resolve_postprocess`);
        anything omitted falls back to the constructor value and then to the
        task default. Because :meth:`compute_track_probs` is memoized, calling
        this repeatedly with different knobs re-runs only the decoder.

        :returns: One dict per track — ``corpus`` plus :data:`SCORE_KEYS` plus
            ``modal_offset``.
        """

        knobs = self.resolve_postprocess(**overrides)
        cached = self.compute_track_probs()
        corpora = self.track_corpora()

        rows = []
        for (beat_times, positions, has_positions, probs), corpus in zip(
            cached, corpora
        ):
            events = self.decode(probs, knobs, advance=advance)

            row = score_events(
                beat_times,
                positions,
                has_positions,
                events,
                tolerance=self.tolerance,
                trim=self.trim,
                group_size=knobs["group_size"],
            )
            rows.append({"corpus": corpus, **row})

        return rows

    def run(self) -> list[dict]:
        """Score the selected split under the configured settings, printing a
        per-track line and a summary block when ``verbose``."""

        _module, task, dataset, indices = self.load()

        if self.verbose:
            print(
                f"[eval] {len(indices)} track(s) from '{self.dataset_name}' "
                f"(split={self.split}, task={task})"
            )

        rows = self.score()

        if self.verbose:
            for i, row in zip(indices, rows):
                label = Path(dataset.samples[i][0]).stem
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


def summary_block(summary: dict) -> str:
    """Render :func:`summarize`'s output as the canonical report block."""

    lines = [
        f"n_tracks={summary['n_tracks']}",
        "BEAT",
        f"  f_beat                    {_fmt(summary['f_beat'])}",
        f"  cmlt / amlt               {_fmt(summary['cmlt'])} / {_fmt(summary['amlt'])}",
        "POSITION",
        f"  position_acc              {_fmt(summary['position_acc'])}",
        f"  position_acc_best_offset  {_fmt(summary['position_acc_best_offset'])}",
        f"    -> anchor_error         {_fmt(summary['anchor_error'])}",
        f"  f_one / f_last            {_fmt(summary['f_one'])} / {_fmt(summary['f_last'])}",
        f"  confusion                 {_fmt(summary['confusion'])}",
    ]

    if summary.get("worst_corpus"):
        lines.append(
            f"WORST CORPUS  {summary['worst_corpus']} "
            f"(position_acc {_fmt(summary['worst_position_acc'])})"
        )

    return "\n".join(lines)
