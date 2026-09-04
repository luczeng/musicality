"""Evaluation engine for beat-only/beat-phase checkpoints on full-length
tracks — not the fixed-duration clips used during training.

:class:`BeatEvaluator` is the library-side counterpart to ``tools/eval_beat.py``,
which is a thin CLI wrapper around it.
"""

from pathlib import Path

import numpy as np
import torch
import yaml

import musicality.dataformats as dataformats
from musicality.inference import load_module, load_track_waveform, run_inference
from musicality.loaders.beat_dataset import BeatDataset, indices_for_split
from musicality.metrics.confusion import confusion_half_cycle_rate
from musicality.metrics.f_measure import beat_f_measure, downbeat_f_measures

DATA_DIR = dataformats.ROOT / dataformats.load().data_dir

# Default CLI/postprocessing values — see configs/eval_beat.yaml for what each means.
DEFAULTS = yaml.safe_load((dataformats.ROOT / "configs" / "eval_beat.yaml").read_text())


def evaluate_track(
    module,
    task: str,
    audio_path: str,
    beat_times: np.ndarray,
    positions: np.ndarray | None,
    has_positions: bool,
    sample_rate: int,
    hop_length: int,
    device: str,
    tolerance: float,
    trim: bool,
    beat_threshold: float,
    min_distance_frames: int,
    gate_tolerance: float,
    anchor_threshold: float,
    group_size: int,
    decoder: str = "greedy",
    switch_penalty: float | None = None,
) -> dict:
    """Run the model on one full track and score its readout against the
    reference. Always returns the same four keys — ``f_one``/``f_last``/
    ``confusion`` stay ``None`` for a beat-only task (no bar-position heads to
    score) or when the track has no position annotations."""

    wav = load_track_waveform(audio_path, sample_rate)
    fps = sample_rate / hop_length

    result = run_inference(
        module,
        task,
        wav,
        fps,
        device,
        beat_threshold=beat_threshold,
        min_distance_frames=min_distance_frames,
        gate_tolerance=gate_tolerance,
        anchor_threshold=anchor_threshold,
        group_size=group_size,
        decoder=decoder,
        switch_penalty=switch_penalty,
    )

    events = result if task == "beat_phase" else None
    pred_times = [e["time"] for e in events] if events is not None else result

    out = {
        "f_beat": beat_f_measure(
            beat_times, pred_times, tolerance=tolerance, trim=trim
        ),
        "f_one": None,
        "f_last": None,
        "confusion": None,
    }

    if task == "beat_phase" and has_positions:
        positions = np.asarray(positions)
        f_one, f_last = downbeat_f_measures(
            beat_times,
            positions,
            events,
            tolerance=tolerance,
            trim=trim,
            group_size=group_size,
        )
        out["f_one"] = f_one
        out["f_last"] = f_last
        out["confusion"] = confusion_half_cycle_rate(
            beat_times, positions, events, tolerance=tolerance, group_size=group_size
        )

    return out


def _mean(results: list[dict], key: str) -> float | None:
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "  n/a"


class BeatEvaluator:
    """Evaluates a beat-only or beat-phase checkpoint (task auto-detected
    from the checkpoint itself) on full-length tracks: beat F-measure, plus
    "1"/"last" F-measure and phase-confusion rate for beat-phase checkpoints.

    Postprocessing knobs left as ``None`` fall back to the tuned defaults for
    the checkpoint's detected task (``configs/eval_beat.yaml``).
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

    def load(self) -> tuple:
        """Load the checkpoint (auto-detecting task) and build the dataset +
        selected split indices. Memoized — safe to call more than once.

        :returns: ``(module, task, dataset, indices)``.
        """

        if self._loaded is None:
            module, task = load_module(self.checkpoint, self.device)
            dataset = BeatDataset(
                name=self.dataset_name,
                data_home=self.data_home,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                group_size=self.group_size if self.group_size is not None else 4,
                binary_only=self.binary_only,
            )
            indices = indices_for_split(
                dataset, self.dataset_name, self.split, self.val_split, self.binary_only
            )
            if self.limit is not None:
                indices = indices[: self.limit]
            self._loaded = (module, task, dataset, indices)

        return self._loaded

    @torch.no_grad()
    def compute_track_probs(self) -> list[tuple]:
        """Run the model once per track, returning cached
        ``(beat_times, positions, has_positions, probs)`` tuples — the raw
        per-track model output, activated and unsliced (shape ``(T,)`` for
        beat-only, ``(3, T)`` for the one/last beat-phase head, or
        ``(1 + group_size, T)`` for the softmax bar-position head), plus the
        reference annotations needed to score it. Lets a caller (e.g. a
        postprocessing hyperparameter sweep) re-score many parameter
        combinations cheaply without re-running the model.

        Mirrors :func:`~musicality.inference.run_inference`'s activation
        choice: a softmax bar-position checkpoint (``group_size`` set in its
        hyperparameters) gets ``sigmoid`` on the ``beat`` channel and
        ``softmax`` over the position channels, not a blanket ``sigmoid`` —
        the position logits compete against each other, not independently.
        """

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

        return cached

    def run(self) -> list[dict]:
        """Evaluate every track in the selected split and return one result
        dict per track (see :func:`evaluate_track`)."""

        module, task, dataset, indices = self.load()
        task_defaults = DEFAULTS[task]

        group_size = (
            self.group_size
            if self.group_size is not None
            else task_defaults.get("group_size", 4)
        )
        beat_threshold = (
            self.beat_threshold
            if self.beat_threshold is not None
            else task_defaults["beat_threshold"]
        )
        min_distance_frames = (
            self.min_distance_frames
            if self.min_distance_frames is not None
            else task_defaults["min_distance_frames"]
        )
        gate_tolerance = (
            self.gate_tolerance
            if self.gate_tolerance is not None
            else task_defaults["gate_tolerance"]
        )
        anchor_threshold = (
            self.anchor_threshold
            if self.anchor_threshold is not None
            else task_defaults.get("anchor_threshold", 0.5)
        )
        decoder = (
            self.decoder
            if self.decoder is not None
            else task_defaults.get("decoder", "greedy")
        )
        # `None` is itself a meaningful switch_penalty (the exact, no-resync
        # decode), so the config value is what selects it — not a sentinel.
        switch_penalty = (
            self.switch_penalty
            if self.switch_penalty is not None
            else task_defaults.get("switch_penalty")
        )

        if self.verbose:
            print(
                f"[eval] {len(indices)} track(s) from '{self.dataset_name}' "
                f"(split={self.split}, task={task})"
            )

        results = []
        for i in indices:
            audio_path, beat_times, positions, has_positions = dataset.samples[i]

            r = evaluate_track(
                module,
                task,
                audio_path,
                beat_times,
                positions,
                has_positions,
                self.sample_rate,
                self.hop_length,
                self.device,
                self.tolerance,
                self.trim,
                beat_threshold,
                min_distance_frames,
                gate_tolerance,
                anchor_threshold,
                group_size,
                decoder=decoder,
                switch_penalty=switch_penalty,
            )
            results.append(r)

            if self.verbose:
                label = Path(audio_path).stem
                if task == "beat_phase":
                    print(
                        f"[{label:30s}] beat={_fmt(r['f_beat'])}  one={_fmt(r['f_one'])}  "
                        f"last={_fmt(r['f_last'])}  confusion={_fmt(r['confusion'])}"
                    )
                else:
                    print(f"[{label:30s}] beat={_fmt(r['f_beat'])}")

        if self.verbose:
            print("-" * 70)
            print(f"n_tracks={len(results)}")
            print(f"mean beat F-measure:  {_fmt(_mean(results, 'f_beat'))}")
            if task == "beat_phase":
                print(f"mean '1' F-measure:   {_fmt(_mean(results, 'f_one'))}")
                print(f"mean 'last' F-measure: {_fmt(_mean(results, 'f_last'))}")
                print(f"mean phase confusion:  {_fmt(_mean(results, 'confusion'))}")

        return results
