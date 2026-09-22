# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`musicality` is a PyTorch-based library for **tempo estimation** from audio. It provides dataset loaders, model architectures, a Lightning training pipeline, and a GUI annotation tool.

## Commands

```bash
# Install dependencies and set up venv
uv sync
uv pip install -e .

# Bootstrap a fresh remote instance (e.g. vast.ai): uv, Backblaze data via DVC, W&B login
bash tools/setup_remote.sh

# Run tests
uv run pytest tests/

# Run a single test
uv run pytest tests/test_tempo_dataset.py

# Format code
uv run ruff format musicality/

# Train a tempo model (Hydra config)
uv run python tools/train.py

# Download datasets
uv run python tools/download_dataset.py

# Launch the annotator GUI
uv run python -m tools.annotator

# Inspect a track
uv run python tools/inspect_track.py <path-to-audio>

# Plot tempo histograms across datasets
uv run python tools/plot_tempo_histograms.py

# Visualize BeatDataset's smeared beat/one/four targets for one clip
uv run python tools/plot_beat_targets.py --dataset ballroom

# Add a dependency
uv add <package>
```

## Architecture

### Models (`musicality/models/`)

- `tcn.py` — Dilated TCN (`TCNTempoNet`), the default architecture. Log-mel → residual dilated convolutions → global pool → regression head.
  Optionally a `Conv2dStem` first (`conv2d_stem: true`): 2D convolutions with frequency-only pooling, so the model sees a time-frequency
  neighbourhood before the band axis is collapsed. Off by default — see `plans/08` §2.1 and `docs/source/configuration.rst`.
- `tempo_net.py` — Alternative tempo model.
- `huggingface.py` — Wraps HuggingFace `transformers` models (e.g. wav2vec2) for tempo estimation.
- `torch_audio.py` — Wraps `torchaudio` pretrained models.

### Training (`musicality/trainers/`)

- `tempo_module.py` — `TempoModule`: Lightning `LightningModule` wrapping a model, loss, and optimizer.
- `train.py` — Core training routine: builds dataloaders, `TempoModule`, W&B logger, callbacks, and calls `L.Trainer.fit()`.

Entry point: `tools/train.py` uses Hydra to compose config and calls `train()`.

### Dataset Loaders (`musicality/loaders/`)

- `tempo_dataset.py` — `TempoDataset`: loads audio + BPM annotations via `mirdata`.
- `beat_dataset.py` — `BeatDataset`: loads audio + beat-level annotations.

### Augmentations (`musicality/augmentations.py`)

`AugmentedDataset` wraps any dataset with configurable time-stretch, gain, and noise augmentation. `build_augmenter(cfg)` constructs it from the Hydra config.

### Losses (`musicality/losses/`)

One module per objective, named after the task it trains.

- `tempo_regression.py` — `absolute_tempo_loss`, `relative_tempo_loss`: MAE on a single BPM value, with or without octave invariance.
- `tempo_classification.py` — `classification_tempo_loss`, `gaussian_soft_target`: tempo as a softmax over BPM bins against a Gaussian soft target, so a neighbouring bin is a near miss rather than an unrelated class.
- `beat_only.py` — `beat_only_loss`: frame-wise beat BCE alone (`configs/train_beat_only.yaml`).
- `beat_phase.py` — `beat_phase_loss`: bar position as two independent sigmoids (`one`/`last`). Superseded — it never asks the discriminative "is this beat a 1 or a 3?" question — but kept so existing checkpoints stay readable.
- `beat_position.py` — `beat_position_loss`: beat BCE plus a softmax over all `G` bar positions, which do compete. **The loss the project trains with** (`configs/train_phase_beat.yaml`).
- `phase_conditioning.py` — `phase_weight`: which frames the bar-position term is supervised on. `"beat"` restricts it to the frames the decoder actually reads; `"mask"` spends ~96% of the gradient re-learning beat detection.
- `pos_weight.py` — `beat_pos_weight`: positive-class weight for a beat BCE term, either passed through or derived per sample (`"auto"`), since the right value is a function of tempo and a fixed one is correct at only one tempo.
- `shift_tolerance.py` — `shift_tolerant_bce`, `sliding_windowed_max`: forgive the beat head a few frames of timing error, by comparing the max-pooled prediction to the label (Beat This!, ISMIR 2024). Selected by `loss: bce | shift_tolerant` in the beat configs, `bce` by default. It *replaces* target smearing rather than adding to it, so pair `shift_tolerant` with `sigma_frames: 0`.

The last three are shared knobs, not losses, and they are coupled in a chain: conditioning on beats removes most of the imbalance `pos_weight` exists to correct, and shift tolerance changes both the gate `phase_conditioning` builds and the imbalance `pos_weight` measures. `docs/source/losses.rst` is the rendered index.

### Metrics (`musicality/metrics/`)

- `f_measure.py` — `beat_f_measure`, `downbeat_f_measures`: event-level F-measure against `mir_eval`, overall and per bar/phrase position.
- `continuity.py` — `beat_continuity`: `mir_eval`'s CMLc/CMLt/AMLc/AMLt. `amlt - cmlt` is the share of a track tracked confidently at the *wrong metrical level* (half-time, double-time, offbeat) — a failure no other metric here can see.
- `confusion.py` — `confusion_half_cycle_rate`: rate of half-cycle ("1" vs. opposite position) phase-parity errors.
- `position_accuracy.py` — `position_accuracy`: bar-position accuracy at every reference beat, plus its offset histogram. Returns `position_acc`, `position_acc_best_offset` (best single rotation per track) and their difference `anchor_error`, which separates a whole-track phase offset (the model can't hear downbeats) from a mid-track flip (the decoder is losing information). Also sees off-by-one errors, which `confusion.py` is blind to.
- `frame_accuracy.py` — `frame_accuracy` (cheap per-frame accuracy) and `peak_f_measure` (peak-picks both curves first, then matches events, so it moves for the same reasons `f_beat` does). Training-time signals only.
- `tempo_acc1.py` — `tempo_acc1`: MIREX Accuracy 1, octave-tolerant tempo accuracy.

Frame metrics and event metrics disagree, in both directions and for four
separate reasons — see `docs/frame_vs_event_metrics.md` before quoting either.

### Callbacks (`musicality/callbacks/`)

- `error_plot.py` — `ErrorVsTempoPlot`: logs a per-epoch error-vs-tempo scatter to W&B.
- `event_metrics.py` — `EventMetricsLogger`: every few epochs, decodes a fixed corpus-stratified slice of the validation split on **full tracks** and logs `val_event/f_beat`, `cmlt`, `amlt`, `position_acc`, `position_acc_best_offset`. Scores through `BeatEvaluator.score`, so these are the same numbers `tools/eval_beat.py` reports afterwards — unlike the frame metrics beside them, which are measured on a 16s clip. Configured by `event_metrics:` in `configs/train_phase_beat.yaml`.
- `metrics_logger.py` — `BestMetricsPrinter`: prints best validation metrics at the end of training.
- `training_report.py` — `TrainingReportLogger`: at `on_fit_end`, writes one `training_report.json` beside the run's checkpoints and uploads it to the W&B run's Files tab. Holds final/best metrics, the per-epoch history, per-track and per-corpus event scores, the resolved config, and the run's identity (W&B id, git commit, best checkpoint) — a run as a single shareable attachment. Decodes nothing; it reuses `EventMetricsLogger`'s last scoring pass.

### Data Formats (`musicality/dataformats/`)

Loads `dataformat.yaml` and exposes hardcoded directory names (data root, splits dir, leaderboard dir) as a typed `DataFormat` object.

### Splits (`musicality/splits/splitter.py`)

`Splitter` manages train/val splits. Pre-computed splits live in `splits/`.
Reading a split verifies every track it lists is on disk and raises
`MissingTrackDataError` otherwise, so a partial `dvc pull` fails a run instead
of silently shrinking its dataset (see `docs/source/data.rst`).

### Tools (`tools/`)

- `annotator/` — PySide6 GUI for manual beat/tempo annotation. Features: waveform display, playback, tap-tempo widget, metronome, recording.
- `download_dataset.py` — Downloads datasets listed in `configs/download.yaml` via `mirdata`.
- `inspect_track.py` — Prints metadata and annotations for a single audio file.
- `plot_tempo_histograms.py` — Plots BPM distributions across datasets.
- `summarize_datasets.py` — Prints summary statistics for all datasets.
- `train.py` — Hydra entry point for training.
- `leaderboard.py` — Compares *several* runs, where `eval_beat.py` scores one. Walks checkpoint folders, re-evaluates every run it finds through `BeatEvaluator` (so the numbers match `eval_beat.py`) on the split `configs/eval_beat.yaml` names, sweeping each checkpoint's own postprocessing on the **train** split first so the reported val numbers stay held out — unlike `eval_beat.py --sweep`, which tunes and reports on the same tracks. Results merge into one **running** board: a single `leaderboard.json` in the DVC-tracked data repo (`musicality_db/leaderboard/`, `--board` for another path), pulled before reading and `dvc push`ed after writing so it outlives the instance that produced it; only the `.dvc` pointer is left to commit. The board is written twice, to the two places it is read from: `leaderboard.json` is the record, DVC-tracked in the data repo; `leaderboard/LEADERBOARD.md` in **this** repo is the page — git-tracked, so the standings render on GitHub with no `dvc pull` (ranking, the ranking metric per corpus, each row's decode, and its checkpoint path). Only the default `--board` writes the page, `--top N` cuts it to the best N runs, and `--render-only` rebuilds it from the JSON alone, scoring nothing. Only the runs named on the command line are evaluated, the rest are carried over, and rows measured under different settings are refused rather than merged.
- `eval_beat.py` — The one evaluation tool for beat-only/beat-phase checkpoints (task auto-detected), on full-length tracks. Default is the canonical metric report; `--per-genre` (automatic on a merged split) breaks it down per corpus, `--profile` prints the phase-offset profile, `--decoders` scores every bar-position decoder against one cached model pass and calls model-vs-decoder, `--sweep` grid-searches the postprocessing knobs that `configs/eval_beat.yaml` holds, `--output` writes per-track rows to CSV. Every mode runs the model once per track and re-uses the cached probabilities.

## Configuration

Hydra configs live in `configs/`. The files hold values only — **every key is
documented in `docs/source/configuration.rst`**, which is the source of truth
for what a key means, which keys are coupled, and the measurements behind the
defaults. Add explanations there, not as YAML comments.

- `train_phase_beat.yaml` — Beat-phase training (`tools/train_beat.py`). The config the project currently trains with.
- `train_beat_only.yaml` — Beat-only training (`tools/train_beat_only.py`).
- `train_tempo.yaml` — Tempo training (`tools/train_tempo.py`).
- `eval_beat.yaml` — `tools/eval_beat.py` defaults. Also read by the annotator and composed into `train_phase_beat.yaml` under `eval`, so it is the project-wide postprocessing default, not just CLI defaults. Nothing under `musicality/` opens it: `BeatEvaluator` takes the per-task block as a required `postprocess` argument, so the tuned numbers have exactly one home and the library holds no defaults to drift from them.
- `download.yaml` — List of datasets to download and their `data_home`.
- `model/` — Backbone overrides selected by each config's `defaults:` list: `tcn.yaml` (tempo), `tcn_frames.yaml` (beat-phase), `tcn_frames_beat.yaml` (beat-only).

## Describing Work

Every proposal, plan, implementation summary and design note gets **two descriptions**, in this order:

1. **High level** — the problem, why it matters, and what changes as a result. No filenames, no function names, no code. Someone who doesn't know the internals should be able to read it and know whether they care.
2. **Technical** — the mechanism. Files and functions touched, data flow, edge cases, and how it was (or will be) verified.

Applies to `plans/*.md`, `docs/*.md`, and end-of-task summaries in chat. Keep them separate — don't interleave the two registers into one blended paragraph, since that produces prose that is too vague to act on and too detailed to skim.

## Code Style

- Use an aerated style: insert blank lines where it improves readability — e.g. between a docstring and the function body, and between logical blocks within a function. Avoid dense, run-on blocks of code.

## Git Commits

- Do not run `git commit` yourself, under any circumstance — the user makes every commit. A `PostToolUse` hook (`.claude/settings.json`) already auto-formats and `git add`s each changed file, so leave changes staged and stop there.
- If asked to commit in a specific instance anyway: only commit once a piece of work is actually done — don't create a commit for every small/incremental change — and commit messages should describe what was added.

## Datasets

Downloaded data lives in the directory configured by `data_dir` in `musicality/dataformats/dataformat.yaml` (currently `../musicality_db/`, a sibling git+dvc repo — see `tools/setup_remote.sh`). Supported datasets (via `mirdata`): ballroom, brid, hainsworth, rwc_classical, rwc_jazz, rwc_popular, groove_midi, guitarset.

## Dependencies

- `torch` / `torchaudio` / `lightning` — model training
- `mirdata` — dataset loading and annotation access
- `librosa` — audio analysis utilities
- `hydra-core` — config composition
- `wandb` — experiment tracking
- `pyside6` — annotator GUI
- `sounddevice` / `soundfile` — audio I/O in the annotator
