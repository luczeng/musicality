# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`musicality` is a PyTorch-based library for **tempo estimation** from audio. It provides dataset loaders, model architectures, a Lightning training pipeline, and a GUI annotation tool.

## Commands

```bash
# Install dependencies and set up venv
uv sync
uv pip install -e .

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

# Add a dependency
uv add <package>
```

## Architecture

### Models (`musicality/models/`)

- `tcn.py` — Dilated TCN (`TCNTempoNet`), the default architecture. Log-mel → residual dilated convolutions → global pool → regression head.
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

### Losses (`musicality/losses.py`)

Supports absolute, relative, and classification loss modes. Classification loss treats tempo as a discretized bin with a Gaussian target distribution.

### Callbacks (`musicality/callbacks/`)

- `error_plot.py` — `ErrorVsTempoPlot`: logs a per-epoch error-vs-tempo scatter to W&B.
- `metrics_logger.py` — `BestMetricsPrinter`: prints best validation metrics at the end of training.

### Data Formats (`musicality/dataformats/`)

Loads `dataformat.yaml` and exposes hardcoded directory names (data root, splits dir) as a typed `DataFormat` object.

### Splits (`musicality/splits/splitter.py`)

`Splitter` manages train/val splits. Pre-computed splits live in `splits/`.

### Tools (`tools/`)

- `annotator/` — PySide6 GUI for manual beat/tempo annotation. Features: waveform display, playback, tap-tempo widget, metronome, recording.
- `download_dataset.py` — Downloads datasets listed in `configs/download.yaml` via `mirdata`.
- `inspect_track.py` — Prints metadata and annotations for a single audio file.
- `plot_tempo_histograms.py` — Plots BPM distributions across datasets.
- `summarize_datasets.py` — Prints summary statistics for all datasets.
- `train.py` — Hydra entry point for training.

## Configuration

Hydra configs live in `configs/`:

- `train.yaml` — Top-level training config (loss, lr, batch size, dataset, augmentations, W&B).
- `download.yaml` — List of datasets to download and their `data_home`.
- `model/` — Per-model overrides: `tcn.yaml`, `beat.yaml`, `cnn.yaml`, `wav2vec2.yaml`.
- `trainer/` — Lightning Trainer overrides.

## Git Commits

- Only commit once a piece of work is actually done — don't create a commit for every small/incremental change.
- Commit messages should describe what was added.

## Datasets

Downloaded data lives in `data/`. Supported datasets (via `mirdata`): ballroom, brid, hainsworth, rwc_classical, rwc_jazz, rwc_popular, groove_midi, guitarset.

## Dependencies

- `torch` / `torchaudio` / `lightning` — model training
- `mirdata` — dataset loading and annotation access
- `librosa` — audio analysis utilities
- `hydra-core` — config composition
- `wandb` — experiment tracking
- `pyside6` — annotator GUI
- `sounddevice` / `soundfile` — audio I/O in the annotator
