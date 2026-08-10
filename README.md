# musicality

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://luczeng.github.io/musicality/)

A Python library for tempo and beat estimation from audio, built on [mirdata](https://mirdata.readthedocs.io), PyTorch, PyTorch Lightning, and Hydra — with desktop and mobile apps for building homemade training data.

## Setup

```bash
uv sync
uv pip install -e .
```

### Fresh machine / remote instance (e.g. vast.ai)

After cloning, with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `WANDB_API_KEY`
set in the environment:

```bash
bash tools/setup_remote.sh
```

This installs `uv` if missing, syncs dependencies, pulls data from Backblaze via
[DVC](https://dvc.org), logs in to Weights & Biases, and fetches the `mirdata` index
for any DVC-pulled dataset (see below for why that last step is needed). Re-running it
on the same machine is safe.

Under the hood:

- Data is stored via DVC on a remote (S3-compatible, Backblaze-hosted) bucket. DVC's S3
  backend reads `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` directly from the
  environment (`uv run dvc pull`, or `uv run dvc pull data/<name>.dvc` for a single
  dataset).
- Training logs to Weights & Biases by default (`uv run wandb login "$WANDB_API_KEY"`).
- `mirdata`'s dataset **index** (small metadata JSON, separate from the audio/annotations
  themselves) is normally fetched the first time `tools/download_dataset.py` runs. Since
  DVC pulls the audio directly and skips that step, training would otherwise fail with
  `FileNotFoundError: This dataset's index must be downloaded` the first time on a new
  machine.

### Vast.ai instance template

To make a rented instance training-ready with no manual steps, set these once in a
reusable vast.ai instance template:

- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `WANDB_API_KEY`.
- On-start script: clone the repo and run the setup script, e.g.
  `git clone <repo-url> musicality && cd musicality && bash tools/setup_remote.sh`.

## Datasets

Training data comes from two sources, both read identically by every loader/tool in
this repo:

- **mirdata datasets** — publicly available beat/tempo-annotated datasets (ballroom,
  brid, hainsworth, rwc_classical, rwc_jazz, rwc_popular, groove_midi, guitarset),
  fetched via [mirdata](https://mirdata.readthedocs.io).
- **Homemade datasets** — audio recorded and beat-tapped by hand with the annotation
  apps below (e.g. a `swing` dataset of hand-recorded dance tracks). These live under
  `data/<name>/tracks/` (audio) and `data/<name>/annotations/*.beats` (tapped beats),
  a plain directory layout rather than a mirdata dataset definition.

### Download a mirdata dataset

```bash
uv run python tools/download_dataset.py <mirdata-name>
# e.g.
uv run python tools/download_dataset.py brid
```

### List downloaded datasets

```bash
uv run python tools/summarize_datasets.py
```

Output includes the dataset name, number of songs, and mirdata annotation types:

```
Dataset    Songs  Annotations
---------------------------------
brid         367  beats, tempo
```

## Splits

Train/val splits are precomputed and stored as plain index lists under
`data/splits/<name>/{train,val}.txt`, read by `Splitter.run()` at train/eval time.
It never generates a split itself — if `data/splits/<name>/` is missing, training
crashes with `FileNotFoundError` rather than silently creating a new (and possibly
different) one. That's what makes a split reproducible across machines: version the
files under `data/splits/` (e.g. via DVC, same as the audio) and pull them rather than
regenerating locally, where a different `mirdata` version or an incomplete download
could silently produce a different split.

### Create a split

```bash
uv run python tools/create_splits.py                          # every dataset in data/
uv run python tools/create_splits.py --datasets ballroom brid  # just these
uv run python tools/create_splits.py --val-split 0.15 --force  # custom split, overwrite
```

Creates two splits per dataset: a tempo split (`data/splits/<name>`, from
`TempoDataset`) and a beat-phase split (`data/splits/beat_phase-<name>`, from
`BeatDataset`). Whichever has no samples for a given dataset (e.g. no tempo
annotations) is skipped. Existing splits are left untouched unless `--force` is passed.

### Binary-meter-only beat-phase splits

Some datasets mix meters — ballroom's waltz/Viennese waltz tracks are annotated with a
triple-meter bar-position cycle (`1, 2, 3, 1, 2, 3, ...`) instead of the binary meter
(beats-per-bar a multiple of 2, e.g. `1, 2, 3, 4, ...`) the beat-phase `one`/`last`
targets assume. Pass `--binary-only` to drop those tracks — and any track with no
position annotation at all, since its meter can't be confirmed — when building the
beat-phase split:

```bash
uv run python tools/create_splits.py --datasets ballroom --binary-only
```

This saves to a separate `data/splits/beat_phase-<name>-binary` split, since it's a
different-length dataset than the unfiltered one. Training/eval must set the matching
flag so they load the split that actually corresponds to the dataset they build:

```bash
uv run python tools/train_beat.py binary_only=true
uv run python tools/eval_beat_phase.py --checkpoint <path> --dataset ballroom --binary-only
```

### Version splits with DVC

```bash
uv run dvc add data/splits
git add data/splits.dvc data/.gitignore
git commit -m "Version train/val splits"
uv run dvc push
```

On another machine, `dvc pull` (see [Fresh machine](#fresh-machine--remote-instance-eg-vastai)
above) fetches the exact same split files, so training and evaluation line up across
machines instead of each generating its own split locally.

## Annotation apps

Two apps produce homemade datasets — audio plus hand-tapped beat annotations, saved
in the same format the mirdata datasets use.

### Desktop annotator

`tools/annotator/` — a PySide6 GUI for browsing datasets, tapping beat annotations by
ear, and recording new tracks from a microphone.

- Waveform display with beat markers and a playback cursor; click to seek,
  Ctrl+click to add a beat, Ctrl+right-click to remove one
- Tap-tempo widget and metronome for annotating a track by ear, with a configurable
  count (4/8) and accent pattern
- Audible click track synced to the annotated beats, with its own volume control
- Record new tracks straight from the microphone into a named dataset folder
- Run inference with a trained beat-phase checkpoint and preview the model's
  predicted beats on a second waveform strip above the track, with an optional click
  track against the prediction instead of the manual annotation
- Per-track metadata (recording device, location, structure) plus a dataset browser
  showing per-track annotation status

```bash
uv run python -m tools.annotator --dataset ballroom
uv run python -m tools.annotator --dataset ballroom --track Media-105901
```

### Mobile companion

`tools/mobile_companion/` — an offline-first PWA + FastAPI backend for recording
audio and tapping tempo from a phone, syncing captures into the same
`data/<dataset>/tracks/` + `annotations/*.beats` structure the desktop annotator
reads. Useful for field recordings (e.g. live dancing) away from a laptop. See
`tools/mobile_companion/README.md` for setup, including remote HTTPS access via
Tailscale.

## Loaders

`musicality/loaders/tempo_dataset.py` — `TempoDataset`, returns `(waveform, tempo)`
pairs for any mirdata dataset exposing a `tempo` attribute per track.

`musicality/loaders/beat_dataset.py` — `BeatDataset`, returns `(waveform, target)`
pairs for any mirdata (or homemade) dataset exposing `beats` per track, where
`target` is a 4-channel `(beat, one, last, mask)` frame-level tensor (see
[Beat-phase detection](#beat-phase-detection) below).

Both resample to a target sample rate and pad/truncate to a fixed clip duration.
Mel-spectrogram extraction happens inside the model, not the loader.

## Training

### Model

`musicality/models/tcn.py` defines `TCNTempoNet`, a dilated TCN (Davies & Böck,
2019) — the default architecture for both tempo estimation and beat-phase detection:

```
log-mel spectrogram → per-clip normalization
  → 1×1 channel projection
  → N dilated residual Conv1d blocks (dilation 1, 2, 4, ..., 2^(N-1))
  → global average pool → FC head            (tempo: scalar / classification bins)
  → or: 1×1 conv head, no pooling            (beat-phase: per-frame beat/one/last)
```

Alternate backbones: `musicality/models/tempo_net.py` (a simpler CNN),
`musicality/models/huggingface.py` (wraps HuggingFace `transformers` models, e.g.
wav2vec2/BEaT), `musicality/models/torch_audio.py` (wraps pretrained `torchaudio`
models).

### LightningModule

`musicality/trainers/tempo_module.py` wraps the model as `TempoModule`:

- Loss: `absolute` (MAE), `relative` (octave-invariant MAE), or `classification`
  (softmax over BPM bins with a Gaussian soft target)
- Metric: MIREX Accuracy 1 (`acc1`), logged alongside MAE
- Optimizer: Adam with a `ReduceLROnPlateau` scheduler

### Configuration

Training is configured with [Hydra](https://hydra.cc). Config files live in
`configs/` and can be overridden on the command line.

Key options in `configs/train.yaml`:

| Key | Default | Description |
|---|---|---|
| `loss` | `classification` | `absolute`, `relative`, or `classification` |
| `lr` | `5e-4` | Learning rate |
| `weight_decay` | `0.0` | L2 regularisation |
| `checkpoint_dir` | `checkpoints/` | Where to save model checkpoints |
| `batch_size` | `32` | Batch size |
| `data.val_split` | `0.2` | Fraction of data held out for validation |
| `data.duration` | `15.0` | Audio clip length in seconds |
| `trainer.max_epochs` | `100` | Maximum training epochs |
| `trainer.accelerator` | `auto` | `cpu`, `gpu`, or `auto` |

### Run training

```bash
uv run python tools/train.py
```

Override any value on the command line:

```bash
# Change batch size and learning rate
uv run python tools/train.py batch_size=16 lr=3e-4

# Train for more epochs on GPU
uv run python tools/train.py trainer.max_epochs=200 trainer.accelerator=gpu

# Use a different model config
uv run python tools/train.py model=cnn n_mels=64
```

Hydra writes logs and run configs to `outputs/<date>/<time>/` by default.
Checkpoints are saved to `checkpoint_dir` (top-3 by `val/loss`, with early stopping
after 10 epochs without improvement).

## Beat-phase detection

A second pipeline, alongside tempo estimation, detects frame-level **beat** /
**"one"** (downbeat) / **"last"** (last beat of the group — bar position 4 by
default) events. It reuses the same dataset/training scaffolding as tempo
estimation (`BeatDataset`, Hydra config, Lightning), configured through
`configs/beat_train.yaml`.

Key options in `configs/beat_train.yaml`:

| Key | Default | Description |
|---|---|---|
| `lr` | `5e-4` | Learning rate |
| `group_size` | `4` | Beats per group: `4` for bar position (1-4), `8` for phrase position (1-8) on a dataset with phrase annotations |
| `binary_only` | `false` | Train on the binary-meter-only split (see [Splits](#splits)); must match how the split was created |
| `pos_weight` | `8.0` | Positive-class weight for the one/last BCE heads |
| `train_subsample` | `null` | Fraction of the training split to use (e.g. `0.2`), for quick smoke runs |
| `data.name` | `ballroom` | mirdata dataset name |
| `checkpoint_dir` | `checkpoints_beat/` | Where to save model checkpoints |

### Run training

```bash
uv run python tools/train_beat.py

# quick smoke test — a couple epochs on a fraction of the data
WANDB_MODE=offline uv run python tools/train_beat.py \
    trainer.max_epochs=2 train_subsample=0.2 checkpoint_dir=checkpoints_beat_test/

# a phrase-position (1-8) dataset instead of the default bar-position (1-4) one
uv run python tools/train_beat.py group_size=8 data.name=<phrase_dataset>
```

### Evaluate a checkpoint

`tools/eval_beat_phase.py` scores a trained checkpoint on full-length tracks
(not the fixed-duration training clips): beat / "1" / "last" F-measure, and
the half-cycle phase-confusion rate (catches a model that's found the right
periodicity but locked onto the wrong phase).

```bash
uv run python tools/eval_beat_phase.py \
    --checkpoint checkpoints_beat/beat-phase-epoch=05-val_loss=0.1234.ckpt \
    --dataset ballroom
```

### Sweep learning rates

`tools/sweep_lr.py` batch-trains across a list of learning rates (reusing the
same dataloaders and seed across runs, so lr is the only thing that varies)
and prints a comparison table of best validation metrics.

```bash
uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3

# save the comparison table instead of only printing it (.csv or plain text)
uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3 --output sweep_results.csv
```

### Visualize targets

```bash
uv run python tools/plot_beat_targets.py --dataset ballroom
```

## Tools

| Tool | Description |
|---|---|
| `tools/train.py` | Hydra entry point for training a tempo model |
| `tools/train_beat.py` | Hydra entry point for training a beat-phase model |
| `tools/create_splits.py` | Create the train/val splits under `data/splits/` that `Splitter.run()` requires (see [Splits](#splits)) |
| `tools/eval_beat_phase.py` | Evaluate a beat-phase checkpoint: beat/"1"/"last" F-measure and phase-confusion rate |
| `tools/sweep_lr.py` | Batch-train the beat-phase model over a list of learning rates and compare results |
| `tools/plot_beat_targets.py` | Visualize a `BeatDataset` clip's waveform against its smeared beat/one/last targets |
| `tools/download_dataset.py` | Download datasets listed in `configs/download.yaml` via mirdata |
| `tools/summarize_datasets.py` | Print summary statistics (song count, annotation types) for all downloaded datasets |
| `tools/inspect_track.py` | Print metadata and annotations for a single audio file |
| `tools/plot_tempo_histograms.py` | Plot BPM distributions across datasets |

See [Annotation apps](#annotation-apps) above for `tools/annotator/` and
`tools/mobile_companion/`.

```bash
uv run python tools/train.py
uv run python tools/train_beat.py
uv run python tools/create_splits.py
uv run python tools/eval_beat_phase.py --checkpoint <path-to-ckpt> --dataset ballroom
uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3
uv run python tools/plot_beat_targets.py --dataset ballroom
uv run python tools/download_dataset.py
uv run python tools/summarize_datasets.py
uv run python tools/inspect_track.py path/to/audio.wav
uv run python tools/plot_tempo_histograms.py
```

## API documentation

Sphinx-generated API reference for the `musicality` package — losses, metrics,
loaders, models, trainers, callbacks — with math equations rendered for the loss
functions. Docstrings are the source of truth; the docs are built from them, not
maintained separately.

```bash
uv run sphinx-build -b html docs/source docs/build && open docs/build/index.html
```

While editing docstrings, use the live-reload server instead — it rebuilds and
refreshes the browser on save:

```bash
uv run sphinx-autobuild docs/source docs/build
```

Rendering the math equations requires internet access (MathJax loads from a CDN);
everything else works fully offline.

## Tests

```bash
uv run pytest tests/
```
