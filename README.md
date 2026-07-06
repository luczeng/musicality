# musicality

A Python library for working with music datasets, featuring a tempo estimation pipeline built on [mirdata](https://mirdata.readthedocs.io), PyTorch, PyTorch Lightning, and Hydra.

## Setup

```bash
uv sync
uv pip install -e .
```

## Project structure

```
configs/             Hydra configuration files
  train.yaml         Top-level config (composes model, data, trainer)
  model/cnn.yaml     TempoNet CNN architecture
  data/brid.yaml     BRID dataset and dataloader settings
  trainer/default.yaml  Lightning trainer settings

musicality/
  loader.py          PyTorch Dataset and DataLoader for the BRID dataset

trainers/
  model.py           TempoNet — a lightweight CNN for tempo regression
  tempo_module.py    LightningModule (loss, metrics, optimizer)

tools/
  download_dataset.py  Download a mirdata dataset into data/
  summarize_datasets.py  List datasets with song counts and annotation types
  train.py           Launch tempo estimation training
```

## Datasets

### Download a dataset

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

## DataLoader

`musicality/loader.py` provides a `BRIDDataset` and a `get_loader` factory.

Each item is a `(mel, tempo)` pair:

- `mel` — log-mel spectrogram of shape `(1, n_mels, T)`, computed with `torchaudio`
- `tempo` — BPM label as a scalar `float32` tensor

```python
from musicality.loader import get_loader

loader = get_loader(
    data_home="data/brid",
    batch_size=32,
    sample_rate=22050,
    n_mels=128,
    duration=10.0,  # seconds — clips are truncated or zero-padded to this length
)

for mel, tempo in loader:
    # mel:   (B, 1, 128, T)
    # tempo: (B,)
    ...
```

## Training

### Model

`trainers/model.py` defines `TempoNet`, a three-block CNN that takes a log-mel spectrogram and regresses a single BPM value.

```
Input  (B, 1, n_mels, T)
  → Conv 1×16 → BN → ReLU → MaxPool
  → Conv 16×32 → BN → ReLU → MaxPool
  → Conv 32×64 → BN → ReLU → AdaptiveAvgPool(4×4)
  → Linear 1024→128 → ReLU → Dropout
  → Linear 128→1
Output (B,)
```

### LightningModule

`trainers/tempo_module.py` wraps `TempoNet`:

- Loss: MSE
- Metric: MSE (logged as `train/mse` and `val/mse`)
- Optimizer: Adam with `ReduceLROnPlateau` scheduler

### Configuration

Training is configured with [Hydra](https://hydra.cc). Config files live in `configs/` and can be overridden on the command line.

Key options in `configs/train.yaml`:

| Key | Default | Description |
|---|---|---|
| `lr` | `1e-3` | Learning rate |
| `weight_decay` | `1e-4` | L2 regularisation |
| `checkpoint_dir` | `checkpoints/` | Where to save model checkpoints |
| `data.batch_size` | `32` | Batch size |
| `data.val_split` | `0.2` | Fraction of data held out for validation |
| `data.duration` | `10.0` | Audio clip length in seconds |
| `trainer.max_epochs` | `50` | Maximum training epochs |
| `trainer.accelerator` | `auto` | `cpu`, `gpu`, or `auto` |

### Run training

```bash
uv run python tools/train.py
```

Override any value on the command line:

```bash
# Change batch size and learning rate
uv run python tools/train.py data.batch_size=16 lr=3e-4

# Train for more epochs on GPU
uv run python tools/train.py trainer.max_epochs=100 trainer.accelerator=gpu

# Use a different model config
uv run python tools/train.py model=cnn data.n_mels=64
```

Hydra writes logs and run configs to `outputs/<date>/<time>/` by default.
Checkpoints are saved to `checkpoints/` (top-3 by `val/loss`, with early stopping after 10 epochs without improvement).

## Tools

| Tool | Description |
|---|---|
| `tools/train.py` | Hydra entry point for training a tempo model |
| `tools/download_dataset.py` | Download datasets listed in `configs/download.yaml` via mirdata |
| `tools/summarize_datasets.py` | Print summary statistics (song count, annotation types) for all downloaded datasets |
| `tools/inspect_track.py` | Print metadata and annotations for a single audio file |
| `tools/plot_tempo_histograms.py` | Plot BPM distributions across datasets |
| `tools/annotator/` | PySide6 GUI for manual beat/tempo annotation (waveform display, playback, tap-tempo, metronome) |

```bash
uv run python tools/train.py
uv run python tools/download_dataset.py
uv run python tools/summarize_datasets.py
uv run python tools/inspect_track.py path/to/audio.wav
uv run python tools/plot_tempo_histograms.py
uv run python -m tools.annotator
```

## Tests

```bash
uv run pytest tests/
```
