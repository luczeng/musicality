<p align="center">
  <img src="assets/logo.png" alt="musicality logo" width="534">
</p>

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://luczeng.github.io/musicality/)

A Python library for music analysis via SOTA methods using ML, bayesian inference and signal processing. Built using PyTorch, PyTorch Lightning, and Hydra — with desktop and mobile apps for building homemade training data.  

Currently supports:  

- Tempo estimation
- Beat estimation
- Tempo phase estimation

## Roadmap

- [ ] App
- [ ] Real time inference
- [ ] Bayesian methods
- [ ] Phrasing, chords methods
- [x] Tempo, beat and beat phase loaders and trainers. Corresponding dataformats. Basic nets and postprocessing tools. Mobile and local annotation apps. 

<details id="setup">
<summary><b>Setup</b></summary>

```bash
uv sync
uv pip install -e .
```

### quick install

For quick setup on a remote instance, a conveniance script is provided: 

```bash
bash tools/setup_remote.sh
```

This also fetches custom dataset from the remote via DVC (currently on Infomaniak s3). Requirements are to setup env variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `WANDB_API_KEY`. The custom datasets might become available on demand.


</details>

<details id="datasets">
<summary><b>Datasets</b></summary>

Training data comes from two sources, both read identically by every loader/tool in
this repo, and both stored under the data directory configured by `data_dir` in
[`musicality/dataformats/dataformat.yaml`](musicality/dataformats/dataformat.yaml)
(currently `../musicality_db` — a sibling git+dvc repo, cloned by `tools/setup_remote.sh`):

- **mirdata datasets** — publicly available beat/tempo-annotated datasets (ballroom,
  brid, hainsworth, rwc_classical, rwc_jazz, rwc_popular, groove_midi, guitarset),
  fetched via [mirdata](https://mirdata.readthedocs.io).
- **Homemade datasets** — audio recorded and beat-tapped by hand with the annotation
  apps below (e.g. a `swing` dataset of hand-recorded dance tracks). These live under
  `../musicality_db/<name>/tracks/` (audio) and
  `../musicality_db/<name>/annotations/*.beats` (tapped beats), a plain directory
  layout rather than a mirdata dataset definition.

### Data format

Both sources read through the same `<time> <position>` `.beats` format plus an
optional `.meta.json` sidecar (device, structure, tempo stats, multi-annotator
support). See the
[data format reference](https://luczeng.github.io/musicality/data.html#data-format)
for the full field list and layout details.

### Splits

Train/val splits should be precomputed and stored as plain index lists under
`../musicality_db/splits/<name>/{train,val}.txt`, read by `Splitter.run()` at
train/eval time. Note: at train time, if `../musicality_db/splits/<name>/` is
missing, training crashes with `FileNotFoundError`. To create a split:

```bash
uv run python tools/create_splits.py                          # every dataset in ../musicality_db/
uv run python tools/create_splits.py --datasets ballroom brid  # just these
uv run python tools/create_splits.py --val-split 0.15 --force  # custom split, overwrite
```

#### Binary-meter-only beat-phase splits

Some datasets mix meters — ballroom's waltz/Viennese waltz tracks are annotated with a
triple-meter bar-position cycle (`1, 2, 3, 1, 2, 3, ...`) instead of the binary meter
(beats-per-bar a multiple of 2, e.g. `1, 2, 3, 4, ...`) the beat-phase `one`/`last`
targets assume. Pass `--binary-only` to drop those tracks — and any track with no
position annotation at all, since its meter can't be confirmed — when building the
beat-phase split:

#### Version splits with DVC

Splits are versioned in the separate [musicality_db](https://github.com/luczeng/musicality_db)
repo alongside the datasets themselves, not in this repo — `uv run --project .`
below reuses this repo's venv (which has `dvc` installed) while running the
command against `../musicality_db`:

```bash
cd ../musicality_db
uv run --project ../musicality dvc add splits
git add splits.dvc .gitignore
git commit -m "Version train/val splits"
uv run --project ../musicality dvc push
cd -
```

On another machine, `tools/setup_remote.sh` (or a manual `dvc pull` in
`../musicality_db`) fetches the exact same split files, so training and
evaluation line up across machines instead of each generating its own split
locally.

The running leaderboard is versioned there the same way, under `leaderboard/` —
but `tools/leaderboard.py` runs its own `dvc pull`/`dvc add`/`dvc push`, so the
only manual step is committing the pointer it names on the way out.

</details>

<details id="annotation-apps">
<summary><b>Annotation apps</b></summary>

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
`../musicality_db/<dataset>/tracks/` + `annotations/*.beats` structure the desktop annotator
reads. Useful for field recordings (e.g. live dancing) away from a laptop. See
`tools/mobile_companion/README.md` for setup, including remote HTTPS access via
Tailscale.

</details>

<details id="train">
<summary><b>Training</b></summary>

### Tempo estimation

`musicality/models/tcn.py` (`TCNTempoNet`) is the default backbone, wrapped by
`musicality/trainers/tempo_module.py` (`TempoModule`) — see the
[API documentation](#api-documentation) for architecture, loss modes, and metrics.
Alternate backbones live in `musicality/models/tempo_net.py` (a simpler CNN),
`musicality/models/huggingface.py` (wraps HuggingFace `transformers` models, e.g.
wav2vec2/BEaT) and `musicality/models/torch_audio.py` (wraps pretrained
`torchaudio` models); they have no config of their own, so point `model._target_`
at one to use it.

Training is configured with [Hydra](https://hydra.cc) and overridable on the
command line. `configs/train.yaml` holds values only — every key is explained in
the [configuration reference](https://luczeng.github.io/musicality/configuration.html).

```bash
uv run python tools/train_tempo.py
```

Override any value on the command line:

```bash
# Change batch size and learning rate
uv run python tools/train_tempo.py batch_size=16 lr=3e-4

# Train for more epochs on GPU
uv run python tools/train_tempo.py trainer.max_epochs=200 trainer.accelerator=gpu

# Train on a different split
uv run python tools/train_tempo.py data.input=merge n_mels=64
```

Hydra writes logs and run configs to `outputs/<date>/<time>/` by default.
Checkpoints are saved to `checkpoint_dir`, one subdirectory per run, top-3 by
`val/loss`.

### Beat-phase detection

A second pipeline, alongside tempo estimation, detects frame-level **beat** /
**"one"** (downbeat) / **"last"** (last beat of the group — bar position 4 by
default) events. It reuses the same dataset/training scaffolding as tempo
estimation (`BeatDataset`, Hydra config, Lightning). Configured through
`configs/beat_train.yaml`; every key is explained in the
[configuration reference](https://luczeng.github.io/musicality/configuration.html).

```bash
uv run python tools/train_beat.py

# quick smoke test — a couple epochs on a fraction of the data
WANDB_MODE=offline uv run python tools/train_beat.py \
    trainer.max_epochs=2 train_subsample=0.2 checkpoint_dir=checkpoints_beat_test/

# a phrase-position (1-8) dataset instead of the default bar-position (1-4) one
uv run python tools/train_beat.py group_size=8 data.name=<phrase_dataset>
```

### Sweep learning rates

`tools/sweep_lr.py` batch-trains across a list of learning rates (reusing the
same dataloaders and seed across runs, so lr is the only thing that varies)
and prints a comparison table of best validation metrics.

```bash
uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3

# name the sweep instead of taking the timestamp
uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 --sweep-id deeper-trunk

# put the comparison table somewhere other than the sweep directory
uv run python tools/sweep_lr.py --lrs 1e-4 5e-4 1e-3 --output ~/sweeps/lr.csv
```

Each sweep gets a directory of its own, stamped with the moment it started, so
two sweeps on the same day don't overwrite each other's checkpoints. Every run
inside it keeps its own `training_report.json`, and the comparison table is
written beside them:

```
checkpoints_beat/lr_sweep-20260918-141530/
    lr_0.0008/   <checkpoints> + training_report.json
    lr_0.002/    <checkpoints> + training_report.json
    sweep_results.csv
```

</details>

<details id="leaderboard">
<summary><b>Leaderboard</b></summary>


| Dataset | Task | Network | Split | Beat F-measure | "1" F-measure | "last" F-measure |
|---|---|---|---|---|---|---|
| ballroom (binary meter) | Beat detection (beat-only) | TCN | val, 104 tracks | 0.896 | — | — |
| ballroom (binary meter) | Beat detection (phase) | TCN | val, 104 tracks | 0.916 | 0.697 | 0.692 |

### Producing one

`tools/leaderboard.py` re-scores a set of trained runs against each other. Each
run's own `training_report.json` was written under that run's split and
postprocessing, so comparing reports compares the settings as much as the
models; this evaluates every checkpoint on one common split instead, sweeping
each one's postprocessing first (the `beat_phase` knobs in
`configs/eval_beat.yaml` are marked UNVERIFIED, and re-sweeping has been worth
more than a retrain).

The sweep runs on **train** (`--sweep-split`, 50 corpus-stratified tracks by
default) and the report on val, so the knobs are never chosen on the tracks
they are then scored on — unlike `eval_beat.py --sweep`, which tunes and reports
on the same split and is optimistic as a result.

Every invocation extends one running board, which lives in the DVC-tracked data
repo (`../musicality_db/leaderboard/`) beside the splits — pulled before reading
and pushed after writing, so a board built on a rented instance survives the
instance. You only ever evaluate what's new:

```bash
# first time — the file doesn't exist yet, so this starts the board
uv run python tools/leaderboard.py checkpoints_deeper checkpoints_norm \
    --dataset merge --split val

# every time after — same command, just the new experiment
uv run python tools/leaderboard.py checkpoints_new --dataset merge --split val
```

Rows for runs not named on the command line are carried over; naming a run
that's already on the board re-measures it and replaces its row. Rows measured
under a different split, tolerance or group size are refused rather than merged,
before any model pass. `--append PATH` keeps a board somewhere else, `--no-append`
makes a local standalone one, and `--no-pull` / `--no-push` skip the DVC sync.

`dvc push` uploads the board's content, but the pointer only becomes the shared
truth once committed — the tool prints the one command to run:

```bash
cd ../musicality_db && git add leaderboard.dvc && git commit -m "Update leaderboard"
```

```bash
# a single checkpoint, at the config's shipped knobs, no W&B, its own board
uv run python tools/leaderboard.py checkpoints/merge_v5.ckpt \
    --no-sweep --no-wandb --no-append

# rank the board by bar-position accuracy instead of beat F-measure
uv run python tools/leaderboard.py checkpoints_deeper --rank-metric position_acc

# tune on val too — no second model pass, but nothing is held out
uv run python tools/leaderboard.py checkpoints_deeper --sweep-split val
```

`--rank-metric` orders the board; the sweep's bar-position stage has its own
`--sweep-rank-metric` (default `position_acc`), because a decoder relabels beats
without moving them and every candidate ties on beat F-measure.

The ranked board is printed and written as one JSON file holding everything —
per-run metrics, the knobs each run was scored at, the per-corpus breakdown, and
a rendered table under `readable` — so a whole comparison travels as a single
attachment.

</details>

<details id="tools">
<summary><b>Tools</b></summary>

| Tool | Description |
|---|---|
| `tools/train_tempo.py` | Hydra entry point for training a tempo model |
| `tools/train_beat.py` | Hydra entry point for training a beat-phase model |
| `tools/create_splits.py` | Create the train/val splits under `../musicality_db/splits/` that `Splitter.run()` requires (see [Splits](#splits)) |
| `tools/eval_beat.py` | The single evaluation entry point for a beat-only or beat-phase checkpoint (task auto-detected), on full-length tracks rather than the fixed-duration training clips. Default: the canonical metric report (`f_beat`, `cmlt`/`amlt`, `position_acc` and its offset-invariant twin). `--per-genre` breaks it down per corpus, `--profile` prints the phase-offset profile, `--decoders` scores every bar-position decoder against one cached model pass and says whether the error is the model's or the decoder's, `--sweep` grid-searches the postprocessing knobs, `--output` writes per-track rows to CSV |
| `tools/sweep_lr.py` | Batch-train the beat-phase model over a list of learning rates and compare results |
| `tools/leaderboard.py` | Re-score every run in one or more checkpoint folders on a common split, sweeping each checkpoint's postprocessing, into one running `leaderboard.json` kept in the DVC-tracked data repo |
| `tools/plot_beat_targets.py` | Visualize a `BeatDataset` clip's waveform against its smeared beat/one/last targets |
| `tools/download_dataset.py` | Download datasets listed in `configs/download.yaml` via mirdata |
| `tools/migrate_mirdata_dataset.py` | Migrate a mirdata dataset's beat annotations into this project's own `tracks/`/`annotations/` layout (see [Data format](#data-format)), so tools that only understand that layout can read it like a homemade dataset |
| `tools/summarize_datasets.py` | Print summary statistics (song count, annotation types) for all downloaded datasets |
| `tools/inspect_track.py` | Print metadata and annotations for a single audio file |
| `tools/plot_tempo_histograms.py` | Plot BPM distributions across datasets |

See [Annotation apps](#annotation-apps) above for `tools/annotator/` and
`tools/mobile_companion/`.

</details>

<details id="api-documentation">
<summary><b>API documentation</b></summary>

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

</details>
