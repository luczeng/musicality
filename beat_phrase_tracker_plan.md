# Beat Phase Detection — Implementation Plan

Goal: given an audio file, output frame-level detections for three things,
building entirely on top of the existing `musicality` package (no new
top-level package, no new pyproject, no external tracker dependencies):

1. **Beat** — every beat.
2. **"1"** — the downbeat (first beat of the 4/4 bar).
3. **"4"** — the last beat of the 4/4 bar (beat before the downbeat).

This supersedes the original 8-beat-phrase / backbeat / particle-filter design
below in scope: v1 targets bar-level position (1–4), not phrase-level (1–8),
and reuses the existing dataset/model/training scaffolding instead of a
parallel implementation. Particle-filter smoothing is kept only as an
optional later refinement (Section 6), not a hard requirement.

---

## 1. What already exists and needs no change

- **`musicality/dataformats/`** — data root / splits dir config loading. Reuse as-is.
- **`musicality/splits/splitter.py`** — persistent train/val split by file. Reuse as-is; works with any `Dataset`.
- **`musicality/trainers/train.py` + `tools/train.py`** — Hydra-driven training entry point, W&B logger, checkpoint/early-stop/callback wiring. The overall shape (`build_dataloaders` / `build_module` / `build_callbacks` / `build_trainer`) is the template for the new training entry point — not something to redesign.
- **mirdata's `BeatData.positions`** — bar-position annotations (1, 2, 3, 4, … per beat; 0 = outside a measure) already ship with several of our configured datasets. Confirmed present in `ballroom`, `hainsworth`, `gtzan_genre`, `beatles`, `brid` (checked against `mirdata.datasets.*` source). **This means no new annotation source is needed** — the original plan's "phrase annotations barely exist" problem doesn't apply at the bar level; `rwc_popular` lacks positions and should just be excluded from the beat-phase splits.
- **`configs/` Hydra layout** (`train.yaml`, `model/*.yaml`) — same pattern extends to the new task; no structural change.
- **Ruff formatting, aerated code style, commit conventions** — unchanged project-wide rules (see `CLAUDE.md`).

## 2. What already exists and needs modification

| File | Current state | What changes |
|---|---|---|
| `musicality/loaders/beat_dataset.py` (`BeatDataset`) | Loads waveform + hard 0/1 per-frame **beat-only** target from `track.beats.times`. Ignores `track.beats.positions` entirely. | Extend to also read `positions`, emit three aligned per-frame target arrays (beat / is-1 / is-4), and Gaussian-smear all three (currently a hard spike train). Tracks whose dataset never populates `positions` should mask the 1/4 heads rather than fail. |
| `musicality/models/tcn.py` (`TCNTempoNet`) | Dilated residual TCN, but **globally average-pools over time** before a scalar/bin regression head — output is one value per clip, not per frame. | Add a frame-level sibling model that reuses the same mel front-end + dilated conv trunk but skips the pooling step, ending in a `Conv1d(channels → 3, kernel_size=1)` + sigmoid instead of the pooled regression head. Share the trunk code rather than duplicating it. |
| `musicality/losses.py` | Has BCE-adjacent machinery (`gaussian_soft_target`, `classification_tempo_loss`) but nothing frame-wise or multi-head. | Add a masked, per-head frame-wise BCE loss (positive-class weighting, since beats/downbeats are a small fraction of frames; mask handles clips/datasets without position labels). |
| `musicality/trainers/tempo_module.py` (`TempoModule`) | Lightning module for scalar tempo regression/classification. | Not modified — instead add a sibling `BeatPhaseModule` following the same structure (`_step`, logging, `configure_optimizers`), since the task (frame-wise multi-head) doesn't fit the scalar-target abstraction cleanly. |
| `musicality/augmentations.py` (`TempoAugmenter` / `AugmentedDataset`) | `TimeStretch` rescales a scalar tempo label to match a resampled waveform; `AugmentedDataset.__getitem__` assumes `label.item()`. | Needs a frame-target-aware variant: when time-stretching the waveform, the per-frame target arrays must be resampled/re-indexed in step (gain/noise augmentations are unaffected and reusable as-is). |

## 3. What needs to be built from scratch

Nothing at the level of a new package — new pieces are all additions inside
`musicality/`:

- A metrics module (or additions to an existing one) for frame-activation → event evaluation: beat F-measure, downbeat ("1") F-measure, beat-4 F-measure, and a **"1-vs-3 confusion" rate** (the bar-level analogue of the original plan's beat-5 confusion metric — the failure mode where the model locks the right periodicity but the wrong phase parity).
- A peak-picking / event-readout step: per-frame sigmoid activations → discrete beat times with a position label. Simpler than the original particle filter: local-max peak-picking on `o_beat`, gated/refined by `o_1` and `o_4`, plus a light periodicity check (e.g. enforce roughly 4 beats between consecutive "1"s) rather than a full particle-filter state estimator.
- A small CLI/inspection script (mirroring `tools/inspect_track.py`) to run the trained model on a track and print/plot beat + "1"/"4" events.

## 4. Step-by-step build order

Implemented incrementally, one step per "next":

1. **Dataset**: extend `BeatDataset` (or add `BeatPhaseDataset` alongside it) to emit `(wav, {beat, is_one, is_four}, mask)` with Gaussian-smeared targets from `times` + `positions`.
2. **Model**: add the frame-level TCN variant (3-head sigmoid output) sharing the existing dilated conv trunk.
3. **Loss**: masked multi-head frame-wise BCE in `losses.py`.
4. **Lightning module**: `BeatPhaseModule` (mirrors `TempoModule`) with per-head loss logging and frame-level accuracy metrics.
5. **Config + training entry**: `configs/model/tcn_frames.yaml`, a `configs/beat_train.yaml`, and `tools/train_beat.py` (or a branch in the existing `train.py`) wired through `musicality/trainers/train.py`-style dataloader/callback builders.
6. **Augmentation fix**: make `TimeStretch`/`AugmentedDataset` target-aware for frame-level labels.
7. **Postprocessing**: peak-picking + periodicity-gated event readout → labeled beat list.
8. **Evaluation**: beat / "1" / "4" F-measure + 1-vs-3 confusion rate script.
9. **(Optional, later)** particle-filter-style temporal smoothing across a full track, only if peak-picking proves too jittery in practice.

## 5. Explicitly dropped or deferred from the original plan

- Separate `beat_phrase_tracker/` package, its own `pyproject.toml`, and CLI scripts under a new `scripts/` tree — everything lives in `musicality/` and `tools/`.
- 8-beat phrase state (`phi` over a full phrase, backbeats at 1/3/5/7) — replaced by plain 4-beat bar position. Revisit phrase-level structure only if bar-level "1"/"4" detection proves solid and there's a concrete need for it.
- `madmom_frontend.py` milestone-0 shortcut — unnecessary; we already have a working backbone (`TCNTempoNet`) and dataset pipeline to build on directly.
- Snare-band auxiliary feature and harmonic-novelty fallback — not needed at bar-level scope; the learned "1" head plus periodicity-gated readout is the first thing to try.
- Full numpy particle filter (state/dynamics/observation/resampling modules) — demoted to an optional stretch goal (Section 4, step 9) behind simpler peak-picking.
- Synthetic click-track test fixtures and the associated property tests — worth revisiting once peak-picking/postprocessing exists, not before.

## 6. Dependencies

No new dependencies expected beyond what's already in the project
(`torch`, `torchaudio`, `lightning`, `mirdata`, `hydra-core`, `wandb`). `mir_eval`
is the one likely addition, for standard beat/downbeat F-measure computation
in the evaluation step.
