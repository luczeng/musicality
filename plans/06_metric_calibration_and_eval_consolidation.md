# Part 6 — Making the metrics mean what they say, and one tool to report them

**Status:** planned, not started. Session of 2026-09-05.
Follows `plans/05_beat_phase_overfitting.md`, whose §1 first flagged that the
logged accuracies are not comparable to the evaluated ones.

---

## Overview

### High level

The numbers in the training log, the numbers the evaluation tools print, and
what you hear when you listen to the output are three different things. The beat
tracking sounds better than its logged accuracy; the bar tracking sounds better
than its logged accuracy; and neither logged number moves for the reasons you
would expect. Every conclusion in `plans/04` and `plans/05` is expressed in these
units, so this is load-bearing rather than cosmetic.

Two separate causes. First, the training-time metrics measure the *shape* of the
model's output frame by frame, while the evaluation measures *where the beats
landed* after the output has been turned into a list of events. Those reward
different things, so a model can improve on one and not the other. Second,
evaluation is spread across three command-line tools that share their scoring
logic by copy-paste, and they have already drifted apart — one of them is
silently scoring a model configuration that is no longer trained.

The outcome: one list of numbers, one tool that reports it, and the same numbers
visible during training as after it. Along the way, two questions that currently
require a bespoke diagnostic run become part of the standard report — whether the
model is tracking at the wrong metrical level, and whether it found the bar but
started counting on the wrong beat.

### Technical

Reference checkpoint `merge_v4.ckpt`. All measurements below: 60 tracks drawn
with `np.random.RandomState(0)` from the 249-track `beat_phase-merge-binary` val
split, clean audio, decoder `global` with `switch_penalty=2.0`.

> **Not comparable to `plans/05` §3**, which used `switch=5` over all 247 tracks.
> These are internally consistent with each other, which is what a calibration
> argument needs.

---

## 1. What was measured

<details>
<summary><b>1.1 — The beat metric is two large biases cancelling</b></summary>

Frame-level, on the 16 s val clips, decomposed:

| tolerance | TPR | TNR | balanced acc | precision | recall | frame F |
|---|---|---|---|---|---|---|
| ±0 frames (23 ms) | 0.915 | 0.799 | **0.857** | 0.487 | 0.915 | 0.636 |
| ±1 frame (46 ms) | 0.947 | 0.898 | 0.922 | 0.761 | 0.947 | 0.844 |
| ±2 frames (70 ms)¹ | 0.957 | 0.947 | 0.952 | 0.886 | 0.957 | 0.920 |
| ±3 frames (93 ms) | 0.964 | 0.954 | 0.959 | 0.908 | 0.964 | 0.935 |

¹ the ±k band is `2k+1` frames wide; ±3 frames is a 7-frame span, so the *band*
that matches `mir_eval`'s 70 ms half-window is ±3 while the *width* that matches
it is ±1. Both are given because the two readings get confused constantly.

Target-positive frames are **17.7%** of the total, so negatives are 82.3% — not
the ~97% assumed earlier in this session.

**Recall was never the problem.** At the currently logged tolerance, recall is
already 0.915. What drags `acc_beat` to 0.857 is **precision 0.487**: the model
fires above 0.5 on roughly twice as many frames as the target marks positive,
because its predicted peak is *wider* than the Gaussian target's half-maximum
band. After `pick_peaks` that width is discarded entirely — only the local
maximum survives — so it costs nothing at inference and everything in the log.

The two errors run in opposite directions and nearly cancel:

| candidate | clip value | vs `f_beat` 0.845 |
|---|---|---|
| balanced acc @ ±0 (**currently logged**) | 0.857 | +0.012 |
| balanced acc @ ±3 | 0.959 | +0.114 |
| frame F @ ±3 (blob overlap) | 0.935 | +0.090 |
| **peak-picked F @ 70 ms** | **0.900** | **+0.055** |
| full-track `f_beat` (mir_eval) | 0.845 | — |

**This is why nobody caught it.** `acc_beat` currently sits 1.2 points from the
number it stands in for, by coincidence. A model that sharpened its peaks without
moving a single beat would gain heavily on `acc_beat` and not at all on `f_beat`.

The residual +0.055 on the peak-picked variant is **material selection, not metric
definition** — see §1.3.

</details>

<details>
<summary><b>1.2 — Nine points of the position error is a wrong anchor</b></summary>

| measurement | value |
|---|---|
| `acc_position`, frame-level, 16 s clip — **what W&B logs** | 0.661 |
| position head argmax at reference beat frames, full track | 0.588 |
| decoded `position_acc` (detector + Viterbi) | **0.581** |
| decoded `position_acc_best_offset` — best single rotation per track | **0.684** |
| head argmax at reference beats, best rotation | 0.671 |
| `f_one` / `f_last` | 0.554 / 0.509 |
| `confusion_half_cycle_rate` | 0.260 |

**The decoder is not the problem.** 0.588 raw → 0.581 decoded: the global Viterbi
decoder costs 0.7 points. The question `position_accuracy.py` was written to answer is
settled, and `plans/04`'s decoder work can be considered closed.

**~9 points is a wrong global anchor.** 0.581 absolute against 0.671
offset-invariant. The model found a consistent bar grid and started it on the
wrong beat. A listener hears "it has the bar, it is phase-shifted", not "it is
wrong 40% of the time" — the most likely reason inference sounds better than the
number. **The best-offset number is therefore not a useless metric; it is the one that
explains the gap.** It is currently computed only inside
`tools/diagnose_beat_phase.py` and never reported.

</details>

<details>
<summary><b>1.3 — The logged val numbers are optimistic, not pessimistic</b></summary>

`BeatDataset` with `random_crop=False` takes a fixed window from the track's
*middle*, deliberately avoiding intros (`configs/beat_train.yaml:125`). So every
`val/*` number is measured on the easiest 16 seconds of each track, while every
evaluated number covers the whole thing including intros, outros and breaks.

| | clip (logged) | full track (evaluated) | gap |
|---|---|---|---|
| beat | 0.900 (peak-picked F) | 0.845 (`f_beat`) | 0.055 |
| position | 0.661 (`acc_position`) | 0.588 (head at ref beats) | 0.073 |

Both heads pay ~6–7 points for it, consistently. Worth stating plainly because it
runs opposite to the intuition that the logged numbers are too harsh: on material
selection they are too kind, and it is the *metric definition* that is too harsh.

</details>

<details>
<summary><b>1.4 — 17 points of beat error is metrical level, not timing</b></summary>

From `mir_eval.beat.continuity`, which nothing in the repo currently calls:

| | value |
|---|---|
| CMLt — correct metrical level, total | 0.665 |
| AMLt — any metrical level, total | **0.839** |
| `f_beat` | 0.845 |

`AMLt ≈ f_beat` says the model's beats are where *some* valid interpretation
wants them. `CMLt` being 17 points lower says a substantial share of tracks are
tracked confidently at half-time, double-time or on the offbeat. That is a
completely different failure from mistiming, with a completely different fix, and
no existing metric can see it.

</details>

<details>
<summary><b>1.5 — Library choice: mir_eval only</b></summary>

`mir_eval` 0.8.2 is already a dependency (`pyproject.toml:16`) and already used
in `musicality/metrics/f_measure.py`. `beat.continuity` supplies CMLc/CMLt/AMLc/AMLt
and is the only thing missing.

**Do not add `madmom`.** It is not installed, needs Cython and a pinned old NumPy,
and its `DownbeatEvaluation` is F-measure over downbeat times — which
`downbeat_f_measures` already computes. `mirdata` has no evaluation module worth
using and is on its way out (`plans/03`).

Deliberately not adopted from `mir_eval.beat`: `cemgil` (measured 0.784 — a finer
timing scale than 70 ms, tells us nothing new), `goto`, `p_score`,
`information_gain`. The goal is fewer numbers, not more.

</details>

---

## 2. The mathematics

Frame rate `fps = 22050 / 512 = 43.066 Hz`, hop `h = 23.22 ms`. Gaussian smear
`σ = 1.5` frames, peak 1.0, group size `G = 4`.

<details>
<summary><b>2.1 — Why the current positive band is ±1 frame</b></summary>

The target is `y[t] = max_k exp(-(t - t_k)² / 2σ²)`, clipped to `[0, 1]`.
Thresholding at `θ = 0.5`:

```
exp(-d² / 2σ²) > 0.5   ⟺   d² < 2σ² ln 2   ⟺   |d| < σ·√(2 ln 2) = 1.5 × 1.1774 = 1.766
```

so integer frames `|d| ≤ 1`: a **3-frame band, ±23.2 ms**. `mir_eval` matches
within **±70 ms = ±3.01 frames**, i.e. a 7-frame band. The current metric is
**3× stricter** on the half-window.

To reach ±3 frames by lowering the threshold instead you would need
`θ = exp(-3² / 2σ²) = e^{-2} = 0.135` — but that also loosens what counts as a
*prediction*, which is not the same change. Tolerance and decision threshold must
be separate parameters.

</details>

<details>
<summary><b>2.2 — Balanced accuracy, and why it flatters</b></summary>

With `P = {t : p[t] > θ}` and `T = {t : y[t] > θ}`:

```
TPR = |P ∩ T| / |T|        TNR = |Pᶜ ∩ Tᶜ| / |Tᶜ|        balanced = ½(TPR + TNR)
```

The floor for a model that never fires is `½(0 + 1) = 0.5`, so the whole scale
lives in `[0.5, 1]` and half of every reported figure is unconditional. Measured
TNR is 0.80–0.95 depending on tolerance, so the metric is roughly
`0.5 + TPR/2 ± 0.1` — and it never sees precision at all, which §1.1 shows is the
only term actually moving.

</details>

<details>
<summary><b>2.3 — Frame F-measure (proposed), and its ceiling</b></summary>

Let `D_τ(S) = {t : ∃ s ∈ S, |t − s| ≤ τ}` be dilation by τ frames, computed as
`F.max_pool1d(x, 2τ+1, stride=1, padding=τ)`.

```
recall    = |D_τ(P) ∩ T| / |T|
precision = |P ∩ D_τ(T)| / |P|
F         = 2·precision·recall / (precision + recall)
```

Dilating *one* side per term is what makes this a tolerance rather than a
smoothing: dilating both would inflate `T` and `P` together and, as measured
earlier this session, actually *lowers* the score.

This fixes the true-negative inflation but not the width penalty — a correctly
centred blob twice as wide as the target still loses precision. Measured 0.935 at
τ=3 against `f_beat` 0.845.

</details>

<details>
<summary><b>2.4 — Peak-picked F, the one that tracks reality</b></summary>

Run the same `pick_peaks(probs, threshold, min_distance)` that inference runs,
on both the prediction and the target, then score the resulting event lists with
`mir_eval.beat.f_measure` at 70 ms:

```
est = pick_peaks(p, θ, d) / fps        ref = pick_peaks(y, θ, d) / fps
F   = mir_eval.beat.f_measure(ref, est, 0.07)
```

Peak width cancels, because only the local maximum survives on both sides. This
is the same operation the evaluation performs, so the number is in the same
units. Measured **0.900** on clips against **0.845** on full tracks — and §1.3
accounts for the remaining 0.055 as material selection.

`pick_peaks` is `O(T)` NumPy; at batch 32 × 689 frames the per-step cost is
negligible next to the forward pass.

</details>

<details>
<summary><b>2.5 — Position accuracy and the anchor decomposition</b></summary>

For each reference beat `(t_i, p_i)`, take the nearest predicted event `e_j` with
`|time(e_j) − t_i| ≤ τ` and a resolved label `ℓ_j`; record the offset

```
o_i = (ℓ_j − p_i) mod G           h[o] = #{i : o_i = o}          N = Σ_o h[o]
```

Then

```
position_acc             = h[0] / N
position_acc_best_offset = max_o h[o] / N
anchor_error             = position_acc_best_offset − position_acc  ≥ 0
```

`position_acc_best_offset` is the accuracy the model would have if you were
allowed to rotate its bar numbering by **one constant per track**. So the
difference is exactly the cost of choosing the wrong global anchor — a
representation failure the model could fix without changing its grid at all.
`h` is the histogram `position_accuracy` already returns.

</details>

<details>
<summary><b>2.6 — What confusion and f_one/f_last cannot see</b></summary>

`confusion_half_cycle_rate` with `q = 1 + G/2 = 3`:

```
eligible = { i : p_i ∈ {1, q} and matched with ℓ_j ∈ {1, q} }
confusion = #{ i ∈ eligible : ℓ_j ≠ p_i } / |eligible|
```

Blind by construction to **offsets 1 and 3** (a beat labelled 2 or 4 is dropped
from the denominator rather than counted wrong) and to **unmatched beats**. For
`G = 4` it observes only `h[2]`, and only over the subset of beats at positions 1
and 3. `position_acc` observes all of `h` over all beats.

`downbeat_f_measures` splits on `ref_positions == 1` and `== group_size`, so
positions `2..G−1` are invisible to it even though the softmax head predicts
them. This is why both are demoted rather than kept as headline.

</details>

<details>
<summary><b>2.7 — Continuity</b></summary>

`mir_eval` calls beat `i` correct when both its own error and the preceding
inter-beat interval fall within a relative tolerance `θ_c = 0.175` of the
reference — phase *and* local period must agree.

- `CMLt` — total fraction correct against the reference as annotated.
- `AMLt` — the same, maximised over reference variants: double-time, half-time
  and offbeat.

Hence `AMLt ≥ CMLt` always, and `AMLt − CMLt` is the share of the track tracked
consistently **but at the wrong metrical level**. Structurally the same
decomposition as §2.5 does for bar phase: an absolute score, an
invariance-forgiving score, and the gap between them as the named quantity.

</details>

---

## 3. The canonical metric set

One list, one order, everywhere beat/phase quality is reported.

```
BEAT
  f_beat                    0.845   mir_eval F-measure @ 70 ms      <- headline
  cmlt / amlt         0.665 / 0.839  metrical-level split           <- new
    -> level_error          0.174   amlt - cmlt
POSITION
  position_acc              0.581   correct bar label at ref beats  <- headline
  position_acc_best_offset  0.671   best single rotation per track
    -> anchor_error         0.090
  f_one / f_last      0.554 / 0.509  (demoted)
  confusion                 0.260   (demoted, kept for continuity)
```

**Pruned:** `modal_fraction` (byte-identical duplicate of the best-offset value in the same
dict); `BASE_VARIANTS` (`diagnose_beat_phase.py:66`, dead constant); the three
copies of `_fmt`/`_mean` (`evaluation.py:105,110`,
`diagnose_beat_phase.py:72,78`, `sweep_beat_postprocess.py:64,148`) collapse to one.

**Not pruned:** the best-offset value — §1.2 makes it load-bearing, not obsolete;
`confusion_half_cycle_rate` — every number in `plans/04` and `plans/05` is in
these units; `f_one`/`f_last`. `acc_one`/`acc_last` in `_TRACKED_KEYS`
(`train_beat_phase.py:21`) are already no-ops under `target_layout: positions`
and `BestMetricsPrinter` skips them silently — left alone for the `one_last`
layout.

---

## 4. Plan

### Phase A — the metric set

| file | change |
|---|---|
| `musicality/metrics/continuity.py` | **new.** `beat_continuity(ref_times, est_times, trim=True) -> dict \| None` with `cmlc/cmlt/amlc/amlt`. Thin wrapper over `mir_eval.beat.continuity`; returns `None` when either sequence has <2 beats after `trim_beats` — mir_eval warns and returns zeros otherwise, which would average in as a real score |
| `musicality/metrics/frame_accuracy.py` | add `peak_f_measure(probs, target, fps, threshold, min_distance, tolerance)` (§2.4) and `frame_f_measure(probs, target, mask=None, threshold=0.5, tolerance_frames=3)` (§2.3). Keep `frame_accuracy` — `beat_module.py` and tests use it — but stop logging it |
| `musicality/metrics/phase_offset.py` | **shipped, and went further than planned:** renamed to `position_accuracy.py`, `phase_offset_profile()` to `position_accuracy()`. Dropped `modal_fraction`; renamed `correct_fraction` → `position_acc` and `stability` → `position_acc_best_offset` rather than only documenting the mapping, so `anchor_error = position_acc_best_offset - position_acc` reads as two accuracies rather than a consistency minus an accuracy. `tests/test_phase_offset.py` → `tests/test_position_accuracy.py` |
| `musicality/metrics/f_measure.py` | docstring only: record that `downbeat_f_measures` scores positions 1 and `group_size` only (§2.6) |
| `musicality/trainers/beat_phase_module.py` | log `{stage}/f_beat` from `peak_f_measure` in place of `{stage}/acc_beat`. **New key, not a redefinition** — old runs keep `acc_beat` meaning what it meant |
| `musicality/trainers/beat_module.py` | same swap for the beat-only task |
| `musicality/trainers/train_beat_phase.py` | `_TRACKED_KEYS`: `acc_beat` → `f_beat` |

### Phase B — one scoring path in `musicality/evaluation.py`

- `score_events(beat_times, positions, has_positions, events, *, tolerance, trim, group_size) -> dict`
  — the single place every metric is computed, returning the §3 dict plus
  `corpus`. Replaces the reimplementations at `diagnose_beat_phase.py:120-177`
  and `sweep_beat_postprocess.py:117-151`.
- `BeatEvaluator.score(*, decoder, switch_penalty, advance) -> list[dict]` — runs
  off the existing `compute_track_probs()` cache → `postprocess.readout` →
  `score_events`. One model pass, N decoder configurations.
- Move `summarize(rows)` and `group_by_corpus(rows)` from
  `diagnose_beat_phase.py:180,218` into `evaluation.py`, extended to the new keys.
- `BeatEvaluator.run()` delegates to `.score()`. `evaluate_track` keeps its
  signature — `tests/test_evaluation.py:68-103` asserts on positional
  `args[11]`..`args[15]` — but its body becomes `run_inference` + `score_events`.
- Fold the 36-line `X if self.X is not None else task_defaults[...]` chain
  (`evaluation.py:297-333`) into one helper.
- Keep `DEFAULTS` / `DATA_DIR` names and top-level YAML keys stable:
  `tools/annotator/main_window.py:1092` reads `EVAL_DEFAULTS` to drive the GUI.

### Phase C — one CLI

`tools/eval_beat.py` becomes the only entry point:

| mode | replaces |
|---|---|
| *(default)* canonical report, per-track lines + summary | `eval_beat.py` |
| `--per-genre` (auto when >1 corpus) | `diagnose_beat_phase.py` PER-GENRE BREAKDOWN |
| `--decoders [--switch-penalties …] [--advance …] [--rank-by …]` | `diagnose_beat_phase.py` DECODER COMPARISON + VERDICT |
| `--profile` | `diagnose_beat_phase.py` PHASE-OFFSET PROFILE |
| `--sweep` | `sweep_beat_postprocess.py` |
| `--output <csv>` | `diagnose_beat_phase.py --output` |

**Delete** `tools/diagnose_beat_phase.py` and `tools/sweep_beat_postprocess.py`.
Fold `configs/sweep_beat_postprocess.yaml` into `configs/eval_beat.yaml` under a
`sweep:` key. Reconcile the default drift: `--split` is `val` in `eval_beat.py`
and `train` in `diagnose_beat_phase.py` — keep `val`.

> **Latent bug, fixed by construction.** `sweep_beat_postprocess.py:121-131`
> hardcodes `probs[0], probs[1], probs[2]` and never passes
> `decoder`/`switch_penalty`/`position_probs`, so it has been sweeping the
> **greedy** decoder against a two-sigmoid head — neither of which the current
> `target_layout: positions` config trains. Routing `--sweep` through
> `BeatEvaluator.score()` removes the divergence. **The tuned values in
> `configs/eval_beat.yaml` came out of that path and must be re-swept**; treat
> them as unverified until they are.

### Phase D — event metrics during training

`musicality/callbacks/event_metrics.py` — `EventMetricsLogger(Callback)`:

- Holds a fixed list of val `TrackRef`s chosen deterministically and **stratified
  across corpora** — the split file is corpus-ordered, so taking the first N
  yields all-ballroom (`plans/05` §6).
- `on_validation_epoch_end`, gated on `every_n_epochs`; skips
  `trainer.sanity_checking`.
- Full-track inference → `readout` → `score_events`, logged as
  `val_event/f_beat`, `val_event/cmlt`, `val_event/amlt`,
  `val_event/position_acc`, `val_event/position_acc_best_offset`.
- **Must not log any key named `val/loss`** — `ReduceLROnPlateau` and
  `ModelCheckpoint` both monitor it and it is spliced into checkpoint filenames
  (`common.py:165`).
- `_LOWER_BETTER` (`metrics_logger.py:22`) is a substring match on
  `("loss", "mae")`; every new key is higher-is-better, so it is already correct.
  If `confusion` is ever added to the logged set it must be added there too.
- Config: an `event_metrics:` block in `configs/beat_train.yaml`
  (`enabled: true`, `n_tracks: 50`, `every_n_epochs: 5`), wired into
  `train_beat_phase.py:build_callbacks`. Add the five keys to `_TRACKED_KEYS`.

This closes the loop: the number quoted in a report and the number on the W&B
chart become the same number, measured on the same material.

### Phase E — docs

- `docs/source/metrics.rst` — add `~musicality.metrics.continuity`.
- `CLAUDE.md` — the `tools/` list names `diagnose_beat_phase.py`; replace with
  the consolidated `eval_beat.py` and its modes, and add `continuity.py` under
  `metrics/`.
- `docs/source/workflows.rst:88-137` — the eval/postprocess narrative names the
  deleted tools.
- Record §1 and §2 in `docs/` so the relationship between 0.857, 0.900 and 0.845
  does not have to be rediscovered.
- Stale references to fix while nearby: `beat_phrase_tracker_plan.md:61` names
  `tools/eval_beat_phase.py` and `musicality/metrics.py`, neither of which exists.

---

## 5. Order

1. **A** — metrics. Self-contained, unit-testable, nothing depends on it yet.
2. **B** — `score_events` + `BeatEvaluator.score`. Both existing tools keep working.
3. **C** — collapse the CLIs, delete two files, re-sweep the postprocess defaults.
4. **D** — the training callback, which needs B.
5. **E** — docs.

---

## 6. Verification

```bash
uv run pytest tests/ -q
uv run ruff format musicality/ tools/ tests/
```

Tests needing updates: `tests/test_sweep_beat_postprocess.py` (retarget at
`--sweep`), `tests/test_evaluation.py` (mocks `evaluate_track`, asserts positional
args), `tests/test_per_genre_eval.py`, `tests/test_position_accuracy.py` (drops
`modal_fraction`), `tests/test_metrics.py` (add `frame_f_measure`,
`peak_f_measure`, `beat_continuity`).

**Regression — the consolidation must not move any number.** `confusion` under
the old and new paths must agree on the same tracks:

```bash
uv run python tools/eval_beat.py --checkpoint checkpoints/merge_v4.ckpt \
    --dataset merge --split val --binary-only --decoders --per-genre
```

`--binary-only` is mandatory; without it ~58% of "val" is training data
(`plans/04` §2.1). `--checkpoint` is a **named flag** — passed positionally it
exits on an argparse error that a piped `tail` swallows, reporting false success.

**Calibration — the new numbers must reproduce.** On the seed-0 60-track subsample
with `--decoder global --switch-penalty 2.0`: `f_beat` 0.845, `cmlt` 0.665,
`amlt` 0.839, `position_acc` 0.581, `position_acc_best_offset` 0.671,
`confusion` 0.260.

**Training metric** — `peak_f_measure` on merge-val clips should land near 0.900,
i.e. within ~0.06 of `f_beat`, with the residual attributable to §1.3's clip
selection rather than to the metric.

---

## 7. Loose ends

- The clip-vs-full-track bias (§1.3) is a property of `random_crop=False` taking
  the track's middle. Worth deciding separately whether val should sample the
  whole track; it would cost ~6 points on every logged number and make them
  comparable to the evaluated ones for free.
- `musicality/callbacks/error_plot.py:37-39` reimplements `tempo_acc1`'s MIREX
  logic inline instead of calling it. Unrelated to the beat path; noted while
  mapping the metrics package.
- `musicality/metrics/` has no `__init__.py` (implicit namespace package).
  Consistent with the rest of the repo, so left alone.
- `plans/05` §1.1's claim that the 3-frame band "is exactly mir_eval's matching
  tolerance" conflates band width with half-window — see §2.1. Its conclusion
  (the metric is ~3× stricter) stands; the arithmetic behind it did not.
