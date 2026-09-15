# Frame metrics vs event metrics

Two families of number describe beat/bar-position quality in this repo, they
disagree, and the disagreement is not noise. This is the reference for what
each one measures, what they read when they were calibrated, and which to
steer by.

Measured on `merge_v4`, 60 tracks drawn with `np.random.RandomState(0)` from
the 249-track `beat_phase-merge-binary` val split, clean audio, decoder
`global` with `switch_penalty=2.0`. Frame rate `fps = 22050 / 512 = 43.07 Hz`,
hop `h = 23.22 ms`, Gaussian smear `σ = 1.5` frames, `group_size = 4`. Source:
`plans/06_metric_calibration_and_eval_consolidation.md` §1-2.

---

## High level

There is one pipeline, and each metric taps into it at a different stage.

```
audio
  |
  +-1-> model ---------------> per-frame probability curve     (one number per 23 ms)
  |                                     |
  +-2-> pick_peaks ----------> list of beat frames             (one entry per beat)
  |                                     |
  +-3-> gate_periodicity ----> cleaned beat list
  |                                     |
  +-4-> Viterbi bar decode --> list of (time, beat_in_bar)     <- drives a metronome
```

| metric | taps in at | unit | material |
|---|---|---|---|
| `frame_accuracy` | **1** | frame | 16 s clip |
| `peak_f_measure` — logged as `val/f_beat` | **2** | event | 16 s clip |
| `val/position_acc` | **1** | frame | 16 s clip |
| `f_beat`, `cmlt`, `amlt`, `position_acc` — `val_event/*` and `tools/eval_beat.py` | **4** | event | full track |

A **frame metric** scores the raw probability curve, one number per 23 ms slot.
An **event metric** scores a discrete list of beats, after the decoder has
committed to where they are.

The naming rule that follows from this: **the metric name says what is
measured, the prefix says how.** `val/position_acc` and
`val_event/position_acc` are the same quantity read at two different stages, so
they share a name and are told apart by namespace
(`musicality/trainers/train_beat_phase.py`).

**The short version:** frame metrics ask *"is this 23 ms slot labelled right?"*
on the easiest 16 seconds of the track. Event metrics ask *"would this play
back correctly?"* on the whole thing. The first is a cheap proxy for the
second, and it was silently wrong in four distinct ways.

---

## Technical

### Why they disagree — four independent causes

#### 1. Balanced accuracy structurally cannot see precision

`frame_accuracy(..., balanced=True)` is `½(TPR + TNR)`. There is no precision
term in it at all — and precision is the only term that actually moves.
Decomposed on the val clips:

| tolerance | TPR | TNR | balanced acc | precision | recall | frame F |
|---|---|---|---|---|---|---|
| ±0 frames (23 ms) | 0.915 | 0.799 | **0.857** | **0.487** | 0.915 | 0.636 |
| ±1 frame (46 ms) | 0.947 | 0.898 | 0.922 | 0.761 | 0.947 | 0.844 |
| ±2 frames (70 ms) | 0.957 | 0.947 | 0.952 | 0.886 | 0.957 | 0.920 |
| ±3 frames (93 ms) | 0.964 | 0.954 | 0.959 | 0.908 | 0.964 | 0.935 |

Recall was never the problem: it is already 0.915 at the strictest tolerance.
What holds the balanced figure at 0.857 is **precision 0.487** — the model
fires above 0.5 on roughly twice as many frames as the target marks positive.

Two things about this table are counter-intuitive enough to state outright.

- **Target-positive frames are 17.7% of the total, not ~3%.** The Gaussian
  smear is what makes the positive class large; it is not one frame per beat.
  A "negatives are ~97% of frames, so TNR is pinned near 1.0" argument is
  wrong here, and the measured TNR of 0.799 shows it. The blind spot is
  structural (no precision term), not a class-imbalance artefact.
- **A ±k band is `2k+1` frames wide.** So the band matching `mir_eval`'s
  ±70 ms half-window is ±3 frames, while the *width* matching it is ±1. The
  two readings get confused constantly.

#### 2. Width is not position

The flip side. The model's predicted peak is *wider* than the Gaussian target's
half-maximum band, which is what costs it that precision. `pick_peaks` keeps
only the local maximum, so the extra width is discarded entirely at inference —
it costs nothing where it matters and everything in the log. `frame_accuracy`
is being pessimistic about something real inference does not care about.

Frame metrics are therefore wrong in *both* directions, and the two errors
nearly cancel:

| candidate | clip value | vs `f_beat` 0.845 |
|---|---|---|
| balanced acc @ ±0 (the original logged metric) | 0.857 | +0.012 |
| balanced acc @ ±3 | 0.959 | +0.114 |
| frame F @ ±3 (blob overlap) | 0.935 | +0.090 |
| **peak-picked F @ 70 ms** (`peak_f_measure`, now logged) | **0.900** | **+0.055** |
| full-track `f_beat` (`mir_eval`) | 0.845 | — |

**This is why nobody caught it.** The old `acc_beat` sat 1.2 points from the
number it stood in for, by coincidence. A model that sharpened its peaks
without moving a single beat would have gained heavily on it and not at all on
`f_beat`.

#### 3. Different audio — the clip is the easy part of the track

Nothing to do with metric design. `BeatDataset` with `random_crop=False` takes
a fixed window from the track's **middle**, deliberately avoiding sparse
intros. So every `val/*` number is measured on the easiest 16 seconds of each
track, while every evaluated number covers intros, outros and breakdowns too.

| | clip (logged) | full track (evaluated) | gap |
|---|---|---|---|
| beat | 0.900 (peak-picked F) | 0.845 (`f_beat`) | 0.055 |
| position | 0.661 (`val/position_acc`) | 0.588 (head at ref beats) | 0.073 |

Both heads pay ~6-7 points, consistently. Worth stating plainly because it runs
opposite to the intuition that logged numbers are too harsh: **on material
selection they are too kind**, and it is the metric definition that is too
harsh.

#### 4. Two error modes have no frame-level expression at all

This is the one that motivated the whole exercise.

**Metrical level.** A model tracking cleanly at half-time is correct on every
frame it fires on; a frame metric barely registers it, and you would hear it
immediately. `mir_eval.beat.continuity` calls beat `i` correct when its own
error *and* the preceding inter-beat interval are both within `θ_c = 0.175`
relative — phase and local period must agree. `AMLt` maximises the same over
double-time, half-time and offbeat variants of the reference, so `AMLt ≥ CMLt`
always:

| | value |
|---|---|
| CMLt — correct metrical level, total | 0.665 |
| AMLt — any metrical level, total | **0.839** |
| `f_beat` | 0.845 |

`AMLt ≈ f_beat` says the beats are where *some* valid interpretation wants
them. `CMLt` 17 points lower says a substantial share of tracks are tracked
confidently at the wrong level. Completely different failure from mistiming,
completely different fix, and no frame metric can represent it — "consistent
grid at the wrong period" is a property of the event sequence, not of any
frame.

**Global anchor.** The bar numbering can be internally perfect and still start
counting on the wrong beat. For each reference beat `(t_i, p_i)`, take the
nearest predicted event `e_j` within `τ` with a resolved label `ℓ_j`:

```
o_i = (ℓ_j − p_i) mod G        h[o] = #{i : o_i = o}        N = Σ_o h[o]

position_acc             = h[0] / N
position_acc_best_offset = max_o h[o] / N
anchor_error             = position_acc_best_offset − position_acc  ≥ 0
```

`position_acc_best_offset` is the accuracy the model would have if allowed to
rotate its bar numbering by **one constant per track**, so the difference is
exactly the cost of a wrong global anchor — a failure the model could fix
without changing its grid at all:

| measurement | value |
|---|---|
| `val/position_acc`, frame-level, 16 s clip | 0.661 |
| position head argmax at reference beat frames, full track | 0.588 |
| decoded `position_acc` (detector + Viterbi) | **0.581** |
| decoded `position_acc_best_offset` | **0.684** |
| head argmax at reference beats, best rotation | 0.671 |
| `f_one` / `f_last` | 0.554 / 0.509 |
| `confusion_half_cycle_rate` | 0.260 |

Two readings fall out of this table.

- **The decoder is not the problem.** 0.588 raw → 0.581 decoded: the global
  Viterbi decoder costs 0.7 points.
- **~9 points is a wrong anchor.** 0.581 absolute against 0.671
  offset-invariant. A listener hears "it has the bar, it is phase-shifted", not
  "it is wrong 42% of the time" — the most likely reason inference sounds
  better than the number.

### Why the positive band is ±1 frame

The target is `y[t] = max_k exp(-(t - t_k)² / 2σ²)`, clipped to `[0, 1]`.
Thresholding at `θ = 0.5`:

```
exp(-d² / 2σ²) > 0.5   <=>   d² < 2σ² ln 2   <=>   |d| < σ·sqrt(2 ln 2) = 1.5 × 1.1774 = 1.766
```

so integer frames `|d| ≤ 1`: a **3-frame band, ±23.2 ms**, against `mir_eval`'s
±70 ms. The frame metric is **3× stricter** on the half-window.

Reaching ±3 frames by lowering the threshold instead would need
`θ = exp(-9 / 2σ²) = e^{-2} = 0.135` — but that also loosens what counts as a
*prediction*, which is a different change. Tolerance and decision threshold
have to be separate parameters, which is why `peak_f_measure` carries both.

### What the legacy position metrics cannot see

Kept for continuity with `plans/04` and `plans/05`, demoted as headline.

`confusion_half_cycle_rate` with `q = 1 + G/2 = 3`:

```
eligible  = { i : p_i ∈ {1, q} and matched with ℓ_j ∈ {1, q} }
confusion = #{ i ∈ eligible : ℓ_j ≠ p_i } / |eligible|
```

Blind by construction to **offsets 1 and 3** — a beat labelled 2 or 4 is
dropped from the denominator rather than counted wrong — and to **unmatched
beats**. For `G = 4` it observes only `h[2]`, and only over beats at positions
1 and 3. `position_acc` observes all of `h` over all beats.

`downbeat_f_measures` splits on `ref_positions == 1` and `== group_size`, so
positions `2..G−1` are invisible to it even though the softmax head predicts
them.

### Library choice

`mir_eval` only. It is already a dependency and already used in
`musicality/metrics/f_measure.py`; `beat.continuity` was the one thing missing
and is now wrapped by `musicality/metrics/continuity.py`.

**Do not add `madmom`**: not installed, needs Cython and a pinned old NumPy,
and its `DownbeatEvaluation` is F-measure over downbeat times, which
`downbeat_f_measures` already computes.

Deliberately not adopted from `mir_eval.beat`: `cemgil` (measured 0.784 — a
finer timing scale than 70 ms, tells us nothing new), `goto`, `p_score`,
`information_gain`. The goal is fewer numbers, not more.

---

## Which numbers to steer by

The canonical set is `musicality.evaluation.SCORE_KEYS`, reported by
`tools/eval_beat.py` and logged during training as `val_event/*` by
`musicality/callbacks/event_metrics.py`. Read it as two pairs plus leftovers:

```
BEAT
  f_beat                    did the beats land
  cmlt / amlt               did they land at the right metrical level
                            (amlt − cmlt = tracked confidently at a wrong one)
POSITION
  position_acc              is the bar numbering right
  position_acc_best_offset  is it at least consistent
    -> anchor_error         the difference: a grid correct except where it starts
  f_one / f_last            (demoted)
  confusion                 (demoted, kept for continuity with plans/04-05)
```

The frame metrics stay — they are free, they are computed every epoch, and
`peak_f_measure` in particular now moves for the same reasons `f_beat` moves.
They are a within-run progress signal. Anything quoted as a result should come
from the event set.

## See also

- `docs/half_cycle_rate_explained.md` — what `confusion_half_cycle_rate` measures.
- `docs/switch_penalty_explained.md` — the one tunable of the global decoder.
- `plans/06_metric_calibration_and_eval_consolidation.md` — the full record,
  including the tooling consolidation these numbers motivated.
