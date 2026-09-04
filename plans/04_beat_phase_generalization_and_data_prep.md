# Part 4 — Beat-phase: the generalization wall, and the road to a general-purpose model

**Status:** analysis complete, nothing implemented. Sessions of 2026-09-03/04.
Follows `docs/beat_phase_improvement_review.md`, whose steps 0–3 are now all
run and measured.

**Scope change (2026-09-04):** the goal is now a **general tool across pop,
jazz, classical and dance** — not a ballroom-tuned model. That reframes the
metric, the sampling strategy, and the loss calibration, and it surfaces one
structural blocker (§2.7). Recommendations below reflect the new goal.

---

## Overview

**The story in five lines:**

1. Steps 2 and 3 worked. They eliminated the fit deficit on ballroom almost
   entirely — train confusion **0.090 → 0.007** — and moved the whole
   remaining error into generalization.
2. That ruled out capacity and a richer mel front-end **for the ballroom
   regime**: with nothing left to fit, both can only widen the gap.
3. Adding data (merge: +jtd, +rwc) put the model back into an **underfit**
   regime — train confusion 0.176 on ballroom, 0.221 on jtd. So capacity is
   back on the table, for the first time.
4. That merge comparison is **not yet trustworthy**: it was measured at 14% of
   its training schedule and changed two variables at once.
5. The "general tool" goal makes **variable meter** the gating problem —
   `binary_only: true` discards 36% of rwc_classical, and `group_size=4`
   cannot represent 3/4 at all.

**Where each lever stands:**

| Lever | Verdict | Why |
|---|---|---|
| Per-genre metric | **Measured, not shipped** | Prototype gave macro confusion 0.311 vs micro 0.283, classical `f_last` 0.066 — then reverted |
| Self-calibrating `pos_weight` | **Required** | No single scalar serves 56–193 BPM |
| Per-item loss normalization | **Required** | Micro-average over beats favours fast genres; metric is macro over tracks |
| Corpus-temperature sampling | **Recommended, α≈0.5** | Track-count gives jazz 65.5% |
| Variable meter support | **Gating for classical** | Structural; nothing else fixes it |
| Model capacity | **Re-opened** | Ruled out at train confusion 0.007; now 0.176–0.221 |
| Richer mel front-end | **Still no** | Fit-improving; not the binding constraint |
| Global duration-proportional sampling | **Rejected** | Would cut ballroom 24%→6% and barely move jtd |
| Chasing `train/acc_beat` = 90% | **No** | It's a ceiling, not a limit — val F exceeds train F |

**Current measurements** — all on `ballroom-binary` (419/104), `viterbi=2`:

| | ballroom-trained (`epoch109`) | merge-trained (`epoch57`) |
|---|---|---|
| train confusion | **0.007** | 0.176 |
| val confusion | **0.130** | 0.260 |
| val f_one / f_last | 0.774 / 0.769 | 0.623 / 0.552 |
| regime | memorizing | **underfitting** |

---

## 1. Context

<details>
<summary><b>1.1 — Where this came from</b></summary>

`docs/beat_phase_improvement_review.md` laid out steps 0–5 for fixing
bar-position estimation. Steps 0 and 1 (diagnostic, global Viterbi decoder)
landed 2026-08-29; step 2 (beat-conditioned phase loss) and step 3
(`group_size`-way softmax over positions) have now both been trained and
measured.

The doc's own error decomposition after step 1 was: decoder loss 6.8 pts, fit
deficit 9.0 pts, generalization gap 9.3 pts. This part re-measures that split
after steps 2 and 3, and it has inverted.

</details>

<details>
<summary><b>1.2 — The goal changed: general tool, not a ballroom model</b></summary>

Everything before 2026-09-04 implicitly optimized ballroom val, because that
was the only benchmark with a clean split. The stated goal is now a model that
works across **pop, jazz, classical and dance**.

Three consequences, developed in §2.7:

- The headline metric must become **macro-averaged across genres** (plus
  worst-genre), not a single blended number.
- Sampling must balance **corpora**, not track counts or audio hours.
- Loss calibration stops being tunable — with tempo spanning 56–193 BPM there
  is no correct single `pos_weight`.

And it promotes one previously-cosmetic issue to a blocker: the model cannot
represent triple meter.

</details>

---

## 2. Analysis

<details>
<summary><b>2.1 — A measurement bug that invalidated the first round of conclusions</b></summary>

**Every ballroom evaluation must pass `--binary-only`.**
`configs/beat_train.yaml` sets `binary_only: true`, and `beat_split_name()`
(`musicality/loaders/beat_dataset.py:295`) appends `-binary` to the split
directory name. The flag selects between two splits that were
**independently randomized, not nested**:

| split | train | val |
|---|---|---|
| `beat_phase-ballroom-binary` (what training uses) | 419 | 104 |
| `beat_phase-ballroom` (default without the flag) | 559 | 139 |

Measured: **81 of the 139** tracks in `beat_phase-ballroom/val.txt` sit in
`beat_phase-ballroom-binary/train.txt`. A no-flag "val" run therefore scores a
set that is ~58% training data.

The flag reads as a pure filter, which suggests the smaller set is a subset of
the larger. It isn't. Nothing errors — the only signal is the
`[Splitter] Loaded existing split ...` line. **Check it says `-binary` with
419/104 before trusting any ballroom number.**

The step-2 write-up's "`f_last` regressed 0.730 → 0.611" was an artifact of
exactly this; on the correct split `f_last` improved.

Verified clean for cross-run comparison: merge-train ⊇ ballroom-train, and
merge-train ∩ ballroom-val = ∅.

</details>

<details>
<summary><b>2.2 — Step 3 solved the fit problem; the wall is now generalization</b></summary>

Checkpoint `beat-phase-epoch109-valloss1.2181.ckpt`, ballroom, `--binary-only`.

| decoder | f_one | f_last | confusion | stability |
|---|---|---|---|---|
| **val (104)** | | | | |
| greedy (anchor=0.8) | 0.748 | 0.736 | 0.173 | 0.786 |
| global (exact, no resync) | 0.666 | 0.653 | 0.203 | 0.734 |
| global + viterbi (switch=1) | 0.772 | 0.761 | 0.132 | 0.854 |
| global + viterbi (switch=2) | 0.774 | 0.769 | 0.130 | 0.862 |
| global + viterbi (switch=5) | 0.769 | 0.768 | **0.125** | 0.861 |
| **train (419)** | | | | |
| greedy (anchor=0.8) | 0.906 | 0.907 | 0.048 | 0.926 |
| global (exact, no resync) | 0.805 | 0.801 | 0.092 | 0.821 |
| global + viterbi (switch=2) | 0.928 | 0.927 | **0.007** | 0.988 |
| global + viterbi (switch=5) | 0.925 | 0.920 | 0.010 | 0.978 |

Progress across stages (val, `viterbi=2`):

| stage | confusion | f_one | f_last |
|---|---|---|---|
| pre-step-2 baseline | 0.185 | 0.756 | 0.730 |
| step 3 (current) | **0.130** | **0.774** | **0.769** |

`switch_penalty` 2 and 5 are within noise; the shipped default of 2.0 holds.

**The decomposition inverted:**

| | pre-step-2 | now |
|---|---|---|
| cannot fit even on train | 9.0 pts | **0.7 pts** |
| generalization gap | 9.3 pts | **12.3 pts** |

Train confusion 0.007 with **100% of train tracks at the correct modal phase**
means the model fully memorizes its training set. Corroborated by W&B:
`acc_position` 98%/80% (18-pt gap) vs `acc_beat` 90%/87% (3-pt gap).

</details>

<details>
<summary><b>2.3 — Capacity and mel resolution: ruled out for ballroom, re-opened for merge</b></summary>

**In the ballroom regime, both were ruled out**, and the argument was sound:
capacity helps only when training performance is the limit, and it was 0.007 /
100%. The model is 2.40M params against 419 tracks (~5,700 per track); more
capacity could only widen a 12.3-pt gap. A richer mel front-end is also a
fit-improving intervention, and near-perfect train performance proves the
current representation already carries the needed information. (43 fps / 23 ms
is well inside mir_eval's 70 ms tolerance, so time resolution isn't binding.)

Measured: **2.40M** params with `use_self_attention: true`, **1.61M** without —
the SSA block is 0.79M, a third of the model.

**In the merge regime that argument no longer holds.** Train confusion is
0.176–0.221 (§2.5) against 55.5 h of audio and 1737 tracks — 16× the audio,
same 2.4M params. The fit deficit is back, so **capacity is a live candidate
again**. It should still come after the cheap calibration fixes and after
confirming the run is actually converged.

**`train/acc_beat` = 90% remains a red herring** regardless of regime. The
review's step-0 table has beat F **train 0.900 / val 0.916** — val above train.
A metric where validation beats training is not a fit problem; 90% is the
balanced-frame-accuracy ceiling from the smeared target (σ=1.5 frames,
thresholded at 0.5) plus annotation jitter. Step 1b showed beat errors explain
only ~14% of phase drift.

</details>

<details>
<summary><b>2.4 — The merge comparison is confounded and premature</b></summary>

Both runs are `tools/sweep_lr.py` at lr 6e-4, same commit (`69ecc58`),
`target_layout: positions`, `batch_size: 64`:

| | ballroom run | merge run |
|---|---|---|
| checkpoint | `epoch109-valloss1.2181` | `epoch57-valloss1.5552` |
| `data.input` | ballroom | merge |
| `n_train` / `n_val` | 419 / 104 | 1670 / 233 |
| steps/epoch | `ceil(419/64)` = 7 | `ceil(1670/64)` = 27 |
| global_step at ckpt | 770 | 1566 |
| **noise augmentation** | **off** | **on** |
| `max_epochs` | 200 | 400 |
| epochs at ckpt | 110 (55% of schedule) | 58 (**14%**) |
| per-track views | 110 | **58** |

**Three confounds, so this settles nothing:**

1. **Opposite regimes.** Ballroom memorizes (train 0.007, 12.3-pt gap); merge
   never fits (train 0.176, 8.4-pt gap) — worse than even the pre-step-2 train
   figure of 0.090.
2. **Half the per-track exposure**, and only 14% through its own schedule. Per
   the W&B curves the merge run's train loss is still descending at 1566 steps
   and doesn't plateau until ~4k.
3. **Noise augmentation differs** — two variables changed in one experiment.

Also: `valloss1.5552` vs `valloss1.2181` is a **meaningless comparison** —
different val sets.

**Outstanding:** get the converged (final-epoch) merge checkpoint and
re-measure. That the best-val-loss checkpoint sits at epoch 57 of a completed
400-epoch run is itself worth understanding — val loss plateaued early while
train kept improving.

</details>

<details>
<summary><b>2.5 — The merge model underfits <i>uniformly</i> — not a jtd problem</b></summary>

Merge checkpoint scored on data it **trained on**, `viterbi=2`:

| subset | confusion | f_one | f_last | stability |
|---|---|---|---|---|
| ballroom train (419) | 0.176 | 0.652 | 0.606 | 0.727 |
| jtd train (150 of 1108) | 0.221 | 0.652 | 0.646 | 0.739 |
| *ballroom-only model, ballroom train* | *0.007* | *0.928* | *0.927* | *0.988* |

It is **not** drowning in jtd while ballroom stays fine, and jtd is **not**
impossible. It underfits everything roughly equally.

This rules out a data-mix explanation for the merge result, and points at
either undertraining (§2.4) or capacity (§2.3) — not at rebalancing. Supporting
signs: Viterbi buys only 13% relative on val here (vs 28% for the ballroom
model), and the exact no-resync decoder nearly ties greedy (0.294 vs 0.297),
meaning there is little coherent per-beat evidence to decode. 59% of *training*
tracks flip phase mid-track.

**Consequence for sampling:** rebalancing an underfit model just moves the
failure around. Sampling changes *what* it sees; this model can't fit what it
already sees.

</details>

<details>
<summary><b>2.6 — Data-preparation defects (measured)</b></summary>

**(a) `sigma_frames` is constant while tempo varies 3.5×.**
σ = 1.5 frames = 35 ms, constant (`beat_dataset.py:265`). The target exceeds
0.5 for ±1.77 frames (±41 ms) and contributes a fixed 3.76 frames of mass per
beat — but beats-per-second is not fixed:

| corpus | BPM | beat period | positive window as % of period | true neg:pos |
|---|---|---|---|---|
| jtd | 193 | 13.4 fr | 26% | 2.6 |
| ballroom | 125 | 20.7 fr | 17% | **4.5** |
| rwc_classical (median) | 105 | 24.6 fr | 14% | 5.5 |
| rwc_classical (p10) | 56 | 46.1 fr | 7.7% | 11.3 |

Important nuance: constant-in-time σ is **correct for the beat head** — it
encodes annotation tolerance, which is absolute (mir_eval uses a flat 70 ms for
the same reason). It is **wrong for the position block**, where "which beat of
the bar is this" is a per-beat property whose natural yardstick is the beat
period. Making σ period-proportional *everywhere* would give ±90 ms at 56 BPM —
sloppier than the metric that grades it.

Two consequences:

1. **`pos_weight` is calibrated for one tempo.** `beat_train.yaml` records the
   ballroom ratio as 4.3 and sets 5. On slow classical the true ratio is 11.3
   (under-fires); on jtd it is 2.6 (over-fires) — and jtd is 64% of merge.
2. **Slow music gets less gradient per second.** `beat_position_loss`
   normalizes globally:
   ```python
   n_weighted = phase_w.sum().clamp(min=1.0)
   position_term = (position_ce * phase_w).sum() / n_weighted
   ```
   Both sums span the batch, so weight ∝ beat count. A 16 s crop holds 51 beats
   at 193 BPM but 15 at 56 BPM. Measured effect on merge: jtd takes **73.7%**
   of position gradient against 63.8% of tracks, while ballroom gets 18.0%
   against 24.1%.

**(b) Meter bug — the uniform-target fallback.** `block` stacks only positions
`1..group_size` (`beat_dataset.py:275`). For a 6- or 8-beat track, beats at
positions 5–8 are zero in *every* channel, so `total ≈ 0` and they hit the
uniform fallback at line 285, receiving a flat `1/G` target. The comment claims
the loss "masks them out by beat weight anyway" — true for non-beat frames,
**false for these**, which are real beats with high `beat_y`. The CE applies at
full weight against a uniform target, actively teaching "all positions equally
likely."

**(c) Sampling is duration-blind.** `__len__` is the track count and
`__getitem__` returns one random 16 s crop, so an epoch is one crop per track
regardless of length. With a 16 s window on a 30 s track the offset ranges over
only 14 s, so two random crops overlap ~11 of 16 seconds — **~70%**. On a 240 s
track they are essentially disjoint. Short-track corpora are therefore both
over-sampled *and* the easy ones to memorize. (But see §3 — the fix is corpus
balancing, not duration weighting.)

**(d) Normalization span mismatch (decoded metrics only).** `tcn.py:255`
standardizes per sample over `(mel, time)` — over a 16 s crop at training, over
the **whole track** at inference. Coincides for 30 s ballroom; diverges for a
300 s track with a quiet intro. Does not touch `val/acc_position` (16 s val
clips) but biases `diagnose_beat_phase.py` / `eval_beat.py`.

**(e) Regularization holes.** The SSA block has **no dropout at all** —
`nn.MultiheadAttention` defaults to `dropout=0.0` and the FFN has none, while
trunk convs get 0.2. That's 0.79M unregularized params in the component with
the most memorization capacity (review §4d). `use_self_attention` is now `true`
(the config comment claiming otherwise is stale) and has never been shown to
earn its parameters.

</details>

<details>
<summary><b>2.7 — What the "general tool" goal exposes</b></summary>

**(a) The metric is currently unusable for this goal.** `val/acc_position` on
merge is a **micro-average dominated by jtd** (64% of tracks, 74% of position
gradient). A model excellent at jazz and useless at classical scores well.

**(b) Neither current nor duration-proportional sampling is genre-balanced.**
Corpus share under `p ∝ n^α`:

| corpus | tracks | α=1.0 (now) | α=0.5 | α=0.3 | α=0 |
|---|---|---|---|---|---|
| jtd | 1108 | 63.8% | 40.5% | 30.1% | 16.7% |
| ballroom | 419 | 24.1% | 24.9% | 22.5% | 16.7% |
| rwc_popular | 80 | 4.6% | 10.9% | 13.7% | 16.7% |
| rwc_genre | 68 | 3.9% | 10.0% | 13.0% | 16.7% |
| rwc_classical | 32 | 1.8% | 6.9% | 10.4% | 16.7% |
| rwc_jazz | 31 | 1.8% | 6.8% | 10.3% | 16.7% |
| **jazz total** | | **65.5%** | 47.3% | 40.4% | 33.3% |

For comparison, weighting by audio hours gives jtd 67.7% and ballroom **6.3%**
— worse on both counts.

**(c) `binary_only: true` makes classical generality impossible.** Measured
discards:

| dataset | position-annotated | kept | **dropped** | meter of dropped |
|---|---|---|---|---|
| ballroom | 698 | 523 | **175 (25%)** | all 3/4 |
| jtd | 1294 | 1204 | 90 (7%) | all 3/4 |
| rwc_genre | 102 | 84 | 18 (18%) | 3, 5, 7, 9 |
| **rwc_classical** | 61 | 39 | **22 (36%)** | 19× 3/4, + 5, 7 |
| rwc_jazz | 50 | 38 | 12 (24%) | 3, 5, 7 |

Triple meter is not an edge case in classical — waltzes, minuets, scherzos. And
`group_size=4` with a 4-way softmax **cannot represent 3/4 even if the tracks
were kept**.

**(d) Receptive field vs tempo.** The trunk's 11.9 s RF covers 6.2 bars at
125 BPM but only **2.8 bars at 56 BPM**. Classical is where the most context is
needed and the least is available.

**(e) Data volume for classical is the binding constraint.** rwc_classical has
**61 position-annotated tracks total, 39 after `binary_only`**. Oversampling 32
tracks teaches you those 32 tracks. Temperature sampling manages imbalance; it
does not manufacture data.

</details>

---

## 3. Proposals

| # | Action | Cost | Gates |
|---|---|---|---|
| 0 | Per-genre val split + macro-average metric — *prototyped, measured, reverted* | free | everything |
| 1 | Get the converged merge checkpoint and re-measure | free | §2.4, §2.5 |
| 2 | Self-calibrating `pos_weight` + per-item loss normalization | small | — |
| 3 | Corpus-temperature sampling (α≈0.5), duration weighting *within* corpus | small | — |
| 4 | **Variable meter support** (drop `binary_only`, max-G head, decode over G) | medium-large | classical/jazz generality |
| 5 | Model capacity | medium | after 1–3 |
| 6 | Period-proportional σ for the position block | medium | target-layout change |
| 7 | Regularization batch (SSA dropout, noise aug, pitch shift) | small | — |
| 8 | Tempo canonicalization / longer RF | large | classical |

Items 2, 4 and 6 all change the loss or target distribution, so they confound
each other within a single run — the same attributability discipline applied to
the step-2 / step-3 sequence.

<details>
<summary><b>#0 — Per-genre metric — <b>prototyped and measured 2026-09-04, then reverted</b></summary>

**Status: the measurements below are real and stand; the code that produced
them was reverted on 2026-09-04 and is not in the tree.** Redo it before
relying on per-genre numbers again.

**What the prototype did**, if it is rebuilt:

- `BeatEvaluator` gained a `track_corpora()` returning the source corpus per
  track, aligned with `compute_track_probs()`. *(The related change that lets
  it build from the split's `TrackRef`s when the dataset has no directory of
  its own — which is what makes `--dataset merge` work at all — was kept.)*
- `tools/diagnose_beat_phase.py` printed a **PER-GENRE BREAKDOWN** with
  per-corpus scores, the macro mean, the micro mean and the worst corpus; with
  >1 corpus it ranked decoders by macro rather than micro confusion.
- `build_beat_dataloaders(cfg, per_genre_val=True)` split validation into one
  loader per corpus; `BeatPhaseModule` logged `val/<corpus>/<metric>` and
  reduced to `val/macro_<metric>` / `val/worst_<metric>`.
- **The training side was kept deliberately additive**, and this constraint
  should survive any rebuild: `val/<metric>` must keep its track-weighted
  (micro) meaning, computed by pooling raw sums so it is bit-for-bit what a
  single blended loader reported. `val/loss` is monitored by
  `ReduceLROnPlateau` and `ModelCheckpoint` *and* spliced into checkpoint
  filenames (`common.py:203`), so redefining it would make runs from before and
  after incomparable while looking identical — the same silent-semantic-drift
  that caused the `--binary-only` misreading in §2.1. To *select* on the
  general-tool objective, point the monitor at `val/macro_loss` explicitly.

**First measurement — merge-trained `epoch57` on merge val, `viterbi=5`:**

| corpus | n | f_one | f_last | confusion | stability |
|---|---|---|---|---|---|
| ballroom | 104 | 0.614 | 0.526 | 0.271 | 0.656 |
| jtd | 96 | 0.524 | 0.525 | 0.278 | 0.756 |
| rwc_popular | 19 | 0.522 | 0.403 | 0.273 | 0.554 |
| rwc_genre | 16 | 0.371 | 0.238 | 0.330 | 0.419 |
| **rwc_classical** | **7** | **0.197** | **0.066** | **0.480** | **0.319** |
| rwc_jazz | 7 | 0.518 | 0.516 | 0.236 | 0.630 |
| **MACRO** | 6 | **0.458** | **0.379** | **0.311** | **0.556** |
| micro | 249 | 0.542 | 0.485 | 0.283 | 0.661 |

The blended number flatters the model: **micro confusion 0.283 vs macro
0.311** (~10% relative). And classical is not merely weaker — `f_last` **0.066**
is essentially no downbeat ability at all, completely invisible in the micro
number. That is the failure mode §2.7a predicted, now measured.

**Caveat: rwc_classical has only 7 val tracks**, so its number is very noisy —
it indicates a real problem but should not be tracked as a precise target until
there is more classical data (§2.7e).

**Second measurement — the same merge val set, both checkpoints.** Confusion,
lower is better; each scored under its own macro-best decoder:

| corpus | n | ballroom-trained | merge-trained | better |
|---|---|---|---|---|
| ballroom | 104 | **0.130** | 0.271 | ballroom, by a lot |
| jtd | 96 | 0.431 | **0.278** | merge, by a lot |
| rwc_popular | 19 | **0.202** | 0.273 | ballroom |
| rwc_genre | 16 | **0.308** | 0.330 | ballroom |
| rwc_classical | 7 | **0.388** | 0.480 | ballroom |
| rwc_jazz | 7 | 0.270 | **0.236** | merge |
| **MACRO** | 6 | **0.288** | 0.311 | **ballroom** |
| micro | 249 | 0.275 | 0.283 | ballroom |

**The merge model is worse on the macro than the ballroom-only model**, and the
pattern is specific: it wins *only* on the two jazz corpora — the ones that make
up 64% of its training set — and loses everywhere else, **including
`rwc_popular` and `rwc_genre`, which are in its own training data**. A
ballroom-only model generalizes to rwc_popular better (0.202) than a model
actually trained on it (0.273).

That is what jtd domination looks like, and it converts §2.7b's sampling
argument from a projection into a measurement: combined with the uniform
underfit of §2.5, the model has too little capacity/training to serve every
corpus and spends what it has on the largest one. It strengthens the case for
**#3 (corpus-temperature sampling)** and for **#5 (capacity)**.

Classical is unlearned by *both* models — `f_last` 0.051 (ballroom-trained) and
0.066 (merge-trained). Nothing in the current setup learns classical downbeats.

</details>

<details>
<summary><b>#2 — Loss calibration (now required, not optional)</b></summary>

**Self-calibrating `pos_weight`.** No single scalar serves 56 BPM (ratio 11.3)
through 125 (4.5) to 193 (2.6). Derive it per sample from the target already in
hand:

```python
pos_frac = beat_y.mean(dim=-1, keepdim=True)          # (B, 1)
pw = alpha * (1 - pos_frac) / pos_frac.clamp(min=1e-6)
beat_term = F.binary_cross_entropy_with_logits(beat_logits, beat_y, pos_weight=pw)
```

`pos_weight` broadcasts against `(B, T)`, so `(B, 1)` works. Keep `alpha ≈ 0.9`
to preserve the existing "shade ~10% under the measured ratio" convention. Side
benefit: it stops needing re-tuning every time the data mix or
`phase_conditioning` changes — which has already bitten once.

**Per-item loss normalization.** Global normalization makes the loss a
micro-average over *beats*, structurally favouring fast genres. Switch to a
macro-average over tracks, matching the §0 metric:

```python
num = (position_ce * phase_w).sum(dim=-1)
den = phase_w.sum(dim=-1)
valid = den > 1e-6
position_term = (num[valid] / den[valid]).mean()
```

The `valid` filter matters — items with `mask=0` would otherwise dilute the
mean with zeros.

</details>

<details>
<summary><b>#3 — Corpus-temperature sampling</b></summary>

Two-level: pick a corpus with probability ∝ `n^α`, then a track within it.
**α ≈ 0.5** to start (see the table in §2.7b) — jazz drops from 65.5% to 47.3%,
and the small rwc corpora go from 1.8–4.6% to 6.8–10.9%, the difference between
"seen occasionally" and "actually learned".

**Duration weighting belongs *inside* a corpus, not across corpora.** Within
jtd a 4-minute take genuinely holds more than a 1-minute one; across corpora it
just rewards whoever recorded longer.

Mechanically:

```python
sampler = WeightedRandomSampler(weights, num_samples=N, replacement=True)
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, ...)
```

`shuffle=True` (`musicality/trainers/common.py:116`) must go — `shuffle` and
`sampler` are mutually exclusive. `replacement=True` is the point: a corpus can
be drawn repeatedly, and since `random_crop=True` re-draws the offset every
`__getitem__`, each draw is a different window.

**Durations for free:** `samples[i][1]` is `beat_times`, and annotation
coverage measured 97.5–99.7% on every corpus, so `beat_times[-1]` is a
~2%-accurate duration proxy with zero I/O.

**What shifts underneath:**
- Epoch length changes meaning — `max_epochs`, `ReduceLROnPlateau(patience=5)`,
  `check_val_every_n_epoch` and checkpoint cadence are all in epochs.
- Build weights **last**: `train_subsample` wraps in `Subset` (remaps indices)
  and `AugmentedBeatDataset` wraps again.
- **Do not touch the val loader** — one fixed centre crop per track, or
  comparability with everything measured so far is lost.

</details>

<details>
<summary><b>#4 — Variable meter (the gating item for classical)</b></summary>

Cleanest incremental path: widen the position head to a fixed **max-G softmax**
(8 or 12), train with the target restricted to each track's true G, and at
inference run the global decoder once per candidate G, picking by total
log-likelihood.

That extends machinery already in place — the decoder already argmaxes over
phase offsets, so an argmax over G is the same shape of search.

Also fixes §2.6b for free: with a max-G head, beats at positions 5–8 stop
falling into the uniform-target fallback.

Reshapes the target layout the same way the `one_last` → `positions` migration
did, so it needs equivalent care.

</details>

<details>
<summary><b>#5 — Capacity (re-opened, but sequenced late)</b></summary>

Ruled out at train confusion 0.007 (ballroom — nothing left to fit). At
0.176–0.221 against 16× the audio with the same 2.4M params, that argument no
longer applies.

Sequence it after #1 (confirm the run is actually converged, not just early)
and #2 (don't add parameters to a miscalibrated objective).

</details>

<details>
<summary><b>Rejected / deferred, with reasons</b></summary>

- **Global duration-proportional sampling.** Original motivation was an 8×
  ballroom oversampling measured against the *old* ballroom+rwc merge. jtd's
  arrival dissolved it: duration weighting now barely moves jtd (63.8% →
  67.7%) and cuts ballroom to 6.3%. Superseded by #3.
- **Richer mel front-end.** Fit-improving; not the binding constraint, and
  43 fps / 23 ms is already well inside the 70 ms scoring tolerance.
- **Chasing `train/acc_beat` = 90%.** A ceiling, not a limit — val F (0.916)
  exceeds train F (0.900).
- **Period-proportional σ for the *beat* head.** Would give ±90 ms at 56 BPM,
  sloppier than mir_eval's 70 ms tolerance. Position block only (#6).
- **Per-mel-bin input normalization** to flatten EQ differences across corpora.
  Would remove absolute low-band energy, a genuine downbeat cue.

</details>

---

## 4. Reference

<details>
<summary><b>Dataset statistics</b></summary>

Measured 2026-09-04, `binary_only=True`. Duration/tempo columns sampled from
the first 150 tracks; `n` is the full count.

| dataset | n | dur_med | dur_p90 | cover% | head_s | tail_s | BPM_med | BPM_p10 |
|---|---|---|---|---|---|---|---|---|
| ballroom | 523 | 30.1 | 30.7 | 97.5 | 0.36 | 0.44 | 125.0 | 100.0 |
| rwc_popular | 99 | 238.2 | 292.6 | 97.9 | 0.05 | 4.60 | 109.1 | 75.9 |
| rwc_genre | 84 | 227.7 | 315.1 | 97.6 | 0.06 | 5.27 | 111.1 | 65.2 |
| rwc_classical | 39 | 304.4 | 598.7 | 97.9 | 0.10 | 5.71 | 105.3 | 56.3 |
| rwc_jazz | 38 | 248.7 | 354.8 | 97.9 | 0.08 | 6.12 | 147.2 | 78.4 |
| gtzan | 940 | 30.0 | 30.0 | 98.5 | 0.25 | 0.21 | 110.1 | 67.5 |
| jtd | 1204 | 122.0 | 225.5 | 99.7 | 0.21 | 0.21 | 193.5 | 127.7 |

`cover%` is `(last_beat - first_beat) / duration` — uniformly ~98%, so
"unannotated regions poison random crops" was checked and rejected.

Beats-per-bar distribution (`binary_only=True`, i.e. what training sees):

| dataset | max(position) distribution |
|---|---|
| ballroom | 4: 100% |
| rwc_popular | 4: 96%, 6: 3%, 8: 1% |
| rwc_genre | 2: 13%, 4: 83%, 6: 4% |
| rwc_classical | 2: 28%, 4: 54%, 6: 13%, 8: 5% |
| rwc_jazz | 4: 97%, 6: 3% |
| gtzan | 2: 1%, 4: 99% |
| jtd | 4: 100% |

</details>

<details>
<summary><b>Splits</b></summary>

| split | train | val |
|---|---|---|
| `beat_phase-ballroom` | 559 | 139 |
| `beat_phase-ballroom-binary` | 419 | 104 |
| `beat_phase-merge` | 809 | 200 |
| `beat_phase-merge-binary` (current, local) | **1737** | **248** |
| `merge` as trained remotely (§2.4) | 1670 | 233 |

Current `beat_phase-merge-binary` train composition — **55.5 h of audio**:

| corpus | tracks | share | audio |
|---|---|---|---|
| jtd | 1108 | 63.8% | 37.6 h |
| ballroom | 419 | 24.1% | 3.5 h |
| rwc_popular | 80 | 4.6% | 5.3 h |
| rwc_genre | 68 | 3.9% | 4.3 h |
| rwc_classical | 32 | 1.8% | 2.7 h |
| rwc_jazz | 31 | 1.8% | 2.1 h |

The remote run used a slightly smaller jtd subset (1670 train). `gtzan` (940
position-annotated tracks) is downloaded but in no merge split. `brid` is
downloaded but yields 0 tracks through the loader — unexplained, worth a
separate look.

</details>

<details>
<summary><b>Tooling fixed during these sessions</b></summary>

- `BeatEvaluator.compute_track_probs()` (`musicality/evaluation.py`) and
  `score_decoder` (`tools/diagnose_beat_phase.py`) applied a blanket `sigmoid`
  to all output channels — correct for the old 3-channel one/last head, wrong
  for a `group_size`-softmax checkpoint, whose position channels need `softmax`
  and number `1 + group_size` rather than 3. It would not have errored, just
  silently scored sigmoid'd logits with positions 3+ never read. Both now
  branch on the checkpoint's own `hparams.group_size`, mirroring
  `musicality.inference.run_inference`. `tools/eval_beat.py` goes through
  `run_inference` and was never affected.
- `_TRACKED_KEYS` (`musicality/trainers/train_beat_phase.py`) listed only
  `acc_one`/`acc_last`, so a step-3 run's terminal output and end-of-run best
  metrics silently omitted `acc_position`. Added.
- Removed a stale "steps 2-3" reference from `print_verdict` in
  `tools/diagnose_beat_phase.py`. Its verdict text is a generic template that
  does not know which step produced the checkpoint being scored — read it as
  such, not as a recommendation.

</details>
