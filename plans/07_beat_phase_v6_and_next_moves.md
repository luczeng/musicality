# Part 7 — v6: what the restored data bought, and where the next gains are

**Status:** analysis complete, nothing implemented. Session of 2026-09-17.
Follows `plans/06_metric_calibration_and_eval_consolidation.md`, whose event
metrics and `training_report.json` are what made this readable at all.

---

## Overview

### High level

The data gap is closed. The run before this one silently trained without two of
its seven corpora; this one has all of them, and the corpus that was missing is
where the gain shows up — the model went from mediocre to respectable on the
blues and country material that was added, roughly a quarter better at naming
which beat is the "one".

Overall performance did not move. What gtzan gained, the older corpora gave back
in small amounts, and turning the attention block off changed nothing
measurable. Meanwhile the validation loss rose steadily through the second half
of training, which looks like the model getting worse and is not: the beat side
kept improving to the last epoch, and the loss climbed because the model became
more confident about bar-position guesses that were already wrong. Checkpoint
selection currently watches that loss, so it is selecting on self-assurance
rather than on accuracy.

The largest gain available right now needs no training at all. The settings that
turn the model's output into beats were tuned for a model the project no longer
trains, and they are far off. Re-tuning them recovers more than the last two
retrains combined. After that, the cheapest real lever is data already sitting
on disk: 853 annotated tracks in seven genres the model has never seen. Only
then does it become worth changing the model, and by then the three remaining
failures — it cannot find the "one" in jazz, it cannot hold a phase for a whole
track, and it does not work on classical at all — are visible as three separate
problems wanting three separate fixes, rather than one vague "make it better".

### Technical

Reference runs and checkpoints. Everything below is scored on the full
277-track `beat_phase-merge-binary` val split with `tools/eval_beat.py` on CPU,
unless a line says otherwise.

| name | file | run | epoch | notes |
|---|---|---|---|---|
| v5 | `checkpoints/merge_v5.ckpt` | `kgncrqxe` (`lr_sweep-0.0008`) | 163 | attention on, lr 8e-4, wd 1e-4, bs 128. **Trained without gtzan/rwc_genre** |
| v6-best | `checkpoints/beat-phase-epoch79-valloss1.4690.ckpt` | `60ujxd72` (`smooth-mountain-98`) | 79 | the run's best `val/loss` |
| v6 | `checkpoints/checkpoint_v6.ckpt` | `60ujxd72` | 107 | attention **off**, lr 5e-4, wd 4e-3, bs 256 |

v6 config deltas against v5: `use_self_attention: false`, `lr` 8e-4 → 5e-4,
`weight_decay` 1e-4 → 4e-3, `batch_size` 128 → 256, `check_val_every_n_epoch`
2 → 4, and all seven corpora present. Four changes at once, so nothing except
the gtzan gain is cleanly attributable to a single cause.

> **The two runs' logged `val/loss` values are not comparable.** v5's validation
> set was missing gtzan and rwc_genre. Scored on the five corpora both runs
> shared, v5 is **1.403** and v6 is **1.441** — a 0.04 gap, not the 0.08 the raw
> report numbers suggest — and their frame position accuracy is identical
> (0.630 vs 0.628).

---

## 1. What the v6 run actually did

<details>
<summary><b>1.1 — The missing corpora are back</b></summary>

`plans`-adjacent history: the previous run dropped gtzan and rwc_genre because
their audio was absent on the remote and `BeatDataset` skips missing audio with
a single printed line. Two independent checks say v6 has them:

- `training_report_vs.json`'s `per_corpus` block lists **all seven** corpora,
  with 50/50 tracks scored (v5 scored 36 of a requested 50).
- `global_step 864` at epoch 107 is `108 × 8` batches. Eight batches at
  `batch_size: 256` implies 1793–2048 train tracks; the split holds **1854**.
  The truncated set would have been 1669 → 7 batches.

</details>

<details>
<summary><b>1.2 — The train/val divergence is the position head alone</b></summary>

From the run history (`training_report_vs.json`):

| epoch | train/loss | val/loss | train/pos_acc | val/pos_acc | train/f_beat | val/f_beat |
|---|---|---|---|---|---|---|
| 43 | 1.549 | 1.545 | 0.591 | 0.594 | 0.882 | 0.871 |
| 79 | 1.406 | **1.469** ← best | 0.651 | 0.617 | 0.909 | 0.885 |
| 107 | 1.267 | 1.477 | 0.710 | 0.633 | 0.919 | 0.891 |
| 155 | 1.094 | 1.579 | 0.785 | 0.603 | 0.933 | 0.895 |
| 199 | 1.013 | 1.630 | 0.808 | 0.614 | 0.938 | **0.900** ← best |

Train loss falls 28% after epoch 79 while val loss rises 11%. The split is not
uniform across the two heads: the train/val gap on **position** goes 0.026 →
0.194, while the gap on **beat** only goes 0.024 → 0.038, and `val/f_beat` sets
its best value at the final epoch. The report's own `best` block agrees — best
`val_event/f_beat` (0.7670) *equals* the final-epoch value, while best
`val_event/position_acc` (0.634) sits above the final 0.610.

</details>

<details>
<summary><b>1.3 — The rising val loss is overconfidence, not error</b></summary>

`beat_position_loss` is beat BCE plus position cross-entropy, and only the sum
is logged. Recomputing both terms over the full val split and over a
size-matched, unaugmented, fixed-crop train subsample (the val totals reproduce
the logged `val/loss` to ±0.01, which is the check that the decomposition is the
same quantity):

| checkpoint | split | beat BCE | position CE | total | pos_acc | confidence when **wrong** |
|---|---|---|---|---|---|---|
| v6-best (e79) | train | 0.615 | 0.765 | 1.379 | 0.669 | 0.468 |
| v6-best (e79) | val | 0.641 | 0.834 | 1.475 | 0.613 | 0.510 |
| v6 (e107) | train | 0.594 | 0.684 | 1.278 | 0.707 | 0.536 |
| v6 (e107) | val | **0.628** | **0.855** | 1.483 | **0.628** | **0.573** |
| v5 (e163) | val | 0.638 | 0.859 | 1.497 | 0.617 | 0.548 |

Between epochs 79 and 107, val beat BCE *improves*, val position accuracy
*improves*, and the total still rises — carried entirely by position CE. The
last column is the mechanism: mean softmax probability on frames the model gets
**wrong** rose from 0.510 to 0.573. Cross-entropy charges for that; accuracy
does not see it.

Consequence for the pipeline: `build_checkpoint_callback` monitors `val/loss`,
so "best checkpoint" means "least overconfident". On full tracks epoch 107 is in
fact the better model (`position_acc` 0.594 vs 0.585).

</details>

---

## 2. Where v6 stands

<details>
<summary><b>2.1 — Full val split, shipped postprocessing defaults</b></summary>

| | f_beat | cmlt | amlt | pos_acc | best_off | anchor_err | confusion |
|---|---|---|---|---|---|---|---|
| v5 (e163) | 0.837 | 0.639 | 0.819 | 0.591 | 0.660 | 0.069 | 0.261 |
| v6-best (e79) | 0.834 | 0.627 | 0.827 | 0.585 | 0.668 | 0.083 | 0.261 |
| v6 (e107) | 0.833 | 0.624 | 0.817 | **0.594** | 0.669 | 0.075 | 0.258 |

Three checkpoints, two architectures, two data regimes, one number. Whatever v6
changed, it did not change the aggregate.

</details>

<details>
<summary><b>2.2 — Per corpus, and what the gtzan data bought</b></summary>

v5 → v6 (e107), shipped defaults:

| corpus | n | f_beat | pos_acc | confusion |
|---|---|---|---|---|
| **gtzan** | 28 | 0.781 → **0.870** | 0.558 → **0.700** | 0.322 → **0.219** |
| ballroom | 104 | 0.827 → 0.802 | 0.654 → 0.635 | 0.235 → 0.246 |
| jtd | 96 | 0.952 → 0.950 | 0.547 → 0.529 | 0.257 → 0.289 |
| rwc_popular | 19 | 0.763 → 0.742 | 0.676 → 0.679 | 0.218 → 0.196 |
| rwc_genre | 16 | 0.596 → 0.572 | 0.469 → 0.532 | 0.312 → 0.243 |
| rwc_classical | 7 | 0.520 → 0.514 | 0.280 → 0.267 | 0.497 → 0.393 |
| rwc_jazz | 7 | 0.724 → 0.706 | 0.692 → 0.653 | 0.228 → 0.263 |

Inside gtzan, blues is where the added data mattered most — `f_beat` 0.734 →
0.865 and `position_acc` 0.504 → 0.695 at epoch 79 (0.832 / 0.651 at e107);
country 0.604 → 0.771 position. Blues `cmlt` nearly doubles (0.445 → 0.767):
it stopped tracking blues at the wrong metrical level.

The cost: on the five corpora both runs shared, micro `f_beat` 0.861 → 0.847 and
`position_acc` 0.603 → 0.586 (n=233). A large in-corpus gain paid for by a
small, broad dip — with the hyperparameter changes confounded into it.

</details>

<details>
<summary><b>2.3 — Three failure modes, not one</b></summary>

Under the shipped defaults, the phase profile reads: modal offset correct on
218/275 tracks (79.3%), half-cycle on 26 (9.5%); within-track stability mean
0.669, **173 tracks (62.9%) flip phase mid-track**.

- **Anchoring, and it is almost entirely jtd.** `position_acc` 0.530 against
  `position_acc_best_offset` 0.699 — a single whole-track rotation would
  recover 0.17. See §2.4: the anchor error is concentrated in one corpus to a
  degree the aggregate hides.
- **Stability.** 63% of tracks change their phase answer partway through. The
  trunk's receptive field is `3 × (2^8 − 1) = 765` frames ≈ **17.8 s**, so the
  model *cannot* enforce consistency beyond ~18 s by construction. Consistency
  has to come from the decoder, and the decoder is demonstrably under-tuned
  (§3.2).
- **A broken corpus.** rwc_classical fails at the beat level, not the phase
  level (`f_beat` 0.514, `cmlt` 0.118). It is 32 train / 7 val tracks, so every
  number on it is noise on a sample of seven.

</details>

<details>
<summary><b>2.4 — The anchor error is one corpus, and the usual explanations do not fit</b></summary>

Per corpus, shipped defaults. `anchor_error` is
`position_acc_best_offset − position_acc`: how much a single whole-track
rotation would recover. The last column is the distribution of each track's
*dominant* phase offset.

| corpus | n | cmlt | pos_acc | best_off | anchor | modal offset 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| **jtd** | 96 | 0.896 | 0.529 | 0.699 | **0.170** | **66% / 12% / 12% / 9%** |
| ballroom | 104 | 0.494 | 0.635 | 0.669 | 0.034 | 86% / 3% / 10% / 2% |
| gtzan | 28 | 0.674 | 0.700 | 0.719 | 0.019 | 86% / 4% / 11% / 0% |
| rwc_popular | 19 | 0.415 | 0.679 | 0.679 | 0.000 | 100% / 0% / 0% / 0% |
| rwc_genre | 16 | 0.261 | 0.532 | 0.536 | 0.004 | 88% / 0% / 6% / 6% |
| rwc_jazz | 7 | 0.525 | 0.653 | 0.653 | 0.000 | 100% / 0% / 0% / 0% |
| rwc_classical | 7 | 0.118 | 0.267 | 0.302 | 0.035 | 50% / 33% / 0% / 17% |

jtd's anchor error is **5× the next largest**, and it is the only corpus where a
third of tracks settle on the wrong count. Four explanations that would
otherwise be reached for are ruled out by the numbers:

- **Not beat tracking.** jtd has the *best*-tracked beats in the set
  (`f_beat` 0.950, `cmlt` 0.896). The model knows where the beats are.
- **Not lack of data.** jtd is 1107 of 1854 train tracks — 60% of everything
  the position head has ever seen.
- **Not track length.** rwc_popular (median 108 bars/track), rwc_genre (104) and
  rwc_jazz (158) are all *longer* than jtd (80 bars) and anchor at 0.000–0.004.
- **Not annotation structure.** Every jtd val track is a strict 4-beat cycle
  with zero breaks — cleaner than rwc_classical (29% of tracks with breaks),
  rwc_genre (25%) or rwc_popular (5%).

What is left is a property of the material, and the errors' shape is the clue:
they are spread roughly evenly across all three wrong offsets (12/12/9) rather
than piling onto one, which is what a systematic convention shift would look
like. That is the signature of a model with no usable evidence, guessing. jtd is
also by far the fastest corpus (median **185 BPM**, up to 300, against 103–158
elsewhere) — a bar lasts under a second — and it is piano-trio swing, where no
kick or snare marks the bar start.

> **Unverified.** "The audio carries weak downbeat evidence" is the hypothesis
> that survives, not a measurement — nobody has listened. The one alternative it
> cannot be separated from by inspection is that jtd's annotated "1" is itself
> debatable in fast swing; that would need an annotator, not a script.

</details>

---

## 3. Free wins, measured

<details>
<summary><b>3.1 — The postprocessing knobs are stale, and it is expensive</b></summary>

`configs/eval_beat.yaml`'s own comment flags the `beat_phase:` block as
UNVERIFIED — swept against a head and a decoder the project no longer trains.
Re-swept on v6 (`--sweep`, one model pass over the 277 val tracks):

| | f_beat | cmlt | amlt | pos_acc | best_off | macro pos |
|---|---|---|---|---|---|---|
| shipped (thr 0.5, min_dist 4, gate 0.1, switch 2.0) | 0.833 | 0.624 | 0.817 | 0.594 | 0.669 | 0.571 |
| **swept (thr 0.8, min_dist 4, gate 0.2, switch 1.0)** | **0.869** | **0.734** | 0.882 | **0.637** | 0.710 | **0.600** |

Per corpus under the swept knobs:

| corpus | n | f_beat | cmlt | amlt | pos_acc | best_off | confusion |
|---|---|---|---|---|---|---|---|
| ballroom | 104 | 0.873 | 0.702 | 0.912 | 0.730 | 0.772 | 0.194 |
| jtd | 96 | 0.950 | 0.903 | 0.950 | 0.530 | 0.680 | 0.289 |
| gtzan | 28 | 0.803 | 0.596 | 0.830 | 0.676 | 0.710 | 0.265 |
| rwc_popular | 19 | 0.874 | 0.720 | 0.894 | 0.774 | 0.774 | 0.166 |
| rwc_genre | 16 | 0.647 | 0.480 | 0.593 | 0.581 | 0.585 | 0.252 |
| rwc_classical | 7 | 0.556 | 0.201 | 0.494 | 0.244 | 0.317 | 0.507 |
| rwc_jazz | 7 | 0.760 | 0.585 | 0.736 | 0.666 | 0.666 | 0.258 |
| **macro** | 7 | **0.780** | **0.598** | 0.773 | **0.600** | 0.644 | 0.276 |

The beat threshold does most of the work, and it is not a lucky grid cell — the
whole `beat_threshold: 0.8` row of the grid is on top, with `gate_tolerance`
0.15–0.30 all within 0.003 of each other. Ballroom carries the gain (`cmlt`
0.494 → 0.702), rwc_popular follows (`cmlt` 0.415 → 0.720).

Two caveats that matter:

1. **gtzan prefers the old threshold** (0.700 → 0.676 position). The optimum is
   corpus-dependent, which argues for an *adaptive* threshold — a quantile of
   each track's own activation distribution — rather than a new constant.
2. The sweep ran on val, so these figures are optimistic. Confirm on a slice
   that was not used for tuning before quoting them as a bar.

</details>

<details>
<summary><b>3.2 — The decoder is worth ~0.13 position accuracy on its own</b></summary>

`--decoders` on v6, one cached model pass, shipped beat knobs:

| decoder | pos_acc | best_off | anchor | confusion |
|---|---|---|---|---|
| greedy (anchor 0.8) | 0.554 | 0.635 | 0.080 | 0.265 |
| global, no resync | 0.491 | 0.599 | 0.107 | 0.318 |
| global + viterbi, switch 0.05 | 0.618 | 0.679 | 0.061 | 0.263 |
| global + viterbi, switch 0.25 | **0.622** | 0.688 | 0.066 | 0.259 |
| global + viterbi, switch 0.5 | **0.623** | 0.691 | 0.069 | 0.250 |
| global + viterbi, switch 2.0 *(shipped)* | 0.594 | 0.669 | 0.075 | 0.258 |
| global + viterbi, switch 20 | 0.548 | 0.646 | 0.098 | 0.295 |

The curve is flat below 0.5 and falls away above it, so the shipped 2.0 sits on
the wrong side of a broad optimum. Note the direction: *cheaper* resyncing is
better, i.e. the model's local phase reads are more trustworthy than its global
one — the same conclusion §2.3 reaches from the stability profile.

The penalty is charged per beat, which means the same constant behaves
differently at 90 BPM and at 220 BPM. Making it tempo-relative is a small change
with a plausible payoff on jtd, the fastest corpus and the one the global decode
helps least.

</details>

<details>
<summary><b>3.3 — Input normalisation depends on input length</b></summary>

`TCNTempoNet.forward` normalises the mel by the mean/std of **its entire
input**. Training inputs are 16 s crops; `run_inference` feeds the whole track in
one pass. A four-minute track with a quiet intro, a loud chorus and a fade
normalises differently from any 16 s window of itself, so the model meets a
distribution at inference that it never saw in training.

Tested by wrapping the module so it runs in 16 s windows and stitches the
central half of each window's logits, then scoring both ways through
`BeatEvaluator.from_module` with the swept knobs:

| | macro f_beat | macro cmlt | macro pos_acc | micro pos_acc |
|---|---|---|---|---|
| one pass (current) | **0.780** | **0.598** | 0.600 | 0.637 |
| 16 s windows | 0.754 | 0.554 | **0.627** | **0.648** |

Position improves (rwc_genre +0.063, rwc_classical +0.055, rwc_popular +0.051)
and beats degrade (rwc_classical −0.081 `f_beat`, rwc_jazz −0.090). So the
mismatch is real, but naive windowing trades one head against the other. The
principled fix is normalisation that does not depend on input length at all —
per-band running statistics, a sliding window, or fixed corpus-level constants —
applied identically in training and inference.

</details>

---

## 4. Ranked next moves

Ordered by expected gain per unit of effort, not by ambition.

### Tier 0 — free, measured, this week

**4.1 Commit the swept postprocessing knobs** (§3.1). `beat_threshold: 0.8`,
`gate_tolerance: 0.2`, `switch_penalty: 1.0` in `configs/eval_beat.yaml`. Worth
+0.036 `f_beat`, +0.110 `cmlt`, +0.043 `position_acc` with no retraining — more
than the last two retrains combined. Follow with an adaptive per-track threshold
so gtzan stops paying for ballroom's optimum.

**4.2 Stop selecting on `val/loss`** (§1.3). Monitor `val_event/position_acc`
instead. Two adjacent fixes: `event_metrics.every_n_epochs: 5` against
`trainer.check_val_every_n_epoch: 4` scores events only on their LCM — **every
20 epochs** — which is far too coarse to select on; and the run should stop
around epoch 110, which halves the cost of every experiment below.

### Tier 1 — cheap training changes

**4.3 Add the rest of gtzan.** `../musicality_db/gtzan/annotations` holds **998**
`.beats` files across all ten genres, with bar positions (checked:
`rock_00001` cycles 2, 3, 4, 1). The split uses blues and country only — **145
tracks, 853 unused**. That is a ~46% larger training set for the cost of a
split regeneration, and it brings in rock, pop, disco, metal, hiphop, reggae and
classical, which the model has never seen. Expected beneficiaries: rwc_genre and
rwc_popular directly; gtzan's 100 classical tracks are the only realistic
near-term help for rwc_classical. Worth a dedup pass — gtzan's duplicate and
corrupt files are well documented.

**4.4 Rebalance the sampler.** jtd is **1107 of 1854 train tracks (60%)**, and
it is the corpus the model is least able to anchor (§2.4: anchor error 0.170
against ≤0.035 everywhere else, despite the best-tracked beats in the set). So
the position head spends most of its gradient on the one corpus where it cannot
find the answer — a plausible, *untested*, driver of the memorisation in §1.2:
a head that cannot generalise on 60% of its input can still drive the loss down
by memorising it. A `WeightedRandomSampler` or a per-corpus per-epoch cap in
`build_beat_dataloaders` is a few lines, and it doubles as the experiment that
tests the hypothesis. Even with all of gtzan, jtd stays at 41% unweighted.

**4.5 Regularise the position head specifically** — that is where the whole
overfit lives (§1.2: gap 0.194 on position vs 0.038 on beat):

- **Label smoothing on the position CE**, which targets the measured pathology
  directly (confidence 0.573 on wrong frames, rising with epochs).
- **SpecAugment.** Augmentation today is time-stretch, gain and noise — nothing
  acts on the spectrogram, and masking is the standard cheap regulariser here.
- **EMA weights**, and more dropout on the position head only. The beat head is
  not the problem; do not handicap it.

### Tier 2 — formulation, bigger swings

**4.6 Attack anchoring and stability separately** (§2.3).

*Anchoring:* the binding constraint is the 16 s clip, not the architecture — the
receptive field is already 17.8 s. Train on 30–40 s clips **and** add one or two
dilation layers so the trunk can use them; a downbeat cue is often a chord
change or a phrase boundary eight bars away. The bigger version of this idea is
a **beat-synchronous position head**: pool trunk features at detected beat
times and run a small sequence model (GRU / transformer / CRF) over *beats*,
turning a 1000-frame phase problem into a ~100-step 4-class one with regular
structure. That is how downbeat trackers are normally built, and it is the
highest-ceiling change in this tier.

*Stability:* a bar-position **DBN with tempo states** instead of the 4-state
Viterbi, plus the tempo-relative switch penalty from §3.2. This also attacks the
residual `amlt − cmlt` gap of 0.148 under the swept knobs.

**Do not** revisit frame-level self-attention: v5 (attention) vs v6 (none) is
0.591 vs 0.594 position accuracy. That experiment is finished, and `plans/05`
reached the same conclusion from the other direction.

**4.7 Fix the normalisation length dependence** (§3.3), in training and
inference together.

**4.8 Try a pretrained encoder.** With fewer than 2000 tracks, a from-scratch
log-mel TCN is fighting its data budget. `musicality/models/huggingface.py`
already wraps HF models; MERT / music2vec features with a light head is the
standard route to a step change on a small beat/downbeat corpus, and it makes
most of the overfitting discussion moot.

**4.9 Tempo-aware decoding.** `amlt − cmlt` says 15% of tracked audio is
confidently tracked at the wrong metrical level even after tuning. The repo
already has tempo models and `tempo_acc1`; a global tempo estimate used as a
prior in the beat decode targets exactly that residual.

### Tier 3 — measurement hygiene, which protects all of the above

**4.10 Seed the split.** `Splitter.create` calls `random_split` without a
generator (`musicality/splits/splitter.py:153`), so every regeneration reshuffles
every corpus and no cross-run comparison survives a data addition. This already
cost a clean v3-vs-v5 comparison.

**4.11 Separate tuning and validation slices**, so sweeps like §3.1 do not leak.

**4.12 Quote macro, not micro.** ballroom + jtd are 200 of 277 val tracks;
micro is mostly a report on those two.

**4.13 Decide about classical.** Either import a real classical beat corpus
(ASAP carries beat and downbeat annotations for ~1000 performances) alongside
gtzan's classical 100, or descope classical explicitly and stop letting seven
tracks define "worst corpus" in every report.

---

## 5. Recommended order

1. §4.1 — commit the swept knobs. Immediate, and it re-baselines every number
   that follows.
2. §4.3 + §4.4 — all of gtzan, and a balanced sampler.
3. §4.2 — select on `val_event/position_acc`, stop at ~110 epochs, then retrain
   and measure. With the decoder fixed first, a model change is finally visible
   instead of being masked by a stale readout.

Everything in Tier 2 should wait until 1–3 have re-baselined the numbers,
because each of them is currently being judged through a decoder that costs
0.043 position accuracy on its own.

---

## 6. Loose ends

- The §3.1 sweep and the §3.2 decoder scan were both run on val. They are the
  right settings to *use*, but they are not clean measurements of the settings'
  value; §4.11 is what fixes that.
- v5's reported `val/loss` (1.386 best) and v6's (1.469 best) were measured on
  different validation sets — v5's was missing two corpora. Do not compare them;
  on the shared five corpora the gap is 1.403 vs 1.441.
- `checkpoint_v6.ckpt` is epoch 107 and `beat-phase-epoch79-valloss1.4690.ckpt`
  is epoch 79 of the same run. The report's `best_checkpoint` field names the
  latter; §1.3 argues the former is the better model.
- rwc_classical's 7 val tracks cannot support any of the conclusions drawn about
  it beyond "it does not work".
- The 36-track slice in v5's report and the 50-track slice in v6's are
  corpus-balanced, so their micro numbers sit much closer to macro than the full
  split's do. Comparing a report's headline to a full-split evaluation compares
  two different weightings.

---

## Appendix — reproducing the measurements

```bash
# the canonical report, per genre, with the phase-offset profile
uv run python tools/eval_beat.py --checkpoint checkpoints/checkpoint_v6.ckpt \
    --dataset merge --split val --binary-only --per-genre --profile \
    --output v6_val.csv

# §3.1 — re-sweep the postprocessing knobs (one model pass)
uv run python tools/eval_beat.py --checkpoint checkpoints/checkpoint_v6.ckpt \
    --dataset merge --split val --binary-only --sweep

# §3.1 — the swept settings, applied
uv run python tools/eval_beat.py --checkpoint checkpoints/checkpoint_v6.ckpt \
    --dataset merge --split val --binary-only --per-genre --profile \
    --beat-threshold 0.8 --min-distance-frames 4 --gate-tolerance 0.2 \
    --switch-penalty 1.0

# §3.2 — every decoder against the same cached probabilities
uv run python tools/eval_beat.py --checkpoint checkpoints/checkpoint_v6.ckpt \
    --dataset merge --split val --binary-only --decoders \
    --switch-penalties 0.05 0.25 0.5 1 2 20
```

§1.3 (loss decomposition) and §3.3 (windowed inference) were one-off scripts.
The decomposition recomputes `beat_position_loss`'s two terms separately —
`F.binary_cross_entropy_with_logits` on channel 0 against
`-(position_y * log_softmax(position_logits)).sum(1)` weighted by `mask * beat`,
averaged per item — plus the mean softmax maximum on frames where
`argmax(position_logits) != argmax(position_y)`, which is the confidence-when-
wrong column. The windowing experiment wraps the module:

```python
class Windowed(nn.Module):
    """Run *module* over 16 s windows and stitch the central half of each."""

    def __init__(self, module):
        super().__init__()
        self.module, self.hparams = module, module.hparams

    def forward(self, wav):
        # 689 frames = 16.0 s at hop 512 / 22050 Hz — the training duration;
        # keep the middle 344 frames of each window, step by that much.
        ...
```

and scores it through `BeatEvaluator.from_module(wrapped, dataset, ...)`, so the
comparison runs the identical decode and scoring path as the CLI.
