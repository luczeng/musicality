# Part 9 — Lessons from the literature: three papers, read in full

**Status:** analysis, no code. Session of 2026-09-21. The three papers our design
descends from were read end to end and put side by side with what we actually
run. Nothing here is a new measurement on our data — every number is either
published or already in `plans/07`, `plans/08` and `leaderboard/LEADERBOARD.md`.

**What it is for.** `plans/08` §1.1 flagged its own literature figures as
*recalled, not measured*, and §6.1 replaced the ones that mattered by benchmarking
two public systems on our split. This document does the other half: it reads the
papers themselves, so the *design* claims — not just the scores — stop being
recalled. Most of what it finds confirms an item already in `plans/07` or
`plans/08` and attaches a published effect size to it, which is what those
documents lack when ordering work. Four items are new, and one recommendation
from the ordering in §5 contradicts nothing here but reverses what I would have
told you after reading only the first two papers.

**The papers.**

- **[DB19]** M. Davies, S. Böck, *Temporal convolutional networks for musical
  audio beat tracking*, EUSIPCO 2019. Our architecture — `TCNTempoNet` cites it.
- **[BD20]** S. Böck, M. Davies, *Deconstruct, Analyse, Reconstruct: How to
  improve Tempo, Beat, and Downbeat Estimation*, ISMIR 2020. The same authors
  taking [DB19] apart, with an ablation.
- **[BT24]** F. Foscarin, J. Schlüter, G. Widmer, *Beat this! Accurate beat
  tracking without DBN postprocessing*, ISMIR 2024. The system already wired into
  `musicality/baselines` and measured on our split in `plans/08` §1.3.

---

## Overview

### High level

Reading all three at once resolves an argument this project has been having with
itself about where its remaining error lives.

Our front end and our decoder are not behind the field — they are, almost line
for line, the choices the *current* state of the art makes. The 2024 system runs
the same sample rate, the same number of mel bands, the same kind of per-band
normalisation, a nearly identical frame rate, and a peak-picker that is our
peak-picker. It reaches state-of-the-art accuracy with no decoder at all. Our
trunk, meanwhile, is the 2019 architecture at roughly the size the 2019 and 2020
papers say is appropriate. So the shape of the system is fine.

What is not fine is how we teach it. All three papers put their biggest reported
gains in the same place, and it is the place we have left alone: the relationship
between the annotation, the target, and the loss. The 2024 paper's ablation is
blunt about it — undoing its two loss ideas costs thirteen points of beat
accuracy, which is more than its entire twenty-million-parameter transformer buys
over a two-million-parameter one. We currently do the thing that paper explicitly
argues against, which is to blur the target instead of forgiving the prediction.
Second on their list is an augmentation we do not perform at all.

There is also a quieter finding that costs nothing to act on. Every one of these
systems is tuned to be *overconfident*, and the 2024 paper says outright that it
keeps training after validation loss starts rising because the score it cares
about keeps improving. We select checkpoints by minimum validation loss, so we
systematically pick the least confident epoch of every run. Our own `plans/07`
measured this exact mechanism a month ago; the literature now says it is not a
quirk of our setup but the normal behaviour of this task.

The ranking that falls out is: fix what the network is asked to predict, then how
we pick the model, then what we feed it. Architecture and decoder come after,
and the decoder comes last.

### Technical

The comparison board in §1 is the artefact — four systems, one row per design
decision. §2 walks the decisions where we differ and says what the papers
measured. §3 collects every published effect size in one table, which is the
thing `plans/08` §7 was ordered without. §4 records where we are already aligned
or ahead. §5 proposes a revised order and states precisely which steps of
`plans/08` §7 it moves.

Our column throughout is `configs/beat_train.yaml` at commit `d654003`
(`group_size: 4`, 32×9 trunk, `conv2d_stem: true`, `pos_weight: auto`,
`position_norm: per_item`), with parameter count and receptive field computed
from the instantiated model rather than quoted, and scores from
`leaderboard/LEADERBOARD.md` row 1 (`checkpoints_change_tcn_stem/lr_0.0008`,
macro over 7 corpora, 277 val tracks, postprocessing swept on `train`).

---

## 1. The board

Four systems, one row per design decision. **Bold** in our column marks a row
where we differ from *all three* papers; a row marked ✓ is one where we already
match the most recent system.

| | **[DB19]** 2019 | **[BD20]** 2020 | **[BT24]** 2024 | **ours** |
|---|---|---|---|---|
| **Task** | beat | beat + downbeat + tempo | beat + downbeat | beat + 4-way bar position |
| **Sample rate** | 44.1 kHz | 44.1 kHz | 22.05 kHz | 22.05 kHz ✓ |
| **Input** | log-mag, 81 log bins, 12/oct | same | 128 mel, 30 Hz–10 kHz | 128 mel, ≤11 kHz ✓ |
| **Magnitude scaling** | log | log | `ln(1+1000x)` (silence → 0) | log-mel |
| **Frame rate** | 100 fps (hop 10 ms) | 100 fps | 50 fps (hop 441) | 43 fps (hop 512) |
| **Normalisation** | — | — | per-band BN in the stem | frozen per-band stats ✓ |
| **Input length** | full track | full track | 30 s | **16 s** |
| **Stem** | 3×3, 3×3, 1×8; mp 1×3 ×2; 16 ch | 3×3, **1×12**, 3×3; mp 1×3 ×3; 20 ch | conv 3×4 + 3 blocks with partial transformers | 3× (3×3 + BN + GELU), mp 3 ×2, 16 ch |
| **Frequency mixing** | 1×8 collapse → 16 feat | 1×12 mid-stem, wide | attention **over the frequency axis** | **fold 16×14 → 224 → 1×1 conv** |
| **Trunk** | TCN, 16 ch, k=5, dil 2⁰–2¹⁰ | TCN, 20 ch, **two dilations per layer** (*d*, 2*d*), 1×1 combine | 6 transformer blocks, 512 d, rotary | TCN, 32 ch, k=3, dil 2⁰–2⁸ |
| **Trunk regularisation** | spatial dropout 0.1 | spatial dropout 0.1 | dropout in transformer blocks | **none (head only)** |
| **Context** | ±41 s (82 s diameter) | ±41 s; *more did not help* | full 30 s sequence (attention) | ±11.9 s (23.8 s diameter) |
| **Params** | 21,809 | 116,302 | 20 M (2 M variant ≈ same F1) | 40,773 |
| **Head** | 1 sigmoid | 2 sigmoids (multi-label) | 2 sigmoids + **Sum Head** | **beat sigmoid + 4-way softmax** |
| **Aux output** | — | tempo, off skip connections | — | **none** |
| **Target** | spike, ±2 frames @ 0.5 | same | **sharp spike** | **Gaussian, σ = 1.5 frames** |
| **Loss** | BCE, unweighted | BCE, unweighted | **shift-tolerant** weighted BCE, *w* = neg/pos | weighted BCE (`pos_weight: auto`) + position CE |
| **Class weighting** | none | none | global neg/pos ratio | **per-sample, derived from the target** |
| **Tempo augmentation** | none | STFT hop jitter, N(tempo, 5%) | precomputed ±4…20% | time-stretch ±15% ✓ |
| **Pitch augmentation** | none | none | **±6/−5 semitones** | **none** |
| **Other augmentation** | none | none | **masking, 0–6 regions, shuffled** | gain ±6 dB |
| **Optimiser** | Adam 1e-3 | **RAdam + Lookahead**, 2e-3, clip 0.5 | AdamW, wd 0.01, warmup → 8e-4, cosine | Adam 5e-4, ReduceLROnPlateau |
| **Batch** | 1, full sequence | 1, full sequence | 8 × accum 8, 30 s excerpts | 16 × 16 s crops |
| **Model selection** | val loss plateau / early stop | unchanged from [DB19] | **trains past rising val loss on purpose** | **`monitor="val/loss"`, min** |
| **Decoder** | DBN (Krebs state space) | DBN, joint or sequential | **none** — local max ±3 frames, p > 0.5 | peak-pick + gate + Viterbi over positions ✓ |
| **Seeds reported** | 1 | 1 | **3, with ±σ** | **1** |
| **Eval** | 8-fold CV + GTZAN held out | 8-fold CV + 3 test sets | 8-fold CV + GTZAN held out | single split, per-corpus macro |
| **Training tracks** | 1 899 over 5 corpora | 6 corpora, counts not stated | **4 556 over 18 corpora** | 1 854, 60 % one corpus |

Published scores, for orientation only — different splits, different corpora,
not comparable to our column except where `plans/08` §1.3 measured the system on
our own tracks. [BD20] reports two decode variants (joint and sequential); the
better of the two is quoted per dataset:

| | Ballroom beat F | Hainsworth beat F | GTZAN beat F | SMC beat F | GTZAN downbeat F |
|---|---|---|---|---|---|
| [DB19] | 0.933 | 0.874 | 0.843 | 0.543 | — |
| [BD20] | 0.962 | 0.904 | 0.885 | 0.552 | 0.672 |
| [BT24] | 0.975 | 0.919 | 0.891 | 0.627 | 0.783 |
| ours (our split) | 0.834 | — | 0.853 | — | — |

Two warnings about that last row. Our ballroom and gtzan numbers are on *our*
val split, not theirs. And gtzan is in our training distribution while it is
held out entirely for all three papers, so our 0.853 is an in-domain number
against their out-of-domain ones — the comparison flatters us.

---

## 2. Design decision by design decision

<details>
<summary><b>2.1 — The loss: they forgive the prediction, we blur the target (largest published effect in any of the three papers)</b></summary>

**What [BT24] does.** Targets stay sharp — one frame per annotation. Before the
BCE, the *predictions* are max-pooled over 7 frames (±3), so only the largest
prediction within ±3 frames of a label is compared to it; negatives are ignored
within ±6 frames, which is how far a max-pooled prediction 3 frames out can
spread. Written out, with `m_k` the k-frame max-pool:

```
L_st(y, ŷ, w) = − Σ_t  w · y_t · log m₇(ŷ)_t  +  (1 − m₁₃(y)_t) · log(1 − m₇(ŷ)_t)
```

The positive weight `w` is the ratio of negative to positive frames over the
training set, and the paper calls it "crucial when not using a DBN".

**Why it is not the same as our smearing.** The paper names our approach and
rejects it: widening the target with extra positive labels "only mitigates the
former problem [slow convergence] without helping with the latter" — the latter
being that the network learns to emit wide, blurred peaks. A blurred activation
then needs a carefully tuned threshold, or a DBN, to become events again.

We are the case being described. `sigma_frames: 1.5` is a Gaussian on the target;
`plans/06` measured frame precision at 0.487, i.e. the predicted peak is wider
than the target's; `musicality/postprocess.py::pick_peaks` opens its own
docstring by explaining that it exists *because* "training targets are
Gaussian-smeared … a real event shows up as a bump spanning several frames"; and
`project_eval_beat_stale_knobs` records that re-sweeping `beat_threshold` beats
any recent retrain. That is one causal chain, and its first link is the target.

**Effect size.** On [BT24]'s single-split validation, 3 seeds: removing shift
tolerance costs 1.4 beat F1 and 3.2 downbeat F1; removing shift tolerance *and*
the positive weighting costs **13.1 beat F1 and 16.7 downbeat F1** (92.6 → 79.5,
85.4 → 68.7). For scale, their 20 M-parameter model beats their 2 M one by 0.3
beat F1 on GTZAN.

**What this changes here.** `plans/08` §7 step 5 already proposes shift-tolerant
BCE and cites this paper's ablation. Nothing about the proposal changes; its
*rank* does. It is the largest published effect in the literature we descend
from, it is a contained change to `musicality/losses.py` and
`musicality/loaders/beat_dataset.py`, and at 43 fps the ±3-frame pooling window
is ±70 ms — exactly the `mir_eval` tolerance we are scored at, so the loss would
forgive precisely what the metric forgives.

**One open question it raises for our second head.** The position CE reads a
*normalised* position block, which is built from the same Gaussian smear. If the
beat target becomes a sharp spike, `BeatDataset`'s position block and the
`phase_w = mask * beat_y` weighting both change shape — `beat_y` currently
doubles as the soft "near a beat" gate with no threshold to invent
(`musicality/losses.py`). Either the gate becomes the max-pooled beat target, or
it becomes an explicit window. This needs deciding before the change is made, not
during.

</details>

<details>
<summary><b>2.2 — Model selection: the field selects for overconfidence, we select against it</b></summary>

[BT24] §4.2, in plain words: "we found that to achieve good results without a
DBN, we need our network to be overconfident in its predictions… we keep training
even after the validation loss starts increasing, which would typically indicate
overfitting. Indeed, we see that the validation F1 score continues to improve
even with increasing validation loss. This means that even with our
modifications, the BCE loss is not a good indicator of the F1 score."

The mechanism they give is the one we need: a peak-picker fed probabilities near
0.5 produces "random oscillations between positive and negative predictions, and
thus erratic beats". They want steady, high-probability output "exactly like the
DBN would".

We have measured the same decoupling from the inside. `plans/07` §1.3 decomposed
v6's loss and found val beat BCE improving, val position accuracy improving, and
the total still rising between epochs 79 and 107 — carried entirely by position
CE, with mean softmax confidence on *wrong* frames rising 0.510 → 0.573. On full
tracks the later checkpoint was the better model. `plans/07` §4.2 concluded "stop
selecting on `val/loss`" and the memory `project_beat_phase_val_loss_decoupled`
records it. `build_checkpoint_callback` still monitors `val/loss`.

**Effect size:** not published as a number, but it is free. `monitor` moves from
`val/loss` to `val_event/f_beat` (max) in `musicality/trainers/common.py`, and
`EventMetricsLogger` already produces that key on scoring epochs — its cadence
(`every_n_epochs: 5`) is the only thing to reconcile with `save_top_k`.

This is also what yesterday's `loss_beat` / `loss_position` split is for: with
both terms logged per epoch, the [BT24] pattern (position CE climbing while
accuracy climbs) is visible on the chart instead of requiring a post-hoc script.

</details>

<details>
<summary><b>2.3 — Augmentation: pitch is the second-biggest published effect, and we do none</b></summary>

[BT24] ablation, ordered by impact: **pitch** (92.6 → 88.3 beat F1, 85.4 → 80.8
downbeat), then **masking** (→ 92.2 / 84.5), then **tempo** (→ 92.5 / 84.9).

- **Pitch, ±6/−5 semitones.** `plans/08` §5.3 already says "it teaches the model
  that rhythm is not pitch. We do none." The literature now prices it: −4.3 beat
  F1 to omit, more than the entire transformer frontend is worth. [BT24]
  precomputes the variants offline (22 per song, tempo and pitch not combined)
  precisely so experiments are reproducible without the source audio — which
  suits our DVC-tracked data repo well.
- **Masking, and its specific form.** 0 to 6 regions of 0.5–2 s; each masked
  region is cut into 5–10 parts which are **randomly reordered** rather than
  zeroed. The stated reason is that zero-masking (SpecAugment) lets the network
  learn a dedicated behaviour for silence, whereas shuffling destroys the local
  audio–beat correspondence while keeping local statistics intact. The purpose is
  to stop the model relying on purely local evidence — which is the *same goal*
  as a longer receptive field, bought without longer input. Under the standing
  short-audio constraint (`plans/08` §7), that is the more interesting of the two
  routes.
- **Tempo.** Ours (±15 % time-stretch via `FrameTimeStretch`) is broader than
  theirs and already in place. [BD20]'s variant — jitter the STFT hop instead of
  the audio, N(annotated tempo, 5 %) — is cheaper and artefact-free, and [BT24]
  report verifying that their precomputed version matches it. No reason to
  switch; worth knowing the trick exists if time-stretch ever shows artefacts.

</details>

<details>
<summary><b>2.4 — The trunk: more context is not the answer, multi-rate context might be</b></summary>

[BD20] §2.3 is a direct negative result on the recommendation `plans/08` §3.1
makes: *"Increasing the temporal context of the TCN by either using larger kernel
sizes or adding more layers (with exponentially increasing dilation rates), did
not improve any of the tasks under investigation."* Their context was already
~40 s one-sided, which they judged sufficient.

What worked instead: **a second dilated convolution inside each TCN layer at
double the dilation rate**, feature maps concatenated, then spatial dropout, ELU,
and a 1×1 convolution to restore the channel count — so parameters grow linearly
with depth rather than exponentially. The stated motivation is musical: metre is
built from time scales that are integer multiples of each other, so let each
layer see two of them at once. A third rate did not help, which they attribute to
the near-absence of compound time signatures in the training data.

For us this is a ~20-line change in `TCNTempoNet`'s layer construction, it is
compatible with 16 s crops (adding a *2d* branch to the deepest layer is the one
place to watch, since 2×256 = 512 frames against a 689-frame crop), and it is a
different axis from the "cycled dilations" `plans/08` §3.1 proposes for depth
past nine layers. Worth noting the two are complementary and should not be
measured in the same run.

Also from this row of the board: both TCN papers put **spatial dropout 0.1 inside
the trunk**; our trunk has dropout only before the head. `plans/07` §1.2 measured
the position head's train/val gap widening to 0.194. Cheap to test, and it is
aimed at something we have quantified.

</details>

<details>
<summary><b>2.5 — The frequency axis: every paper mixes across frequency early, and we mix last</b></summary>

Three different mechanisms, one shared idea:

- [BD20] moves a **1×12 frequency-only convolution** into the middle of the stem,
  arguing from the literature that frequency-only filters capture harmonic and
  timbral structure, and that this matters both for music without drums and for
  downbeats, "where harmonic changes often occur at bar boundaries".
- [BT24] interleaves **attention over the frequency axis** with the convolutions
  in its frontend, adopted from Band Split RoFormer. Removing the partial
  transformers costs 0.4 beat F1 and 1.5 downbeat F1.
- [DB19] collapses frequency with a 1×8 convolution at the end of the stem.

Our `Conv2dStem` is three local 3×3 blocks with frequency pooling; every filter
in it is three bins wide, and the whole frequency axis is only mixed at the end,
by folding 16 channels × 14 bins into 224 and projecting. Nothing in our front end
can see a chord.

Our two worst corpora are `rwc_classical` (0.555 `f_beat`) and `rwc_genre`
(0.634). `plans/08` §2.2 proposes multi-resolution STFT for the same failure —
that is a different mechanism (window length, so time–frequency trade-off) for
the same goal (harmonic evidence). A 1×12 convolution inside the existing stem is
cheaper than both and composes with either.

</details>

<details>
<summary><b>2.6 — The head: nobody else makes the network do the bar arithmetic</b></summary>

[BD20] and [BT24] independently choose **two binary sigmoids** (beat, downbeat)
over a multi-class head, and both give the same reason: a multi-class formulation
"cannot fully leverage the information if a dataset contains only beat or
downbeat annotations" [BD20] / "to be able to train on datasets that do not
include downbeat annotations, we stick to binary classifiers" [BT24].

That argument does not bind us — `BeatDataset`'s `mask` channel solves exactly
that problem, and it is why `beat_position_loss` weights by `mask * beat_y`. But
a second consequence does: **in both systems the network is never asked "is this
beat a 2 or a 3"**. It emits beat and downbeat likelihoods; bar position between
downbeats is either implied or, in the DBN systems, inferred by a state space
that knows the bar length.

Our `group_size: 4` softmax asks that question directly. It was the right fix for
the defect `docs/beat_phase_improvement_review.md` step 3 diagnosed — under the
old one/last sigmoids, positions 2 and 3 carried identical supervision — and it
measurably improved fit. But the board makes visible that we chose the harder
network task and the weaker decoder at the same time, and `position_acc` 0.594 is
our weakest headline number. `plans/08` §4.1's "dedicated downbeat head" item is
the same observation arrived at from our side.

Two concrete details worth stealing regardless of that choice:

- **The Sum Head.** [BT24] add the downbeat logit into the beat logit, so a
  confident downbeat pushes up the beat prediction at the same frame. It nearly
  halves downbeats that land more than 70 ms from any beat (1.1 % → 0.62 %).
  Their ablation gives it 0 F1 — they keep it for musical validity, not score.
  Our decoder enforces the same invariant structurally, by reading positions only
  at detected beats, so this is a non-issue for us. Worth recording that it is
  *already* handled rather than rediscovering it.
- **Snap, then report.** They move every downbeat prediction to the nearest beat
  prediction as a final step. Same invariant, enforced twice.

</details>

<details>
<summary><b>2.7 — The decoder: [BT24] explains <i>why</i> <code>plans/08</code> §4.2 was right to reverse itself</b></summary>

`plans/08` §4.2 already killed the bar-pointer DBN as a priority, on measurement:
madmom carries our exact `amlt − cmlt` gap (0.149 vs our 0.148) while Beat This!,
with no decoder at all, has 0.042. This paper supplies the mechanism and the
price, from the other side.

Their argument against the DBN is that it hard-codes musical assumptions — a
tempo range (55–215 BPM), a fixed list of beats-per-bar, a single tempo-
variability prior tuned on pop/rock/dance, and periodicity itself. Pieces outside
those assumptions are mispredicted by construction, and they are a minority of
standard datasets, so "in terms of evaluation metrics, it usually does not pay to
remove the DBN" — but keeping it blocks progress on exactly the corner cases.

And they measure both sides. Bolting a DBN onto their own model, on GTZAN:

| | beat F1 | beat CMLt | beat AMLt | DB F1 | DB CMLt | DB AMLt |
|---|---|---|---|---|---|---|
| Beat This!, no DBN | **89.1** | 79.8 | 89.8 | **78.3** | 67.3 | 79.1 |
| the same model + DBN | 88.1 | **80.5** | **91.1** | 77.4 | **73.3** | **87.8** |

So a DBN is a trade, not an upgrade: it buys continuity and costs F-measure. Our
deficit *is* continuity (`cmlt` 0.514 against `amlt` 0.746), so the trade points
our way — but it stays where `plans/08` §7 put it, last, because §2.1 of this
document is worth an order of magnitude more in published effect and because
`plans/08` §1.3 showed a full DBN does not close that gap anyway.

The honest reading of all three papers together: **[BT24] did not remove the DBN
by improving the decoder, they removed it by improving the loss.** Their
postprocessing is strictly simpler than ours — local maximum within ±3 frames,
probability above 0.5, no periodicity gate — and it works because the
activations are sharp and confident. That is §2.1 and §2.2, not a decoder
project.

</details>

<details>
<summary><b>2.8 — Frame rate: a smaller problem than the 2019 paper alone suggests</b></summary>

[DB19] and [BD20] both run 100 fps, which makes our 43 fps look coarse:
quantising an annotation onto our grid costs up to ±11.6 ms against a ±70 ms
scoring window, and `min_distance_frames: 4` is 93 ms for us where it would be
40 ms for them.

[BT24] runs **50 fps** and is the state of the art. Their peak-picking
neighbourhood of ±3 frames is ±70 ms at their rate; ±3 frames at our 43 fps is
±70 ms too, and our swept grid brackets it (`[1, 2, 4]`, with 4 selected on most
runs — 93 ms). The two systems are doing the same thing at nearly the same
resolution.

This demotes the hop-256 idea in `plans/08` §7 step 4, which already carries a
gate ("histogram per-beat timing errors; if under ~2 % of matched beats sit in
the outer 11.6 ms band, the justification collapses"). Keep the gate, and note
that the strongest available evidence is that 50 fps is enough. The 2× compute
this would cost is better spent on §2.3.

</details>

<details>
<summary><b>2.9 — Measurement hygiene: three seeds, and what our leaderboard spread means</b></summary>

[BT24] runs every configuration **3 times** and reports mean ± standard
deviation; on GTZAN their beat F1 is 89.1 ± 0.3, and their ablation differences
of 0.1–0.4 are reported next to σ of 0.0–0.4, i.e. explicitly at the edge of
resolution.

Our leaderboard's top four runs span 0.763 / 0.761 / 0.756 / 0.755 macro
`f_beat`. That is 0.008 across four different architectures and learning rates —
within the seed noise band of a published system, and we have one seed each. We
cannot currently tell those four apart, and any conclusion drawn from their
ordering is unsupported.

[BD20] is relevant here too: they adopted RAdam + Lookahead specifically because
the combination made models "less sensitive to different random initialisations",
alongside gradient clipping at norm 0.5 and lr 2e-3. Our top run uses lr 8e-4,
the same value [BT24] warm up to.

**The cheap fix is not three seeds on everything** — it is three seeds on the
next decision that matters, and a note on the leaderboard that rows within ~0.01
are tied. `tools/leaderboard.py` reports macro means with no dispersion; a
seed column would make the tie visible.

</details>

---

## 3. Every published effect size, in one table

Ranked by magnitude. This is the column `plans/08` §7 was ordered without.

| change | source | measured effect | do we do it? |
|---|---|---|---|
| shift-tolerant loss **+** positive weighting | [BT24] ablation | **−13.1 beat F1, −16.7 downbeat F1** when both removed | weighting ✓, shift tolerance ✗ |
| pitch augmentation | [BT24] ablation | **−4.3 beat F1, −4.6 downbeat F1** when removed | ✗ |
| shift-tolerant loss alone | [BT24] ablation | −1.4 beat F1, −3.2 downbeat F1 | ✗ |
| partial (frequency-axis) transformers | [BT24] ablation | −0.4 beat F1, −1.5 downbeat F1 | ✗ (attention exists, off) |
| masking augmentation | [BT24] ablation | −0.4 beat F1, −0.9 downbeat F1 | ✗ |
| tempo augmentation | [BT24] ablation | −0.1 beat F1, −0.5 downbeat F1 | ✓ (broader) |
| Sum Head | [BT24] ablation | 0.0 beat F1, −0.4 downbeat F1; halves invalid downbeats | n/a — our decoder enforces it |
| 20 M → 2 M parameters | [BT24] Table 2 | −0.3 beat F1, −1.1 downbeat F1 | we are at 0.04 M |
| adding a DBN to a well-trained model | [BT24] Table 2 | **−1.0 beat F1, +0.7 CMLt, +1.3 AMLt** | ✗ (correctly, per `plans/08` §4.2) |
| larger kernels **or** more TCN layers | [BD20] §2.3 | **no improvement on any task** | — |
| second dilation rate per TCN layer | [BD20] Fig. 3 | CMLt and tempo Acc1 improve; no per-item number given | ✗ |
| 1×12 frequency conv mid-stem | [BD20] Fig. 3 | same, no per-item number | ✗ |
| adding the downbeat task to a beat model | [BD20] Fig. 3 | improves *beat* CMLt and tempo Acc1 | ✓ (as bar position) |
| TCN instead of BLSTM | [DB19] Table III | parity on F, 60× faster training, 1/3 the weights | ✓ |

Two cautions. [BT24]'s ablation is on their single-split validation set with a
20 M transformer and 4 556 tracks; effect sizes do not transfer verbatim to a
40 k-parameter TCN on 1 854 tracks. And [BD20]'s ablation is a figure without a
table, so those three rows are directional only — the paper's own summary is that
there is "no magic bullet" among them and that the combination is what works.

---

## 4. Where we are already aligned, or ahead

Recording these so they stop being re-litigated:

- **Front end.** 22.05 kHz, 128 mel bands, frozen per-band normalisation, ~43–50
  fps — the [BT24] configuration, arrived at independently.
- **Decoder.** Threshold + local maximum + non-maximum suppression *is* [BT24]'s
  postprocessing. `plans/08` §4.2 established the DBN is not our gap; this
  confirms the positive form of that claim — the decoder we have is the one the
  state of the art uses.
- **Positive class weighting.** [BT24] call it "crucial", computed as neg/pos over
  the training set. `beat_pos_weight(…, "auto")` derives it *per sample* from the
  target, so it tracks tempo and time-stretch augmentation instead of being one
  constant correct at one tempo. This is a refinement of their approach, not a
  gap.
- **Short inputs.** [BT24] train on 30 s excerpts and predict on non-overlapping
  30 s chunks; no system here needs the full track at inference. The standing
  short-audio constraint is not in tension with the field.
- **Model size.** At 40,773 parameters we sit between [DB19] (21,809) and [BD20]
  (116,302), and two orders below [BT24]. `plans/08` §1.2's "roughly 150 times
  larger than the reference architecture" was written before the 32×9 trunk
  landed and no longer describes the model — if anything there is now room to
  spend.
- **jtd.** Our best run scores 0.957 `f_beat` on jtd against Beat This!'s 0.945
  on the same 96 tracks, a corpus that is clean for them. That is a genuine win
  and worth remembering — with the caveat that jtd is ~64 % of our training
  tracks, so it reads as specialisation rather than capability. On `rwc_genre`,
  also clean for them, we score 0.634 against 0.843.

And one thing we do that none of them do: **Gaussian-smeared targets combined
with per-sample weighting**. [BT24]'s conclusion asks for "new losses that enforce
periodicity during training" and [BD20]'s asks for "a fundamentally different way
in which to present targets to the network which is better able to model temporal
uncertainty in the annotations". Our smearing is an attempt at the second — it is
just, per §2.1, the attempt that paper argues is the wrong shape.

---

## 5. Revised order

This does not replace `plans/08` §7. It re-ranks it, and every item below already
exists there or in `plans/07` except where marked **new**.

0. **Change the checkpoint monitor** (§2.2; `plans/07` §4.2). One line in
   `build_checkpoint_callback`, plus reconciling `EventMetricsLogger`'s
   `every_n_epochs: 5` with `save_top_k`. Do it before the next training run,
   because every run made under `monitor="val/loss"` has to be re-selected by
   hand otherwise. **Free.**

1. **Shift-tolerant BCE with sharp targets** (§2.1; `plans/08` §7 step 5,
   promoted from 5 to 1). Largest published effect in the literature we descend
   from, contained change, and the tolerance window coincides with the metric's.
   Decide the position-head gating question in §2.1 first. **Days.**

2. **Pitch augmentation, then masking** (§2.3; `plans/08` §5.3 / §7 step 6,
   promoted). Second-largest published effect, and masking substitutes for
   receptive field under the short-audio constraint. Precompute offline into the
   data repo as [BT24] do. **Days, mostly data plumbing.**

3. **Three seeds on whatever 1 and 2 produce** (§2.9, **new**). Without this,
   steps 1–2 land in the same 0.008 band the current top four runs sit in and we
   learn nothing. Cheap insurance, and it also prices the seed noise once for
   everything after.

4. **Trunk and stem, as two separate runs** (§2.4, §2.5, both **new** in this
   form): a second dilation rate per TCN layer, and a 1×12 frequency-only
   convolution mid-stem. Both are ~20-line changes with a stated musical
   rationale, and §2.4 comes with [BD20]'s negative result telling us *not* to
   spend the same effort on depth or kernel size instead. Spatial dropout in the
   trunk rides along with the first of these.

5. **Data** (`plans/08` §5.1, unchanged in rank). 1 854 tracks against 4 556; no
   loss function closes that.

6. **Architecture: frequency-axis attention** (§2.5; `plans/08` §3.3). Worth 0.4
   beat F1 in [BT24]'s ablation — real, but below everything above it, and
   `plans/08` §7's warning stands: do not compare architectures through a readout
   that is still being fixed.

7. **The bar-pointer DBN, last and expect little** (§2.7; `plans/08` §7 step 7,
   unchanged). Now with a published price tag: −1.0 beat F1 for +0.7 CMLt on a
   well-trained model.

**What this moves.** `plans/08` §7 has the loss at step 5 and augmentation at
step 6, behind two front-end/trunk steps. The published effect sizes invert that:
the loss and the augmentation are the two largest measured effects in the
literature, and the front-end work — which is already built and merely untrained
— is worth less than either. Steps 2 and 3 of `plans/08` §7 are config changes
that have landed and simply need a run; they are not displaced by this document,
they are waiting on GPU time. Step 4's hop-256 half is demoted by §2.8.

---

## 6. Relationship to the other plans

- **`plans/07`** — §4.2 (stop selecting on `val/loss`) is confirmed by [BT24] §4.2
  with a mechanism; §1.3's loss decomposition is the same phenomenon that paper
  describes, and is now logged per epoch as `{train,val}/loss_beat` and
  `{train,val}/loss_position`.
- **`plans/08`** — §4.1 (shift-tolerant BCE, dedicated downbeat head) and §5.3
  (augmentation) are confirmed and promoted. §4.2 (no DBN) is confirmed with the
  mechanism and the price. §3.1's "more context" framing is contradicted by
  [BD20] §2.3, which found larger kernels and more layers helped nothing;
  §2.2's hop-256 half is weakened by [BT24] running at 50 fps. §1.2's "150 times
  larger than the reference architecture" is stale after the 32×9 trunk.
- **`docs/frame_vs_event_metrics.md`** — [BT24] §4.2 ("the BCE loss is not a good
  indicator of the F1 score") is a fifth reason frame and event measurements
  disagree, and the only one that is about the *loss* rather than the metric.
