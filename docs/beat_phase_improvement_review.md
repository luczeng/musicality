# Beat-phase: review of `beat_phase_context_ideas.md`, and what I'd do instead

A critique of `docs/beat_phase_context_ideas.md` and a ranked plan for improving
phase (bar-position) estimation.

> **Status:** steps 0 and 1 have since been run. See
> [RESULTS](#results-step-0-and-step-1-run-2026-08-29) below — the headline is
> that the decoder, not the model, was responsible for a large part of the
> error, and the priorities in section 7 changed as a result. The analysis
> above the RESULTS section is preserved as written, before the measurements. Written after reading the ideas doc, the TCN
(`musicality/models/tcn.py`), the loss (`musicality/losses.py`), the
postprocessor (`musicality/postprocess.py`), and the eval path
(`musicality/inference.py`, `tools/eval_beat.py`,
`tools/sweep_beat_postprocess.py`).

## The core challenge: the diagnosis was never actually tested

`beat_phase_context_ideas.md`'s whole plan (longer context -> SSA -> chunked
inference) rests on one inference: *"frame accuracy is fine, decoded phase is
bad -> therefore the model lacks long-range context."* That's one of at least
three explanations, and the cheapest discriminating experiment was never run.

The evidence in the doc actually argues *against* the context story:

- train `acc_one` 93% / val `acc_one` 89-90%. That's a ~3pt gap — the model is
  **underfitting**, not failing to generalize. A "can't carry a phase belief
  across the clip" model would be bad at phase on *training* data too, and
  nobody measured that.
- 25.3% half-cycle rate against a 50% random-phase floor means phase is right
  ~76% of the time given the beat is right. That's not "misses occasional
  ambiguous stretches", that's structurally broken.

### Step 0, before writing any architecture code

One `eval_beat.py --split train` run, ~an hour:

1. `confusion_half_cycle_rate` on the **train** split.
   - train ~= val ~= 25% -> it's the objective or the decoder, and neither
     context nor data will fix it.
   - train << val -> it's generalization, and the doc's deferred data
     expansion becomes the top lever.
2. Per track, compute the **modal phase offset** between prediction and truth,
   and the fraction of beats deviating from it. This separates two totally
   different failures:
   - *"the whole track is offset by 2"* — the model genuinely can't hear
     downbeats (acoustic/objective problem).
   - *"phase flips mid-track"* — the decoder's greedy resync (postprocessing
     problem).

   The fix is completely different in each case and the current metrics can't
   tell them apart.

My bet: flips, for the reason in section 1.

## 1. The decoder throws away most of the evidence — fix it first, no retraining

`label_bar_position` (`musicality/postprocess.py:157`) is a greedy forward-count:

- Only beats with `p > 0.8` vote (`musicality/postprocess.py:212`). Everything
  softer — `p_one=0.45` vs `p_last=0.15`, which is real evidence — is discarded.
- **Any single false vote resyncs the counter and corrupts every beat
  downstream until the next vote** (`musicality/postprocess.py:221`). One bad
  anchor wrecks a long stretch. That is exactly the shape of a 25% confusion
  rate.
- It never revises backwards. A confident anchor at beat 100 can't correct
  beats 1-99.
- It can't use the fact that `last` sits immediately *before* `one`. It treats
  them as two independent vote sources into one counter.

The function's own docstring admits `anchor_threshold` is a non-monotonic,
two-sided knob with an interior optimum. That's not a parameter to tune — it's
the algorithm signalling that it's the wrong algorithm.

### Replacement: a global decode over bar-position states

With deterministic `+1 mod G` transitions this collapses to something almost
trivial — for each of the `G` candidate phase offsets, sum `log p` over *every*
beat in the track and take the argmax:

```
for a beat at frame f:
  ll(pos=1)     = log p_one[f]      + log(1 - p_last[f])
  ll(pos=G)     = log p_last[f]     + log(1 - p_one[f])
  ll(pos=other) = log(1 - p_one[f]) + log(1 - p_last[f])

offset* = argmax_offset  sum_beats ll(pos = (i + offset) mod G)
```

O(G*N), uses every beat's soft evidence, and `anchor_threshold` disappears
entirely. Upgrade to a proper Viterbi with a small "phase change" transition
penalty if tolerance for genuine meter changes and intros is needed.

Crucially: `tools/sweep_beat_postprocess.py` already caches per-track frame
probabilities and re-scores decoders cheaply. **This can be A/B'd against the
existing checkpoint without a single training step.** If the errors are
decoder-driven, this halves the confusion rate for a day's work.

## 2. The phase heads are trained on the wrong frames

`beat_phase_loss` (`musicality/losses.py:145`) supervises `one`/`last` on
**every frame** of the clip. But at inference, `label_bar_position` only ever
samples those curves **at detected beat frames**. So ~97% of the phase heads'
gradient goes into learning "this is not a beat at all" — which the beat head
already does at 91.6% F — and ~3% into the question that actually matters:
*given this is a beat, is it a 1?*

`pos_weight=18` patches the class imbalance; it doesn't remove the mismatch
between what's optimized and what's read.

Fix — weight the `one`/`last` BCE terms by the beat target (or a slightly
dilated version) in addition to `mask`:

```python
phase_w = mask * beat_y          # or mask * dilate(beat_y, +/-2 frames)
n_phase = phase_w.sum().clamp(min=1.0)
one_term  = (one_loss  * phase_w).sum() / n_phase
last_term = (last_loss * phase_w).sum() / n_phase
```

This turns 1-vs-3 from a rare-event detection problem into a balanced 1-in-4
classification problem, and `pos_weight` for those heads should then drop from
18 to ~3. Small diff, one retrain, aimed squarely at the broken metric.

## 3. Two independent sigmoids is the wrong parameterization for a cyclic variable

Positions 2 and 3 receive *identical* supervision today: negative on both
heads. The model is literally never asked "is this a 1 or a 3?" — the
discriminative question the metric measures.

Make the phase output a `group_size`-way softmax over bar position, with
cross-entropy weighted by the beat target (i.e. section 2's weighting). Then
position 3 has its own logit and competes directly with position 1 inside one
normalization. It also hands the decoder in section 1 a proper emission
distribution for free, and generalizes to `group_size=8` phrases without
reshaping the loss. `beat` stays a separate sigmoid — it's fine at 91.6%.

## 4. On the SSA implementation as it stands

Four concrete problems, roughly in order of how much they matter:

### (a) Absolute sinusoidal PE is the wrong encoding here, and may be a net negative

`random_crop: true` means the same downbeat lands at absolute frame 37 one
epoch and 412 the next — absolute position carries *zero* musical information
in this setup. What matters for phase is relative distance in beats. So the PE
is pure nuisance the attention has to learn to ignore. Then at inference it's
evaluated ~15x outside its trained range on a full track.

Use a **relative position bias** (learned per-head scalar over clipped relative
distance, T5-style): cheap, length-generalizes, and expressed in exactly the
units phase lives in. Minimal-change alternative: randomize the PE's phase
offset per training sample so the model can't lean on absolute position at all.

The PE math itself (`musicality/models/tcn.py:50`) is correct — the problem is
the choice of encoding, not the implementation.

### (b) A mechanism by which the SSA head may be silently degrading full-track eval

Trained at T=690, run at T~=10,000, the softmax normalizes over ~15x more keys
with logits at the same scale — attention flattens, and the block's output
collapses toward "add the sequence mean". Training-time frame accuracy would
look completely healthy while the inference-time head does nothing useful.

`beat_phase_context_ideas.md` treats this as speculative extrapolation risk;
it's a specific, testable failure. Compare val frame accuracy on 16s clips
against the decoded full-track metric with the SSA path fed 16s chunks — that
isolates it.

### (c) Post-LN on an unnormalized input

`SelfAttentionBlock.forward` (`musicality/models/tcn.py:108`) feeds `x`
straight into `mha` with no normalization, and `x` arrives from 8 additive
residual conv layers so its scale is arbitrary and grows with depth. Post-LN
needs LR warmup to train stably; `BeatPhaseModule.configure_optimizers` uses
plain Adam + `ReduceLROnPlateau` with no warmup.

Switch to pre-LN — `h = x + mha(norm1(x))`, then `h = h + ff(norm2(h))`.
Strictly more robust, essentially a one-line change.

### (d) No dropout inside the block

`nn.MultiheadAttention` defaults to `dropout=0.0` and the FFN has none, while
the head convs get `dropout=0.2`. Tie them to the same config value.

## 5. The real architectural fix: attend over *beats*, not frames

The biggest available win, and the thing `beat_phase_context_ideas.md` doesn't
consider.

Frame-rate attention over 690 tokens to discover bar-level structure is a poor
use of attention: at 43fps, comparing bar N to bar N+16 is a ~1,300-frame
reach, and even at `duration: 16.0` there aren't 16 bars in the window.
**16s barely clears the trunk's own 11.9s RF — it is not enough to test the
doc's own hypothesis.** Frame-level SSA would need 30-40s crops, which is
expensive.

Instead: pool trunk features **at beat positions** (annotated beats at training
time; the beat head's peaks at inference — or a differentiable
beat-probability-weighted pooling), then run a small transformer over ~60-120
*beat tokens* with a `G`-way softmax per beat.

- Sequence length drops 10-40x, so a **full 4-minute track fits in one
  attention window** — which kills the entire planned chunked-inference MR and
  the length-generalization problem with it.
- Relative position measured in beats *is* the phase variable. The inductive
  bias is exact rather than approximate.
- The model can directly compare bar 1 to bar 20 — the "carry a phase belief
  past an ambiguous stretch" capability the doc wanted, and which frame-level
  SSA at 16s cannot deliver.
- "`last` is adjacent to `one`" becomes trivially learnable.

This is where the downbeat-tracking literature landed (beat-synchronous
features for downbeat estimation), for the same reasons.

## 6. Drop the chunked-inference MR

It's a workaround for a self-inflicted problem, and
`beat_phase_context_ideas.md` already concedes it just relocates boundary
effects to every 16s. With section 5 it's unnecessary; with relative position
bias (4a) it's mostly unnecessary. Don't spend the MR.

## 7. On the deferred data expansion

Mostly endorse the deferral, but not for the doc's stated reason. The 3pt
train/val gap says underfitting, so more data isn't the current bottleneck.

But the split hides a real risk: **ballroom's downbeats are uniquely
stereotyped** (fixed dance-genre patterns), so a ballroom+rwc model overfits
*genre*, and a within-ballroom val split cannot reveal that. If the end goal is
a general tool, gtzan/hainsworth/jtd matter — just don't expect them to move
the ballroom val number, and don't reach for them until step 0 says
generalization is the problem.

## RESULTS (step 0 and step 1, run 2026-08-29)

Both cheap steps are done. Checkpoint `checkpoints_beat/loss=1.6565.ckpt`,
ballroom, `binary_only=True`, via `tools/diagnose_beat_phase.py`.

### Step 0 verdict: a fit problem, but only half as much as it looked

| split | confusion (greedy decoder) | beat F | `1` F | `last` F |
|---|---|---|---|---|
| train (419) | 0.178 | 0.900 | 0.761 | 0.773 |
| val (104) | 0.253 | 0.916 | 0.697 | 0.692 |

Train confusion is 17.8%, not ~2%, so the model fails on tracks it has seen
hundreds of times. Data expansion cannot address that part. **But see the
re-decomposition below — the decoder was hiding half of it.**

Note also that `configs/eval_beat.yaml`'s beat-detection knobs were originally
swept on the ballroom *val* split, so val is optimistically biased and the
true train/val gap is at least as large as measured.

### Step 1 verdict: CAUSE (B) — the decoder, decisively

Same checkpoint, same cached probabilities, **no retraining**:

| decoder | train confusion | val confusion | val `1` F | val `last` F |
|---|---|---|---|---|
| greedy (`anchor=0.8`) — old default | 0.178 | 0.253 | 0.697 | 0.692 |
| global, exact (no resync) | 0.134 | 0.227 | 0.673 | 0.649 |
| **global + Viterbi (`switch=2.0`)** | **0.090** | **0.185** | **0.756** | **0.730** |

`switch_penalty` was tuned on **train** and reported on **val**, so the val
column is not self-selected. The optimum is genuinely interior (val confusion
is 0.274 at `switch=0.25` and 0.219 at `switch=40`), not a grid edge.

Now the default in `configs/eval_beat.yaml` (`decoder: global`,
`switch_penalty: 2.0`).

### The offset profile: the model was never the main problem

Dominant per-track phase offset under the old greedy decoder:

| offset | train | val |
|---|---|---|
| **0 (correct)** | **96.7%** | **87.5%** |
| 1 (off-by-one) | 1.4% | 2.9% |
| 2 (half-cycle) | 1.2% | 7.7% |
| 3 (off-by-one) | 0.7% | 1.9% |

Almost every track's *dominant* phase is correct. The 25% confusion was never
"the model can't hear downbeats" — it was instability *within* tracks. Phase
stability (fraction of beats agreeing with their own track's modal offset):
mean 0.768 train / 0.718 val, with 45% of train and 62% of val tracks below
0.80.

### Underneath it: the beat sequence drifts, and beat F-measure can't see it

The *exact* single-offset decoder scores stability 0.757 train / 0.784 val.
That decoder assigns one fixed offset to a whole track by construction, so it
is mathematically incapable of flipping phase on its own. The instability
therefore cannot come from the labeler — it must come from **insertions and
deletions in the predicted beat sequence**, each of which permanently shifts
every position after it by one.

Beat F-measure cannot detect this: it is set matching, indifferent to order
and count, so 0.90 looks healthy. But every decoder here *counts* beats, so a
beat error that barely dents F-measure scrambles the bar grid from that point
on. This also explains the shape of the decoder table — the rigid decoder
cannot recover from a beat-count error, the greedy one resyncs on noise, and
the Viterbi resyncs only when the evidence justifies it.

**New recommendation (was not in the original plan): make the phase decode
time-based rather than index-based.** Advance the position by
`round(dt / period)` instead of always `+1`, so a missed beat advances two
positions and a spurious beat advances zero. That targets the measured
mechanism directly and is a modest change to `label_bar_position_global`.

### Re-decomposition of the val error

| | before step 1 | after step 1 |
|---|---|---|
| decoder loss | (invisible) | **6.8 pts** |
| cannot fit even on train | 17.8 pts | **9.0 pts** |
| generalization gap | 7.5 pts | **9.3 pts** |

Half of what looked like a model failure was decoder loss. What remains splits
roughly **50/50** between fit and generalization, where it was 70/30 before —
so **section 7's data expansion is meaningfully more attractive than stated
there**, now addressing about half the remaining error rather than a third.

### Updated order of work

| # | Change | Status |
|---|---|---|
| 0 | Train-split diagnostic | **done** — fit problem, halved by step 1 |
| 1 | Global Viterbi decoder | **done** — val confusion 0.253 -> 0.185, shipped as the default |
| 1b | Time-based (not index-based) phase transitions | **new, next** — targets the measured beat-drift mechanism |
| 2 | Beat-conditioned phase loss (section 2) | pending |
| 3 | `G`-way softmax over positions (section 3) | pending |
| 4 | pre-LN + relative position bias + dropout in SSA (section 4) | pending |
| 5 | Beat-synchronous phase head (section 5) | pending |
| 7 | Data expansion (section 7) | **promoted** — now ~50% of the remaining error |

### Tooling added

- `musicality/postprocess.py` — `label_bar_position_global`; `readout` gained
  `decoder=` / `switch_penalty=`.
- `musicality/metrics/phase_offset.py` — `phase_offset_profile`.
- `tools/diagnose_beat_phase.py` — runs both experiments in one command.
- Threaded through `musicality/inference.py`, `musicality/evaluation.py`,
  `tools/eval_beat.py` and the annotator, defaulting from
  `configs/eval_beat.yaml`.

Known follow-up: `tools/sweep_beat_postprocess.py` still sweeps
`anchor_threshold`, which only the (now non-default) greedy decoder reads. It
should learn to sweep `switch_penalty` instead.

## Recommended order of work (original, superseded above)

| # | Change | Retrain? | Cost |
|---|--------|----------|------|
| 0 | Train-split confusion + per-track phase-offset diagnostic | no | ~1h |
| 1 | Global Viterbi / offset-argmax decoder (section 1) | no | ~1d |
| 2 | Beat-conditioned phase loss (section 2) | yes | small diff |
| 3 | `G`-way softmax over positions (section 3) | yes | moderate |
| 4 | pre-LN + relative position bias + dropout in SSA (section 4) | yes | small diff |
| 5 | Beat-synchronous phase head (section 5) | yes | large |

Steps 0 and 1 require no training at all and can be evaluated against the
existing checkpoint's cached probabilities via
`tools/sweep_beat_postprocess.py`. Do them before committing to anything in
steps 3-5.

## Quirks worth flagging

- `anchor_threshold` being a documented non-monotonic knob is a symptom of the
  wrong decoding algorithm, not a hyperparameter to sweep harder.
- The `PositionalEncoding` math is correct — the problem is that absolute
  position is meaningless under `random_crop: true`, not that it's implemented
  wrong.
- The attention-flattening-at-long-T mechanism (4b) means the SSA head could be
  a no-op (or worse) at full-track inference while every training-time metric
  looks fine. Frame accuracy will never show this.
- `data.duration: 16.0` is too short to actually test
  `beat_phase_context_ideas.md`'s own SSA hypothesis — it barely clears the
  trunk's 11.9s receptive field.
