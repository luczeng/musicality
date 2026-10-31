# Part 11: the periodicity loss

**Status:** plan, not started. Needs plan 10. Implements the position
periodicity term of `plans/softargmax_periodicity_math.md` (the theory doc)
§4.2, and the wiring both new losses share.

---

## Overview

### High level

The position head names each beat's place in the bar, and it is right about
three times in four. It is trained one beat at a time, so nothing asks it to
*count*: a head that slips from 1 2 3 4 to 3 4 1 2 halfway through a clip is
penalised for the wrong beats, but not for breaking the count. The periodicity
loss adds that. It asks all beats of a clip to be shifted from the annotation
by the same amount, whichever amount it is, and leaves "which beat is the 1"
to the existing loss.

The loss can only fix one kind of error: a beat whose label disagrees with the
rest of its clip (a *slip*). It cannot fix a whole clip that counts correctly
from the wrong beat (a *shift*). So the plan first measures how the current
model's errors split between the two. If slips are rare, the loss is not worth
building.

What changes: a new training term, off by default, which ramps in after the
first epochs. The model and inference are unchanged. Three new numbers are
logged in every run, from now on, saying how the position head's errors split
into slips and shifts.

### Technical

- A new module, `musicality/losses/periodicity.py`, with three functions:
  offset probabilities, the loss in its two candidate forms, and the
  slip/shift diagnostics.
- `BeatPhaseModule` gets a weight, a form and a ramp schedule. The ramp is
  shared with plan 12.
- Order of work: diagnostics, then a measurement on v7, then a decision, then
  the loss, then three training runs.

---

## 1. Diagnostics first

Each beat of a clip has an offset: its predicted position minus its annotated
one, around the bar. Take the predicted position as the head's argmax at the
beat's frame, not the decoded label: the decoder smooths slips away, which is
why `position_acc_best_offset` cannot answer this question. Per clip, let
$P(o)$ be the share of beats with offset $o$ and $\hat o$ the most common
offset:

| Logged key | Value | Meaning |
|---|---|---|
| `head_acc` | $P(0)$ | per-beat accuracy of the raw head |
| `head_slip` | $1 - P(\hat o)$ | beats off their clip's own count, the loss's target |
| `head_shift` | $P(\hat o) - P(0)$ | beats on a count that starts on the wrong beat |

`head_slip + head_shift = 1 - head_acc`, and both are non-negative. Each is
averaged per clip, then over the clips with at least $2G$ annotated beats, as
the loss is. They are a per-clip, raw-head analogue of `position_accuracy`'s
`position_acc_best_offset` and `anchor_error`. They are logged as
`{stage}/head_*` in every run, whatever the loss weight.

**Measurement (open question 1).** Run these diagnostics once over the v7
checkpoint's validation clips, with a scratch script that loads the module and
calls `trainer.validate`. Record the three numbers here.

**Decision.** Decided together, on those numbers. As a rough bar: if
`head_slip` is under a third of the per-beat error, most of the error is
shifts, which this loss cannot fix, and plan 12 goes first.

## 2. The loss

`musicality/losses/periodicity.py`:

- `offset_probabilities(position_logits, beats) -> (pi, valid)`.
  - Reads the position softmax at $f_k = \mathrm{clip}(\mathrm{round}(b_k), 0, T-1)$.
  - Rotates each beat's $G$ probabilities to start at its reference position,
    `gather` with index $(n_k - 1 + o) \bmod G$.
  - Shapes `(B, K, G)` and `(B, K)`. A beat is valid if it is not padding and
    carries a position.
- `periodicity_loss(position_logits, beats, form, min_beats) -> scalar`.
  - $\bar\pi_b$ is the mean of `pi` over a clip's valid beats.
  - The clips counted are those with at least `min_beats` valid beats,
    $2G$ by default.
  - It returns a zero that keeps its gradient when no clip qualifies.
- `head_slip_rates(position_logits, beats) -> dict`: the §1 diagnostics, no
  gradient.

The two forms are the two sides of open question 2:

| `form` | Clip loss | Behaviour |
|---|---|---|
| `spread` | $1 - \sum_o \bar\pi_b(o)^2$ | theory doc §4.2 as written; rewards agreement *and* confidence |
| `agreement` | $1 - \sum_o \bar\pi_b(o)^2 \big/ \overline{\textstyle\sum_o \pi_k(o)^2}$ | agreement only; the theory doc's likely default |

**A risk in `agreement`, not stated in the theory doc.** Its minimum is any
clip whose beats all share one distribution, including the uniform one.
Flattening every beat is therefore a way down, as well as agreeing. Two
sure beats that disagree cost 0.5 and cost 0 once both are uniform. The
cross-entropy resists that, but how strongly is unknown. The guard is a logged
value: the mean, over beats, of the probability of the head's top position. If
it falls in the `agreement` run while `head_acc` does not rise, the form is
flattening. The first fix to try is detaching the denominator.

Read frame (open question 3): the annotated frame $f_k$. The current config
trains with `loss: bce`, where that is also where the cross-entropy is centred.

## 3. Wiring into training

`BeatPhaseModule` gets three hyperparameters. Defaults reproduce today's
objective exactly, and old checkpoints load with them:

| Hyperparameter | Default | Config key |
|---|---|---|
| `period_weight` | 0.0 | `period_weight` |
| `period_form` | `"agreement"` | `period_form` |
| `aux_ramp` | `(5, 15)` | `aux_ramp` (shared with plan 12) |

- **The ramp** is a plain function, `ramp(epoch, start, end) -> float`. It is
  0 before `start`, 1 after `end` and linear in between (theory doc §5).
- **Training loss**: beat term + position term +
  `period_weight * ramp(current_epoch) * period_term`.
- **`val/loss` stays beat term + position term.** It drives
  `ReduceLROnPlateau`, and a term ramping up over epochs 5–15 would read as a
  plateau and cut the learning rate. It also keeps `val/loss` comparable with
  earlier runs.
- **Logged in every run**, whatever the weight:
  - `{stage}/loss_period`, unweighted;
  - the three `head_*` diagnostics;
  - `{stage}/head_confidence`, the mean top-position probability from §2.

The new keys go in `configs/train_phase_beat.yaml`, documented in
`docs/source/configuration.rst`. The module goes in `docs/source/losses.rst` and
CLAUDE.md's loss list.

## 4. Verification

**Unit tests**, in `tests/test_periodicity.py`:

- **The theory doc's tables.** One-hot predictions on a clip annotated
  1 2 3 4 1 2 3 4. Correct and shifted predictions give `spread` 0. A slip by
  two, a slip by one and a restart every two beats give 0.5. Uniform gives
  0.75.
- **The two-beat examples**, 70% sure: 0.48 and 0.66 under `spread`, 0 and
  0.35 under `agreement`.
- **Offset invariance.** Rolling every beat's logits by the same amount leaves
  both forms unchanged.
- **Gradient.** Nonzero only at the read frames' position logits, and zero on
  the beat logit.
- **Edges.** Clips under `min_beats` and padded or unannotated beats are
  ignored. A batch with no qualifying clip gives 0 with a gradient.
- **Diagnostics.** Known offsets give known `head_slip` and `head_shift`, and
  the two sum to `1 - head_acc`.
- **Ramp.** 0, 0.5 and 1 at epochs 5, 10 and 15.

**Training**: three runs on the current config, identical apart from the term:

| Run | `period_weight` | `period_form` |
|---|---|---|
| baseline | 0 | — |
| A | 1 | `agreement` |
| B | 1 | `spread` |

The baseline is rerun rather than taken from the leaderboard, so that all three
share code, seed and data. Compare them on `val_event/position_acc`,
`position_acc_best_offset` and `val/head_slip`, and check that
`val_event/f_beat` does not move. Final numbers come from `tools/leaderboard.py`.
The term works if `head_slip` falls and `position_acc` rises. If `head_shift`
rises instead, the term is pulling the minority towards a wrong majority
(theory doc §6.2), and open question 4's later ramp is the next thing to try.
