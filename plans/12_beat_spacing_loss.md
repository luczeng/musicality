# Part 12: the beat-spacing loss

**Status:** plan, not started. Needs plan 10, and reuses plan 11's ramp and
logging. Implements the beat term of `plans/softargmax_periodicity_math.md`
(the theory doc), §1 to §3.

---

## Overview

### High level

The beat head is trained one frame at a time: is there a beat here or not?
Nothing asks consecutive beats to be as far apart as the annotated ones, yet
that is exactly what the continuity metrics check, since they reject a beat
whose gap to the previous one is more than 17.5% off. The beat-spacing loss
adds that check to training. It finds where the network puts each beat, to a
fraction of a frame, and penalises gaps that differ from the annotated gaps by
more than annotation noise.

It only moves peaks that are already there. Deciding whether a beat exists
stays the existing loss's job, so pairs of beats without a clear peak are
faded out rather than trained on.

What changes: a new training term, off by default, ramping in on the same
schedule as the periodicity loss. The model and inference are unchanged. Two
new numbers are logged in every run: how many beat gaps the metric would
accept, and how strongly the loss's gate weighs them.

### Technical

- A new module, `musicality/losses/beat_spacing.py`, built in two milestones:
  1. pure, testable building blocks: the windows and the soft-argmax;
  2. the loss on top of them.
- Two measurements come before the training runs, one on the model and one on
  the annotations. They set the gate and the deadzone.

---

## 1. Milestone A: windows and soft-argmax

Pure functions, each tested on its own:

- `beat_windows(beats, n_frames, half_width) -> (frames, in_window, valid)`.
  - Each beat's window is its Voronoi cell, capped at `half_width` ($H$) frames
    on either side, with the edge rules of theory doc §2.1.
  - `frames` holds integer frame indices, shape `(B, K, W)` with
    $W = 2H + 1$. They start at $\lceil b_k - H \rceil$, and `in_window` masks
    the frames outside the cell or the clip.
  - `valid` is §2.1's $v_k$: not cut by the clip edge, at least 3 frames.
  - Padding slots, and a clip with a single beat (no gap to borrow), are
    invalid.
- `adaptive_softmax(logits, mask, kappa, tau_min) -> probs`.
  - Implements theory doc §1. The max, mean and sum run over masked-in frames
    only.
  - The temperature is computed under `torch.no_grad()`, so it is detached.
- `soft_argmax(beat_logits, frames, in_window) -> (centres, peak_logits)`.
  These are $\hat b_k$ and $m_k$, through one `gather` of the beat logits.

The theory doc's defaults (§8) become module-level constants: $H = 34$,
$\kappa_b = 8$, $\tau_{\min} = 0.5$, $\varepsilon = 0.05$, $\gamma = \log 1.175$.

## 2. Milestone B: the loss

`beat_spacing_loss(beat_logits, beats) -> scalar`, theory doc §3 line by line:

1. The log-interval residual $\rho_k$, with both gaps clamped at 0.5 frames.
2. The excess $x_k = [\thinspace\lvert\rho_k\rvert - \varepsilon\thinspace]_+ / \gamma$.
3. The Huber penalty $h(x_k)$.
4. The gate $g_k = \mathrm{sg}[\sigma(m_k)\thinspace\sigma(m_{k+1})]$.
5. A mean over the pairs whose two windows are valid, pooled over the batch,
   divided by the number of pairs, not by the gate sum.
6. A zero that keeps its gradient when no pair qualifies.

`beat_spacing_diagnostics(beat_logits, beats) -> dict`, no gradient:

| Logged key | Value | Why |
|---|---|---|
| `spacing_in_tol` | share of pairs with $\lvert\rho_k\rvert < \gamma$ | the continuity metrics' own acceptance test, per gap, on clips |
| `spacing_gate` | mean $g_k$ over pairs with $\lvert\rho_k\rvert < \gamma$ | open question 8: how much correct pairs count |

## 3. Measurements before training

Both are one-off scripts in a scratch folder. Their numbers are recorded here.

- **Gate on v7** (open question 8). Run the diagnostics over the v7
  checkpoint's validation clips. If `spacing_gate` is low, around the 0.2 the
  theory doc estimates, the term is several times weaker than its weight
  suggests. Then the choice before training is between a larger weight and the
  hard gate.
- **Annotation jitter per corpus** (open question 6). This needs annotations
  only, no model. Take the distribution of
  $\lvert\log(\Delta_{k+1}/\Delta_k)\rvert$ over each corpus, along with the
  same quantity in milliseconds. It gives the deadzone $\varepsilon$ a measured
  value. It also shows whether the noise is closer to constant in percent, so a
  relative deadzone, or in milliseconds. If the second, switching to a deadzone
  in milliseconds is decided with the user before training.

## 4. Wiring into training

As plan 11, with one more hyperparameter, `spacing_weight`, default 0.0 (config
key `spacing_weight`), on the shared `aux_ramp`.

- The training loss adds
  `spacing_weight * ramp(current_epoch) * spacing_term`.
- `val/loss` stays beat term + position term, for the same reasons as in plan
  11.
- Logged in every run, whatever the weight: `{stage}/loss_spacing`,
  unweighted, and the two diagnostics.
- Documented in `docs/source/configuration.rst`, `docs/source/losses.rst` and
  CLAUDE.md, as in plan 11.

## 5. Verification

**Unit tests**, in `tests/test_beat_spacing.py`:

- **Windows.** Disjoint and ordered; never wider than $2H + 1$. First and last
  beats borrow the neighbouring gap. Windows cut by the clip edge, windows
  under 3 frames and padding are invalid.
- **Softmax.** When the floor is inactive, the scaled mean logit sits exactly
  $\kappa_b$ below the scaled max. The floor applies on flat logits.
  Masked-out frames get probability 0. No gradient flows through the
  temperature.
- **Soft-argmax.** A single sharp peak gives $\hat b_k$ within 0.01 frames of
  it. Two symmetric peaks give their midpoint.
- **Loss values against the theory doc's table** of $\rho$, $x$ and $h$:
  - a 3% error costs 0, with zero gradient;
  - 10% costs 0.04;
  - an octave error costs 3.49.
- **Invariances** (theory doc §7). Shifting every peak and beat by a constant
  leaves the loss unchanged, and so does scaling all times.
- **Redistribution.** The gradient of the loss with respect to the beat logits
  sums to zero over each window (§6.1). This also checks that the gate and the
  temperature are detached: adding a constant to a window's logits changes
  nothing.
- **Edges.** Perfectly spaced peaks give 0. A batch with no valid pair gives
  0 with a gradient.

**Training**: on the current config, a baseline and `spacing_weight: 1`. The
baseline can be plan 11's, if nothing else has changed since.

- Compare `val_event/cmlt` and `amlt`, the metrics this term targets, and
  `val/spacing_in_tol`.
- Check that `val_event/f_beat` does not fall.
- A second pair with `loss: shift_tolerant` tests open question 9, where the
  adaptive temperature is meant to matter.

Once both losses work on their own, one run with both settles whether they
add up.
