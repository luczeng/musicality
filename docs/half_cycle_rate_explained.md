# What `confusion_half_cycle_rate` measures

`confusion_half_cycle_rate` (`musicality/metrics/confusion.py`) measures one
specific failure: **the model found the right beat periodicity but locked onto
the wrong phase — exactly half a bar off.**

## What "half a cycle" means

With `group_size=4`, a bar cycles `1 -> 2 -> 3 -> 4 -> 1`. Half a cycle away
from position 1 is position 3 — the diametric opposite
(`musicality/metrics/confusion.py:47`: `opposite = 1 + group_size // 2`). So the
error is calling beat 3 the downbeat, or calling the downbeat beat 3. For
`group_size=8` (phrases) the opposite is position 5.

This is a *parity* error. It's musically distinctive: the model is tapping in
time, at the right tempo, with a consistent bar grid — it just started counting
on the wrong beat. Perceptually that's a much worse and more specific mistake
than a generic mislabel, which is why it's measured separately from
`musicality.metrics.f_measure.downbeat_f_measures`.

## How it's computed

1. **Eligible** = reference beats whose true position is 1 or 3
   (`confusion.py:49`). Positions 2 and 4 are ignored.
2. For each, find the nearest predicted event (`confusion.py:63`).
3. **Skip** it if the nearest prediction is more than `tolerance` (0.07s) away
   (`confusion.py:64`) — that's a detection/timing failure, not a phase failure.
4. **Skip** it if the predicted label isn't 1 or 3 (`confusion.py:68`) — an
   unresolved `None` or an off-by-one label is a different failure.
5. Whatever survives is **matched** (the denominator). It's **swapped** if the
   predicted label is the *other* one of the pair (`confusion.py:72-73`).

```
rate = n_swapped / n_matched
```

Returns `None` when there is nothing eligible or nothing matched — i.e. the
metric declines to report rather than fabricating a 0.

The tight denominator is deliberate: misses and unresolved labels are
*excluded*, not scored as correct, so they can't dilute the signal. It answers a
narrow question — *among beats where the model committed to a downbeat-axis
label and got the timing right, how often was it exactly half a bar out?*

## Reading the number

- **0% = perfect.**
- **50% = the floor for a phase-blind model.** If the model picks a bar-phase
  offset at random from `{0, 1, 2, 3}`: offsets 1 and 3 produce labels 2/4 on
  eligible beats and get filtered out at step 4 entirely; offset 0 scores 0%,
  offset 2 scores 100%. Averaged over the two offsets that survive filtering ->
  50%.
- **The 25.3% reported in `docs/beat_phase_context_ideas.md` sits halfway
  between correct and coin-flip.** Roughly one in four eligible beats is on the
  wrong half of the bar. That's why it's the headline problem rather than the
  91.6% beat F-measure.

## Two blind spots worth knowing

- **It cannot see quarter-cycle (off-by-one) errors.** A model consistently one
  beat early emits labels 2/4 on eligible beats, which step 4 discards — so
  those tracks contribute *nothing* to the metric. A model could post a healthy
  half-cycle rate while being badly off by one. `downbeat_f_measures` is what
  catches that.
- **Matching is nearest-neighbour, not one-to-one** (unlike `mir_eval`'s
  assignment). One prediction can be the nearest match for several reference
  beats. Fine in practice given a 70ms window, but it's not a bijection.

## Related

- `docs/beat_phase_context_ideas.md` — the original plan this metric is
  diagnosing.
- `docs/beat_phase_improvement_review.md` — critique of that plan and a ranked
  set of fixes aimed at this metric.
