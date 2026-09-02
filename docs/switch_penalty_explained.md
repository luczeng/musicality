# What `switch_penalty` does

Reference for the one tunable parameter of
:func:`musicality.postprocess.label_bar_position_global`, set in
`configs/eval_beat.yaml` under `beat_phase`.

## The rule it governs

The decoder assumes bar position simply counts forward — `1 → 2 → 3 → 4 → 1 →
2 …`, one step per beat, forever.

`switch_penalty` is **the price of breaking that rule**: jumping to some other
position mid-track instead of just incrementing.

In the code it is one subtraction, evaluated at every beat:

```python
switch_score = delta[other_idx] - switch_penalty   # jump, and pay the fee
take_switch  = switch_score > natural_score        # is it still worth it?
```

The decoder compares "keep counting" against "jump", with the fee charged
against the jump, and keeps whichever scores higher.

## The dial

| `switch_penalty` | behaviour |
|---|---|
| `null` / `None` | jumping is **forbidden** — one phase offset for the entire track |
| very high (40) | almost never jumps |
| **2.0** (shipped) | jumps only when the evidence clearly justifies it |
| very low (0.25) | jumps constantly, chases noise |

## What the number means

The decoder accumulates **log**-probabilities, so a penalty of 2.0 is 2 log
units. In plain probability that is `e² ≈ 7.4`.

> `switch_penalty = 2.0` means: only resync if doing so makes the rest of the
> track's evidence at least ~7× more likely.

## Why the parameter is needed at all

The detected beat sequence is not perfect. If the beat tracker **misses one
beat**, every position after it is off by one — permanently. With `null` the
decoder is locked into a single offset for the whole track and can never
recover, so the remainder of the track is wrong.

The resync is what lets it recover; each missed or spurious beat costs exactly
one resync.

Make it too cheap, though, and you have rebuilt the old greedy decoder's bug
(:func:`musicality.postprocess.label_bar_position`) — it resyncs on noise and
the phase flips around mid-track.

## Measured curve

Checkpoint `checkpoints_beat/loss=1.6565.ckpt`, ballroom, `binary_only=True`,
via `tools/diagnose_beat_phase.py`. Confusion is
`confusion_half_cycle_rate` — lower is better.

### Train (419 tracks) — the split the value was tuned on

| `switch_penalty` | confusion | `1` F | `last` F |
|---|---|---|---|
| 0.25 | 0.186 | 0.772 | 0.773 |
| 0.5 | 0.165 | 0.787 | 0.788 |
| 1.0 | 0.123 | 0.821 | 0.812 |
| 1.5 | 0.100 | 0.834 | 0.822 |
| **2.0** | **0.090** | **0.841** | **0.825** |
| 3.0 | 0.091 | 0.833 | 0.818 |
| 5.0 | 0.100 | 0.814 | 0.805 |
| 10.0 | 0.111 | 0.791 | 0.796 |
| 20.0 | 0.114 | 0.777 | 0.782 |
| 40.0 | 0.124 | 0.757 | 0.763 |
| `null` (exact) | 0.134 | 0.726 | 0.734 |

### Val (104 tracks) — reported, not tuned on

| `switch_penalty` | confusion | `1` F | `last` F |
|---|---|---|---|
| 0.25 | 0.274 | 0.700 | 0.686 |
| 0.5 | 0.257 | 0.711 | 0.693 |
| 1.0 | 0.229 | 0.737 | 0.711 |
| 1.5 | 0.201 | 0.749 | 0.723 |
| **2.0** | **0.185** | **0.756** | **0.730** |
| 3.0 | 0.172 | 0.752 | 0.728 |
| 5.0 | 0.183 | 0.735 | 0.709 |
| 10.0 | 0.189 | 0.731 | 0.701 |
| 20.0 | 0.201 | 0.706 | 0.684 |
| 40.0 | 0.219 | 0.693 | 0.668 |
| `null` (exact) | 0.227 | 0.673 | 0.649 |

```
val confusion
 0.25  0.274  ████████████████████  too eager — chases noise
 0.5   0.257  ██████████████████
 1.0   0.229  ██████████████
 1.5   0.201  █████████
 2.0   0.185  ██████                ← shipped
 3.0   0.172  ███
 5.0   0.183  ██████
10.0   0.189  ███████
20.0   0.201  █████████
40.0   0.219  ████████████
 null  0.227  █████████████         too rigid — cannot recover
```

A genuine U: bad at both ends, best in the middle — an interior optimum, not a
grid edge.

**Why 2.0 and not 3.0**, even though val confusion is lower at 3.0: the train
minimum is cleanly at 2.0 on both confusion and mean F, and 2.0 also wins on
mean(`1` F, `last` F) on val (0.743 vs 0.740). Taking the train-chosen value
avoids tuning on the split being reported — a flaw the beat-detection knobs in
`configs/eval_beat.yaml` already have, since they were swept on val.

## Retuning it

Tune on **train**, verify on **val**, then update `configs/eval_beat.yaml`:

```bash
uv run python tools/diagnose_beat_phase.py \
    --checkpoint <ckpt> --dataset ballroom --binary-only \
    --split train --switch-penalties 1 1.5 2 3 5

uv run python tools/diagnose_beat_phase.py \
    --checkpoint <ckpt> --dataset ballroom --binary-only \
    --split val --switch-penalties 1 1.5 2 3 5
```

The tool runs the model once per track and re-scores every penalty against the
cached probabilities, so adding values to the grid is nearly free.

Expect the optimum to move if the beat-detection quality changes: the penalty
is trading off against how often the beat sequence gains or loses a beat (see
below), so a better beat tracker should push the optimum **higher** (fewer
resyncs needed), and a worse one lower.

## The twist: it is not doing the job it was added for

### What it was expected to do

The jump was added for cases where **the music** does something unusual:

- a song switches from 4/4 to 3/4
- an intro starts mid-bar
- a bridge has one odd-length bar

That is rare, especially in ballroom. So the expectation was a **high** penalty
— meaning "almost never jump" — and a no-jump (`null`) decode that would be
nearly as good.

### What it actually does

The jump is needed because **the beat detector** makes mistakes. The music is
doing nothing unusual at all.

Concrete example. The song is plain 4/4 throughout:

```
true beats      0.0   0.5   1.0   1.5   2.0   2.5   3.0   3.5
true positions   1     2     3     4     1     2     3     4
```

Now the detector misses the beat at 2.0. That is the only thing that goes
wrong:

```
detected beats  0.0   0.5   1.0   1.5    x    2.5   3.0   3.5
decoder counts   1     2     3     4          1     2     3
truth            1     2     3     4          2     3     4
                                             ^^^^^^^^^^^^^^^
                                        wrong from here to the end
```

The decoder is counting perfectly. It simply has one fewer beat than the music
does, so its count slips by one — and stays slipped for the rest of the track.

The jump is what lets it notice that its count no longer matches the evidence
and correct. **It is fixing the beat list, not the song.**

### How we know

Three signs, in increasing strength:

1. **The best penalty is low (2.0), not high.** Low means "jump often". If
   jumps were only for rare musical oddities, a high value would have won.

2. **Forbidding jumps entirely is much worse** — val confusion 0.227 vs 0.185.
   If the music genuinely never broke the counting rule, banning jumps would
   cost almost nothing. It costs a lot.

3. **The stability argument, which is a proof rather than an inference.** The
   exact (`null`) decoder assigns one fixed offset per track *by construction*
   and therefore cannot flip phase on its own — its within-track phase
   stability should be 1.0. It measures 0.757 train / 0.784 val. The
   instability cannot come from the labeller, so it comes from insertions and
   deletions in the predicted beat sequence, each of which permanently shifts
   every position after it.

Beat F-measure cannot detect any of this: it is set matching, indifferent to
order and count, which is why 0.916 looked healthy.

### Why the distinction matters

If this were a *music* problem there would be nothing to fix — some songs
change meter, you handle it and move on.

Because it is a *beat detection* problem, a second fix exists: stop counting
beats and start counting time. Advance by `round(Δt / period)` instead of always
`+1`, so a missed beat leaves a two-period gap and the decoder moves two
positions instead of slipping.

That is implemented — `phase_advances` and
`label_bar_position_global(advance="time")` — and it works, but **it does not
replace this parameter, because the two are substitutes rather than
complements.** Measured on ballroom train:

| `switch_penalty` | `advance="index"` | `advance="time"` | time helps by |
|---|---|---|---|
| `null` (exact) | 0.134 | 0.093 | **+0.041** |
| 20 | 0.114 | 0.097 | +0.017 |
| 5 | 0.100 | 0.113 | -0.013 |
| 2 | **0.090** | 0.100 | -0.010 |

The cheaper resyncing gets, the less time-based advances add — and past a point
they inject noise that the Viterbi then pays to undo. Both mechanisms repair
the same beat-count errors, so running both at low penalty over-corrects.

**Two predictions in the original version of this section were wrong**, and are
kept here rather than quietly edited:

- *"this parameter's optimum should move higher"* — it did not. `2.0` remains
  the best value with `advance="time"` disabled, and enabling it does not shift
  the optimum, it just makes every low-penalty setting slightly worse.
- *"the count never slips in the first place"* — overstated. Time-based
  advances close only about 14% of the measured within-track drift (exact-decoder
  stability 0.757 -> 0.791). The rest comes from somewhere else, most likely
  `gate_periodicity`'s ambiguous branch, which accepts an off-grid beat without
  updating its period estimate.

`switch_penalty: 2.0` with `advance: index` therefore remains the shipped
configuration. See docs/beat_phase_improvement_review.md's step-1b results.

## Related

- `docs/beat_phase_improvement_review.md` — the full analysis and ranked plan.
- `docs/half_cycle_rate_explained.md` — the metric this parameter is tuned
  against.
