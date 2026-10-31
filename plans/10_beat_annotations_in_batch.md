# Part 10: beat annotations in the training batch

**Status:** plan, not started. The first of three plans implementing
`plans/softargmax_periodicity_math.md` (the theory doc). Plans 11 and 12 add the
two losses, and both need this one.

---

## Overview

### High level

Both new losses compare the network's output with individual annotated beats:
the beat-spacing loss with each beat's exact time, the periodicity loss with
each beat's bar position. Training never sees those beats today. The loader
turns them into curves over frames (one curve for "a beat is here", one per bar
position), blurred and rounded to the frame, and the loss only ever sees the
curves. The exact beat times are lost along the way, and so is the list itself.

This plan carries the list of beats through to the loss, next to the curves,
and keeps it aligned with the audio when tempo augmentation stretches a clip.
Nothing reads it yet. Training, the model and every metric stay exactly as
they are; the plan is done when tests show the list agrees with the curves.

### Technical

`BeatDataset` gains a `return_beats` flag. When it is set, each item carries a
third tensor, `beats`, of fixed shape `(2, MAX_BEATS)`. The time-stretch
augmentation moves those beats with the same mapping it applies to the target.
Only beat-phase training asks for it. Evaluation, the plotting tools and
beat-only training keep receiving `(wav, target)`.

---

## 1. The `beats` tensor

| Row | Content | Padding |
|---|---|---|
| 0 | beat time in frames since the clip start, sub-frame ($b_k$ in the theory doc) | $-1$ |
| 1 | folded bar position $1..G$ ($n_k$), or 0 for a track without positions | 0 |

- **Which beats.** Exactly the beats `_times_to_spike` writes into the clip,
  those whose rounded frame falls in `[0, n_frames)`. The list and the beat
  channel then agree by construction, which is what the tests check.
- **Fixed size.** `MAX_BEATS = 128`, a module-level constant in
  `beat_dataset.py`. A fixed shape keeps PyTorch's default collate and needs no
  custom one. 128 beats in a 16 s clip is 480 BPM, far above any corpus.
  A clip with more raises `ValueError` rather than being truncated.
- **Padding is a negative time.** A padded slot is recognised by `time < 0`.
  The padding value is finite, so masked arithmetic cannot spread a NaN into the
  gradient.
- **Positions come from `self.samples`**, already folded at load time by
  `fold_positions`. Tracks with no usable positions get 0, the same case in
  which `mask` is 0.

## 2. Time-stretch

`FrameTimeStretch` resamples the target with
`F.interpolate(size=T', align_corners=False)`, where `T' = int(T / rate)`.
That call reads input position $(i + 0.5)\thinspace T/T' - 0.5$ for output
frame $i$, so a beat at input time $b$ moves to

$$
b' = (b + 0.5)\thinspace \frac{T'}{T} - 0.5
$$

This uses the actual ratio $T'/T$, not `rate`, so the beats follow the target's
peaks exactly. After the re-crop to `n_frames`, beats whose rounded frame no
longer falls inside the clip become padding.

The beats are optional all the way down:

- `FrameTimeStretch` and `BeatPhaseAugmenter` take `beats=None` and return it,
  moved, or still `None`;
- `AugmentedBeatDataset` forwards a third element only when its subset yields
  one.

## 3. Trainer

- `build_beat_dataloaders(cfg, return_beats=False)`. `train_beat_phase.py`
  passes `True` for both splits, so validation clips carry beats as well.
- `BeatPhaseModule._step` unpacks `wav, target, *extra = batch`. `extra` is
  unused until plans 11 and 12.
- `fit_input_stats` iterates `for wav, *_ in …` instead of `for wav, _ in …`.
- `align_time` can make the logits one frame shorter than the target. The two
  losses deal with this themselves: they clip read frames to `T - 1` (theory doc
  §2.2) and test window validity against the logits' `T`. Nothing to change here.

The alternative is always returning the triple and updating every consumer of
`BeatDataset`. That touches evaluation and plotting code that has no use for
beats, so a flag is the smaller change.

## 4. Verification

Done when these pass, plus the full suite unchanged:

- **Agreement with the target** (`tests/test_beat_dataset.py`). With
  `sigma_frames=0`, `round(beats[0])` of the valid slots equals the nonzero
  frames of the beat channel, and `beats[1]` equals the argmax of the position
  block at those frames.
- **Padding.** A clip with no beats gives all $-1$. A track without positions
  gives row 1 all 0. A clip with more than `MAX_BEATS` beats raises.
- **Stretch** (`tests/test_beat_phase_augmentations.py`). With `random.uniform` fixed to a
  known rate, each moved beat lies within one frame of the corresponding peak of
  the stretched beat channel, and beats pushed past `n_frames` are padding.
- **Flag off.** Without `return_beats`, items are still `(wav, target)`.
- **Smoke run.** One epoch of `tools/train.py` with `train_subsample: 0.02`
  runs and logs the same keys as before.
