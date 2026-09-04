# Beat-Phase Model: Eval Results, Postprocessing, and `pos_weight` Notes

Notes from a debugging/analysis session on the beat-phase model (`BeatPhaseModule`,
trained on ballroom). Captures why the training-time metrics look much better
than the event-level evaluation, and a concrete `pos_weight` recommendation.

> **Read §6 first.** Every imbalance ratio below was measured on ballroom and
> only on ballroom. Under the general-tool goal that turns out to be the
> document's load-bearing assumption: the `beat` head's `pos_weight` is a
> function of *tempo*, not of the task, so a single tuned constant is correct
> at exactly one BPM. §6 derives it and records what replaced it. §1–§5 are
> kept as the historical record.

## 1. Starting point: LR sweep + eval results

LR sweep on ballroom, `lr` from `1e-5` to `4e-3`. Some overfitting at high lr
(val loss slightly increases), but best performance was still at the high end:

- `val/acc_one` ≈ 93%
- `val/acc_last` ≈ 95%
- `val/acc_beat` ≈ 77%

Running the actual evaluation tool (`tools/eval_beat_phase.py`, event-level
F-measure via `musicality/metrics.py`) on the same model gave much worse
numbers:

- mean beat F-measure: 0.726
- mean '1' F-measure: 0.488
- mean 'last' F-measure: 0.519
- mean phase confusion: 0.371

The `'1'`/`'last'` gap (93%/95% train-time vs. 49%/52% eval-time) looks like a
regression but isn't — see below.

## 2. Why the train-time and eval-time numbers disagree

Two different metrics are being compared:

- **`frame_accuracy`** (`musicality/trainers/beat_phase_module.py:31`) — thresholds
  both predicted probability and the Gaussian-smeared target at 0.5 and checks
  per-frame agreement. Its own docstring flags this as "a cheap per-epoch
  signal, dominated by the true-negative rate since beat/one/last frames are a
  small minority" — **not** the metric the project ultimately cares about.
- **Event-level F-measure** (`musicality/metrics.py`) — converts frame
  probabilities into discrete labeled beat events (via `musicality/postprocess.py:readout`)
  and matches them against reference events within a tolerance window
  (`mir_eval.beat.f_measure`). This is the metric that actually reflects
  real-world usefulness.

High frame accuracy with low event F-measure means the frame-level heads are
*roughly* in the right place a lot of the time, but the discrete event
extraction on top of them is unreliable — see next section.

## 3. Likely bottleneck: `label_bar_position` anchor-counting

`musicality/postprocess.py:157` (`label_bar_position`) assigns each gated beat
a bar position by:

1. Casting an "anchor vote" for position 1 or `group_size` whenever `one_probs`
   / `last_probs` clears `anchor_threshold` (default 0.5) at that beat's frame.
2. Between confident anchors, just **counting forward** (1, 2, 3, 4, 1, 2, ...)
   with no error correction.

This means a single missed or wrong anchor doesn't cost one beat — it
mislabels every beat until the next confident anchor resyncs the count. That
matches the observed 0.371 phase-confusion rate well: weak anchor
precision/recall compounds into long mislabeled stretches, rather than
isolated errors.

**Update:** swept `anchor_threshold` jointly with the beat-detection grid via
`tools/sweep_beat_postprocess.py` against a later checkpoint
(`checkpoints_beat/checkpoints_beat/lr_sweep/lr_0.03/beat-phase-epoch=126-val/loss=1.6565.ckpt`,
`binary_only=True`, ballroom val split). Confirms this is at least partly
postprocessing-limited, not purely a representation problem: the sweep found
a real interior optimum rather than "higher is always better" — `0.7` scored
f_one=0.679/f_last=0.682, `0.8` scored f_one=0.697/f_last=0.692 (best),
`0.9` scored f_one=0.698/f_last=0.676 (worse than `0.8`), exactly the
two-sided tradeoff described above (too low = noisy false votes corrupt long
stretches; too high = even real anchors stop firing). Tuned value now lives
in `configs/eval_beat.yaml`'s `beat_phase.anchor_threshold` (`0.8`).

## 4. `pos_weight` investigation

`beat_phase_loss` (`musicality/losses.py:90`) already supports a per-head
`pos_weight` (scalar or 3-element list/tensor for beat/one/last), but at the
time of writing `configs/beat_train.yaml` set a single flat value shared across
all three heads:

```yaml
pos_weight: 8.0
```

*(Now `[5, 4, 4]`. The one/last values dropped because `phase_conditioning`
moved to `beat`, which removes most of the imbalance those heads face — see the
comment block in `configs/beat_train.yaml`.)*

The plan doc (`beat_phrase_tracker_plan.md`, step 3) already flagged this as
an untuned placeholder, since one/last positives are rarer than beat
positives.

### What `pos_weight` actually does

Each head is frame-level binary classification via `BCEWithLogitsLoss`:

```
loss = -[ pos_weight * y * log(p) + (1-y) * log(1-p) ]
```

- `y=1` frame (real event): only the first term matters — punishes low `p`.
- `y=0` frame (no event): only the second term matters — punishes high `p`.
- `pos_weight` scales *only* the positive-frame term, making it more costly to
  miss a positive.

**Why this matters here:** a 10s clip at `sample_rate=22050`, `hop_length=512`
has ~430 frames. A beat event is smeared into a ~3-frame Gaussian bump
(`gaussian_smear`), so only a small fraction of frames are positive for any
given head — the rest are silence-between-events, and for the `one`/`last`
heads, *the other beats in the bar are also negative*. With `pos_weight=1`,
the model can minimize total loss by suppressing positive predictions almost
everywhere (cheap win on the huge negative majority, small cost on the few
positives it misses) — i.e. it never learns to fire the rare heads.
`pos_weight ≈ (#negative frames) / (#positive frames)` roughly balances the
total loss contribution from positives vs. negatives.

### Measured class imbalance (real data, not estimated)

Computed directly from `BeatDataset("ballroom")` by summing/thresholding the
actual per-frame targets:

**`binary_only=false`** (698 tracks, includes triple-meter waltz/Viennese
waltz tracks):

| head | positive frame fraction | neg:pos ratio |
|---|---|---|
| beat | 14.3% | ~6:1 |
| one  | 4.1%  | ~23:1 |
| last | 2.6%  | ~38:1 |

**`binary_only=true`** (523 tracks, waltz/Viennese waltz dropped):

| head | positive frame fraction | neg:pos ratio |
|---|---|---|
| beat | 14.6% | ~6:1 |
| one  | 3.9%  | ~24:1 |
| last | 3.5%  | ~28:1 |

Takeaways:

- `beat`'s current `pos_weight=8.0` was already roughly right (measured ratio
  ~6:1).
- `one` and `last` are badly underweighted at 8.0 — they need something like
  3-5x more.
- `last` is disproportionately rare under `binary_only=false` because
  triple-meter tracks contribute masked-in frames where `last` (position 4)
  never fires at all. Switching to `binary_only=true` partially closes this
  (28:1 vs 38:1) but doesn't fully close the gap to `one`, and costs 175
  tracks of training data — not an obvious win on its own, worth checking
  against a real val split before committing to it.

### Recommendation

```yaml
pos_weight: [5.0, 18.0, 25.0]  # beat, one, last
```

Shaded a bit below the raw measured ratios — exact inverse-frequency
weighting on a soft/smeared target tends to overshoot into false positives.

### Implementation gotcha — checked, and it is a non-issue

The concern was that `build_module` passes `cfg.pos_weight` straight through,
so a YAML list arrives as an OmegaConf `ListConfig` rather than a plain list,
and `beat_phase_loss`'s `torch.as_tensor(pos_weight, ...)` might not accept it.
Verified 2026-09-04: it does. `ListConfig` is a `MutableSequence` of numbers and
`torch.as_tensor` handles it, so no `OmegaConf.to_container` is needed.

There is a *different* trap in the same place, which is real. `[5, 4, 4]` is
meaningful only for `target_layout: one_last`. The `positions` head
(`beat_position_loss`) has a single BCE term, so a 3-element weight reaches
`binary_cross_entropy_with_logits` as a shape-`(3,)` tensor against `(B, T)`
and dies on a broadcast error naming neither the config key nor the head.
`BeatPhaseModule.__init__` now rejects it at construction instead. Note that
the check tests `collections.abc.Sequence`, not `list`, precisely because
`ListConfig` is not a `list`.

## 5. Open next steps

- [x] Wire per-head `pos_weight` through config → `BeatPhaseModule`. Done; the
      `ListConfig` gotcha turned out not to exist (see §4).
- [x] Retrain on ballroom with per-head weights. Done, but with `[5, 4, 4]`
      rather than `[5.0, 18.0, 25.0]` — `phase_conditioning: beat` landed in
      between and removed most of the one/last imbalance those numbers were
      compensating for.
- [x] Re-run the event-level eval and compare against the 0.726 / 0.488 /
      0.519 / 0.371 baseline above. Current ballroom val (`epoch109`,
      `--binary-only`, global+viterbi decoder): f_one 0.774, f_last 0.769,
      confusion 0.130. See `plans/04_beat_phase_generalization_and_data_prep.md`
      §2.2. Note the tool is now `tools/eval_beat.py`.
- [x] Sweep `anchor_threshold` in `musicality/postprocess.py:readout` — check
      whether the F-measure gap is more postprocessing- or model-limited.
      Done via `tools/sweep_beat_postprocess.py`; see §3 update above — a real
      interior optimum was found (not boundary-saturated), so at least
      partly postprocessing-limited.
- [ ] Decide on `binary_only` (data-loss vs. cleaner `last` signal) based on
      actual val-split results, not just the frame-balance numbers above.

## 6. Update 2026-09-04 — `pos_weight` is a function of tempo, and is now derived

### High level

Everything above is measured on ballroom, and only on ballroom. That was fine
while ballroom was the only training corpus. Under the general-tool goal it is
not, because `pos_weight` turns out not to be a property of the *task* at all —
it is a property of the *tempo*.

The reason is that the two quantities whose ratio it compensates for scale
differently. A beat contributes a fixed amount of positive target no matter how
fast the music is, while the gap between beats does not. Fast music therefore
carries proportionally more positive target than slow music, and a single
constant is correct at exactly one tempo: the one it was measured at. Across
the corpora now in the merge split, the configured value is wrong by up to 2.5×
in one direction and 1.7× in the other. Time-stretch augmentation compounds it,
since it changes the tempo of every augmented clip — so even on ballroom alone
the tuned number is only correct for the unaugmented case.

The same asymmetry does *not* affect the `one`/`last` heads, for a reason worth
stating plainly: under `phase_conditioning: beat` those heads are only weighted
where a beat already is, so tempo cancels out of their imbalance entirely. It
is set by how many beats there are per bar, which is a constant. That is why
only the beat head is self-calibrated and the others keep configured constants.

The fix is to stop tuning it. The ratio is a function of the target the model is
already being shown, so it can simply be computed per clip.

### Technical

The beat target is a Gaussian smeared to peak 1.0, so its mass per beat is
`sigma * sqrt(2*pi)` ≈ 3.75 frames at the configured `sigma_frames: 1.5`,
independent of tempo. The beat period is `60 * fps / BPM` frames, and at
`sample_rate: 22050` / `hop_length: 512` that is `2584 / BPM`. With
`f = mass / period` the positive fraction, the imbalance is `(1 - f) / f`, a
pure function of BPM:

| corpus | BPM | period (frames) | neg:pos |
|---|---|---|---|
| jtd | 193 | 13.4 | 2.6 |
| ballroom | 125 | 20.7 | 4.5 |
| rwc_classical (median) | 105 | 24.6 | 5.5 |
| rwc_classical (p10) | 56 | 46.1 | 11.3 |

These are *mass* ratios, which is what `BCEWithLogitsLoss` actually weights.
§4's 6:1 counts frames whose target clears 0.5, over a real BPM distribution
rather than at the median — the two are measured differently and both are
ballroom-only. The conclusion is unaffected either way.

**Why `one`/`last` are exempt.** Under `phase_conditioning: beat` the weight is
`beat_y` and the positive is `one_y * beat_y`, a product of two Gaussians whose
mass is `sigma * sqrt(pi)` ≈ 2.66 per downbeat. Per bar that gives
`(4 * 3.75 - 2.66) / 2.66` = 4.66, matching the 4.7 measured in
`configs/beat_train.yaml`. No period appears, so the ratio is fixed by
`group_size` and the Gaussian overlap — not by tempo.

**Implementation.** `musicality.losses.beat_pos_weight`, selected by
`pos_weight: auto`:

```python
pos_frac = beat_y.mean(dim=-1, keepdim=True)          # (B, 1)
weight = alpha * (1.0 - pos_frac) / pos_frac.clamp(min=1e-6)
return weight.clamp(*AUTO_POS_WEIGHT_RANGE)
```

Per sample, shape `(B, 1)`, which broadcasts against `(B, T)` inside
`binary_cross_entropy_with_logits`. `AUTO_POS_WEIGHT_RANGE = (1.0, 20.0)` never
binds on real music — the slowest case is 12.5, and 14.8 once the augmenter's
0.85 floor applies — it exists for degenerate crops, where a window holding a
single beat derives ~200.

`alpha` defaults to `AUTO_POS_WEIGHT_ALPHA = 1.11`, which reproduces the
configured `5` at ballroom's median tempo. Anchoring there makes switching to
`auto` a pure cross-tempo change, neutral on the corpus every measurement in
this document was taken on. `alpha = 1.0` is exact inverse-frequency weighting.

**The other half.** The same tempo asymmetry hits the bar-position term of
`beat_position_loss`, which normalized over the whole batch and so weighted each
clip by its beat count — making two clips' shares of that gradient stand in
exactly the ratio of their tempos. `position_norm: per_item` divides each clip
by its own weight first, giving every annotated clip `1/n_valid` regardless of
tempo. Both switches default off; see
`plans/04_beat_phase_generalization_and_data_prep.md` §2.6a and §3 #2, and
`tests/test_loss_calibration.py`.

**Caveat.** Both change the value of `val/loss`, which `ModelCheckpoint`
splices into checkpoint filenames — runs from before and after are not
loss-comparable, and will look comparable.
