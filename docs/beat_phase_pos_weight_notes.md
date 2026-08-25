# Beat-Phase Model: Eval Results, Postprocessing, and `pos_weight` Notes

Notes from a debugging/analysis session on the beat-phase model (`BeatPhaseModule`,
trained on ballroom). Captures why the training-time metrics look much better
than the event-level evaluation, and a concrete `pos_weight` recommendation.

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
`pos_weight` (scalar or 3-element list/tensor for beat/one/last), but
`configs/beat_train.yaml:5` currently sets a single flat value shared across
all three heads:

```yaml
pos_weight: 8.0
```

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

### Implementation gotcha (not yet fixed)

`musicality/trainers/train_beat_phase.py:133` passes `cfg.pos_weight`
straight through to `BeatPhaseModule`. If `pos_weight` becomes a YAML list,
Hydra/OmegaConf will hand it over as a `ListConfig`, not a plain Python list —
`beat_phase_loss`'s `torch.as_tensor(pos_weight, ...)` may not handle that
directly. May need `OmegaConf.to_container(cfg.pos_weight)` first, mirroring
the existing pattern in `beat_phase_module.py` where `model_cfg` is converted
the same way.

## 5. Open next steps

- [ ] Wire per-head `pos_weight` through config → `BeatPhaseModule` (handle
      the `ListConfig` gotcha above).
- [ ] Retrain on ballroom with `pos_weight: [5.0, 18.0, 25.0]`.
- [ ] Re-run `tools/eval_beat_phase.py` and compare F-measure / phase
      confusion against the 0.726 / 0.488 / 0.519 / 0.371 baseline above.
- [x] Sweep `anchor_threshold` in `musicality/postprocess.py:readout` — check
      whether the F-measure gap is more postprocessing- or model-limited.
      Done via `tools/sweep_beat_postprocess.py`; see §3 update above — a real
      interior optimum was found (not boundary-saturated), so at least
      partly postprocessing-limited.
- [ ] Decide on `binary_only` (data-loss vs. cleaner `last` signal) based on
      actual val-split results, not just the frame-balance numbers above.
