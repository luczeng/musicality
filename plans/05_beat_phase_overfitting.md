# Part 5 — Beat-phase: preventing overfitting

**Status:** analysis complete, implementation starting. Session of 2026-09-05.
Follows `plans/04_beat_phase_generalization_and_data_prep.md`, whose #2 (loss
calibration) and #4b (position folding) have shipped.

---

## Overview

### High level

The model stops getting better at music it has not heard roughly halfway
through training, and everything after that point is spent memorising the
training set. The accuracy numbers in the logs disguise this twice over: one of
them is already at its practical ceiling and looks stuck, and the other is
measured on deliberately harder material than the number it is compared
against. Corrected for both, the model fits its training data considerably
better than it appears to, and generalises considerably worse.

That rules out the two changes that looked obvious. Making the network bigger
does not help, because it already has spare capacity that it spends on
memorisation rather than on learning. Restoring the attention block does not
help either — it is a third of the model for a difference too small to justify
it.

What is left is making the training material harder to memorise and stopping
the run once it stops improving. Those are cheap, and one of them halves the
cost of every experiment that follows.

### Technical

Reference run `vocal-violet-88` (`merge_v4`: `use_self_attention=false`,
position folding on, `pos_weight: auto`, `position_norm: per_item`, lr 8e-4).

| step | train acc_pos | val acc_pos | gap | val loss |
|---|---|---|---|---|
| 1000 | 0.608 | 0.606 | 0.002 | 1.498 |
| **2268** | **0.698** | **0.653** | **0.045** | **1.406** ← saved checkpoint |
| 3000 | 0.767 | 0.650 | 0.116 | 1.470 |
| 3800 | 0.801 | 0.640 | 0.160 | 1.524 |

`val/loss` minimum and `val/acc_position` maximum both land on **step 2267**.
Past it, val loss rises **+9.8%** while train loss falls 0.22. `merge_v4.ckpt`
is from step 2268, so the evaluated model is the pre-overfitting one.

---

## 1. Why the training metrics look low

Two independent measurement artifacts, one per metric. Neither is a model
problem, and together they explain the apparent paradox of overfitting at 80%
training accuracy.

<details>
<summary><b>1.1 — <code>acc_beat</code> is ~3× stricter than the metric we evaluate on</b></summary>

The Gaussian target (σ = 1.5 frames, peak 1.0) thresholded at 0.5 is positive
only for `|offset| ≤ 1` frame — a **3-frame band, 70 ms wide**. That is exactly
`mir_eval`'s matching tolerance, so `acc_beat` demands ±1 frame where the real
metric allows ±3.

Balanced `acc_beat` of a *perfect* model with pure timing jitter:

| jitter | balanced `acc_beat` |
|---|---|
| 0 ms | 1.000 |
| 11.6 ms | 0.950 |
| **23.2 ms** (±1 frame) | **0.907** |
| 34.8 ms | 0.853 |
| 46.4 ms | 0.806 |

`train/acc_beat` ≈ 0.88 corresponds to ~28 ms mean error — inaudible, and
comfortably inside the 70 ms tolerance. Corroborated by the beat F-measure
already on record: train 0.900 / val 0.916. **The beat head is finished; there
is nothing to chase here.**

</details>

<details>
<summary><b>1.2 — <code>acc_position</code> is measured on augmented audio, val is not</b></summary>

`build_beat_dataloaders` wraps only the train dataset in
`AugmentedBeatDataset`. So `train/*` is computed on time-stretched, gain-varied
audio and `val/*` on clean audio — the logged curves do not compare like with
like.

Measured on `merge_v4` over 120 tracks from its own training split, identical
fixed crops, augmentation the only difference:

| | `acc_beat` | `acc_position` |
|---|---|---|
| clean | 0.877 | **0.731** |
| augmented (what gets logged) | 0.871 | **0.594** |
| penalty | 0.006 | **0.137** |

The position head pays 13.7 points; the beat head pays nothing. So at the
checkpoint the true gap is **0.078**, not the 0.045 the logs show —
augmentation was hiding about half of it.

**Not a target bug.** A perfect model still scores ≥0.997 against the
interpolated target at every stretch rate in 0.85–1.15, so `FrameTimeStretch`
is not corrupting position labels. It is genuine added difficulty: every
augmented sample is effectively novel, which makes the train metric closer to a
held-out measurement than a training one.

**Not measurable at end of training.** Only best-val checkpoints are saved, so
the final model does not exist on disk. Any statement about how far the fit
eventually got is extrapolation — see §4's `save_last`.

</details>

---

## 2. What this rules out

| lever | verdict | why |
|---|---|---|
| Wider trunk (`channels` 256→384) | **No** | Spare capacity is already being spent on memorisation; more buys a faster route to the same ceiling |
| Deeper trunk (`n_layers` 8→9) | **No** | RF would be 23.7 s against a 16 s clip; longer clips ruled out, so depth is capped at 8 |
| Restore self-attention | **No** | 0.79M params (+45%) for ≤0.023 macro, itself confounded with the folding fix |
| Longer clips | **No** | User decision |

The capacity argument in `04`'s §2.3 was re-opened on the grounds that
merge-trained models showed a large training-fit deficit. That reasoning does
not survive §1: `ModelCheckpoint` selects at the val optimum, so *every*
early-stopped checkpoint looks underfit. The `v4` curves show the model taking
train `acc_position` from 0.698 to 0.801 with **zero** val gain — spare capacity,
spent on memorisation. **Capacity is closed again.**

---

## 3. Measured checkpoint status

`merge` val, `--binary-only`, each under its own macro-best decoder
(`global + viterbi (switch=5)`). Confusion, lower better.

| corpus | n | `v2.0` | `v3.0` | `v4` |
|---|---|---|---|---|
| ballroom | 104 | 0.271 | **0.170** | 0.242 |
| jtd | 96 | 0.278 | **0.259** | 0.291 |
| rwc_popular | 18 | 0.275 | **0.222** | 0.240 |
| rwc_genre | 16 | 0.330 | **0.283** | 0.339 |
| rwc_jazz | 7 | 0.236 | **0.219** | 0.246 |
| rwc_classical | 6 | 0.485 | 0.484 | **0.420** |
| **MACRO** | 6 | 0.312 | **0.273** | 0.296 |
| micro | 247 | 0.282 | **0.224** | 0.272 |

Ballroom-train fit: `v2.0` 0.176, `v3.0` **0.119**, `v4` 0.161; ballroom-only
`v1.0` reaches **0.007**.

**`v3.0` remains the best model.** Phase 2 (loss calibration) took macro
0.312 → 0.273 and, for the first time, beat the ballroom-only model's 0.288 —
reversing `04` §3 #0's second finding. Confounded by lr (8e-4 vs 6e-4).

`v4` changed **two** things at once (attention removed *and* folding), so its
classical gain is unattributable as it stands. The fold touches 19% of
classical training tracks, which makes it the likelier cause.

---

## 4. Plan

Ordered by effect per unit of work, one substantive change per run.

<details>
<summary><b>#1 — <code>EarlyStopping</code> + <code>save_last</code> (do first)</b></summary>

Best checkpoint at step 2268; the run continued to 4751. **Over half of every
run is spent getting worse.**

- `musicality/trainers/train_beat_phase.py` — add
  `EarlyStopping(monitor="val/loss", patience=10, mode="min")` to
  `build_callbacks`. `patience=10` against `ReduceLROnPlateau(patience=5)` lets
  the LR drop fire and the model recover before the run is cut.
- `musicality/trainers/common.py` — `save_last=True` on the
  `ModelCheckpoint`, so end-of-training behaviour becomes measurable at all.

Improves no metric; halves the cost of everything below.

</details>

<details>
<summary><b>#2 — Log a clean-train metric</b></summary>

So the real gap (§1.2) is visible during a run instead of reconstructed
afterwards. A callback holding a fixed, unaugmented, fixed-crop subsample of
the training split, scored each validation epoch and logged as
`train_clean/acc_beat` / `train_clean/acc_position`.

Must **not** touch `val/loss` — it is monitored by `ReduceLROnPlateau` *and*
`ModelCheckpoint` *and* spliced into checkpoint filenames.

</details>

<details>
<summary><b>#3 — Frequency masking (SpecAugment)</b></summary>

The position head's likely failure mode is timbral — "this snare is beat 3"
works on a seen track and nowhere else. Frequency masking destroys that cue
while leaving rhythm intact.

- `musicality/models/tcn.py` — apply `torchaudio.transforms.FrequencyMasking`
  in `TCNTempoNet.forward` **after** the existing per-sample normalisation, so
  masked bins sit at the distribution mean rather than skewing the mean/std.
  Gate on `self.training`.
- `configs/model/tcn_frames.yaml` — expose the widths; default **disabled** in
  the Python signature so existing checkpoints reconstruct bit-for-bit.

**Frequency masking only to start.** Time masking removes frames the beat head
needs, and beat is the healthy half of the model — run it separately if at all.

</details>

<details>
<summary><b>#4 — Augmentation already wired but off</b></summary>

`configs/beat_train.yaml` currently has `noise.enabled: false`.

- Enable noise.
- Widen `time_stretch` to 0.8–1.25. Phase 2 unlocked this: with
  `pos_weight: auto` deriving from the target, stretching no longer invalidates
  a hand-tuned calibration. Safe on the target side (ceiling ≥0.997, §1.2).

Then the blunt knobs, folded into the same run rather than as separate
experiments: `model.dropout` 0.2 → 0.3, `weight_decay` 1e-4 → 1e-3.

</details>

<details>
<summary><b>#5 — Corpus-temperature sampling</b></summary>

`04` §3 #3, unchanged. Without gtzan (§5 below) jtd stays at 63.8% of the
training set, so this matters *more* than it would have, not less.

</details>

---

## 5. Data — what is actually available

<details>
<summary><b>gtzan — CONTESTED, blocked pending a call</b></summary>

940 usable tracks across 10 genres, in no split. **User has listened and found
tracks mislabeled.** For the record, what the audit found:

*Internal annotation consistency — gtzan is the 2nd cleanest corpus here:*

| corpus | n | IBI cv | cv>.15% | clean cycle% | dup sigs |
|---|---|---|---|---|---|
| gtzan | 992 | 0.017 | 1.7 | 98.7 | 12 |
| ballroom | 698 | 0.021 | 0.4 | 99.9 | 1 |
| jtd | 1294 | 0.035 | 1.2 | 100.0 | 3 |
| rwc_classical | 61 | 0.242 | 65.6 | 77.0 | 0 |

*Model agreement — poor (confusion 0.422, and the decoder gains 0% where it
buys 12–31% elsewhere), but the error signature does not match
mis-annotation:*

```
offset 0  correct   : 50.7%      stability mean 0.496
offset 1  off-by-1  : 15.4%      flipping (<0.80): 90.4%
offset 2  half-cycle: 18.4%      stable  (>=0.95):  2.9%
offset 3  off-by-3  : 15.4%
```

A systematic annotation offset would show as *high stability plus one
concentrated wrong offset*. What is there is near-uniform spread with 90% of
tracks flipping phase mid-track — the model failing to form coherent phase, not
disagreeing consistently with a shifted reference.

**That reasoning is indirect** and cannot rule out per-track errors, which is
exactly what listening finds. Note also that documented GTZAN genre
mislabeling is **irrelevant here** — training reads beat times and bar
positions only, never genre labels.

**Nothing in §4 depends on gtzan.**

</details>

<details>
<summary><b>Other sources on disk</b></summary>

| source | state |
|---|---|
| `brid` | **367 wav + 93 annotation files** in raw `BRID_1.0/{Annotations,Data}` layout, never migrated — this is why the loader reports 0 tracks. ~93 usable; needs a `tools/migrate_*` script. Brazilian percussion, genuinely different rhythmic material |
| `MTG-JAAH-7686b91` | 113 annotations, **0 audio** — unusable as-is |
| `swing` | 23 tracks / 35 annotations |
| `groove_midi`, `field_recordings` | not migrated / no tracks |

`brid` is the only real untapped source outside gtzan, but at ~5% of the
training set it will not move overfitting on its own.

</details>

---

## 6. Verification

```bash
uv run pytest tests/ -q
uv run ruff format musicality/ tests/
```

```bash
# fit — on a corpus inside the training set
uv run python tools/diagnose_beat_phase.py --checkpoint <ckpt> \
    --dataset ballroom --split train --binary-only

# generalization, per genre — read MACRO and the worst corpus
uv run python tools/diagnose_beat_phase.py --checkpoint <ckpt> \
    --dataset merge --split val --binary-only
```

`--checkpoint` is a **named flag, not positional**; passed positionally it exits
on an argparse error that a piped `tail` swallows, reporting false success.
`--binary-only` is mandatory (`04` §2.1: without it ~58% of "val" is training
data).

Current bar: macro **0.273** (`v3.0`), worst corpus `rwc_classical` **0.420**
(`v4`), ballroom train fit ceiling **0.007**.

**The signal that any of this worked is `val/loss`'s minimum going lower**, not
the train/val gap narrowing — a gap closes just as easily by making training
worse.

---

## 7. Loose ends

- `tcn.py`'s receptive-field docstring is wrong: it claims
  `kernel_size × (2^n_layers − 1)` = 765 frames. Measured empirically by
  gradient reachability it is `1 + (k−1)(2^n − 1)` = **511 frames = 11.87 s**
  (0.74× the 16 s clip; 6.2 bars at 125 BPM, 2.8 at 56).
- `brid` yielding 0 tracks is now explained (never migrated) — `04`'s reference
  section still records it as unexplained.
- Classical remains unlearned by every checkpoint. `04` #4 (variable meter) is
  still the only lever that addresses it, and is still deferred.
