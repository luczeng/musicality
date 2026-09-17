# Part 8 — Rethinking the approach: input, architecture, decoder, data

**Status:** brainstorm, nothing implemented and nothing decided. Session of
2026-09-17. Follows `plans/07_beat_phase_v6_and_next_moves.md`, which ranks the
*incremental* moves; this one asks whether the incremental moves are aimed at
the right thing at all.

---

## Overview

### High level

We are not slightly behind the published state of the art — we are behind it in
a way that tuning cannot close, and two measurements say where the mismatch
lives.

The model is roughly 150 times larger than the reference architecture for this
task, and it sees eight times less of the music at once. The beat-tracking
literature's standard convolutional model is tiny and very deep, which buys it
two minutes of temporal context; ours is wide and shallow, which buys 18 seconds
and a great deal of memorisation. Every failure documented in `plans/07` — the
position head memorising, the inability to find the "one" in a jazz tune, the
phase flipping halfway through a track — is what that shape predicts.

The second measurement is where the gap actually is. On beat tracking we are a
few points behind; on downbeat tracking we are roughly three times further
behind. Beat tracking is respectable and improves with tuning. Downbeat tracking
is a different league, and it is where all the remaining value sits.

So the most informative next step is not to build anything. Run a public
pretrained beat tracker over the same held-out tracks. If it beats us by the
margin the literature implies, the approach is the problem and the path is to
copy the field's recipe. If it struggles where we struggle, our annotations or
our split are the problem. One afternoon decides which of the two documents —
this one or `plans/07` — is the one to work from.

### Technical

Everything measured here is `checkpoints/checkpoint_v6.ckpt` on the 277-track
`beat_phase-merge-binary` val split, under the swept postprocessing of
`plans/07` §3.1 (`beat_threshold: 0.8`, `min_distance_frames: 4`,
`gate_tolerance: 0.2`, `switch_penalty: 1.0`).

> **Literature figures in this document are recalled, not measured.** They are
> given as ranges and are load-bearing only for *direction*. §6.1 exists to
> replace them with numbers measured on our own split, and nothing here should
> be quoted as a target until it has.

---

## 1. Where we stand against the field

<details>
<summary><b>1.1 — The downbeat gap is roughly three times the beat gap</b></summary>

`f_one` is the F-measure of position-1 events at a 70 ms tolerance — the same
quantity the literature calls downbeat F-measure.

| corpus | n | beat F (ours) | downbeat F (ours) | published beat F | published downbeat F |
|---|---|---|---|---|---|
| ballroom | 104 | 0.873 | **0.687** | ~0.93–0.96 | ~0.90 |
| jtd | 96 | 0.950 | 0.505 | — | — |
| gtzan | 28 | 0.803 | **0.480** | ~0.85–0.89 | ~0.70–0.76 |
| rwc_popular | 19 | 0.874 | 0.756 | — | — |
| rwc_genre | 16 | 0.647 | 0.478 | — | — |
| rwc_jazz | 7 | 0.760 | 0.518 | — | — |
| rwc_classical | 7 | 0.556 | 0.243 | — | — |

On the two corpora with widely published numbers we are ~6–9 points behind on
beats and ~20–25 points behind on downbeats. jtd has no published baseline
(it is a recent corpus), which is part of why §6.1 matters — we cannot tell
whether 0.505 is bad or whether jazz trio downbeats are simply hard.

</details>

<details>
<summary><b>1.2 — The model is ~150× the reference size and has 8× less context</b></summary>

Parameter counts and receptive fields for `TCNTempoNet`, measured by
instantiating it (`3 × (2^n_layers − 1)` frames at 23.2 ms/frame):

| config | params | receptive field |
|---|---|---|
| **current** (256 ch × 8 layers) | **1.613 M** | **17.8 s** |
| v5, with attention | 2.403 M | 17.8 s |
| 64 ch × 8 | 0.108 M | 17.8 s |
| 32 ch × 11 | 0.039 M | 142.6 s |
| **16 ch × 11** (madmom-TCN-like) | **0.011 M** | **142.6 s** |

The reference convolutional beat tracker in the literature is in the tens of
thousands of parameters and reaches beat F in the low 0.90s on ballroom. We have
1.6 M parameters on 1854 training tracks and reach 0.873.

The shape is backwards in both directions at once: wide layers spend parameters
on memorisation (`plans/07` §1.2 measured the position head's train/val gap
widening to 0.194), while too few layers starve the model of context. Thin and
deep fixes both with one config change — and note what 142 s buys musically: a
jazz standard is a 32-bar form (≈40 s at jtd's median 185 BPM) and a blues is 12
bars (≈15 s). **A 142 s receptive field can see the form repeat; 17.8 s cannot.**
`plans/07` §2.4's unexplained jtd anchor failure and our context budget may well
be the same fact.

</details>

---

## 2. Input representation

<details>
<summary><b>2.1 — The first operation destroys frequency locality</b></summary>

`TCNTempoNet.forward` normalises the mel and then applies
`input_proj = Conv1d(n_mels, channels, kernel_size=1)`: all 128 mel bands are
collapsed into a per-frame linear mixture **before anything sees a
time-frequency neighbourhood**. There are no 2D convolutions anywhere in the
model.

Every published beat tracker begins with local spectro-temporal processing —
madmom's TCN opens with 3×3 convolutions and frequency max-pooling; Beat This
uses frequency-wise partial attention — because an onset *is* a local event:
energy rising across a limited band over ~20 ms. Asking a 1×1 mixer to
represent that is the top suspect for the beat-F gap in §1.1, and the fix is
small: two or three `Conv2d` layers with frequency pooling down to ~8–16 bands,
then the existing TCN trunk on the flattened result.

</details>

<details>
<summary><b>2.2 — Frame rate and resolution</b></summary>

- **23.2 ms per frame** (hop 512 at 22050 Hz) against a **70 ms** evaluation
  tolerance leaves ±1 frame of quantisation noise in every predicted beat. The
  field runs 10 ms (100 fps) or 20 ms (50 fps). Hop 256 doubles sequence length
  and is the cheapest precision available.
- **One STFT window is the worst of both worlds.** madmom feeds three window
  sizes (≈23/46/93 ms) at once: short windows localise onsets, long windows
  resolve harmony. We use a single 93 ms window — blurry onsets *and* mediocre
  harmonic resolution.
- 128 mel bands is fine (the field uses 81–128); this is not where the problem
  is.

</details>

<details>
<summary><b>2.3 — Two representations aimed at our actual failure</b></summary>

Both target downbeats specifically, which §1.1 says is where the gap is:

- **HPSS two-stream input.** Percussive stream for beats, harmonic stream for
  downbeats. Cheap, classical, and aligned exactly with the beat-works /
  downbeat-fails split we measure.
- **Chroma or CQT stream.** Downbeats correlate with *harmonic change*, which is
  the dominant cue wherever no drum marks the bar — jazz trio and classical,
  i.e. our two worst corpora. Chroma plus a novelty or self-similarity feature is
  how downbeat trackers found the "one" before deep learning, and it remains the
  right prior.

</details>

---

## 3. Architecture

<details>
<summary><b>3.1 — Shrink it, deepen it, and feed it longer crops</b></summary>

The one-line version of §1.2: go to 16–32 channels and 11 layers, and train on
60–90 s crops so the receptive field has something to look at. This attacks
capacity and context simultaneously and is a config change, not a rewrite.

Expect it to *also* change what the regularisation discussion in `plans/07` §4.5
is for: label smoothing and SpecAugment are treatments for a model that is too
big. Fix the size first, then see what is left.

</details>

<details>
<summary><b>3.2 — Split beats and downbeats into two stages</b></summary>

Beat tracking works (0.87–0.95 on the good corpora); downbeat tracking does not.
One frame-level head is being asked to do both, and they are not the same
difficulty.

1. **Stage 1** — beat activations → beats. The current model, shrunk, plus a
   real decoder (§4).
2. **Stage 2** — pool trunk features **at the detected beats** and run a small
   sequence model over the beat sequence: ~100 steps instead of ~5000 frames, a
   4-class output, and regular metrical structure. Phrase-level patterns become
   learnable instead of hopeless.

This is how Durand, Krebs and Böck all built downbeat trackers, and it is the
same idea as `plans/07` §4.6's beat-synchronous head, promoted here from "one
option" to the structural recommendation.

</details>

<details>
<summary><b>3.3 — Other architectures, in priority order</b></summary>

- **Small transformer with rotary position embeddings** over 30–60 s at 50 fps —
  the Beat This shape, current SOTA, and notable for dropping the DBN entirely
  in favour of a better-trained model and a tolerant loss.
- **SpecTNT** — a frequency-axis transformer nested in a temporal one; the
  strongest downbeat architecture before that.

Both want more data than 1854 tracks (§5), so they belong after the data work,
not before it.

**Do not revisit frame-level self-attention on the current trunk.** Measured
twice — `plans/05` from one direction, v5-vs-v6 from the other (0.591 vs 0.594
position accuracy). That question is closed.

</details>

---

## 4. Targets, loss, and decoding

<details>
<summary><b>4.1 — Sharper targets, more tolerant loss</b></summary>

- **Shift-tolerant BCE instead of Gaussian-smeared targets.** `plans/06`
  measured frame precision at 0.487: the predicted peak is wider than the
  target's, because a σ=1.5-frame Gaussian teaches the model to spread. A loss
  that forgives errors *inside* the tolerance window while keeping the target
  sharp fixes the cause rather than the symptom.
- **A dedicated downbeat head.** madmom predicts beat and downbeat as two
  activations with independent peak-picking. Our 4-way softmax makes "is this a
  downbeat" a byproduct of a harder question.
- **Circular phase regression as an auxiliary** — predict `(sin θ, cos θ)` of
  bar phase alongside the softmax, so a half-cycle error is a 180° error rather
  than one of three equally wrong classes.
- **Multi-task with tempo.** The repo already has tempo models and
  `tempo_acc1`; a tempo head regularises the trunk *and* supplies the prior the
  decoder needs.
- Label smoothing on the position CE, for the overconfidence measured in
  `plans/07` §1.3.

</details>

<details>
<summary><b>4.2 — Our decoder has no tempo model, and it shows</b></summary>

We run a 4-state Viterbi over bar positions with a hand-set switch penalty and a
fixed threshold. The field runs a **bar-pointer DBN** (Krebs/Böck,
`DBNDownBeatTrackingProcessor`): a joint state space over tempo × position-in-bar
with transitions encoding how tempo actually moves.

Our `amlt − cmlt` gap of **0.148** is that missing tempo model stated as a
number — 15% of tracked audio is confidently tracked at the wrong metrical
level, and nothing in the decode knows what tempo the piece is.

Cheaper intermediate steps, in increasing order of effort: a per-track adaptive
threshold (a quantile of the track's own activation, instead of an absolute 0.8
that ballroom likes and gtzan hates — `plans/07` §3.1); a tempo-relative switch
penalty (ours is charged per beat, so it behaves differently at 185 BPM than at
105); a global tempo estimate constraining the beat DP; then the full DBN, which
is importable rather than reimplementable.

</details>

---

## 5. Data is where the field's advantage actually comes from

<details>
<summary><b>5.1 — Corpora, unpulled and unused</b></summary>

The SOTA systems train on the order of a thousand hours across a dozen-plus
annotated corpora. We train on 1854 tracks, 60% of them one corpus. No
architecture closes that.

**Already on disk, unused:** 853 annotated gtzan tracks (`plans/07` §4.3), and
`brid`, `groove_midi`, `MTG-JAAH` directories with nothing fetched into them.

**Public, not yet here:**

| corpus | ≈size | why it matters here |
|---|---|---|
| Harmonix Set | 912 | pop, beats + downbeats + segments; distributed with precomputed features, which also sidesteps audio licensing |
| ASAP | ~1000 performances | classical beats *and* downbeats — the only real fix for rwc_classical |
| Beatles / Isophonics | 180 | the field's standard pop benchmark |
| Hainsworth, SMC | 222 / 217 | deliberately hard material; SMC is where models are separated |
| Candombe, HJDB | 35 / 236 | rhythmic traditions far from ballroom's 4/4 |
| TapCorrect, MedleyDB, Filosax, GuitarSet | — | breadth, and several are jazz-adjacent |

</details>

<details>
<summary><b>5.2 — Distillation is the affordable 10×</b></summary>

Run a strong public model over a large *unlabeled* collection, keep the
high-confidence tracks, train on those. This is how a small team gets to
corpus scale without an annotation budget, and it costs compute rather than
money. It pairs naturally with §6.1, which requires standing up a public model
anyway.

</details>

<details>
<summary><b>5.3 — Augmentation is under-powered</b></summary>

We do time-stretch ±15%, gain ±6 dB, and noise. The literature also does:

- **Pitch shifting (±5 semitones)** — standard, and more important than it
  sounds: it teaches the model that rhythm is not pitch. We do none.
- Wider tempo ranges, and resampling-based stretching rather than phase-vocoder.
- Spectrogram masking (SpecAugment).

</details>

<details>
<summary><b>5.4 — One split is not an evaluation</b></summary>

The field reports 8-fold cross-validation per dataset. We report single-split
numbers on corpora with 7 val tracks. Combined with the unseeded
`random_split` noted in `plans/07` §4.10, no result here survives contact with a
data addition.

</details>

---

## 6. Pretrained or from scratch

<details>
<summary><b>6.1 — The experiment that decides everything else</b></summary>

**Benchmark a public model (madmom, Beat This, or both) on our own 277-track val
split.** This is the highest-information action available and it builds nothing.

It answers three questions at once:

1. **How large is the real gap?** §1.1's comparison is against recalled numbers
   on *their* splits; this measures it on ours.
2. **Are our annotations sound?** If a SOTA model also scores ~0.50 downbeat F
   on jtd, then jtd is genuinely hard and `plans/07` §2.4's open question closes
   in favour of "the material", not "our pipeline".
3. **Is any of §2–§4 worth doing?** If a public model is 20 points ahead, copy
   the recipe. If it is 5 points ahead, our architecture is not the bottleneck
   and §5 is.

</details>

<details>
<summary><b>6.2 — On pretrained encoders, honestly</b></summary>

Pretrained music encoders (MERT, MULE, BEATs) are worth exactly one experiment:
freeze the features, train a light head, see where it lands. They help most on
tonal and semantic tasks, and beat tracking is the task where hand-crafted
spectrograms have held up best — current SOTA is trained from scratch. Do not
bet the project on it.

The genuinely promising form of "use someone else's model" here is distillation
(§5.2), not feature extraction.

</details>

---

## 7. Recommended order

1. **§6.1 — benchmark a public model on our val split.** One afternoon. It
   converts every estimate in this document into a measurement and decides
   whether the rest of it applies.
2. **§3.1 — shrink and deepen** (16–32 channels, 11 layers), train on 60 s
   crops. One config change; expect the largest single gain in this document.
3. **§2.1 — add a 2D convolutional frontend** with frequency pooling, and move
   to 50–100 fps (§2.2).
4. **§4.2 — replace the decoder** with a bar-pointer DBN, and **§4.1** — adopt
   shift-tolerant BCE.
5. **§5.1 — add Harmonix, ASAP and the rest of gtzan**, plus pitch-shift
   augmentation (§5.3). Only then consider the beat-synchronous second stage
   (§3.2) or a transformer (§3.3).

Steps 1–3 are days of work and hold most of the leverage. Everything in §3.3 is
worth doing only once the data and the decoder have stopped being the
bottleneck — otherwise we compare architectures through a broken readout, which
is exactly the mistake `plans/07` §3.1 documents.

---

## 8. Relationship to `plans/07`

`plans/07` is the incremental path: re-sweep the decoder, add the gtzan tracks
already on disk, fix checkpoint selection, rebalance the sampler. Every item in
it remains correct and cheap, and §7 step 1 here does not block any of it.

Where they disagree is emphasis. `plans/07` §4.5 treats overfitting as something
to regularise; §1.2 here argues the model is simply the wrong size and shape, and
that regularisation is a treatment for a self-inflicted problem. `plans/07` §4.8
lists a pretrained encoder as a high-ceiling option; §6.2 here downgrades it
relative to distillation. Resolve both with measurements, not argument.
