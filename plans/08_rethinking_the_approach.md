# Part 8 — Rethinking the approach: input, architecture, decoder, data

**Status:** brainstorm, with one item measured. Session of 2026-09-17, revised
the same day after §6.1 was run. Follows
`plans/07_beat_phase_v6_and_next_moves.md`, which ranks the *incremental* moves;
this one asks whether the incremental moves are aimed at the right thing at all.

**Revision log.** §6.1 (benchmark a public tracker on our own split) is done —
`tools/eval_baseline.py`, results in §1.3. It reversed §4.2 (a bar-pointer DBN
is *not* the fix), promoted §2.1 (input representation) to the first thing to
build, lowered the target §1.1 sets, closed `plans/07` §2.4 in favour of "the
material is hard", and re-ordered §7.

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
and a great deal of memorisation. Most of the failures documented in `plans/07`
— the position head memorising, the phase flipping halfway through a track — are
what that shape predicts. (One of them is *not*: the inability to find the "one"
in a jazz tune turns out to be shared by the state of the art — see §1.3 — so it
is a property of the music, not of our shape.)

The second measurement is where the gap actually is. On beat tracking we are a
few points behind; on downbeat tracking we are roughly three times further
behind. Beat tracking is respectable and improves with tuning. Downbeat tracking
is a different league, and it is where all the remaining value sits.

So the most informative next step was not to build anything: run a public
pretrained beat tracker over the same held-out tracks and see whether it beats
us by the margin the literature implies, or struggles where we struggle.

**That has now been done** (§1.3, §6.1). Two results changed this document.
The state of the art struggles where we struggle — it scores 0.553 bar-position
accuracy on jazz trio, so that corpus is hard material rather than a broken
pipeline. And it does so **without a decoder at all**, which reverses §4.2:
the bar-pointer DBN this document originally recommended importing is not what
separates us from the field, and the input representation (§2.1) is now the
best-evidenced place to start. §7 is revised accordingly.

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

> **Superseded by §1.3.** The right-hand columns are *recalled* figures from
> papers reporting on *their* splits, and they set the target too high. §6.1 has
> since measured two public trackers on these exact rows: on gtzan the state of
> the art reaches 0.818 bar-position accuracy, and on jtd 0.553. Read §1.3
> before quoting any gap from this table. The "is 0.505 bad?" question is
> answered there: no — jazz trio downbeats are simply hard.

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


<details>
<summary><b>1.3 — Measured: two public trackers on our own 277 tracks</b></summary>

§6.1 has been run. `tools/eval_baseline.py`, 2026-09-17, `merge` val,
`--binary-only`, scored through `musicality.evaluation.score_events` — the same
scorer `tools/eval_beat.py` uses, so these rows sit in the same units as a
checkpoint's.

**Read the exposure column before anything else.** *seen* means the tracker was
trained on that corpus, so the row is recall, not generalisation. Only the
**clean** rows mean anything. madmom's `?` rows are *unknown*, not clean: it
ships no training manifest, and its exposure row in
`musicality.baselines.base.CORPUS_EXPOSURE` is recalled from Böck et al. rather
than read off anything.

**Beat This! `final0`, its own peak-picking postprocessor, no DBN:**

| corpus | n | exposure | f_beat | cmlt | amlt | pos_acc | best_off | confusion |
|---|---|---|---|---|---|---|---|---|
| ballroom | 104 | seen | 0.988 | 0.981 | 0.986 | 0.999 | 0.999 | 0.000 |
| jtd | 96 | **clean** | 0.945 | 0.859 | 0.907 | **0.553** | 0.825 | 0.236 |
| gtzan | 28 | **clean** | 0.935 | 0.852 | 0.921 | 0.818 | 0.865 | 0.112 |
| rwc_popular | 19 | seen | 0.995 | 0.992 | 0.992 | 0.972 | 0.972 | 0.024 |
| rwc_genre | 16 | **clean** | 0.843 | 0.709 | 0.773 | 0.741 | 0.744 | 0.171 |
| rwc_classical | 7 | seen | 0.905 | 0.747 | 0.747 | 0.858 | 0.858 | 0.047 |
| rwc_jazz | 7 | seen | 0.774 | 0.420 | 0.945 | 0.697 | 0.697 | 0.026 |
| **MACRO** | 7 | — | 0.912 | 0.794 | 0.896 | 0.805 | 0.851 | 0.088 |

**madmom, downbeat variant, `beats_per_bar=[3, 4]`, stock settings:**

| corpus | n | exposure | f_beat | cmlt | amlt | pos_acc | best_off | confusion |
|---|---|---|---|---|---|---|---|---|
| ballroom | 104 | seen | 0.977 | 0.946 | 0.984 | 0.914 | 0.972 | 0.077 |
| jtd | 96 | ? | 0.857 | 0.629 | 0.939 | 0.626 | 0.688 | 0.194 |
| gtzan | 28 | **clean** | 0.902 | 0.776 | 0.918 | 0.751 | 0.825 | 0.215 |
| rwc_popular | 19 | seen | 0.992 | 0.994 | 0.994 | 0.971 | 0.973 | 0.029 |
| rwc_genre | 16 | ? | 0.851 | 0.797 | 0.875 | 0.759 | 0.797 | 0.145 |
| rwc_classical | 7 | ? | **0.648** | 0.416 | 0.543 | **0.420** | 0.573 | 0.433 |
| rwc_jazz | 7 | ? | 0.831 | 0.567 | 0.802 | 0.714 | 0.741 | 0.201 |
| **MACRO** | 7 | — | 0.866 | 0.732 | 0.865 | 0.736 | 0.796 | 0.185 |

**The comparison against our own checkpoints is deliberately not here** — it is
§7 step 1. Everything below is what the two baselines say about *each other*,
which needs no reference to us and is therefore already safe to act on.

1. **The published numbers in §1.1 are not the target.** Those were recalled
   from papers reporting on *their* splits, and the ~0.90 downbeat F they quote
   is ballroom-shaped data. On our clean corpora the state of the art reaches
   0.818 `position_acc` on gtzan and **0.553** on jtd. The ceiling is a lot
   lower than §1.1 implied, and every "we are 20 points behind" statement in
   this document should be re-derived against §7 step 1 rather than against the
   literature.

2. **jtd is genuinely hard — `plans/07` §2.4 closes in favour of the material.**
   jtd is clean for this Beat This! checkpoint, and it is the state of the
   art's *weakest* corpus by a wide margin: `position_acc` 0.553 against a
   `best_off` of 0.825. That shape is a bar grid that is largely consistent and
   simply anchored in the wrong place — anchor error 0.272, nearly three times
   its own 0.100 average. madmom lands in the same place from a different
   direction (0.626 / 0.688). A SOTA tracker cannot reliably find the "one" in a
   piano-trio recording either. `plans/07` §2.4 explicitly refused to assert
   that without evidence; there is now evidence.

3. **The beat/downbeat split is a property of the task, not of our
   architecture.** On clean corpora `f_beat` holds at 0.84–0.95 while
   `position_acc` falls to 0.55–0.82, for both trackers. §3.2's premise — that
   one frame-level head is being asked to do two jobs of very different
   difficulty — survives contact with the state of the art.

4. **A bar-pointer DBN is not what we are missing.** See §4.2, which this
   measurement rewrites.

</details>

---

## 2. Input representation

<details>
<summary><b>2.1 — The first operation destroys frequency locality</b></summary>

**What the code actually does.** `TCNTempoNet.forward` is four lines before the
trunk:

```python
x = self.mel(wav)                          # (B, 128, T) — 93 ms window, 23.2 ms hop, dB
x = (x - x.mean((1, 2))) / x.std((1, 2))   # ONE scalar mean/std for the whole input
x = self.input_proj(x)                     # Conv1d(128, channels, kernel_size=1)
for layer in self.layers:
    x = x + layer(x)                       # dilated Conv1d — over TIME only
```

`input_proj` is a **1×1 convolution over frequency**. At each frame it computes
`channels` fixed linear combinations of the 128 mel bands and discards the band
axis entirely. Every layer after it is a `Conv1d` over time. **There is no 2D
convolution anywhere in the model**, so nothing in it ever sees a
time-frequency neighbourhood.

**Why that is the wrong first operation.** An onset is a *local, relative*
event: energy rising, inside a limited band, over roughly 20 ms. Four separate
things go wrong.

1. **A 1×1 mixer is frequency-absolute; onsets are frequency-relative.** The
   projection learns one fixed weight per mel band and applies it identically at
   every frame. It can learn "bands 20–40 matter on average". It cannot learn
   "energy rose *in whichever band it rose in*". A kick drum and a walking
   double-bass note are the same event shape at different absolute frequencies,
   so a mixer has to spend separate output channels on each pitch region to
   detect the same thing twice — and it has `channels` of them to spend on
   every event type at every register at once. A `Conv2d` shares one kernel
   across the frequency axis and gets that equivariance for free, which is the
   whole reason to put audio on a log-frequency axis in the first place.

2. **Mixing before differencing lets onsets cancel.** Onset strength is
   classically spectral flux — a difference *across time, within a band*. Our
   first temporal operation happens only after the bands have been summed, so if
   band A rises and band B falls by the same weighted amount, the mixture is
   flat and the event is gone before any layer could have seen it. The case
   where this bites hardest is a harmonic change with no percussive attack:
   energy moving *between* bands rather than into them. That is the dominant
   downbeat cue wherever no drum marks the bar — jazz trio and classical, which
   §1.3 confirms are the hardest corpora for everyone, and §2.3 is the
   representation-level answer to the same problem.

3. **One analysis window, wrong for both jobs.** `n_fft=2048` at 22050 Hz is a
   93 ms window (§2.2). Even a perfect frequency-local front end would be
   reading a spectrum whose onsets have already been smeared across four output
   frames.

4. **The normalisation statistics change between training and inference.**
   `mean`/`std` are taken over `dim=(1, 2)` — both axes of the whole input. In
   training that is a 16 s clip; at inference `musicality.inference.run_inference`
   passes the entire track in one forward call. The same eight bars therefore
   get a different input scale depending on what surrounds them, and one quiet
   intro re-scales every frame of the track. This is not the frequency-locality
   problem, but it lives in the same four lines, it is a genuine train/inference
   mismatch, and it is the cheapest fix in this entire document — a running or
   per-band normalisation, or simply normalising over a fixed window.

**What the field does instead, and what §1.3 measured about it.** Every
published tracker begins with local spectro-temporal processing: madmom's TCN
opens with 3×3 convolutions and frequency max-pooling, and Beat This! applies
frequency-wise partial attention to an 81-band log-mel at 50 fps. §1.3 is the
evidence that this is where their advantage lives — Beat This! reaches 0.935
`f_beat` and 0.818 `position_acc` on gtzan **with no DBN at all**, peak-picking
its own frame probabilities. Whatever produces that number is in the front end
and the trunk, because there is nothing else in the system. That is the
strongest argument in this document for fixing the input representation before
anything else.

**The fix, concretely.** Keep the mel; insert a small 2D stem in front of the
existing trunk:

| stage | output shape | note |
|---|---|---|
| log-mel | `(B, 1, 128, T)` | unchanged |
| `Conv2d(1, 16, 3×3)` + BN + GELU | `(B, 16, 128, T)` | the first time-frequency neighbourhood in the model |
| `MaxPool2d((3, 1))` | `(B, 16, 42, T)` | pool **frequency only** |
| `Conv2d(16, 16, 3×3)` + BN + GELU | `(B, 16, 42, T)` | |
| `MaxPool2d((3, 1))` | `(B, 16, 14, T)` | |
| `Conv2d(16, 16, 3×3)` + BN + GELU | `(B, 16, 14, T)` | |
| flatten frequency into channels | `(B, 224, T)` | |
| `Conv1d(224, channels, 1)` | `(B, channels, T)` | the existing `input_proj`, kept as the adapter |
| existing dilated trunk | `(B, channels, T)` | unchanged |

**Implemented** as `musicality.models.tcn.Conv2dStem`, off by default behind
`conv2d_stem:` in `configs/model/*.yaml`. Measured cost at `n_mels=128,
channels=256`: **1.613 M parameters to 1.643 M** — the stem itself is 4.9 k and
the rest is `input_proj` widening from 128 to 224 inputs. (This block previously
estimated "~10 k"; that was the stem alone and ignored the wider projection.)
Negligible beside the trunk either way, and it *composes* with §3.1's
shrink-and-deepen rather than competing for the budget — shrinking `channels`
shrinks the projection too. Two rules: pool frequency aggressively, and **never
pool time** — the frame rate is the output resolution (§2.2), and §3.1 is
already trying to buy context a cheaper way.

**What would falsify it.** If a 2D stem does not move `f_beat` on the corpora
where beats are weakest for everyone — rwc_genre and rwc_classical in §1.3 —
then the front end is not the bottleneck and §5 (data) is. That is a single
training run, it needs nothing else in this document to have happened first, and
it is why §7 puts it before the architecture work.

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
<summary><b>4.2 — A bar-pointer DBN is <i>not</i> the fix (measured, and this reverses the original claim)</b></summary>

> **This block previously argued the opposite.** It read our `amlt − cmlt` gap of
> 0.148 as "the missing tempo model stated as a number" and put importing
> `DBNDownBeatTrackingProcessor` near the top of the list. §1.3 measured that
> claim and it does not survive. The original text is preserved in the git
> history of this file.

We run a 4-state Viterbi over bar positions with a hand-set switch penalty and a
fixed threshold. The field's reference decoder is a **bar-pointer DBN**
(Krebs/Böck, `DBNDownBeatTrackingProcessor`): a joint state space over tempo ×
position-in-bar, with transitions encoding how tempo actually moves. The
inference drawn from that used to be that our metrical-level errors are caused
by not having one.

**What §1.3 shows.** `amlt − cmlt` is the share of a track tracked confidently at
the *wrong* metrical level, on the same 277 tracks:

| system | decoder | `amlt − cmlt` |
|---|---|---|
| ours (v6, re-swept) | 4-state Viterbi, no tempo model | 0.148 |
| **madmom** | **full bar-pointer DBN** | **0.149** |
| Beat This! | peak-picking, no DBN at all | **0.042** |

The system *with* the decoder we were told to import has our gap, to three
decimal places. The system with **no decoder at all** has a third of it. Whatever
produces a confident half-time or double-time reading, the presence or absence
of a bar-pointer DBN is not it.

The head-to-head on gtzan — the one corpus that is clean for both — says the
same thing without the metric subtlety: the DBN system loses on every number
(`f_beat` 0.902 vs 0.935, `cmlt` 0.776 vs 0.852, `pos_acc` 0.751 vs 0.818,
`confusion` 0.215 vs 0.112), and its phase-offset profile is worse too (8.7%
half-cycle errors vs 5.5%; 25.5% of tracks flipping phase mid-track vs 19.3%).

**What this means for the decoder work.** A decoder can only reorganise the
evidence it is handed. Beat This! demonstrates that when the frame probabilities
are good enough, peak-picking is sufficient — which relocates the problem
upstream, to §2.1 and §4.1, and is the single most useful thing §6.1 bought.

The cheap intermediate steps keep their value and are independently motivated,
so they stay:

- a **per-track adaptive threshold** — a quantile of the track's own activation
  instead of an absolute 0.8 that ballroom likes and gtzan hates
  (`plans/07` §3.1 measured exactly that disagreement);
- a **tempo-relative switch penalty** — ours is charged per beat, so the same
  number behaves differently at 185 BPM than at 105.

The full DBN drops to "importable, an afternoon, expect little" — worth trying
once, near the end, and worth abandoning quickly. Note also that madmom's model
files are CC BY-NC-SA (non-commercial), which matters for the decoder only if we
were to ship its weights rather than reimplement the state space.

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
<summary><b>6.1 — The experiment that decides everything else — <i>done</i></b></summary>

**Benchmark a public model (madmom, Beat This, or both) on our own 277-track val
split.** This was the highest-information action available and it built nothing.

**Run on 2026-09-17.** `tools/eval_baseline.py` (branch
`feature/lz_baseline_benchmark`) wraps both trackers and scores them through
`musicality.evaluation.score_events`, the same scorer `tools/eval_beat.py` uses.
Results are in **§1.3**; the three questions it was meant to answer:

1. **How large is the real gap?** Partly answered, and the answer moved the
   goalposts: the ceiling on our clean corpora is far below §1.1's recalled
   literature numbers (0.818 `position_acc` on gtzan, 0.553 on jtd). The
   remaining half — our own checkpoint on the same rows — is §7 step 1.
2. **Are our annotations sound?** Yes, with a caveat that is now a finding
   rather than a worry: a SOTA tracker also scores ~0.55 bar-position accuracy
   on jtd, so jtd is hard material, not a broken pipeline. `plans/07` §2.4
   closes.
3. **Is any of §2–§4 worth doing?** §2.1 gained the strongest evidence in this
   document; §4.2 was *refuted* and reversed. That single swap is worth more
   than the afternoon it cost.

**The caveat that now governs every number here.** Both trackers were trained on
most of our validation split. Only gtzan, rwc_genre and jtd are clean; see
`musicality.baselines.base.CORPUS_EXPOSURE`, which the tool prints before its
output. Reading the mean instead of the per-genre table is the easiest wrong
conclusion this experiment makes available.

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

*Revised 2026-09-17, after §6.1 was run. The previous order put the decoder at
step 4 and the front end at step 3; §1.3 reversed both.*

> **Standing constraint (2026-09-17, from Luc).** The project wants to work with
> *shorter* audio over time, not longer. That does not veto §3.1 — receptive
> field and training crop length are different things, and a thin deep TCN can
> be trained on 60 s crops and still run on 15 s of audio — but it does mean
> any design whose *accuracy depends on* having a minute of music in hand is the
> wrong direction. Read every "longer context" recommendation below as "more
> context per second of audio", and measure short-input performance before
> adopting one.

1. **Finish §6.1: score our own checkpoint on the same rows.** The baselines are
   measured (§1.3); we are not, on those exact tracks with those exact knobs.
   `tools/eval_beat.py --checkpoint checkpoints/checkpoint_v6.ckpt --dataset
   merge --split val --binary-only --per-genre`, re-swept first
   (`project_eval_beat_stale_knobs`), then put the per-genre tables side by
   side. Nothing below should be prioritised on the strength of §1.1's recalled
   literature numbers now that measured ones are available. **Half a day, and
   it is the only step that is pure measurement.**

2. **§2.1 — add a 2D convolutional stem** with frequency pooling. Promoted from
   step 3 to the first thing built, because §1.3 removed the competing
   explanation: Beat This! reaches 0.935 `f_beat` / 0.818 `position_acc` on
   gtzan with *no decoder*, so its advantage is in the front end and trunk.
   ~10 k parameters, composes with step 3, and §2.1 states what would falsify
   it in one training run.

3. **§3.1 — shrink and deepen** (16–32 channels, 11 layers), trained on 60 s
   crops but **evaluated at 15 s and 30 s as well**, per the constraint above.
   Still expected to be a large gain and still one config change; it is second
   rather than first only because step 2 is cheaper and better evidenced.

4. **§2.2 + the normalisation fix.** Move to 50–100 fps (hop 256), and fix the
   whole-input `mean`/`std` in `TCNTempoNet.forward` that makes the same eight
   bars normalise differently in training (16 s clip) and at inference (full
   track). The second of those is a few lines and is the cheapest item in this
   document.

5. **§4.1 — shift-tolerant BCE**, plus a dedicated downbeat activation. Beat
   This!'s own ablations single this out, and §1.3 is consistent with it: their
   frame probabilities are good enough to peak-pick.

6. **§5.1 — add Harmonix, ASAP and the rest of gtzan**, plus pitch-shift
   augmentation (§5.3). Then the beat-synchronous second stage (§3.2), whose
   premise §1.3 confirmed, or a transformer (§3.3).

7. **§4.2 — the bar-pointer DBN, last and expect little.** Demoted from step 4.
   madmom has one and carries our exact `amlt − cmlt` gap of ~0.149 anyway. The
   two cheap items inside §4.2 — a per-track adaptive threshold and a
   tempo-relative switch penalty — are independently motivated and can be done
   at any time.

**What changed and why.** The old order assumed our decoder and our size were
the two bottlenecks. §1.3 kept the size argument, killed the decoder argument,
and promoted the input representation into the gap. Steps 1–4 are days of work
and now hold most of the leverage. Everything in §3.3 still waits until the data
and the readout have stopped being the bottleneck — otherwise we compare
architectures through a broken readout, which is exactly the mistake
`plans/07` §3.1 documents.

**One target to stop chasing.** jtd `position_acc` is no longer evidence that
something is broken here: the state of the art scores 0.553 on it (§1.3). Track
it, do not optimise for it, and do not let it set the agenda.

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

**One disagreement is now resolved.** `plans/07` §2.4 left open whether jtd's
anchor failure was the material or our pipeline, and explicitly refused to
assert the former without evidence. §1.3 supplies it: a state-of-the-art tracker
that has never seen jtd scores 0.553 bar-position accuracy on it, with a
`best_off` of 0.825 — the same "consistent grid, wrong anchor" shape we show.
That question closes in favour of the material, and §4.4's memorisation
hypothesis for jtd loses its motivating puzzle.
