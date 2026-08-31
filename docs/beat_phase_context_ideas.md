# Beat-phase: ideas for fixing bar-position confusion

Notes from a discussion of what to try next on the beat-phase model, beyond
training 50 more epochs. Scope: architecture/context ideas only (data
expansion was considered and deliberately deferred).

## Diagnosis

LR-sweep run on `merge` (ballroom + rwc, `binary_only=True`), 150 epochs.
Frame-level metrics look healthy and show no overfitting:

- `train`: acc_beat 88%, acc_one 93%, acc_last 93%
- `val`: acc_beat 87%, acc_one 89-90%, acc_last 89-90%

But the downstream, decoded metrics (`configs/eval_beat.yaml`'s
postprocess-sweep notes, same checkpoint family) tell a very different story:

- beat F-measure: **91.6%**
- one/last F-measure: **69.5%**
- `confusion_half_cycle_rate`: **25.3%**

So the model finds beats reliably, but ~1 in 4 eligible beats has position 1
and position 3 (the diametric opposite in a 4-beat bar) swapped. This is a
**bar-phase disambiguation** problem, not a frame-accuracy problem — frame
accuracy is a poor proxy for it (see `frame_accuracy.py` docstring).

Why the architecture is prone to this: `label_bar_position`
(`musicality/postprocess.py`) resolves phase by counting forward from local
per-frame "anchor votes," and `TCNTempoNet` is fully convolutional with a
**fixed, bounded receptive field** — it has no way to use information from
outside that window, and no memory that persists across the clip.

Exact receptive field for the current `tcn_frames.yaml` (`kernel_size=3`,
`n_layers=8`, dilations `1,2,...,128`):

```
diameter = 1 + (kernel_size - 1) * sum(dilations) = 1 + 2*255 = 511 frames
         ≈ 11.9s  (at hop_length=512, sr=22050 → ~43.07 fps)
radius R ≈ 255 frames ≈ 5.9s each side
```

(The model docstring's `kernel_size * (2^n_layers - 1)` ≈ 17.8s estimate is
a looser upper-bound approximation — ~1.5x the true figure. Use ~5.9s/side
for any sizing math.)

Two candidate fixes, both aimed at giving the model access to context beyond
that ~5.9s/side window — not mutually exclusive.

## 1. Longer context (receptive-field / context padding)

### Naive version: just bump `duration`

`beat_train.yaml` currently trains on 10s clips (`data.duration: 10.0`).
Bumping this (e.g. toward 18s) is a one-line config change, no architecture
change — cheap to test.

**Caveat — it under-delivers on its face value.** Every `Conv1d` in the
trunk uses `padding=2**i` ("same" padding), so a frame only gets a
zero-free receptive field once it's ≥R (~5.9s) from *both* edges of
whatever's fed to the model. At `duration=18s` (776 frames), the fully-clean
central region is only `776 - 2*255 ≈ 266 frames ≈ 6.2s` — about a third of
the clip. The rest sits on a gradient of partial zero-contamination (worse
near the edges). Real benefit, but far short of "the whole clip gets full
context."

### Correct version: context padding

Feed the model more real audio than you intend to score, so the "same"-padding
zeros land *outside* the scored window:

1. Crop a wider waveform window: `D + 2R` instead of `D`. For the current
   `D=10s`: `10 + 2*5.9 ≈ 21.8s` (~22s) of input.
2. Run it through `TCNTempoNet` unchanged — it's fully convolutional and
   doesn't care about input length.
3. Slice the model's per-frame output down to the central `D`-second region
   before computing loss. Every frame in that slice is provably ≥R from
   either edge of the wide input, so its receptive field never touches a
   synthetic zero — 100% real audio, guaranteed.
4. Build/slice the target (`beat`/`one`/`last`/`mask`) the same way so it
   lines up with the cropped output.

**Implementation shape:** `BeatDataset` crops `n_samples + 2*margin_samples`
and builds targets over the wider frame grid. `TCNTempoNet.forward` needs no
change. Augmentations (`FrameTimeStretch` etc.) run on the whole wide window
first (already supports arbitrary-length wav+target pairs); slicing to the
central region happens *after* the model call, in the train/val step, right
before `beat_phase_loss`.

**Cost:** margin (`2R ≈ 11.8s`) is a *fixed* overhead, so it amortizes
better with a larger scored window — e.g. scoring `D=10s` costs ~2.2x
today's compute (10s→21.8s total), but scoring `D=20s` only costs ~1.6x
(20s→31.8s total). If building this, pair it with a larger `D` rather than
keeping `D=10s`.

### Suggested test order

Run the naive duration bump first (cheap, near-zero effort) and check
whether `confusion_half_cycle_rate` / one-last F-measure move at all via
`eval_beat.py` — not just frame accuracy.

- **Moves the needle** → context helps; worth building context padding
  properly (with a larger `D` to amortize the margin).
- **Doesn't move** → evidence the half-cycle confusion is more a
  locally-ambiguous-audio problem (e.g. symmetric four-on-the-floor passages
  where 1 and 3 genuinely sound alike within *any* local window) than a
  missing-context problem. Context padding likely won't fix that either —
  points toward option 2 instead.

## 2. Longer-range head: BiGRU or self-attention

Both remove the TCN's fixed-window ceiling by giving a frame's
representation access to *any* other frame in the clip, not just neighbors
within ~5.9s. Useful if the model needs to carry an unambiguous phase belief
(e.g. from a clear intro) through a later locally-ambiguous stretch — a
"memory" problem a fixed-window conv structurally cannot solve no matter how
big the window gets.

### BiGRU

- Processes the sequence step-by-step, maintaining a hidden-state summary
  that gated updates blend new frames into — this is what lets it retain a
  signal ("we're on beat 1") over many steps without washing out.
- Bidirectional = one GRU left-to-right, one right-to-left, concatenated —
  each frame sees everything before *and* after it in the clip.
- Cheap in parameters/FLOPs, but sequential (can't parallelize over time),
  so slower per training step than the TCN.
- Memory does still degrade over very long sequences, just far less rigidly
  than a fixed conv window.
- **This is the standard architecture pattern in the beat-tracking
  literature** (e.g. Böck's trackers): CNN for local acoustic features, then
  BiLSTM/BiGRU for temporal/phase integration — a proven fit for exactly
  this failure mode.

### Self-attention

- Every frame directly compares itself against every other frame in one
  shot (query/key/value dot-product), rather than propagating information
  step-by-step. Better suited to precise long-range matching ("compare me to
  that frame 40s away right now") since it doesn't rely on a running summary
  that can dilute over many steps.
- Needs an explicit positional encoding (unlike conv/RNN, attention has no
  built-in sense of frame order) and is typically used as multi-head
  attention + feedforward sublayer, with residuals/LayerNorm.
- Cost is O(T²), but for clip lengths in play here (400-1300 frames) that's
  still tractable — not the limiting factor.
- Weaker built-in structural bias than conv/recurrence → generally needs
  more data/tuning to train well (well-documented pattern, e.g. ViT vs CNN
  on small datasets). Real risk given the current dataset size.

### Recommendation

Try BiGRU first: proven fit for this exact task, smaller implementation and
tuning footprint, better suited to the current (comparatively small) dataset
than self-attention's weaker inductive bias. Reserve self-attention for if
BiGRU's gains plateau and long-range *precise* pattern-matching still looks
like the bottleneck.

**Placement:** after the TCN trunk (local acoustic features), before the
frame head — `(B, channels, T)` → transpose → BiGRU → transpose back →
`frame_head`. Consider scoping it to just the `one`/`last` heads (leaving
`beat`, already at 91.6% F, reading straight off the trunk) to limit blast
radius and target the diagnosed weak point directly rather than the whole
model.

## Decision: self-attention head

Going with self-attention over BiGRU. Key correction from working through the
details before implementing:

**Self-attention only helps if training clips are longer than the TCN's own
receptive field.** `beat_train.yaml` currently trains on 10s crops
(`data.duration: 10.0`) — already *shorter* than the trunk's own nominal RF
(~11.9s, see above). Attention doesn't invent context that isn't in its
input; it only removes the *fixed-window* ceiling, and that only matters if
it's actually handed a longer sequence than the window it's replacing. So
adding an SSA head on top of the current 10s crops would **not** give any
long-range benefit beyond what the TCN trunk already nominally covers.

What SSA *would* still buy at the current 10s, even with no other change:
no boundary degradation (every frame gets full, non-degraded access to the
whole crop, unlike the trunk's "same"-padded convs near the crop edges) and
content-based routing instead of convolution's fixed distance-based mixing.
Real, but not the "carry a phase belief past a locally ambiguous stretch"
benefit that motivated the idea.

**Action: SSA and "longer context" (section 1) compose — they're not
independent.** To get the long-range benefit, pair the SSA head with a real
`data.duration` bump in `beat_train.yaml` (well beyond 10s — 20-30s+, per
compute budget). The trunk's own edge-degradation cost becomes a smaller
*fraction* of a longer clip (`2R≈11.8s` out of 30s ≈ 39%, out of 60s ≈ 20%),
so the full context-padding machinery from section 1 probably isn't needed —
some residual edge softness at the two ends is acceptable and shrinks
proportionally as clip length grows.

**Inference-time length generalization (attention-specific, optional):**
unlike the TCN (hard-capped at its architectural RF) or BiGRU (recurrent
state that empirically degrades over very long sequences), a self-attention
layer has no hardcoded max distance — it can in principle be *run* at
inference on a longer sequence than it was trained on (e.g. train on 20-30s
crops, run inference on a full 3-4 minute track). Requires a positional
encoding that generalizes past its training length (relative position
encoding, not naive absolute/sinusoidal used out-of-range), and isn't
guaranteed to work — the model never saw long-range examples during
training, so whether it actually learned to use far-away context at
inference is an empirical question. Validate via `eval_beat.py` on real
full-track runs rather than assuming it. Compute is not a blocker either way
(~60-100M ops for one attention layer over a full 3-4 min track at the
current channel width).

**Next step:** pick a training `duration` (candidate: 24-30s, with
`batch_size` reduced to fit), then implement the SSA head — after the TCN
trunk, before the frame head, per the placement notes above.

## Implementation plan (ordered by difficulty)

BiGRU dropped — going with self-attention only. Steps below build on each
other; do them in order.

### 1. Bump training clip duration (easiest) — done

- `configs/beat_train.yaml`: `data.duration` bumped `10.0 → 16.0` (clears
  the trunk's ~11.9s receptive field, satisfying the prerequisite below).
  `batch_size` left at `16` — only ~1.6x more frames than the 10s default,
  smaller memory bump than a more aggressive 24-30s duration would cause;
  revisit if training hits memory pressure.
- Prerequisite for step 2 — SSA over crops shorter than the trunk's own RF
  has no long-range benefit to give (see "Decision" above).

### 2. Add the SSA head to `TCNTempoNet` (moderate) — done

- `musicality/models/tcn.py`: `PositionalEncoding` + `SelfAttentionBlock`
  (hand-built from `nn.MultiheadAttention` + residual/LayerNorm + feedforward
  sublayer, not `nn.TransformerEncoderLayer` directly) added, wired into
  `TCNTempoNet` via a new `use_self_attention` flag.
- Scoped, per the placement notes above: `beat_head` reads straight off the
  trunk unchanged (already at 91.6% F); only `one`/`last` route through
  `phase_head` (positional encoding, applied once, then a stack of
  `n_attn_layers` `SelfAttentionBlock`s). Verified by test that `beat`'s
  gradient never reaches `phase_head`'s parameters.
- `configs/model/tcn_frames.yaml`: `use_self_attention` (now `true`),
  `n_attn_layers` (1), `n_attn_heads` (4) added. Verified end-to-end through
  real Hydra composition + `BeatPhaseModule` (instantiate → forward → loss →
  backward).
- Tests: `tests/test_tcn.py` (`TestPositionalEncoding`,
  `TestSelfAttentionBlock`, `TestSelfAttentionIntegration`).
- Not yet done: retrain and evaluate via `eval_beat.py` /
  `tools/sweep_beat_postprocess.py` — watch `confusion_half_cycle_rate` and
  one/last F-measure specifically, not just frame accuracy.

### 3. Proper context padding (harder — conditional)

Only pursue if step 2's results still show edge-degradation symptoms (e.g.
metrics on tracks/crops sensitive to where the crop boundary falls).

- `musicality/loaders/beat_dataset.py`: crop `n_samples + 2*margin_samples`
  instead of `n_samples`; build `beat`/`one`/`last`/`mask` targets over the
  wider frame grid.
- `musicality/trainers/beat_phase_module.py`: `align_time` currently crops
  the longer of `(logits, target)` to match the shorter (off-by-one fix) —
  extend this step to slice the wide model output down to the central
  scored region before `beat_phase_loss`, instead of/in addition to the
  existing shorter-side crop.
- Augmentations (`BeatPhaseAugmenter`/`FrameTimeStretch` in
  `musicality/augmentations.py`) already operate on arbitrary-length
  wav+target pairs — should need no change, just confirm they run on the
  wide window before the central-region slice happens.

### 4. Inference-time length generalization to full tracks (hardest, exploratory)

**Confirmed this isn't hypothetical — it's the current default behavior.**
`musicality/inference.py`'s `load_track_waveform` loads the *entire* track
("no cropping/padding", per its own docstring) and `run_inference` feeds it
through the model in one unchunked forward pass. Measured frame counts at
`hop_length=512, sr=22050`: a 16s training clip is 690 frames; a 3-4 minute
track is ~7,750-10,300 frames — **~11-15x longer than anything the SSA head
was ever trained on.** So as soon as this checkpoint is loaded for
`eval_beat.py` or any inference script, the attention block is already
extrapolating to sequence lengths it never saw, with no explicit choice
required to trigger it. Two concrete risks, not just theoretical:

- Query/key projections were only optimized on relative distances up to
  ~690 frames apart; interactions thousands of positions apart are pure
  extrapolation — whether they're meaningful is an open empirical question.
- Memory: a ~10,000-frame attention matrix is `10,000² × n_heads(4) ≈ 400M`
  floats ≈ 1.6GB in fp32 for one layer's attention weights alone. FLOPs are
  cheap (~60-100M ops), but this is a real memory consideration on
  constrained inference hardware.

Two ways to handle it — not mutually exclusive, but different effort levels:

- **(a) Relative positional encoding** (this section, as originally
  scoped): swap sinusoidal/absolute encoding for a relative scheme (e.g.
  relative position bias), then validate directly on full-track
  `eval_beat.py` runs against the crop-length-only baseline. Doesn't fix the
  underlying train/inference length mismatch, just gives the model a better
  chance of coping with it. No guarantee of a win.
- **(b) Chunked inference** — see the dedicated plan below. Sidesteps the
  extrapolation question entirely by keeping inference sequence length equal
  to training sequence length. **Chosen for the next MR** — see below.

## Chunked inference (planned — separate MR)

Goal: keep the SSA head's inference-time input length consistent with what
it was trained on, instead of feeding it a full track ~11-15x longer (see
step 4 above). Sidesteps the extrapolation question entirely rather than
attempting to fix it.

**Tradeoff going in:** boundary effects don't disappear, they just move.
Splitting a track into `duration`-length chunks (16s, matching
`beat_train.yaml`) reintroduces an edge-degradation-like discontinuity at
each chunk boundary — same category of problem the SSA head/context padding
(section 1) target for the TCN trunk, just recurring every ~16s instead of
every ~12s (the plain trunk's RF). Overlapping chunks (below) mitigate this
but don't eliminate it — worth checking empirically whether chunked
inference actually beats full-track inference before treating this as
strictly better in all cases, rather than just "distribution-matched but
choppier at the seams."

### Design

- **Where:** `musicality/inference.py`. `load_track_waveform` stays as-is
  (still loads the full track). Add the chunking logic to `run_inference` —
  or a wrapper it delegates to — since that's the one place both
  `tools/eval_beat.py` and `tools/annotator/inference.py` ultimately call
  through `musicality.inference`.
- **Chunk size:** match `beat_train.yaml`'s `data.duration` (16s → 690
  frames) exactly, so inference sequence length equals training sequence
  length — the whole point of this change.
- **Overlap:** non-overlapping chunks make every chunk boundary a hard cut
  with no context on either side — worse than the trunk's own edge
  degradation, not better. Use overlapping chunks (e.g. 50% stride) and
  either (a) keep only the central portion of each chunk's output — same
  "context padding" idea from section 1, just applied at inference instead
  of training — or (b) blend overlapping regions (e.g. linear crossfade on
  the per-frame probabilities). Start with (a): simpler, and directly reuses
  the section-1 idea instead of introducing a new blending mechanism.
- **Stitching:** concatenate each chunk's kept (central) region of `beat`/
  `one`/`last` probabilities in track order to rebuild one full-track
  per-frame probability curve, then feed that into `postprocess.readout`
  unchanged — `pick_peaks`/`gate_periodicity`/`label_bar_position` all
  already operate on an arbitrary-length curve, so nothing downstream of
  probability reconstruction needs to change.
- **Scope check:** only the beat-phase (`task="beat_phase"`) path needs
  this — `BeatModule`'s beat-only head has no SSA/attention component, so
  `readout_beat_only`'s full-track inference is unaffected by the length
  mismatch and doesn't need chunking for this reason (may still want it for
  memory reasons on very long tracks, but that's a separate, lower-priority
  motivation).

### Last-chunk padding: why it needs a decision, not just zero-padding

Track length almost never divides evenly into `duration`-length (16s)
windows at whatever stride is chosen. Interior chunks are always full, real
16s slices — no padding needed. The **last chunk of a track** is the
exception: it typically has less real audio left than `duration` (e.g. 5s
real audio, needing 11s of zero-padding to reach the 690 frames the model
expects). `BeatDataset` already zero-pads short tracks the same way at
training time — what's new here is what zero-padding does once it reaches
the SSA head specifically.

**Why self-attention is worse off than the TCN trunk here.** The trunk
already tolerates zero-padding fine, because convolution's contamination is
*local and decaying* — a "same"-padded conv only pollutes frames within its
kernel's reach of the padded region, attenuating with distance. Self-attention
has no such locality: `nn.MultiheadAttention`'s softmax normalizes over
*every* key in the sequence, including padded frames, so **every real frame
in the chunk gets some nonzero attention weight pointing at the fake padded
frames** — not just frames near the boundary. The padded frames steal a
slice of attention probability mass from every position in the whole 16s
chunk, globally, not just at the edge. This is exactly what the
`SelfAttentionBlock.forward` comment (`musicality/models/tcn.py`) flags:
"attention will silently attend into padding frames." Cropping the *output*
at padded positions afterward doesn't fix this — real frames near the
padded tail already had their representations diluted during the forward
pass, before any output gets cropped.

**Two ways to handle it — pick one when building this MR:**

1. **Thread a `key_padding_mask` through.** `nn.MultiheadAttention` already
   accepts this (`(B, T)` boolean, True = ignore). Requires real plumbing,
   not a one-line fix: `SelfAttentionBlock.forward` needs a `mask`
   parameter, `TCNTempoNet.forward` needs to build and pass one through
   `phase_head`'s attention loop, and the chunking helper in
   `musicality/inference.py` needs to know, per chunk, how many trailing
   frames are padding vs. real.
2. **Don't pad the last chunk at all — feed it at its true, shorter
   length.** `TCNTempoNet` is fully convolutional + attention, both
   shape-agnostic in `T` (`BatchNorm1d` doesn't care about `T` either), so
   the last chunk can just be fed as whatever real audio remains, no
   padding, no mask. Simpler to implement, but reintroduces a — much
   smaller — length-generalization gap: the model was only ever trained on
   exactly-`duration` inputs, so a shorter final chunk is slightly
   out-of-distribution too, just far less dramatically than the
   full-track-at-once problem in step 4 above.

**Leaning towards option 2** — simpler, avoids adding masking plumbing
across three files for what's ultimately a one-chunk-per-track edge case.
Worth revisiting if the shorter-final-chunk distribution shift turns out to
matter empirically once this is built.

### Steps

1. Add a `chunk_waveform_and_stitch`-style helper (exact name TBD) near
   `run_inference` in `musicality/inference.py`: splits a full-track
   waveform into overlapping `duration`-length windows, runs each through
   the model, keeps each window's central region, concatenates into one
   full-track probability curve per channel (`beat`/`one`/`last`).
2. Wire it into `run_inference`'s `task == "beat_phase"` branch, replacing
   the current single unchunked forward pass. Gate behind a parameter (e.g.
   `chunk_duration: float | None = None`) so `task == "beat_only"` and any
   caller that wants the old unchunked behavior are unaffected —
   backward-compatible default.
3. New/updated tests in `tests/test_inference.py`: chunked output on a
   short synthetic track matches (within tolerance) the unchunked output
   for a track short enough to fit in one chunk (sanity check the stitching
   logic doesn't distort the trivial case); output length matches the
   full-track frame count; no discontinuity artifacts at chunk boundaries
   beyond what overlap-and-trim is expected to leave.
4. Re-run `eval_beat.py` (full-track, real checkpoint) both ways — chunked
   vs. the current unchunked full-track pass — and compare
   `confusion_half_cycle_rate` / one-last F-measure. This is the actual
   test of whether the mismatch flagged in step 4 above was hurting
   real performance, not just a theoretical concern.

## Deferred

Expanding training data beyond `merge` (ballroom + rwc) to the other
position-annotated datasets already sitting in `musicality_db/` (brid,
hainsworth, swing, groove_midi, guitarset, gtzan, jtd, MTG-JAAH, personal
annotator tracks) — lowest-risk lever given no overfitting sign, but
explicitly deferred for now in favor of the architecture ideas above.
