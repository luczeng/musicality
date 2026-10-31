# Periodicity losses for the beat and bar-position heads: mathematical formulation

<!-- Math has to render on GitHub and in markdown-it/KaTeX previews (nvim), so it uses only $...$ and $$...$$.
GitHub strips the backslash from \, \; \! \{ \} \\ inside math: write \thinspace, \negthinspace, "\ " (a space),
\lbrace, \rbrace, \cr. In inline math, an underscore right after } can start GitHub italics: write z^b_t, not z^{b}_t. -->

## Overview

### High level

Our model has two heads. The beat head says, for every moment, whether a beat is there. The position head says, for every beat, which position of the bar it is: 1, 2, 3 or 4. Both are trained one moment or one beat at a time. Nothing in training asks consecutive beats to be as far apart as the annotated ones, and nothing asks the positions to *count*: 1, 2, 3, 4, then 1 again.

The position head gets a single beat right about three times in four, and each beat is judged alone: a head that is right on one beat and wrong on the next is penalised for the wrong beat, but not for breaking the count. The aim is a network whose answers already count, so that less has to be repaired after training.

This document specifies two losses that teach these relationships during training, added to the existing ones rather than replacing them:

- **Beat spacing.** The gap between neighbouring predicted beats should match the annotated gap, within annotation jitter. The continuity metrics only accept a beat whose distance to the previous one is within 17.5% of the annotated distance.
- **Position periodicity.** The positions the head predicts should advance by one at every beat. This term does not care which beat the network calls the 1; the existing position loss already decides that. It only asks that, whatever the network's choice, the choice stays the same from one bar to the next.

The existing losses keep deciding whether a beat is there and which label each beat carries; the new terms decide how beats relate to each other. Both act only during training, the model is unchanged, and both are switched on gradually after the first epochs, once the model has learned to produce peaks. How much of the position head's error the periodicity term can address is the main open question (§9).

### Technical

It targets the baseline beat-phase model: a frame-level network with one beat logit and $G$ position logits per frame, trained by a beat binary cross-entropy plus a position cross-entropy. It defines two auxiliary terms added to that loss:

- $\mathcal L_{\text{beat}}$ works on the **beat head, on the time axis**. A soft-argmax inside per-beat windows gives sub-frame beat centres. Their inter-beat intervals are compared with the ground-truth intervals in log space.
- $\mathcal L_{\text{period}}$ works on the **position head, across beats**. At each annotated beat it measures how far the head's count is shifted from the annotation, the *offset*, and asks all beats of a clip to share one offset, whichever it is. A whole clip shifted by a constant offset costs nothing, which leaves the anchoring to the existing position cross-entropy.

The setup is the project's: 43.07 fps (22050 Hz, hop 512), 16 s clips, $G = 4$ positions per bar, annotated positions folded onto $1..G$, and every training track carries bar positions.

The evidence behind the overview's number is a one-off measurement on the v7 checkpoint (2026-09-23, 40 val and 39 train tracks, annotated beat frames). The position head is 73% accurate per beat, on train and val alike, and underconfident (best temperature about 0.65).

The sections build on each other. §0 fixes notation and §1 defines the soft-argmax the beat term relies on. §2 builds, from the annotations alone, what both terms read: a window of frames per beat, and each beat's reference position. §3 and §4 define the two terms, and §5 adds them to the existing loss. §6 derives their gradients, which show what each term can and cannot do to the logits. §7 collects their invariances and their link to the evaluation, §8 lists the defaults, and §9 the open questions.

---

## 0. Conventions

Everything below shares this notation. Times are counted in frames, but annotations keep their sub-frame precision, because the beat term measures positions more finely than one frame. The constant $\gamma$ is the evaluation's own tolerance on beat spacing: §3 uses it as the unit of error, so that the loss and the metric agree on what counts as close.

- The frame rate is $22050 / 512 \approx 43.07$ fps and all times are in frames. Ground-truth positions are real-valued (sub-frame).
- $\mathrm{sg}[\cdot]$ is **stop-gradient**: $\mathrm{sg}[a]$ has the value of $a$ in the forward pass, but no gradient flows through it in the backward pass, as if $a$ were a constant. It lets a quantity computed from the network's output *weight* a loss without becoming something the network can change to lower that loss.
- $[x]_+ = \max(x, 0)$, $\mathbb{1}[\cdot]$ is the indicator and $\sigma$ is the logistic sigmoid.
- $\gamma := \log 1.175 \approx 0.1613$ is the upper bound of the continuity metrics' tolerance in log space.
- The beat term's sums over beat pairs run over the **whole batch**: it is a batch-pooled mean. The periodicity term is averaged per clip first (§4.2), like the existing position loss.

| Symbol | Meaning | Shape |
|---|---|---|
| $z_t$ | beat head logit | $(B, T)$ |
| $y_{t,j},\ j = 1..G$ | position head logits | $(B, G, T)$ |
| $b_k,\ k = 1..K$ | ground-truth beat (sub-frame) | $(B, K)$ |
| $\Delta_k = b_{k+1} - b_k$ | ground-truth inter-beat interval | $(B, K-1)$ |
| $n_k \in \lbrace 1, \dots, G \rbrace$ | ground-truth bar position, folded (the reference) | $(B, K)$ |
| $\mathcal{W}_k$ | frame window of beat $k$ | $(B, K, W)$ |
| $\hat b_k$ | soft-argmax beat centre | $(B, K)$ |
| $f_k$ | frame of beat $k$ | $(B, K)$ |
| $q_{k,j}$ | position probabilities at beat $k$ | $(B, K, G)$ |
| $\pi_k(o),\ o = 0..G-1$ | probability that beat $k$'s count is shifted by $o$ | $(B, K, G)$ |

For a 16 s clip: $T \approx 689$, $W = 2H + 1 = 69$, $K \approx 15\text{–}51$ (56 to 193 BPM).

---

## 1. Shared operator: masked softmax with adaptive temperature

The beat term needs to know where a peak is inside its window (§3). A hard argmax answers that but has no gradient. A soft-argmax does: turn the logits into a softmax distribution and take the expected position, which moves smoothly as the logits change. This section defines that softmax. The index set $\mathcal{I}$ is a window's frames ("masked" because, in code, windows of different sizes are padded to one shape). Its one non-standard part is the temperature, which follows the scale of the logits so that the softmax does not collapse to one-hot on an overconfident beat head. The periodicity term does not need it: it reads the position head's own softmax as it is (§4.2).

Take logits $a_i$ over a finite index set $\mathcal{I}$ and a range parameter $\kappa$:

$$
m = \max_{i \in \mathcal{I}} a_i, \qquad
\bar a = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} a_i, \qquad
\tau = \mathrm{sg}\negthinspace\left[ \max\negthinspace\left( \tau_{\min},\ \frac{m - \bar a}{\kappa} \right) \right]
$$

$$
\mathrm{softmax}_{\kappa}(a)_i = \frac{\exp\big( (a_i - m)/\tau \big)}{\sum_{i' \in \mathcal{I}} \exp\big( (a_{i'} - m)/\tau \big)}
$$

**Why adaptive.** Peak-sharpening training (shift tolerance in particular) produces overconfident logits ($|a| \approx 10$). With a fixed $\tau$, the softmax becomes one-hot and the gradients, which are proportional to $p_i$ (see §6), vanish. With the adaptive $\tau$, whenever the floor is inactive the scaled mean logit sits exactly $\kappa$ below the scaled maximum, $(m - \bar a)/\tau = \kappa$, whatever the logit scale. The temperature is detached, so it is never a path for the network to lower the loss.

---

## 2. Target construction

Before either term can be computed, the annotations have to be turned into what the terms read. The beat term needs, for every annotated beat, a window of frames expected to hold that beat's peak and no other (§2.1). The periodicity term needs, for every annotated beat, the frame where the position head is read and the position that beat should have (§2.2). All of this depends only on the annotations, not on the network's output.

### 2.1 Beat windows (Voronoi, capped)

Each beat gets the frames that are closer to it than to its neighbours, a one-dimensional Voronoi cell, capped at $H$ frames on either side. Splitting at midpoints means no frame is counted for two beats, and the cap keeps windows bounded at slow tempos and across gaps in the annotation. The first and last beats have only one neighbour, so they borrow the adjacent interval for their open side.

$$
\ell_k = \tfrac{1}{2} (b_{k-1} + b_k), \quad \ell_1 = b_1 - \tfrac{1}{2} \Delta_1, \qquad
r_k = \ell_{k+1}, \quad r_K = b_K + \tfrac{1}{2} \Delta_{K-1}
$$

$$
\mathcal{W}_k = \lbrace\thinspace t \in \mathbb{Z} \ :\ \ell_k \le t < r_k,\ \ \lvert t - b_k \rvert \le H,\ \ 0 \le t < T \thinspace\rbrace
$$

The windows are disjoint and ordered, so every frame belongs to at most one window.

A beat is **valid** if its window was not cut by the clip edge and has at least 3 frames. A cut window would bias its soft-argmax towards the inside of the clip, and fewer than 3 frames are too few to locate a peak:

$$
v_k = \mathbb{1}[\ell_k \ge 0]\ \mathbb{1}[r_k \le T]\ \mathbb{1}\big[ |\mathcal{W}_k| \ge 3 \big]
$$

### 2.2 Read frames and reference positions

The position head is read once per annotated beat, at its nearest frame, $f_k = \mathrm{clip}\big( \mathrm{round}(b_k),\ 0,\ T - 1 \big)$, which is where the position cross-entropy is centred.

The periodicity term compares the head's answer at each beat with the position that beat should have, its **reference position** $n_k$, the folded annotated position. A pickup, a missing annotated beat or a short bar is then harmless, because the reference follows the annotation rather than a counter.

---

## 3. Beat term

The first term checks the spacing of the predicted beats. Inside each beat's window, a soft-argmax locates the predicted peak to sub-frame precision. The gap between consecutive peaks is then compared with the annotated gap as a ratio, so a given relative error costs the same at any tempo. Errors within annotation jitter are free, larger ones are measured in units of the evaluation's tolerance, and pairs where the network shows no clear peak are skipped. The four steps below follow that order.

**Soft-argmax inside each window**

$$
p_{k,\cdot} = \mathrm{softmax}_{\kappa_b}\big( z_t \big)_{t \in \mathcal{W}_k}, \qquad
\hat b_k = \sum_{t \in \mathcal{W}_k} p_{k,t}\ t, \qquad
m_k = \max_{t \in \mathcal{W}_k} z_t
$$

$\hat b_k$ is where the network places beat $k$. $m_k$, the largest logit in the window, measures how confident that peak is; the gate below uses it.

**Log-interval residual**

The residual is 0 when the predicted gap equals the annotated one, and depends only on their ratio, so a 10% error costs the same at 60 BPM as at 180 BPM.

$$
\rho_k = \log \max\big( \hat b_{k+1} - \hat b_k,\ 0.5 \big) - \log \max\big( \Delta_k,\ 0.5 \big)
$$

Because the windows are disjoint and ordered, $\hat b_{k+1} > \hat b_k$ always holds. The clamp at 0.5 frames is only there for numerical safety.

**Excess beyond the deadzone, in tolerance units**

$\rho_k$ says how wrong a gap is, but it is not yet the thing to penalise. Two adjustments come first: ignore errors that are only annotation noise, and measure the rest in a unit the evaluation gives meaning to. Both happen in one line:

$$
x_k = \frac{\big[\thinspace |\rho_k| - \varepsilon \thinspace\big]_+}{\gamma}
$$

- **The deadzone $\varepsilon$ (default 0.05, about 5%).** The annotated gaps are themselves noisy, because each annotated beat is placed by a person, or by a tapping device, with some error. A gap inherits the error of both its ends. If each beat is off by about 15 ms, a gap is off by about $\sqrt{2} \times 15 \approx 21$ ms, which is 4% of a gap at 120 BPM. Penalising differences that small would train the network to reproduce the annotator's noise instead of the music's beat, so the first $\varepsilon$ of every residual is subtracted and costs nothing. The value 0.05 is a judgement, not a measurement (§9). Because it is relative, it is also tempo-dependent in milliseconds: 5% of a gap is 50 ms at 60 BPM but 17 ms at 180 BPM, whereas annotation noise is probably closer to constant in milliseconds.
- **The unit $\gamma$ (default $\log 1.175 \approx 0.161$).** The continuity metrics reject a beat whose gap to the previous one is more than 17.5% off the annotated gap. In log terms that limit is $\gamma$. Dividing by $\gamma$ expresses what is left after the deadzone in multiples of that tolerance, so $x_k$ reads directly as "how far past the deadzone, in metric tolerances". It also fixes where the penalty changes shape in the next step, $x_k = 1$, from the evaluation instead of from a free constant.

What this gives for a gap that is too long by a given amount (too short behaves the same, up to the small log asymmetry):

| Predicted gap vs annotated | $\rho_k$ | $x_k$ | Penalty $h(x_k)$ | Pull $h'(x_k)$ |
|---|---|---|---|---|
| 3% too long | 0.030 | 0 (inside the deadzone) | 0 | 0 |
| 10% too long | 0.095 | 0.28 | 0.04 | 0.28 |
| 17.5% too long (the metric's limit) | 0.161 | 0.69 | 0.24 | 0.69 |
| 23.5% too long | 0.211 | 1 | 0.5 | 1 |
| doubled or halved (octave error) | ±0.693 | 3.99 | 3.49 | 1 |

The metric's own limit lands at $x_k = 0.69$, not at 1, because the deadzone is subtracted first. Dividing by $\gamma - \varepsilon$ instead would put it exactly at 1; which normalisation to use is open (§9).

**Penalty, gate and loss**

What remains is how fast the penalty grows, which pairs count, and how they are averaged.

$$
h(x) = \begin{cases} \tfrac{1}{2} x^2 & x \le 1 \cr x - \tfrac{1}{2} & x > 1 \end{cases}
\qquad
g_k = \mathrm{sg}\big[ \sigma(m_k)\thinspace \sigma(m_{k+1}) \big]
\qquad
\mathcal{P} = \lbrace k : v_k = v_{k+1} = 1 \rbrace
$$

$$
\boxed{\ \mathcal{L}_{\text{beat}} = \frac{1}{|\mathcal{P}|} \sum_{k \in \mathcal{P}} g_k\ h(x_k)\ }
$$

- $h$ is the Huber function with threshold 1: quadratic up to $x_k = 1$, linear beyond. What matters for training is its slope, the pull $h'(x) = \min(x, 1)$ in the table above, which is capped at 1. Below the cap, the pull grows with the error, so a gap that is a little off gets a little push and a gap that is badly off a bigger one. Above it, every gross error (an octave jump, or a window whose soft-argmax lands far off because of a missed or extra peak) pulls equally hard. Without the cap, an octave error would pull 14 times harder than a 10% error, against 3.6 times with it; a handful of such pairs would then dominate the gradient, although they are mostly not fixable by moving a peak inside its window and are the BCE's job.
- **The gate $g_k$ decides how much pair $k$ counts.** $m_k$ is the largest beat logit in window $k$, so $\sigma(m_k)$ is the beat head's probability at its most confident frame there: roughly, "is there a peak in this window at all?". The product $\sigma(m_k)\thinspace \sigma(m_{k+1})$ is close to 1 only when both windows hold a confident peak, and close to 0 as soon as either is empty. A soft-argmax over a window with no peak returns the window's centre of mass over background noise, and the interval it implies says nothing about the music, so such pairs are faded out instead of being trained on. Missing peaks are the BCE's job.

  | Largest logits $(m_k, m_{k+1})$ | Probabilities | $g_k$ | Meaning |
  |---|---|---|---|
  | (10, 10) | 1.00, 1.00 | 1.00 | two sharp peaks: counts fully |
  | (4, 4) | 0.98, 0.98 | 0.96 | two clear peaks |
  | (2, 2) | 0.88, 0.88 | 0.78 | two moderate peaks |
  | (0, 0) | 0.50, 0.50 | 0.25 | two undecided windows: counts a quarter |
  | (4, −4) | 0.98, 0.02 | 0.02 | one window empty: almost ignored |

  **Why the gate is detached.** Without $\mathrm{sg}$, the gate would be a second way to lower $\mathcal L_{\text{beat}}$: whenever a pair is badly spaced, the network could shrink $g_k$ by *lowering its beat peaks*, since the gradient on $m_k$ would be $h(x_k)\thinspace \sigma'(m_k)\thinspace \sigma(m_{k+1}) > 0$. A spacing loss would then teach the beat head to hide beats, against the BCE. Detached, the gate only reads how confident the peaks are; the network can lower the loss only by moving its peaks. The temperature $\tau$ in §1 is detached for the same reason.
- $\mathcal{P}$ keeps the pairs whose two windows are both valid (§2.1).
- The loss is normalised by $|\mathcal{P}|$ rather than by $\sum g_k$, so gated pairs reduce the loss instead of reweighting the remaining pairs.

---

## 4. Position terms

The position head answers, for each beat, "which position of the bar are you?". Two different things can be asked of those answers:

- the **anchor**: each beat should carry its annotated position (§4.1). This is the existing position cross-entropy, one beat at a time;
- the **periodicity**: the positions should advance by one at every beat, wherever the count starts (§4.2). This links the beats to each other, which is what "after 1, 2, 3, 4 comes 1 again" means.

Neither implies the other during training. The cross-entropy scores beats independently, so it says nothing about whether the errors that remain are scattered or form a consistent shift. The periodicity term is blind to which beat is the 1. Together they are minimised only by a count that starts on the right beat *and* keeps counting.

### 4.1 Anchor: the existing position cross-entropy

$\mathcal L_{\text{pos}}$ is the existing position term: a soft-target cross-entropy between the softmax over $y_{t,\cdot}$ and the annotated position, restricted to the frames at and around annotated beats (the beat mask) and averaged per clip. It is unchanged by this plan. It treats every wrong position as equally wrong; whether a cost that grows with the distance between labels would be better is a separate question (§9).

### 4.2 Periodicity term

**The idea in three steps.**

1. **Compare each beat's label with the annotation.** The difference, counted around the bar, is the beat's *offset*: how far the head's count is shifted at that beat. Calling a 1 "1" is offset 0; calling a 1 "3", a 2 "4", and so on, is offset 2.
2. **A head that counts keeps the same offset at every beat.** The offset is 0 if the count starts on the right beat and something else if the count is consistently shifted, but either way it does not change within the clip. A slip, a restart or a guess changes it.
3. **Score the clip by how often its beats disagree.** Pick two beats of the clip at random (the same beat may be picked twice): the loss is the probability that their offsets differ. It is 0 for a head that keeps one offset, whatever that offset is, and grows the more the offset changes from beat to beat.

The term never asks *which* offset; that is the cross-entropy's job (§4.1). It only asks that there be one.

**Worked example.** On a clip annotated 1 2 3 4 1 2 3 4, a head that predicts 1 2 3 4 3 4 1 2 has offsets 0 0 0 0 2 2 2 2. Two random beats have the same offset half the time, so the loss is 0.5. The cross-entropy sees four wrong beats, but not that they are all wrong *in the same way*; the periodicity term sees exactly that, a count that slipped once.

**How it ranks predictions** on an annotated 4/4 clip whose positions run 1, 2, 3, 4, 1, …, with each beat's probability entirely on one position:

| Positions the head predicts | Offsets | $\mathcal L_{\text{pos}}$ (cross-entropy) | $\mathcal L_{\text{period}}$ |
|---|---|---|---|
| 1 2 3 4 1 2 3 4 (correct) | 0 0 0 0 0 0 0 0 | low | 0 |
| 3 4 1 2 3 4 1 2 (counting, but shifted) | 2 2 2 2 2 2 2 2 | high | 0 |
| 1 2 3 4 3 4 1 2 (count slips by two mid-clip) | 0 0 0 0 2 2 2 2 | high on half the beats | 0.5 |
| 1 2 3 4 2 3 4 1 (count slips by one mid-clip) | 0 0 0 0 1 1 1 1 | high on half the beats | 0.5 |
| 1 2 1 2 1 2 1 2 (count restarts every two beats) | 0 0 2 2 0 0 2 2 | high on half the beats | 0.5 |
| uniform (no opinion) | — | medium | 0.75 |

The two terms disagree on the shifted and the slipping predictions, which is the point: each catches the failure the other is blind to. In terms of evaluation, the anchor targets whole-track offsets and the periodicity term targets mid-track slips.

**Computing it from probabilities.** The model gives each beat a probability for every position, not a single answer, so the loss works with those probabilities, in three steps.

**Step 1: reorder each beat's probabilities so that each slot means the same kind of mistake at every beat.** Probabilities over positions cannot be compared across beats: position 3 is a correct answer at a beat whose true position is 3, but a +2 mistake at a beat whose true position is 1. Reordering them to start from the beat's reference position $n_k$ turns them into probabilities over offsets, where slot 0 is always "ok", slot 1 always "+1", and so on:

$$
q_{k,j} = \mathrm{softmax}_j\big( y_{f_k, j} \big), \qquad
\pi_k(o) = q_{k,\thinspace 1 + (n_k - 1 + o) \bmod G}, \qquad o = 0, \dots, G - 1
$$

| Beat | True position $n_k$ | Probabilities over positions 1, 2, 3, 4 | Reordered, over ok, +1, +2, +3 |
|---|---|---|---|
| first | 1 | 0.7, 0.1, 0.1, 0.1 | 0.7, 0.1, 0.1, 0.1 |
| second | 2 | 0.1, 0.7, 0.1, 0.1 | 0.7, 0.1, 0.1, 0.1 |

The two beats put their confidence on different positions, but on the same kind of mistake. Only the reordered numbers show it.

**Step 2: average the reordered probabilities over the clip.** Write $\mathcal{K}_b$ for the annotated beats of clip $b$, and keep the clips with at least two bars' worth of them, $\mathcal{T} = \lbrace b : |\mathcal{K}_b| \ge 2G \rbrace$. The average says how the clip's confidence is spread over the $G$ kinds of mistake:

$$
\bar\pi_b(o) = \frac{1}{|\mathcal{K}_b|} \sum_{k \in \mathcal{K}_b} \pi_k(o)
$$

**Step 3: measure how concentrated that average is, with its sum of squares.** The sum is 1 if all the confidence sits on one kind of mistake and $1/G$ if it is spread evenly. The loss is 1 minus it, averaged over clips:

$$
\boxed{\ \mathcal{L}_{\text{period}} = \frac{1}{|\mathcal{T}|} \sum_{b \in \mathcal{T}} \Big( 1 - \sum_{o=0}^{G-1} \bar\pi_b(o)^2 \Big)\ }
$$

If the beats agree on the kind of mistake, the average stays concentrated and the loss is low; if they disagree, it spreads out and the loss is high:

| Clip | Average over ok, +1, +2, +3 | Sum of squares | Loss |
|---|---|---|---|
| the two beats above, agreeing on "ok" | 0.7, 0.1, 0.1, 0.1 | 0.52 | 0.48 |
| one beat 70% "ok", one 70% "+2" | 0.4, 0.1, 0.4, 0.1 | 0.34 | 0.66 |
| the worked example: half the beats sure of "ok", half of "+2" | 0.5, 0, 0.5, 0 | 0.5 | 0.5 |

The first row costs 0.48 although the beats agree, because each is only 70% sure: the loss rewards confidence as well as agreement (§9). The sum of squares is also the probability that two beats drawn at random from the clip, each drawing its kind of mistake from its own probabilities, land on the same one: step 3 of the idea above. The same beat may be drawn twice, and that is where the confidence reward comes from: a beat 70% sure of "ok" agrees with itself only $0.7^2 + 3 \times 0.1^2 = 0.52$ of the time. The loss ranges from 0 to $1 - 1/G$ (0.75 for $G = 4$).

**Design choices.**

- **Averaged per clip, not over the batch.** An offset is the head's shift *within one clip*; offsets of different clips are unrelated, so pooling them would compare nothing. Per-clip averaging also matches how the existing position loss is averaged.
- **It rewards confidence as well as agreement.** $\sum_o \bar\pi_b(o)^2$ is at most the beats' average $\sum_o \pi_k(o)^2$, so the term only reaches 0 if every beat is confident *and* the beats agree. The head is underconfident by design, and sharpening it is not the lever, so a pure-agreement variant may be the better default (§9).
- **Every slip costs the same.** Offsets are categories, not distances: a count that slips by one position costs as much as one that slips by two. Periodicity is about whether the count slipped, not how far.

**Diagnostics (no gradient).** The clip's most likely offset, $\hat o_b = \arg\max_o \bar\pi_b(o)$, is how far the head's count is shifted from the annotation, and $\bar\pi_b(0)$ is the clip's average probability of the correct label, the quantity the anchor raises. Comparing $\sum_o \bar\pi_b(o)^2$ with the beats' average $\sum_o \pi_k(o)^2$ separates disagreement between beats from indecision at each beat.

---

## 5. Total objective

The new terms are added to the existing objective rather than replacing it. The existing loss stays as it is: its beat term keeps deciding whether a beat is present, and its position cross-entropy which label each beat carries. The two new terms come in with their own weights, which ramp up from zero over the early epochs.

$$
\mathcal{L} = \mathcal{L}_{\text{bce}} + \mathcal{L}_{\text{pos}} + \lambda_b(e)\thinspace \mathcal{L}_{\text{beat}} + \lambda_p(e)\thinspace \mathcal{L}_{\text{period}}
$$

$\mathcal L_{\text{bce}} + \mathcal L_{\text{pos}}$ is the existing loss as it stands: the beat head's weighted binary cross-entropy, plain or shift-tolerant, plus the position cross-entropy of §4.1.

$$
\lambda(e) = \lambda \cdot \mathrm{clip}\negthinspace\left( \frac{e - e_0}{e_1 - e_0},\ 0,\ 1 \right)
$$

The ramp over the epoch index $e$ switches the new terms on only after peaks have formed. On a flat, untrained output the soft-argmax is noise, and the position head has no count yet to make consistent.

---

## 6. Gradients

This section checks that each term pushes only where it should. The beat term moves logit mass around inside each beat's window but cannot add or remove it, so missing beats stay the BCE's job. The periodicity term touches only the position logits, and only at the annotated beats' frames. The derivation also shows why §1's adaptive temperature matters for the beat term: each frame's gradient is proportional to its softmax weight, so a collapsed softmax would leave all but the peak frame with none.

The temperatures $\tau$ and the gates $g_k$ are detached, so they contribute no gradient.

### 6.1 Beat term

The soft-argmax derivative has the form of a covariance:

$$
\frac{\partial \hat b_k}{\partial z_t} = \frac{1}{\tau_k}\thinspace p_{k,t}\thinspace \big( t - \hat b_k \big) \quad (t \in \mathcal{W}_k), \qquad
\sum_{t \in \mathcal{W}_k} \frac{\partial \hat b_k}{\partial z_t} = 0
$$

The derivatives sum to zero over the window. So the beat term only **redistributes** logit mass inside a window and cannot create or remove a peak in aggregate.

Let $s_k = \mathrm{sign}(\rho_k)\thinspace \mathbb{1}\big[ |\rho_k| > \varepsilon \big]$ and $h'(x) = \min(x, 1)$. Since the windows are disjoint, frame $t \in \mathcal{W}_k$ receives gradient only through $\hat b_k$, which enters the two neighbouring pairs:

$$
\frac{\partial \mathcal{L}_{\text{beat}}}{\partial \hat b_k} = \frac{1}{\gamma\thinspace |\mathcal{P}|} \left[ \frac{g_{k-1}\thinspace h'(x_{k-1})\thinspace s_{k-1}}{\hat b_k - \hat b_{k-1}} - \frac{g_k\thinspace h'(x_k)\thinspace s_k}{\hat b_{k+1} - \hat b_k} \right]
$$

A term is dropped when its pair is not in $\mathcal{P}$. Then

$$
\frac{\partial \mathcal{L}_{\text{beat}}}{\partial z_t} = \frac{\partial \mathcal{L}_{\text{beat}}}{\partial \hat b_k}\ \frac{\partial \hat b_k}{\partial z_t}
$$

Frames on the side of $\hat b_k$ that reduces $|\rho|$ are pushed up, and frames on the other side are pushed down. This is how an inserted peak inside a window gets suppressed.

**Temperature leverage.** The gradient scales as $p_{k,t}/\tau_k$. With a fixed $\tau$ and $|z| \approx 10$, off-peak frames have $p \lesssim e^{-8}$. With the adaptive $\tau_k$, the relative logits of background frames are set by $\kappa_b$, not by the logit scale.

### 6.2 Periodicity term

For beat $k$ of clip $b$, write $o_{k,j} = (j - n_k) \bmod G$ for the offset that position $j$ stands for at that beat, and $A_k = \sum_o \bar\pi_b(o)\thinspace \pi_k(o)$ for how much beat $k$ agrees with the clip on average. Differentiating through the position softmax gives, at the beat's frame $f_k$,

$$
\frac{\partial \mathcal{L}_{\text{period}}}{\partial y_{f_k, j}} = -\frac{2}{|\mathcal{T}|\thinspace |\mathcal{K}_b|}\thinspace q_{k,j}\thinspace \Big( \bar\pi_b\big( o_{k,j} \big) - A_k \Big)
$$

and zero at every other frame. It has the covariance form of §6.1 again. Each beat is pulled towards the offsets that are popular in its clip: a position is raised if the offset it stands for is more common in the clip than the beat's current agreement, and lowered otherwise. The majority of beats sets what is popular. When most beats are right, this pulls the stray ones into line; when most are wrong, it pulls the right ones towards the wrong count, and only the cross-entropy pushes back (§9). The beat head gets no gradient from this term.

---

## 7. Properties

This section collects what the terms ignore on purpose, what they deliberately do not ignore, and how their thresholds line up with the evaluation's.

All of these follow directly from the definitions above.

- **Latency invariance (beat).** $\hat b_k \to \hat b_k + c$ leaves $\rho_k$ unchanged, which absorbs constant annotator or tapping latency.
- **Tempo-scale invariance.** $(\hat b_k, b_k) \to (s \hat b_k, s b_k)$ leaves $\rho_k$ unchanged, which covers tempo augmentation and comparisons across pieces.
- **Octave symmetry.** Doubling or halving the predicted interval gives $\rho = \mp \log 2$, so both are penalised equally.
- **Tempo curves.** The term is anchored to the ground-truth $\Delta_k$, so a ritardando that is in the annotation costs nothing.
- **Link to the metric's tolerance.** The continuity metrics' condition $|\hat\Delta/\Delta - 1| < 0.175$ becomes, in log space, $\rho \in (\log 0.825,\ \log 1.175) = (-0.192,\ 0.161)$; using the symmetric bound $\gamma = 0.161$ is slightly stricter on the short side. In terms of the loss:
  - the penalty starts at $|\rho| = \varepsilon$, about 5%;
  - the metric boundary $|\rho| = \gamma$ sits at $x = 1 - \varepsilon/\gamma \approx 0.69$;
  - the Huber penalty turns linear at $|\rho| = \varepsilon + \gamma \approx 0.211$, about 23.5%.
- **Periodicity is deliberately offset-invariant.** Shifting the head's count by the same number of positions on every beat relabels the offsets of a clip and leaves $\sum_o \bar\pi_b(o)^2$ unchanged. Fixing the offset is the cross-entropy's job; the periodicity term only asks that the offset stay the same.
- **Bounds.** $0 \le \mathcal L_{\text{period}} \le 1 - 1/G$, and $\mathcal L_{\text{beat}} \ge 0$ with bounded gradient.

---

## 8. Defaults

Every parameter above with its default. Frame counts assume 43.07 fps.

| Symbol | Default | Meaning |
|---|---|---|
| $H$ | 34 frames (0.79 s) | half-window cap; a full Voronoi window needs $\Delta \le 2H$, i.e. tempo $\ge 38$ BPM |
| $\kappa_b$ | 8 | softmax range inside a beat window |
| $\tau_{\min}$ | 0.5 | temperature floor |
| $\varepsilon$ | 0.05 | log-interval deadzone, about annotation jitter |
| $\gamma$ | $\log 1.175$ | residual unit |
| $g_k$ | on | presence gate (§3) |
| $G$ | 4 | positions per bar, shared with the position head |
| $\min_b \lvert \mathcal{K}_b \rvert$ | $2G = 8$ | annotated beats a clip needs to enter $\mathcal L_{\text{period}}$ |
| $\lambda_b,\ \lambda_p$ | $\approx 1$ (untuned) | loss weights |
| $e_0,\ e_1$ | 5, 15 | ramp start / end epoch, shared by both terms (§9) |

---

## 9. Open questions

The periodicity term (§4.2) is a proposal; these need settling before or while it is implemented.

| # | Question | Term | Settled by | Default until then | Alternative |
|---|---|---|---|---|---|
| 1 | How much of the error is slips? | periodicity | measuring current checkpoints, before building | — | — |
| 2 | Push for confidence? | periodicity | comparing the two variants in training | agreement only: $1 - \sum_o \bar\pi_b(o)^2 \big/ \overline{\textstyle\sum_o \pi_k(o)^2}$, the bar a mean over the clip's beats | §4.2 as written, $1 - \sum_o \bar\pi_b(o)^2$, or a hinge on it, $[\mathcal L_{\text{period}} - \delta]_+$ |
| 3 | Where to read the head | periodicity | experiment | annotated frame $f_k$ | average over the tolerance window: $\bar q_{k,j} = \frac{1}{2r+1} \sum_{\lvert t - f_k \rvert \le r} q_{t,j}$, with $q_{t,j}$ the position softmax at frame $t$ |
| 4 | Weight and schedule | both | sweep | $\lambda \approx 1$, ramp over epochs 5–15 | ramp periodicity in later than the beat term: $e_0^p > e_0^b$ |
| 5 | Distance-aware anchor | anchor | separate experiment | plain cross-entropy | expected distance around the bar: $\sum_j q_{k,j}\thinspace d(j, n_k)$ with $d(j, n) = \min(\lvert j - n \rvert,\ G - \lvert j - n \rvert)$ |
| 6 | Deadzone value | beat | measuring annotation noise | $\varepsilon = 0.05$, relative | constant in time: $\varepsilon_k = d / \Delta_k$, with $d$ a fixed number of frames |
| 7 | Where the penalty bends | beat | experiment | divide by $\gamma$ | $x_k = [\thinspace\lvert \rho_k \rvert - \varepsilon\thinspace]_+ / (\gamma - \varepsilon)$ |
| 8 | Gate on correct beats | beat | measuring gate values on v7 | soft gate $g_k = \mathrm{sg}[\sigma(m_k)\thinspace\sigma(m_{k+1})]$ | hard gate: $g_k = \mathbb{1}[\sigma(m_k) > \theta]\ \mathbb{1}[\sigma(m_{k+1}) > \theta]$ |
| 9 | Adaptive temperature? | beat | experiment; matters only with shift tolerance | adaptive $\tau$, $\kappa_b = 8$ (§1) | fixed temperature: $p_{k,t} \propto \exp(z_t / \tau_{\min})$ |
| 10 | Cap the penalty? | beat | experiment | Huber $h$, slope capped at 1 (§3) | quadratic, $h(x) = \tfrac{1}{2} x^2$, or dropping gross errors, $\tfrac{1}{2} x^2\thinspace \mathbb{1}[x \le x_{\max}]$ |
| 11 | Normalise by pairs or by gate? | beat | decided with #8 | $1 / \lvert \mathcal P \rvert$ (§3) | $\mathcal L_{\text{beat}} = \sum_{k \in \mathcal P} g_k h(x_k) \big/ \sum_{k \in \mathcal P} g_k$ |
| 12 | Pool the beat term per clip? | beat | experiment | pooled over the batch (§0); fast clips have more pairs and weigh more | per clip, like periodicity: $\frac{1}{B} \sum_b \frac{1}{\lvert \mathcal P_b \rvert} \sum_{k \in \mathcal P_b} g_k h(x_k)$ |
| 13 | Minimum beats per clip | periodicity | experiment | $\lvert \mathcal K_b \rvert \ge 2G$ (§4.2) | weight clips by beat count: $\sum_b \lvert \mathcal K_b \rvert \big( 1 - \sum_o \bar\pi_b(o)^2 \big) \big/ \sum_b \lvert \mathcal K_b \rvert$ |

Rows 9–13 are design choices explained where they are defined, so they have no entry below.

1. **How much of the error is slips?** The term can only fix beats whose label disagrees with the rest of their clip. Before building it, measure on current checkpoints, from the head's own answers at the annotated beats, how much of the per-beat error is mid-clip slips (beats off the clip's most common offset) and how much is a whole-clip shift (the most common offset itself being wrong). The periodicity term cannot fix the second kind, and on those clips (§6.2) it pulls against the cross-entropy.
2. **Should the term also push for confidence?** As defined, the loss is 0 only when the beats agree *and* each beat is fully sure of its answer. Two beats that agree on "ok" but are each 70% sure still cost 0.48 (§4.2). So besides making the beats consistent, the term pushes every beat towards one-hot. The v7 measurement suggests that push is not wanted: the head is underconfident, but what holds it back is per-beat accuracy, not sharpness. Two ways to remove it:
   - **Measure agreement only.** Divide $\sum_o \bar\pi_b(o)^2$ by the average over the clip's beats of $\sum_o \pi_k(o)^2$, and take 1 minus the ratio as the loss. The ratio is at most 1, and exactly 1 when every beat has the same probabilities, however unsure. The two 70%-sure beats above then cost 0 if they agree and 0.35 if they disagree.
   - **Stop pushing once the loss is small**, with a hinge $[\mathcal L_{\text{period}} - \delta]_+$. This leaves some indecision unpunished, but still mixes agreement and confidence.

   Measuring agreement only is the likely default.
3. **Where to read the head.** §2.2 reads it at the annotated frame. With shift tolerance on, the cross-entropy is supervised over a window $\pm r$ frames wide, and at inference the head is read at the detected beat, which may sit up to $r$ frames away. Averaging the position probabilities over that window would match inference better, at the cost of mixing in frames where the head is near-uniform.
4. **Weight and schedule.** Should $\lambda_p$ ramp in later than the cross-entropy has settled, so that the consensus it pulls towards is mostly right? Both weights are untuned.
5. **A distance-aware anchor.** The cross-entropy treats a beat labelled 3 instead of 1 as wrong in the same way as one labelled 2. A cost that grows with how far the predicted label is from the annotated one around the bar (2 and 4 are one position from 1, 3 is two) would weight errors by that distance. This is independent of the periodicity term and could be tried on its own.
6. **Measure the deadzone instead of guessing it.** $\varepsilon = 0.05$ stands for annotation noise, which our own annotations can measure: on steady stretches of a track, how much do consecutive annotated gaps differ, $|\log(\Delta_{k+1}/\Delta_k)|$? That distribution, per corpus, would give $\varepsilon$ a value, and would show whether the noise is closer to constant in milliseconds (then the deadzone should be in milliseconds, not relative) or in percent.
7. **Where the penalty changes shape.** Dividing by $\gamma$ puts the switch from quadratic to linear one full tolerance past the deadzone, at a 23.5% error, and the metric's limit at $x_k = 0.69$. Dividing by $\gamma - \varepsilon$ would put the switch exactly at the metric's limit, 17.5%, so errors the metric still accepts are penalised quadratically and errors it rejects linearly.
8. **Does the gate silence correct beats?** The gate (§3) weights each pair of beats by how confident the beat head is at both: two sure peaks count fully, two hesitant ones count little. That assumes a correctly placed beat comes with a confident peak, and on v7 it often does not. At annotated beats, the beat head's probability averaged 0.44, so a typical correct pair would get a gate of about $0.44^2 \approx 0.2$ (somewhat more in practice, since the gate reads the highest value in each window). The term would then be several times weaker than $\lambda_b$ suggests, and it would train mostly on tracks where the beat head is already confident, the ones that need it least. Measure the gate's values on correctly placed pairs before choosing $\lambda_b$. If they are low, the alternative is a hard threshold: a pair counts fully when both peaks clear a fixed probability, and not at all otherwise.
