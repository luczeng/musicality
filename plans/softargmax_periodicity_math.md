
<!-- Math has to render on GitHub and in markdown-it/KaTeX previews (nvim), so it uses only $...$ and $$...$$.
GitHub strips the backslash from \, \; \! \{ \} \\ inside math: write \thinspace, \negthinspace, "\ " (a space),
\lbrace, \rbrace, \cr. In inline math, an underscore right after } can start GitHub italics: write z^b_t, not z^{b}_t. -->

## Overview

### High level

Beat This!'s loss judges each moment of a track on its own: should there be a beat here or not? It never compares two predicted beats with each other, so nothing in it asks that consecutive beats be as far apart as the annotated ones, or that the downbeat fall on the first beat of the bar. The evaluation asks both. The continuity metrics only accept a beat whose distance to the previous one is within 17.5% of the annotated distance, and a downbeat placed on the wrong beat puts every beat of that bar in the wrong position. The Beat This! authors end their paper by asking for losses that enforce periodicity during training.

This document specifies three such losses, added to the existing one rather than replacing it. The first compares the gap between neighbouring predicted beats with the annotated gap, and penalises differences larger than annotation jitter. The other two are about the downbeat, and they split it into two questions. Which beat is the 1? The anchor term looks at each annotated bar and penalises downbeat predictions that are not concentrated on its first beat. Does the 1 come back every bar? The periodicity term asks that the model place the downbeat at the same position in every bar of a track, whichever position that is. The anchor alone does not ask this: it scores each bar on its own, so it even rates a prediction that flips between beats 1 and 3 as better than one that sits consistently on beat 3. The periodicity term needs no downbeat annotation, only the beats and the bar length. The existing loss keeps deciding whether a beat is there; the new terms decide where beats sit relative to each other. All three act only during training, and they switch on gradually, once the model has learned to produce clear peaks. Several questions about the periodicity term are still open (§9).

### Technical

This is the companion to `softargmax_periodicity.py`. It defines three auxiliary terms that are added to the Beat This! shift-tolerant BCE (`plans/09` §2.1):

- $\mathcal{L}_{\text{beat}}$ works at **beat level, on the time axis**. A soft-argmax inside per-beat windows gives sub-frame beat centres. Their inter-beat intervals are compared with the ground-truth intervals in log space.
- $\mathcal{L}_{\text{anchor}}$ works at **bar level, on the beat-index axis**. It takes a circular soft-argmax of the downbeat logits over the beats of each bar, and pulls each bar's circular mean towards beat 1.
- $\mathcal{L}_{\text{period}}$ works at **track level, across bars**. It averages the same per-bar circular means over all bars of a track and asks the average to be long, i.e. every bar to point the same way. It is rotation-invariant: it never says which beat is the 1.

The BCE decides *whether* a peak exists. These three terms decide *where peaks sit relative to each other*.

The setup is Beat This!'s: 50 fps, 30 s excerpts, and separate beat and downbeat heads whose logits are summed into the beat prediction (the Sum Head, `plans/09` §2.6).

The sections build on each other. §0 fixes notation and §1 defines the soft-argmax all three terms rely on. §2 builds, from the annotations alone, the sets that soft-argmax runs over: a window of frames per beat, and complete bars. §3 defines the beat term and §4 the two downbeat terms, and §5 adds them to the BCE. §6 derives their gradients, which show what each term can and cannot do to the logits. §7 collects their invariances and their link to the evaluation's tolerances, §8 lists the defaults, and §9 the open questions.

---

## 0. Conventions

Everything below shares this notation. Times are counted in frames, but annotations keep their sub-frame precision, because the beat term measures positions more finely than one frame. The constant $\gamma$ is the evaluation's own tolerance on beat spacing: §3 uses it as the unit of error, so that the loss and the metric agree on what counts as close.

- The frame rate is 50 fps and all times are in frames. Ground-truth positions are real-valued (sub-frame).
- $\mathrm{sg}[\cdot]$ is stop-gradient, $[x]_+ = \max(x, 0)$, $\mathbb{1}[\cdot]$ is the indicator and $\sigma$ is the logistic sigmoid.
- $\gamma := \log 1.175 \approx 0.1613$ is the upper bound of mir_eval's continuity tolerance in log space.
- The sums over beat pairs and over bars run over the **whole batch**. The losses are batch-pooled means, not means of per-item means.

| Symbol | Meaning | Code | Shape |
|---|---|---|---|
| $z^{b}_t,\ z^{d}_t$ | beat / downbeat head logits | — | $(B, T)$ |
| $z_t = z^{b}_t + z^{d}_t$ | Sum Head beat logits | `beat_logits` | $(B, T)$ |
| $b_k,\ k = 1..K$ | ground-truth beat (sub-frame) | `t_gt` | $(B, K)$ |
| $\Delta_k = b_{k+1} - b_k$ | ground-truth inter-beat interval | — | $(B, K-1)$ |
| $n_k \in \lbrace 1, \dots, N \rbrace$ | ground-truth beat-in-bar number | `beat_numbers` | $(K,)$ per item |
| $\mathcal{W}_k$ | frame window of beat $k$ | `win_idx`, `win_mask` | $(B, K, W)$ |
| $\hat b_k$ | soft-argmax beat centre | `t_hat` | $(B, K)$ |
| $\mathcal{B}_j$ | beat indices of bar $j$ | `bar_beat`, `bar_mask` | $(B, J, N_{\max})$ |
| $\theta_k$ | angle of beat $k$ in its bar | `bar_theta` | $(B, J, N_{\max})$ |

For a 30 s excerpt: $T = 1500$, $W = 2H + 1 = 81$, $K \approx 30\text{–}120$, $J \approx 8\text{–}30$.

---

## 1. Shared operator: masked softmax with adaptive temperature

Both terms need to know where a peak is: where the beat peak sits inside its window (§3), and which beat of a bar carries the downbeat peak (§4). A hard argmax answers that but has no gradient. A soft-argmax does: turn the logits into a softmax distribution and take the expected position, which moves smoothly as the logits change. This section defines the softmax both terms use. The index set $\mathcal{I}$ is a window's frames in §3 and a bar's beats in §4 ("masked" because, in code, sets of different sizes are padded to one shape). Its one non-standard part is the temperature, which follows the scale of the logits so that the softmax does not collapse to one-hot on an overconfident network.

Take logits $a_i$ over a finite index set $\mathcal{I}$ and a range parameter $\kappa$:

$$
m = \max_{i \in \mathcal{I}} a_i, \qquad
\bar a = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} a_i, \qquad
\tau = \mathrm{sg}\negthinspace\left[ \max\negthinspace\left( \tau_{\min},\ \frac{m - \bar a}{\kappa} \right) \right]
$$

$$
\mathrm{softmax}_{\kappa}(a)_i = \frac{\exp\big( (a_i - m)/\tau \big)}{\sum_{i' \in \mathcal{I}} \exp\big( (a_{i'} - m)/\tau \big)}
$$

**Why adaptive.** Beat This!-style training produces overconfident logits ($|a| \approx 10$). With a fixed $\tau$, the softmax becomes one-hot and the gradients, which are proportional to $p_i$ (see §6), vanish. With the adaptive $\tau$, whenever the floor is inactive the scaled mean logit sits exactly $\kappa$ below the scaled maximum, $(m - \bar a)/\tau = \kappa$, whatever the logit scale. The temperature is detached, so it is never a path for the network to lower the loss. In the code, `kappa=None` gives a fixed $\tau = \tau_{\min}$.

---

## 2. Target construction

Before either term can be computed, the annotations have to be turned into the sets the soft-argmax runs over. The beat term needs, for every annotated beat, a window of frames expected to hold that beat's peak and no other (§2.1). The downbeat terms need whole bars: which beats form each bar, and where each beat sits in it as an angle on a circle (§2.2–2.3). All of this depends only on the annotations, not on the network's output.

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

A beat is **valid** if its window was not cut by the excerpt edge and has at least 3 frames. A cut window would bias its soft-argmax towards the inside of the excerpt, and fewer than 3 frames are too few to locate a peak:

$$
v_k = \mathbb{1}[\ell_k \ge 0]\ \mathbb{1}[r_k \le T]\ \mathbb{1}\big[ |\mathcal{W}_k| \ge 3 \big]
$$

### 2.2 Bar lengths (on the full song, before cropping)

Placing a beat on the circle needs the length of its bar: beat 2 is a quarter of the way round in 4/4 but a third in 3/4. Lengths are computed on the whole song, before cropping to an excerpt, because the excerpt usually cuts its first and last bars and their length cannot be read from what remains.

Let $s_1 < \dots < s_M$ be the downbeat indices ($n_{s_m} = 1$). Bar $m$ covers beats $[s_m, s_{m+1})$ and has length

$$
N_m = \max\negthinspace\Big( s_{m+1} - s_m,\ \max_{s_m \le k < s_{m+1}} n_k \Big)
$$

Taking the larger of the beat count and the highest beat number means a bar whose annotation skips a beat still gets its full length. Pickup beats ($k < s_1$) take $N_1$. The final bar takes $\max(N_M, N_{M-1})$, because songs and annotations often end mid-bar. In the code this is `bar_lengths_from_numbers`.

### 2.3 Complete bars inside the excerpt

The downbeat terms work one bar at a time, so they keep only the bars the excerpt contains whole. Each beat of a kept bar becomes an angle: the downbeat at 0, the other beats evenly spaced around the circle. A circle rather than a line from 1 to $N$ encodes that the bar repeats: the last beat is as close to the next downbeat as beat 2 is to this one, and the middle of the bar is the furthest from beat 1.

Bar $j$ is kept only if it is complete and consecutively numbered:

$$
\mathcal{B}_j = \lbrace k_j, \dots, k_j + N_j - 1 \rbrace, \qquad n_{k_j + i} = i + 1 \ \ (i = 0, \dots, N_j - 1), \qquad N_j \ge 2
$$

$$
\theta_k = \frac{2\pi\thinspace (n_k - 1)}{N_j}, \qquad k \in \mathcal{B}_j
$$

Bars cut by the excerpt edge are dropped, because a partial bar would bias the circular mean. One-beat bars are dropped too ($N_j \ge 2$): a single beat has no phase to get wrong.

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

Residuals below $\varepsilon$, about 5%, are within annotation jitter and cost nothing. What exceeds it is divided by $\gamma$, so $x_k = 1$ means one full metric tolerance beyond the deadzone.

$$
x_k = \frac{\big[\thinspace |\rho_k| - \varepsilon \thinspace\big]_+}{\gamma}
$$

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

- $h$ is the Huber function with threshold 1. It is quadratic for small errors, and for gross errors (octave jumps, missed beats) it grows linearly, so the gradient is bounded: $|h'| \le 1$.
- The gate removes pairs where either window has no confident peak, so soft-argmax does not chase noise in empty windows. Missing peaks are the BCE's job.
- $\mathcal{P}$ keeps the pairs whose two windows are both valid (§2.1).
- The loss is normalised by $|\mathcal{P}|$ rather than by $\sum g_k$, so gated pairs reduce the loss instead of reweighting the remaining pairs.

---

## 4. Downbeat terms

Timing is the beat term's job, so both downbeat terms read a single downbeat logit at each annotated beat and only ask *which* beat carries the downbeat. A soft-argmax over a bar's beats, taken on the circle from §2.3, turns that into one arrow per bar: its direction is where the model puts the downbeat, its length how decided it is. The two terms then ask different things of these arrows:

- the **anchor term** (§4.1) pulls every arrow towards beat 1. It is supervised, one bar at a time;
- the **periodicity term** (§4.2) asks all arrows of a track to point the same way, wherever that is. It links the bars to each other, which is what "after 1, 2, 3, 4 comes 1 again" means.

Neither implies the other during training. The anchor scores bars independently, so a flipping prediction can score better than a consistent but shifted one; the periodicity term is blind to which beat is the 1. Together they are minimised only by a downbeat that is on beat 1 *and* stays there.

### 4.1 Anchor term

The anchor term checks where the network puts the downbeat within each bar. The loss is the distance of the bar's arrow from beat 1, with a small hinge so that a nearly right bar costs nothing.

**Downbeat logits sampled at the ground-truth beats** (max-pooling over $\pm 3$ frames, as in the shift-tolerant BCE, with $-\infty$ padding):

$$
\tilde z^{d}_t = \max_{|s| \le 3} z^{d}_{t+s}, \qquad
u_k = \tilde z^{d}_{f_k}, \qquad
f_k = \mathrm{clip}\big( \mathrm{round}(b_k),\ 0,\ T - 1 \big)
$$

Pooling means a downbeat peak up to 3 frames from the annotated beat still counts, as it does for the BCE.

**Circular soft-argmax over each bar**

The softmax says how strongly each beat of the bar claims the downbeat. $(C_j, S_j)$ is the resulting mean of the beats' angles, a point inside the unit circle. Beat 1 sits at $(1, 0)$, so $C_j$ approaches 1 as the mass concentrates on beat 1, and the loss is a hinge on how far it falls short.

$$
q_{j,\cdot} = \mathrm{softmax}_{\kappa_\phi}\big( u_k \big)_{k \in \mathcal{B}_j}, \qquad
C_j = \sum_{k \in \mathcal{B}_j} q_{j,k} \cos \theta_k, \qquad
S_j = \sum_{k \in \mathcal{B}_j} q_{j,k} \sin \theta_k
$$

$$
\boxed{\ \mathcal{L}_{\text{anchor}} = \frac{1}{J} \sum_{j=1}^{J} \big[\thinspace 1 - C_j - \delta \thinspace\big]_+\ }
$$

Here $J$ is the number of complete bars in the batch. Datasets without downbeat annotations give $J = 0$, and then $\mathcal{L}_{\text{anchor}} = 0$.

**Interpretation.** $1 - C_j$ is the expected circular distance from beat 1:

$$
1 - C_j = \sum_{k \in \mathcal{B}_j} q_{j,k}\thinspace \big( 1 - \cos \theta_k \big) \in [0, 2]
$$

| Downbeat mass on | 4/4: $1 - C$ | 3/4: $1 - C$ | $R$ |
|---|---|---|---|
| beat 1 | 0 | 0 | 1 |
| beat 2 | 1 | 1.5 | 1 |
| beat 3 | 2 | 1.5 | 1 |
| beat 4 | 1 | — | 1 |
| beats 1 and 3 equally | 1 | — | 0 |
| uniform (no downbeat peak) | 1 | 1 | 0 |

For small errors $1 - \cos\varphi \approx \varphi^2/2$. The hinge $\delta = 0.05$ corresponds to $|\varphi| \le \arccos(0.95) \approx 18.2°$, which is about 0.2 beat in 4/4.

**Diagnostics (no gradient)**

The loss reads only $C_j$, so it scores a bar confidently on beat 2 and a bar with no downbeat peak the same (both 1 in the table). Two quantities that are not trained on tell them apart:

$$
\hat\varphi_j = \mathrm{atan2}(S_j, C_j), \qquad
R_j = \sqrt{C_j^2 + S_j^2}, \qquad
\text{err}_j = \frac{|\hat\varphi_j|}{2\pi} \ \ \text{(fraction of a bar)}
$$

$R_j \approx 1$ means the mass is concentrated (confidently placed), and $R_j \approx 0$ means it is split or absent. Together, $(\hat\varphi_j, R_j)$ separate "confidently on the wrong beat" from "undecided".

### 4.2 Periodicity term

The anchor term looks at one bar at a time, so it never asks whether the downbeat *recurs*. The periodicity term does. It reuses the per-bar arrows of §4.1 and averages them over all bars of a track. If every bar puts its downbeat at the same position, the arrows line up and their average is as long as they are. If the downbeat wanders from bar to bar, the arrows point different ways and partly cancel. The loss is how far that average falls short of length 1.

**Chunks.** The term needs groups of beats that should each contain one downbeat, and an angle for each beat within its group:

- On tracks **with downbeat annotations**, the chunks are the complete bars $\mathcal{B}_j$ of §2.3, with the angles $\theta_k$ from the annotated beat numbers. A meter change is harmless: every bar has its downbeat at angle 0, whatever its length.
- On tracks **with beats but no downbeats**, the chunks are consecutive groups of $N$ beats starting at the first beat in the excerpt, and beat $i$ of a group gets $\theta = 2\pi i / N$. Because the term is rotation-invariant, it does not matter that the groups start on an arbitrary beat: a periodic prediction lines up its arrows whatever the offset. This needs a bar length $N$, which these tracks do not annotate (§9).

Each chunk $j$ gets its arrow $(C_j, S_j)$ exactly as in §4.1, from its own softmax $q_{j,\cdot}$. Write $\mathcal{J}_b$ for the chunks of track $b$, and keep the tracks with at least two of them, $\mathcal{T} = \lbrace b : |\mathcal{J}_b| \ge 2 \rbrace$:

$$
\bar C_b = \frac{1}{|\mathcal{J}_b|} \sum_{j \in \mathcal{J}_b} C_j, \qquad
\bar S_b = \frac{1}{|\mathcal{J}_b|} \sum_{j \in \mathcal{J}_b} S_j, \qquad
\bar R_b = \sqrt{\bar C_b^2 + \bar S_b^2}
$$

$$
\boxed{\ \mathcal{L}_{\text{period}} = \frac{1}{|\mathcal{T}|} \sum_{b \in \mathcal{T}} \big( 1 - \bar R_b \big)\ }
$$

In code the square root takes a small $\epsilon$ inside, since its gradient is undefined at $\bar R_b = 0$.

- **Averaged per track, not over the batch.** Unlike every other term (§0), the arrows are averaged within a track first. An arrow's angle only means something relative to the other bars of the same track, so pooling bars across tracks would compare unrelated phases.
- **One isolated downbeat cannot fake periodicity.** Each chunk has its own softmax, so every chunk has to vote. Pooling raw downbeat probabilities over the whole track instead would let a single peak look perfectly periodic.
- **It rewards confidence as well as agreement.** $\bar R_b$ is at most the mean of the chunks' own lengths $R_j$, so the term can only reach 0 if every chunk is concentrated *and* they agree (§9).

How it ranks the predictions the anchor term gets wrong, in 4/4, with each bar's votes entirely on one beat (anchor values include the hinge $\delta$):

| Downbeat prediction over the bars | $\mathcal{L}_{\text{anchor}}$ | $\mathcal{L}_{\text{period}}$ |
|---|---|---|
| always on beat 1 | 0 | 0 |
| always on beat 3 (consistently shifted) | 1.95 | 0 |
| alternating beat 1, beat 3 (flips) | 0.975 | 1 |
| uniform (no downbeat peak) | 0.95 | 1 |

The two rankings disagree on the shifted and the flipping predictions, which is the point: each term catches the failure the other is blind to. In terms of the project's metrics, the anchor targets whole-track offsets (`anchor_error`) and the periodicity term targets mid-track flips (`position_acc_best_offset`).

**Diagnostics (no gradient).** The track's consensus phase $\bar\Phi_b = \mathrm{atan2}(\bar S_b, \bar C_b)$ is where the model thinks the downbeat is; on annotated tracks it should approach 0. Comparing $\bar R_b$ with the mean of the chunks' $R_j$ separates disagreement between bars from indecision within them.

---

## 5. Total objective

The periodicity terms are added to the existing objective rather than replacing it. The shift-tolerant BCE stays on both heads and keeps deciding whether a beat or downbeat is present at all. The three new terms come in with their own weights, which ramp up from zero over the early epochs.

The Beat This! shift-tolerant weighted BCE, with $m_k(\cdot)$ denoting max-pooling over $k$ frames, is

$$
\mathcal{L}_{\text{st}}(y, \hat y, w) = -\sum_t \Big[\thinspace w\thinspace y_t \log m_7(\hat y)_t + \big( 1 - m_{13}(y)_t \big) \log\big( 1 - m_7(\hat y)_t \big) \Big]
$$

Here $w$ is the ratio of negative to positive frames over the training set. With the Sum Head, $\hat y^{\text{beat}} = \sigma(z)$ and $\hat y^{\text{down}} = \sigma(z^{d})$.

$$
\mathcal{L} = \mathcal{L}_{\text{st}}\big( y^{\text{beat}}, \hat y^{\text{beat}}, w_b \big) + \mathcal{L}_{\text{st}}\big( y^{\text{down}}, \hat y^{\text{down}}, w_d \big) + \lambda_b(e)\thinspace \mathcal{L}_{\text{beat}} + \lambda_a(e)\thinspace \mathcal{L}_{\text{anchor}} + \lambda_p(e)\thinspace \mathcal{L}_{\text{period}}
$$

$$
\lambda(e) = \lambda \cdot \mathrm{clip}\negthinspace\left( \frac{e - e_0}{e_1 - e_0},\ 0,\ 1 \right)
$$

The ramp over the epoch index $e$ switches the periodicity terms on only after peaks have formed. On a flat, untrained output the soft-argmax is noise.

---

## 6. Gradients

This section checks that each term pushes only where it should. The beat term moves logit mass around inside each beat's window but cannot add or remove it, so missing beats stay the BCE's job. The downbeat terms touch only the downbeat logits, at the annotated beats. The derivation also shows why §1's adaptive temperature matters: each frame's gradient is proportional to its softmax weight, so a collapsed softmax would leave all but the peak frame with none.

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

**Sum Head.** $\partial \mathcal L_{\text{beat}} / \partial z^b_t = \partial \mathcal L_{\text{beat}} / \partial z^d_t = \partial \mathcal L_{\text{beat}} / \partial z_t$, so the beat term also shapes the downbeat head.

### 6.2 Anchor term

The same covariance form as in §6.1, over a bar's beats instead of a window's frames:

$$
\frac{\partial C_j}{\partial u_k} = \frac{1}{\tau_j}\thinspace q_{j,k}\thinspace \big( \cos\theta_k - C_j \big), \qquad
\frac{\partial \mathcal{L}_{\text{anchor}}}{\partial u_k} = -\frac{1}{J}\thinspace \mathbb{1}\big[ 1 - C_j > \delta \big]\thinspace \frac{\partial C_j}{\partial u_k}
$$

This raises every beat whose angle is closer to beat 1 than the current mean, and lowers the others. Through the max-pool, $\partial u_k / \partial z^d_t = 1$ only at $t = f_k + \arg\max_{|s| \le 3} z^d_{f_k + s}$ and is 0 elsewhere. $\mathcal{L}_{\text{anchor}}$ touches only $z^{d}$.

### 6.3 Periodicity term

For beat $k$ in chunk $j$ of track $b$, write $\bar\Phi_b$ for the consensus phase (§4.2) and $P_j = C_j \cos\bar\Phi_b + S_j \sin\bar\Phi_b$ for the length of chunk $j$'s arrow along the consensus direction. Differentiating $\bar R_b$ through $\bar C_b$ and $\bar S_b$ gives

$$
\frac{\partial \mathcal{L}_{\text{period}}}{\partial u_k} = -\frac{1}{|\mathcal{T}|\thinspace |\mathcal{J}_b|\thinspace \tau_j}\thinspace q_{j,k}\thinspace \Big( \cos\big( \theta_k - \bar\Phi_b \big) - P_j \Big)
$$

This is the anchor's gradient (§6.2) with beat 1 replaced by the track's own consensus: each bar is pulled towards wherever the other bars put the downbeat, raising the beats closer to that consensus than the bar's current arrow and lowering the others. The consensus itself is set by the majority of bars. When most bars are right, this pulls the stray ones back into line; when most bars are wrong, it pulls the right ones towards the wrong consensus, and only the anchor term pushes back (§9). Like the anchor, it touches only $z^{d}$, at the annotated beats.

---

## 7. Properties

This section collects what the terms ignore on purpose, what they deliberately do not ignore, and how their thresholds line up with the evaluation's.

All of these follow directly from the definitions above.

- **Offset invariance (beat).** $\hat b_k \to \hat b_k + c$ leaves $\rho_k$ unchanged, which absorbs constant annotator or tapping latency.
- **Tempo-scale invariance.** $(\hat b_k, b_k) \to (s \hat b_k, s b_k)$ leaves $\rho_k$ unchanged, which covers tempo augmentation and comparisons across pieces.
- **Octave symmetry.** Doubling or halving the predicted interval gives $\rho = \mp \log 2$, so both are penalised equally.
- **Tempo curves.** The term is anchored to the ground-truth $\Delta_k$, so a ritardando that is in the annotation costs nothing.
- **Link to the metric's tolerance.** mir_eval's continuity condition $|\hat\Delta/\Delta - 1| < 0.175$ becomes, in log space, $\rho \in (\log 0.825,\ \log 1.175) = (-0.192,\ 0.161)$; using the symmetric bound $\gamma = 0.161$ is slightly stricter on the short side. In terms of the loss:
  - the penalty starts at $|\rho| = \varepsilon$, about 5%;
  - the metric boundary $|\rho| = \gamma$ sits at $x = 1 - \varepsilon/\gamma \approx 0.69$;
  - the Huber penalty turns linear at $|\rho| = \varepsilon + \gamma \approx 0.211$, about 23.5%.
- **The anchor is deliberately not offset-invariant.** A consistent 2-beat shift in 4/4 gives $C_j = -1$, which is the maximal penalty $2 - \delta$.
- **Periodicity is deliberately offset-invariant.** Shifting the downbeat by the same number of beats in every bar rotates all arrows of a track together and leaves $\bar R_b$ unchanged. It is the anchor's complement: the anchor fixes which beat is the 1, the periodicity term that it stays the same beat.
- **Bounds.** $0 \le \mathcal L_{\text{anchor}} \le 2 - \delta$, $0 \le \mathcal L_{\text{period}} \le 1$, and $\mathcal L_{\text{beat}} \ge 0$ with bounded gradient.

---

## 8. Defaults

Every parameter above, with the code argument that sets it and its default. Frame counts assume 50 fps.

| Symbol | Code argument | Default | Meaning |
|---|---|---|---|
| $H$ | `max_half` | 40 frames (0.8 s) | half-window cap; a full Voronoi window needs $\Delta \le 2H$, i.e. tempo $\ge 37.5$ BPM |
| $\kappa_b$ | `kappa_beat` | 8 | softmax range inside a beat window |
| $\kappa_\phi$ | `kappa_bar` | 4 | softmax range inside a bar (few slots) |
| $\tau_{\min}$ | `tau_min` | 0.5 | temperature floor |
| $\varepsilon$ | `eps_ibi` | 0.05 | log-interval deadzone, about annotation jitter |
| $\gamma$ | `ibi_tol` | $\log 1.175$ | residual unit |
| $\delta$ | `anchor_hinge` | 0.05 | anchor hinge, $\approx 18°$ |
| — | `pool` | 7 ($\pm 3$ frames) | downbeat sampling pool |
| — | `gate` | `True` | presence gate $g_k$ |
| $\lambda_b,\ \lambda_a,\ \lambda_p$ | — | $\approx 1$ (untuned) | loss weights |
| $e_0,\ e_1$ | — | 5, 15 | ramp start / end epoch, shared by all three terms (§9) |
| — | `period_min_chunks` | 2 | chunks a track needs to enter $\mathcal{L}_{\text{period}}$ |
| $N$ | — | annotated tracks only | bar length for tracks without downbeats (§9) |

---

## 9. Open questions

The periodicity term (§4.2) is a proposal; these need settling before or while it is implemented.

1. **Bar length on tracks without downbeat annotations.** The chunks there need an $N$ that nothing annotates. Options: take the smaller loss over $N \in \lbrace 3, 4 \rbrace$ (the model is periodic with *some* common bar length); assume one meter per corpus; or apply the term only to annotated tracks, where it duplicates part of what the anchor already supervises. The last is the default in §8 until this is decided.
2. **Meter changes and the pooling span.** On annotated tracks a meter change is harmless (each bar has its own angles). On unannotated tracks it misaligns every later chunk, and pooling over the whole 30 s excerpt then penalises a correct prediction. Pooling over a sliding window of a few bars instead would limit the damage, at the cost of a weaker signal.
3. **Agreement or agreement plus confidence.** $\bar R_b$ also rewards each bar being concentrated, which adds pressure towards overconfidence on top of the anchor's. A pure agreement measure would divide by the mean chunk length $R_j$, and a hinge like the anchor's $\delta$ would stop the push towards one-hot. Which one to use is open.
4. **When the majority of bars is wrong.** The gradient pulls every bar towards the track's consensus (§6.3). If most bars are on beat 3, it pulls the correct bars towards beat 3, against the anchor. On annotated tracks the anchor has to outweigh it, which bears on the next question.
5. **Weight and schedule.** Should $\lambda_p$ ramp in later than $\lambda_a$, so that the anchor has set the consensus before the bars are pulled towards it? Is it per-track mean or a mean weighted by the number of chunks? Both are untuned.
6. **Does the anchor's blind spot matter in practice?** The anchor prefers flipping to a consistent shift only in principle. Before building the term, measure on current checkpoints how much of the bar-position error is mid-track flips: the gap between `position_acc_best_offset` and 1. If it is small, the periodicity term mainly earns its place on tracks without downbeat annotations.
