# Soft-argmax periodicity losses: mathematical formulation

This is the companion to `softargmax_periodicity.py`. It defines two auxiliary terms that are added to the Beat This! shift-tolerant BCE:

- $\mathcal{L}_{\text{beat}}$ works at **beat level, on the time axis**. A soft-argmax inside per-beat windows gives sub-frame beat centres. Their inter-beat intervals are compared with the ground-truth intervals in log space.
- $\mathcal{L}_{\text{phase}}$ works at **bar level, on the beat-index axis**. It takes a circular soft-argmax of the downbeat logits over the beats of each bar.

The BCE decides *whether* a peak exists. These two terms decide *where peaks sit relative to each other*.

---

## 0. Conventions

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
| $n_k \in \{1, \dots, N\}$ | ground-truth beat-in-bar number | `beat_numbers` | $(K,)$ per item |
| $\mathcal{W}_k$ | frame window of beat $k$ | `win_idx`, `win_mask` | $(B, K, W)$ |
| $\hat b_k$ | soft-argmax beat centre | `t_hat` | $(B, K)$ |
| $\mathcal{B}_j$ | beat indices of bar $j$ | `bar_beat`, `bar_mask` | $(B, J, N_{\max})$ |
| $\theta_k$ | angle of beat $k$ in its bar | `bar_theta` | $(B, J, N_{\max})$ |

For a 30 s excerpt: $T = 1500$, $W = 2H + 1 = 81$, $K \approx 30$–$120$, $J \approx 8$–$30$.

---

## 1. Shared operator: masked softmax with adaptive temperature

Take logits $a_i$ over a finite index set $\mathcal{I}$ and a range parameter $\kappa$:

$$
m = \max_{i \in \mathcal{I}} a_i, \qquad
\bar a = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} a_i, \qquad
\tau = \mathrm{sg}\!\left[ \max\!\left( \tau_{\min},\ \frac{m - \bar a}{\kappa} \right) \right]
$$

$$
\operatorname{softmax}_{\kappa}(a)_i = \frac{\exp\big( (a_i - m)/\tau \big)}{\sum_{i' \in \mathcal{I}} \exp\big( (a_{i'} - m)/\tau \big)}
$$

**Why adaptive.** Beat This!-style training produces overconfident logits ($|a| \approx 10$). With a fixed $\tau$, the softmax becomes one-hot and the gradients, which are proportional to $p_i$ (see §6), vanish. With the adaptive $\tau$, whenever the floor is inactive the scaled mean logit sits exactly $\kappa$ below the scaled maximum, $(m - \bar a)/\tau = \kappa$, whatever the logit scale. The temperature is detached, so it is never a path for the network to lower the loss. In the code, `kappa=None` gives a fixed $\tau = \tau_{\min}$.

---

## 2. Target construction

### 2.1 Beat windows (Voronoi, capped)

$$
\ell_k = \tfrac{1}{2} (b_{k-1} + b_k), \quad \ell_1 = b_1 - \tfrac{1}{2} \Delta_1, \qquad
r_k = \ell_{k+1}, \quad r_K = b_K + \tfrac{1}{2} \Delta_{K-1}
$$

$$
\mathcal{W}_k = \{\, t \in \mathbb{Z} \;:\; \ell_k \le t < r_k,\ \ \lvert t - b_k \rvert \le H,\ \ 0 \le t < T \,\}
$$

The windows are disjoint and ordered, so every frame belongs to at most one window.

A beat is **valid** if its window was not cut by the excerpt edge and has at least 3 frames:

$$
v_k = \mathbb{1}[\ell_k \ge 0]\ \mathbb{1}[r_k \le T]\ \mathbb{1}\big[ |\mathcal{W}_k| \ge 3 \big]
$$

### 2.2 Bar lengths (on the full song, before cropping)

Let $s_1 < \dots < s_M$ be the downbeat indices ($n_{s_m} = 1$). Bar $m$ covers beats $[s_m, s_{m+1})$ and has length

$$
N_m = \max\!\Big( s_{m+1} - s_m,\ \max_{s_m \le k < s_{m+1}} n_k \Big)
$$

Pickup beats ($k < s_1$) take $N_1$. The final bar takes $\max(N_M, N_{M-1})$, because songs and annotations often end mid-bar. In the code this is `bar_lengths_from_numbers`.

### 2.3 Complete bars inside the excerpt

Bar $j$ is kept only if it is complete and consecutively numbered:

$$
\mathcal{B}_j = \{ k_j, \dots, k_j + N_j - 1 \}, \qquad n_{k_j + i} = i + 1 \ \ (i = 0, \dots, N_j - 1), \qquad N_j \ge 2
$$

$$
\theta_k = \frac{2\pi\, (n_k - 1)}{N_j}, \qquad k \in \mathcal{B}_j
$$

Bars cut by the excerpt edge are dropped, because a partial bar would bias the circular mean.

---

## 3. Beat term

**Soft-argmax inside each window**

$$
p_{k,\cdot} = \operatorname{softmax}_{\kappa_b}\big( z_t \big)_{t \in \mathcal{W}_k}, \qquad
\hat b_k = \sum_{t \in \mathcal{W}_k} p_{k,t}\ t, \qquad
m_k = \max_{t \in \mathcal{W}_k} z_t
$$

**Log-interval residual**

$$
\rho_k = \log \max\big( \hat b_{k+1} - \hat b_k,\ 0.5 \big) - \log \max\big( \Delta_k,\ 0.5 \big)
$$

Because the windows are disjoint and ordered, $\hat b_{k+1} > \hat b_k$ always holds. The clamp at 0.5 frames is only there for numerical safety.

**Excess beyond the deadzone, in tolerance units**

$$
x_k = \frac{\big[\, |\rho_k| - \varepsilon \,\big]_+}{\gamma}
$$

**Penalty, gate and loss**

$$
h(x) = \begin{cases} \tfrac{1}{2} x^2 & x \le 1 \\[2pt] x - \tfrac{1}{2} & x > 1 \end{cases}
\qquad
g_k = \mathrm{sg}\big[ \sigma(m_k)\, \sigma(m_{k+1}) \big]
\qquad
\mathcal{P} = \{ k : v_k = v_{k+1} = 1 \}
$$

$$
\boxed{\ \mathcal{L}_{\text{beat}} = \frac{1}{|\mathcal{P}|} \sum_{k \in \mathcal{P}} g_k\ h(x_k)\ }
$$

- $h$ is the Huber function with threshold 1. It is quadratic for small errors, and for gross errors (octave jumps, missed beats) it grows linearly, so the gradient is bounded: $|h'| \le 1$.
- The gate removes pairs where either window has no confident peak, so soft-argmax does not chase noise in empty windows. Missing peaks are the BCE's job.
- The loss is normalised by $|\mathcal{P}|$ rather than by $\sum g_k$, so gated pairs reduce the loss instead of reweighting the remaining pairs.

---

## 4. Phase term

**Downbeat logits sampled at the ground-truth beats** (max-pooling over $\pm 3$ frames, as in the shift-tolerant BCE, with $-\infty$ padding):

$$
\tilde z^{d}_t = \max_{|s| \le 3} z^{d}_{t+s}, \qquad
u_k = \tilde z^{d}_{f_k}, \qquad
f_k = \operatorname{clip}\big( \operatorname{round}(b_k),\ 0,\ T - 1 \big)
$$

**Circular soft-argmax over each bar**

$$
q_{j,\cdot} = \operatorname{softmax}_{\kappa_\phi}\big( u_k \big)_{k \in \mathcal{B}_j}, \qquad
C_j = \sum_{k \in \mathcal{B}_j} q_{j,k} \cos \theta_k, \qquad
S_j = \sum_{k \in \mathcal{B}_j} q_{j,k} \sin \theta_k
$$

$$
\boxed{\ \mathcal{L}_{\text{phase}} = \frac{1}{J} \sum_{j=1}^{J} \big[\, 1 - C_j - \delta \,\big]_+\ }
$$

Here $J$ is the number of complete bars in the batch. Datasets without downbeat annotations give $J = 0$, and then $\mathcal{L}_{\text{phase}} = 0$.

**Interpretation.** $1 - C_j$ is the expected circular distance from beat 1:

$$
1 - C_j = \sum_{k \in \mathcal{B}_j} q_{j,k}\, \big( 1 - \cos \theta_k \big) \in [0, 2]
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

$$
\hat\varphi_j = \operatorname{atan2}(S_j, C_j), \qquad
R_j = \sqrt{C_j^2 + S_j^2}, \qquad
\text{err}_j = \frac{|\hat\varphi_j|}{2\pi} \ \ \text{(fraction of a bar)}
$$

$R_j \approx 1$ means the mass is concentrated (confidently placed), and $R_j \approx 0$ means it is split or absent. Together, $(\hat\varphi_j, R_j)$ separate "confidently on the wrong beat" from "undecided".

---

## 5. Total objective

The Beat This! shift-tolerant weighted BCE, with $m_k(\cdot)$ denoting max-pooling over $k$ frames, is

$$
\mathcal{L}_{\text{st}}(y, \hat y, w) = -\sum_t \Big[\, w\, y_t \log m_7(\hat y)_t + \big( 1 - m_{13}(y)_t \big) \log\big( 1 - m_7(\hat y)_t \big) \Big]
$$

Here $w$ is the ratio of negative to positive frames over the training set. With the Sum Head, $\hat y^{\text{beat}} = \sigma(z)$ and $\hat y^{\text{down}} = \sigma(z^{d})$.

$$
\mathcal{L} = \mathcal{L}_{\text{st}}\big( y^{\text{beat}}, \hat y^{\text{beat}}, w_b \big) + \mathcal{L}_{\text{st}}\big( y^{\text{down}}, \hat y^{\text{down}}, w_d \big) + \lambda_b(e)\, \mathcal{L}_{\text{beat}} + \lambda_\phi(e)\, \mathcal{L}_{\text{phase}}
$$

$$
\lambda(e) = \lambda \cdot \operatorname{clip}\!\left( \frac{e - e_0}{e_1 - e_0},\ 0,\ 1 \right)
$$

The ramp over the epoch index $e$ switches the periodicity terms on only after peaks have formed. On a flat, untrained output the soft-argmax is noise.

---

## 6. Gradients

The temperatures $\tau$ and the gates $g_k$ are detached, so they contribute no gradient.

### 6.1 Beat term

The soft-argmax derivative has the form of a covariance:

$$
\frac{\partial \hat b_k}{\partial z_t} = \frac{1}{\tau_k}\, p_{k,t}\, \big( t - \hat b_k \big) \quad (t \in \mathcal{W}_k), \qquad
\sum_{t \in \mathcal{W}_k} \frac{\partial \hat b_k}{\partial z_t} = 0
$$

The derivatives sum to zero over the window. So the beat term only **redistributes** logit mass inside a window and cannot create or remove a peak in aggregate.

Let $s_k = \operatorname{sign}(\rho_k)\, \mathbb{1}\big[ |\rho_k| > \varepsilon \big]$ and $h'(x) = \min(x, 1)$. Since the windows are disjoint, frame $t \in \mathcal{W}_k$ receives gradient only through $\hat b_k$, which enters the two neighbouring pairs:

$$
\frac{\partial \mathcal{L}_{\text{beat}}}{\partial \hat b_k} = \frac{1}{\gamma\, |\mathcal{P}|} \left[ \frac{g_{k-1}\, h'(x_{k-1})\, s_{k-1}}{\hat b_k - \hat b_{k-1}} - \frac{g_k\, h'(x_k)\, s_k}{\hat b_{k+1} - \hat b_k} \right]
$$

A term is dropped when its pair is not in $\mathcal{P}$. Then

$$
\frac{\partial \mathcal{L}_{\text{beat}}}{\partial z_t} = \frac{\partial \mathcal{L}_{\text{beat}}}{\partial \hat b_k}\ \frac{\partial \hat b_k}{\partial z_t}
$$

Frames on the side of $\hat b_k$ that reduces $|\rho|$ are pushed up, and frames on the other side are pushed down. This is how an inserted peak inside a window gets suppressed.

**Temperature leverage.** The gradient scales as $p_{k,t}/\tau_k$. With a fixed $\tau$ and $|z| \approx 10$, off-peak frames have $p \lesssim e^{-8}$. With the adaptive $\tau_k$, the relative logits of background frames are set by $\kappa_b$, not by the logit scale.

**Sum Head.** $\partial \mathcal{L}_{\text{beat}} / \partial z^{b}_t = \partial \mathcal{L}_{\text{beat}} / \partial z^{d}_t = \partial \mathcal{L}_{\text{beat}} / \partial z_t$, so the beat term also shapes the downbeat head.

### 6.2 Phase term

$$
\frac{\partial C_j}{\partial u_k} = \frac{1}{\tau_j}\, q_{j,k}\, \big( \cos\theta_k - C_j \big), \qquad
\frac{\partial \mathcal{L}_{\text{phase}}}{\partial u_k} = -\frac{1}{J}\, \mathbb{1}\big[ 1 - C_j > \delta \big]\, \frac{\partial C_j}{\partial u_k}
$$

This raises every beat whose angle is closer to beat 1 than the current mean, and lowers the others. Through the max-pool, $\partial u_k / \partial z^{d}_t = 1$ only at $t = \arg\max_{|s| \le 3} z^{d}_{f_k + s}$ and is 0 elsewhere. $\mathcal{L}_{\text{phase}}$ touches only $z^{d}$.

---

## 7. Properties

All of these follow directly from the definitions above.

- **Offset invariance (beat).** $\hat b_k \to \hat b_k + c$ leaves $\rho_k$ unchanged, which absorbs constant annotator or tapping latency.
- **Tempo-scale invariance.** $(\hat b_k, b_k) \to (s \hat b_k, s b_k)$ leaves $\rho_k$ unchanged, which covers tempo augmentation and comparisons across pieces.
- **Octave symmetry.** Doubling or halving the predicted interval gives $\rho = \mp \log 2$, so both are penalised equally.
- **Tempo curves.** The term is anchored to the ground-truth $\Delta_k$, so a ritardando that is in the annotation costs nothing.
- **Link to the metric's tolerance.** mir_eval's continuity condition $|\hat\Delta/\Delta - 1| < 0.175$ becomes, in log space, $\rho \in (\log 0.825,\ \log 1.175) = (-0.192,\ 0.161)$; using the symmetric bound $\gamma = 0.161$ is slightly stricter on the short side. In terms of the loss:
  - the penalty starts at $|\rho| = \varepsilon$, about 5%;
  - the metric boundary $|\rho| = \gamma$ sits at $x = 1 - \varepsilon/\gamma \approx 0.69$;
  - the Huber penalty turns linear at $|\rho| = \varepsilon + \gamma \approx 0.211$, about 23.5%.
- **Phase is deliberately not offset-invariant.** A consistent 2-beat shift in 4/4 gives $C_j = -1$, which is the maximal penalty $2 - \delta$.
- **Bounds.** $0 \le \mathcal{L}_{\text{phase}} \le 2 - \delta$, and $\mathcal{L}_{\text{beat}} \ge 0$ with bounded gradient.

---

## 8. Defaults

| Symbol | Code argument | Default | Meaning |
|---|---|---|---|
| $H$ | `max_half` | 40 frames (0.8 s) | half-window cap; a full Voronoi window needs $\Delta \le 2H$, i.e. tempo $\ge 37.5$ BPM |
| $\kappa_b$ | `kappa_beat` | 8 | softmax range inside a beat window |
| $\kappa_\phi$ | `kappa_bar` | 4 | softmax range inside a bar (few slots) |
| $\tau_{\min}$ | `tau_min` | 0.5 | temperature floor |
| $\varepsilon$ | `eps_ibi` | 0.05 | log-interval deadzone, about annotation jitter |
| $\gamma$ | `ibi_tol` | $\log 1.175$ | residual unit |
| $\delta$ | `phase_hinge` | 0.05 | phase hinge, $\approx 18°$ |
| — | `pool` | 7 ($\pm 3$ frames) | downbeat sampling pool |
| — | `gate` | `True` | presence gate $g_k$ |
| $\lambda_b,\ \lambda_\phi$ | — | $\approx 1$ (untuned) | loss weights |
| $e_0,\ e_1$ | — | 5, 15 | ramp start / end epoch |
