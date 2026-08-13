"""Turn per-frame beat/one/last probability curves into a labeled beat list.

Pipeline: :func:`pick_peaks` (continuous curve -> discrete timestamps) ->
:func:`gate_periodicity` (clean up the timestamp list using beat regularity) ->
:func:`label_bar_position` (assign each timestamp a position 1-4 in the bar) ->
all three composed in :func:`readout`.
"""

import numpy as np


def pick_peaks(
    probs: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 1,
) -> np.ndarray:
    """Find local-maximum frame indices in a per-frame probability curve.

    Because training targets are Gaussian-smeared (see
    :func:`musicality.loaders.beat_dataset.gaussian_smear`), a real event
    shows up as a *bump* spanning several frames, not a single spike.
    Thresholding alone would return every frame in that bump; this finds the
    single frame at each bump's summit instead.

    :param probs: Per-frame probability curve, shape ``(T,)``.
    :param threshold: Minimum probability to be considered a peak at all.
    :param min_distance: Minimum frame gap enforced between returned peaks.
        When two candidate peaks are closer than this, only the higher one is
        kept (greedy non-max suppression) — a cheap safety net against
        multiple noisy local maxima inside one bump.
    :returns: Sorted frame indices of accepted peaks, shape ``(n_peaks,)``.
    """

    n = len(probs)
    if n == 0:
        return np.array([], dtype=int)

    candidates = np.where(probs > threshold)[0]
    peaks = [
        i
        for i in candidates
        if (i == 0 or probs[i] > probs[i - 1])
        and (i == n - 1 or probs[i] > probs[i + 1])
    ]

    if len(peaks) <= 1 or min_distance <= 1:
        return np.array(sorted(peaks), dtype=int)

    peaks = np.array(peaks, dtype=int)
    order = np.argsort(-probs[peaks])  # highest probability first
    suppressed = np.zeros(len(peaks), dtype=bool)
    kept = []

    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(peaks[idx])
        suppressed |= np.abs(peaks - peaks[idx]) < min_distance

    return np.array(sorted(kept), dtype=int)


def gate_periodicity(
    times: np.ndarray,
    tolerance: float = 0.2,
    ema_alpha: float = 0.2,
    bootstrap_window: int = 8,
) -> np.ndarray:
    """Clean up a raw peak-picked beat-time sequence using local periodicity.

    Beats are not independent events — they're (locally) evenly spaced. This
    walks the raw timestamps in order, maintaining a running estimate of the
    current beat period, and uses it to both drop peaks that can't be real
    beats and fill in beats the peak-picker missed. It is a light-weight,
    single-pass stand-in for full tempo tracking (a particle filter would be
    the principled version — see the beat-phase plan's postprocessing
    section); it assumes the tempo is roughly constant within ``tolerance``
    from one beat to the next, not that it's constant across the whole track.

    Algorithm, per candidate timestamp ``t`` (with ``last`` = the most
    recently *accepted* timestamp, and ``period`` = the current running
    period estimate, seeded from the median of the first
    ``bootstrap_window`` raw intervals):

    1. Compute ``ratio = (t - last) / period`` — how many beat-periods have
       elapsed since the last accepted beat.
    2. **Too soon (``ratio < 1 - tolerance``):** less than one period has
       elapsed. A real next beat can't be this close — ``t`` is almost
       certainly a spurious duplicate (e.g. a loud snare hit inside the same
       beat re-triggering the detector). Drop it: don't accept it, don't
       move ``last``, don't touch ``period``.
    3. **Matches ``k`` periods (``round(ratio) = k`` and
       ``|ratio - k| <= tolerance``):** the gap is consistent with exactly
       ``k`` beats' worth of time.

       - ``k == 1``: an ordinary, expected next beat. Accept ``t``.
       - ``k >= 2``: the peak-picker likely missed ``k - 1`` beats in a row
         (e.g. a quiet passage where ``p_beat`` never crossed
         ``threshold``). Rather than leave a hole, synthesize ``k - 1``
         evenly-spaced timestamps filling the gap (assuming constant tempo
         *within this one gap*, which is a much narrower assumption than
         constant tempo for the whole track), then accept ``t``.
       - Either way, refresh ``period`` as an exponential moving average
         (weight ``ema_alpha``) using the *implied single-beat interval*
         ``(t - last) / k`` — so tempo can drift slowly over a track without
         this gate's expectation going stale, but no single noisy interval
         can swing it too far in one step.
    4. **Ambiguous (fits no integer ``k`` within tolerance):** doesn't look
       like a clean duplicate or a clean multi-beat gap — e.g. the model's
       timing was just imprecise on this one beat. Rather than guess, accept
       ``t`` as-is (never silently drop a real-looking detection on a
       best-effort heuristic) but leave ``period`` unchanged, since neither
       explanation was confident enough to learn from.

    :param times: Raw peak timestamps (seconds), any order — sorted internally.
    :param tolerance: Relative slack around an integer multiple of the current
        period for a candidate to count as "matching" (e.g. ``0.2`` = ±20%).
    :param ema_alpha: Weight given to each new observed interval when updating
        the running period estimate. Higher = adapts faster to tempo drift,
        but is noisier.
    :param bootstrap_window: Number of leading raw intervals averaged
        (median) to seed the initial period estimate.
    :returns: Cleaned, sorted timestamps (seconds), including any
        gap-filled beats. Shape ``(n_out,)`` — not generally equal to the
        input length.
    """

    times = np.sort(np.asarray(times, dtype=float))
    if len(times) < 3:
        return times

    bootstrap_intervals = np.diff(times[: bootstrap_window + 1])
    period = max(float(np.median(bootstrap_intervals)), 1e-6)

    accepted = [times[0]]
    for t in times[1:]:
        last = accepted[-1]
        d = t - last
        ratio = d / period

        if ratio < 1 - tolerance:
            continue  # too soon since the last accepted beat — spurious

        k = max(1, round(ratio))
        if abs(ratio - k) <= tolerance:
            single_beat = d / k
            for i in range(1, k):
                accepted.append(last + i * single_beat)
            accepted.append(t)
            period = (1 - ema_alpha) * period + ema_alpha * single_beat
        else:
            accepted.append(t)  # ambiguous — keep, but don't update the period

    return np.array(accepted)


def label_bar_position(
    beat_times: np.ndarray,
    one_probs: np.ndarray,
    last_probs: np.ndarray,
    fps: float,
    anchor_threshold: float = 0.5,
    group_size: int = 4,
) -> list[int | None]:
    """Assign each gated beat time a group position (1-``group_size``), or ``None`` if unresolved.

    Each beat casts an "anchor vote" by sampling ``one_probs``/``last_probs``
    at its nearest frame: a confident vote for position 1 or ``group_size`` if
    the corresponding probability clears ``anchor_threshold`` and beats the
    other; no vote otherwise (expected for the majority of beats — the
    in-between positions have no dedicated head). Positions are then assigned
    by counting forward from confident votes (1, 2, ..., ``group_size``, 1, 2,
    ...), with every new confident vote resyncing the count — so a stretch of
    ambiguous beats between two anchors is still labeled by counting, but the
    moment a strong anchor disagrees with where counting drifted to, the
    anchor wins. Beats before the first confident anchor are left unresolved
    (``None``).

    :param beat_times: Gated beat timestamps (seconds), sorted.
    :param one_probs: Per-frame "one" probability curve, shape ``(T,)``.
    :param last_probs: Per-frame "last" (position ``group_size``) probability
        curve, shape ``(T,)``.
    :param fps: Frames per second (``sample_rate / hop_length``), used to map
        a beat time to its nearest frame.
    :param anchor_threshold: Minimum probability for a beat to be trusted as
        a confident 1/``group_size`` anchor.
    :param group_size: Number of positions per group — ``4`` for bar position
        (default), ``8`` for phrase position.
    :returns: One entry per beat time: ``1``..``group_size``, or ``None``.
    """

    n_frames = len(one_probs)
    labels: list[int | None] = [None] * len(beat_times)

    votes: list[int | None] = []
    for t in beat_times:
        frame = min(max(int(round(t * fps)), 0), n_frames - 1)
        p_one, p_last = one_probs[frame], last_probs[frame]
        if p_one > anchor_threshold and p_one >= p_last:
            votes.append(1)
        elif p_last > anchor_threshold and p_last > p_one:
            votes.append(group_size)
        else:
            votes.append(None)

    expected = None
    for i, vote in enumerate(votes):
        if vote is not None:
            expected = vote
            labels[i] = expected
        elif expected is not None:
            expected = expected % group_size + 1
            labels[i] = expected

    return labels


def readout(
    beat_probs: np.ndarray,
    one_probs: np.ndarray,
    last_probs: np.ndarray,
    fps: float,
    beat_threshold: float = 0.3,
    min_distance_frames: int = 1,
    gate_tolerance: float = 0.2,
    anchor_threshold: float = 0.5,
    group_size: int = 4,
) -> list[dict]:
    """End-to-end: per-frame probability curves -> a labeled beat list.

    Composes :func:`pick_peaks` on ``beat_probs``, :func:`gate_periodicity`
    on the resulting timestamps, then :func:`label_bar_position` using
    ``one_probs``/``last_probs``.

    :param beat_probs: Per-frame beat probability curve, shape ``(T,)``.
    :param one_probs: Per-frame "one" probability curve, shape ``(T,)``.
    :param last_probs: Per-frame "last" (position ``group_size``) probability
        curve, shape ``(T,)``.
    :param fps: Frames per second (``sample_rate / hop_length``).
    :param beat_threshold: Passed to :func:`pick_peaks`.
    :param min_distance_frames: Passed to :func:`pick_peaks`.
    :param gate_tolerance: Passed to :func:`gate_periodicity` as ``tolerance``.
    :param anchor_threshold: Passed to :func:`label_bar_position`.
    :param group_size: Passed to :func:`label_bar_position` — ``4`` for bar
        position (default), ``8`` for phrase position.
    :returns: One dict per detected beat, sorted by time:
        ``{"time": float, "beat_in_bar": int | None}``.
    """

    peak_frames = pick_peaks(
        beat_probs, threshold=beat_threshold, min_distance=min_distance_frames
    )
    peak_times = peak_frames / fps

    beat_times = gate_periodicity(peak_times, tolerance=gate_tolerance)

    labels = label_bar_position(
        beat_times,
        one_probs,
        last_probs,
        fps,
        anchor_threshold=anchor_threshold,
        group_size=group_size,
    )

    return [
        {"time": float(t), "beat_in_bar": label} for t, label in zip(beat_times, labels)
    ]


def readout_beat_only(
    beat_probs: np.ndarray,
    fps: float,
    beat_threshold: float = 0.3,
    min_distance_frames: int = 1,
    gate_tolerance: float = 0.2,
) -> np.ndarray:
    """End-to-end: a per-frame beat probability curve -> beat timestamps.

    Same first two stages as :func:`readout` (:func:`pick_peaks` then
    :func:`gate_periodicity`), minus the bar-position labeling step — that
    needs "one"/"last" probability curves, which a beat-only model doesn't
    produce.

    :param beat_probs: Per-frame beat probability curve, shape ``(T,)``.
    :param fps: Frames per second (``sample_rate / hop_length``).
    :param beat_threshold: Passed to :func:`pick_peaks`.
    :param min_distance_frames: Passed to :func:`pick_peaks`.
    :param gate_tolerance: Passed to :func:`gate_periodicity` as ``tolerance``.
    :returns: Sorted beat timestamps (seconds).
    """

    peak_frames = pick_peaks(
        beat_probs, threshold=beat_threshold, min_distance=min_distance_frames
    )
    peak_times = peak_frames / fps

    return gate_periodicity(peak_times, tolerance=gate_tolerance)
