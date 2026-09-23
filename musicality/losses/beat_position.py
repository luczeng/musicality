"""Bar position as a softmax over the ``G`` positions in a bar.

The current beat-phase objective, paired with
:class:`~musicality.trainers.beat_phase_module.BeatPhaseModule` and a backbone
emitting ``1 + G`` frame-level channels.
"""

import torch
import torch.nn.functional as F

from musicality.losses.phase_conditioning import phase_weight
from musicality.losses.pos_weight import AUTO_POS_WEIGHT_ALPHA
from musicality.losses.shift_tolerance import (
    TOLERANCE_FRAMES,
    resolve_tolerance,
    shift_tolerant_bce,
    sliding_windowed_max,
)

POSITION_NORMS = ("global", "per_item")


def beat_position_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | float | str = 5.0,
    phase_conditioning: str = "beat",
    pos_weight_alpha: float = AUTO_POS_WEIGHT_ALPHA,
    position_norm: str = "global",
    loss: str = "bce",
    tolerance_frames: int = TOLERANCE_FRAMES,
    ignore_frames: int | None = None,
    return_terms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Beat BCE plus a softmax cross-entropy over bar position.

    The successor to :func:`~musicality.losses.beat_phase.beat_phase_loss`.
    That loss models bar position as two *independent* binary detectors
    (``one`` and ``last``), which leaves the positions in between with
    identical supervision — negative on both heads — so the model is never
    asked the discriminative question the metric measures, "is this beat a 1
    or a 3?". Here every position gets its own logit and they compete inside a
    single softmax, so raising the score for position 1 necessarily lowers
    position 3.

    ``beat`` stays an independent sigmoid: "is there a beat here?" is a
    genuine binary question over time, not a pick-one-of-``G``, and folding it
    into the softmax would couple a head that already works to one that
    doesn't.

    .. math::

        \mathcal{L} = \underbrace{\frac{1}{BT} \sum_{i,t} \ell(\hat{b}_{i,t}, b_{i,t})}_{\text{beat}}
        - \underbrace{\frac{\sum_{i,t} w_{i,t} \sum_{p} q_{i,t,p} \log \hat{q}_{i,t,p}}{\sum_{i,t} w_{i,t}}}_{\text{position}}

    where :math:`\ell` is weighted binary cross-entropy — shift-tolerant once
    ``tolerance_frames`` is set — :math:`q` is the
    target's normalized position block, :math:`\hat{q}` the softmax over the
    model's position logits, and :math:`w` the per-frame phase weight (see
    ``phase_conditioning``).

    No ``pos_weight`` is needed on the position term: bar positions occur
    equally often, so the softmax is already balanced. ``pos_weight`` here is
    a scalar for the ``beat`` head alone.

    :param logits: Raw per-frame model output, shape ``(B, 1 + G, T)`` — beat
        first, then one logit per bar position (see
        :class:`musicality.models.tcn.TCNTempoNet` with ``frame_level=True``
        and ``n_outputs=1 + G``).
    :param target: Ground-truth target, shape ``(B, 2 + G, T)`` — beat, the
        normalized position block, then mask. Built by
        :class:`~musicality.loaders.beat_dataset.BeatDataset` with
        ``target_layout="positions"``.
    :param pos_weight: Positive-class weight for the ``beat`` BCE term only.
        A number, or ``"auto"`` to derive one per sample from the target — see
        :func:`~musicality.losses.pos_weight.beat_pos_weight`.
    :param phase_conditioning: ``"beat"`` (default) weights the position term
        by ``mask * beat``, so it is optimized only where a beat actually is —
        which is where :mod:`musicality.postprocess` reads it. ``"mask"``
        weights by ``mask`` alone, supervising every frame. See
        :func:`~musicality.losses.phase_conditioning.phase_weight` for why the
        former matters.
    :param pos_weight_alpha: Scale on the derived ``pos_weight``. Read only
        when ``pos_weight == "auto"``.
    :param position_norm: How the position term is averaged.

        - ``"global"`` (default): one weighted mean over the whole batch. That
          makes it a micro-average over *beats*, so a clip's influence is
          proportional to how many beats it happens to contain — and a 16 s
          crop holds 51 beats at 193 BPM but 15 at 56. Measured on the merged
          split, jtd takes 73.7% of the position gradient against 63.8% of the
          tracks, while ballroom gets 18.0% against 24.1%.
        - ``"per_item"``: normalize each clip by its own weight first, then
          average over clips. Every annotated clip carries exactly
          ``1/n_valid`` whatever its tempo, which is a macro-average over
          tracks — the same shape as the per-genre metric this is graded by.

        See plans/04_beat_phase_generalization_and_data_prep.md §2.6a.
    :param loss: Which objective the ``beat`` term uses — ``"bce"`` (the
        default, the pre-existing loss bit for bit) or ``"shift_tolerant"``.
        It also switches the ``position`` term's widening on, since the two
        are the same decision seen from either head; see ``tolerance_frames``.
        ``"shift_tolerant"`` wants ``sigma_frames: 0`` alongside it.
    :param tolerance_frames: Half-width, in frames, of the timing error both
        heads are forgiven. Read only under ``loss="shift_tolerant"``; the
        default is
        :data:`~musicality.losses.shift_tolerance.TOLERANCE_FRAMES`. It means
        two different things to the two terms, because the decoder reads them
        two different ways:

        - The ``beat`` term becomes
          :func:`~musicality.losses.shift_tolerance.shift_tolerant_bce`. That
          head is *scanned* over time by
          :func:`musicality.postprocess.pick_peaks`, so where its peak sits is
          the answer, and tolerance means forgiving a peak that is a frame or
          two off.
        - The ``position`` term is *widened* instead — both the gate and the
          target. That head is never scanned: the decoder rounds a beat time to
          a frame and reads that one column. Its answer is a label, not a time,
          so there is no peak to forgive; what it needs is to be right across
          the whole window the lookup might land in, which shift tolerance on
          the beat head has just made ``r`` frames wide.

        See :mod:`musicality.losses.shift_tolerance` on why smearing and
        tolerance do not compose.
    :param ignore_frames: Half-width of the band around each beat where the
        ``beat`` term's negative half is switched off. ``None`` derives it as
        ``2 * tolerance_frames``. Read only under ``loss="shift_tolerant"``,
        and it does not touch the position term, which has no negative class.
    :param return_terms: Return the two terms separately instead of their sum,
        as ``(beat_term, position_term)``. The sum is what optimisation needs;
        the split is what tells a rising loss apart from a rising *error* —
        position CE can climb on overconfidence alone while beat BCE and
        position accuracy both improve (plans/07 §1.3, measured between epochs
        79 and 107 of v6). Logged per epoch by
        :class:`~musicality.trainers.beat_phase_module.BeatPhaseModule`.
    :returns: Scalar mean loss, shape ``()`` — or, with ``return_terms``, the
        pair of scalars that sums to it.
    """

    if position_norm not in POSITION_NORMS:
        raise ValueError(
            f"Unknown position_norm {position_norm!r} — expected 'global' or 'per_item'"
        )

    tolerance_frames = resolve_tolerance(loss, tolerance_frames)

    beat_logits, position_logits = logits[:, 0], logits[:, 1:]
    beat_y, position_y, mask = target[:, 0], target[:, 1:-1], target[:, -1]

    if position_logits.shape[1] != position_y.shape[1]:
        raise ValueError(
            f"logits carry {position_logits.shape[1]} position channels but the "
            f"target carries {position_y.shape[1]} — logits should be "
            "(B, 1 + G, T) against a (B, 2 + G, T) target"
        )

    phase_w = phase_weight(beat_y, mask, phase_conditioning, tolerance_frames)

    beat_term = shift_tolerant_bce(
        beat_logits,
        beat_y,
        pos_weight=pos_weight,
        pos_weight_alpha=pos_weight_alpha,
        tolerance_frames=tolerance_frames,  # already resolved from `loss`
        ignore_frames=ignore_frames,
    )

    if tolerance_frames:
        # Widen the target alongside the gate, never on its own. BeatDataset
        # gives frames away from a beat a uniform 1/G row — "no information
        # here" — so a wider gate against the untouched target would supervise
        # the frames the decoder reads towards maximum uncertainty.
        #
        # Masking by `beat_y` before the window max is what keeps those uniform
        # rows out of the result. Pooling the block directly would carry their
        # 1/G into every channel the beat leaves at zero, and renormalising a
        # (1, 1/G, 1/G, 1/G) column yields (0.57, 0.14, 0.14, 0.14) — a target
        # that punishes a confidently *correct* prediction. Masked first, the
        # window holds the beat's own column and nothing else.
        #
        # Frames the window never reaches fall back to uniform, exactly as
        # BeatDataset builds them: under ``phase_conditioning="beat"`` their
        # gate is zero anyway, but under ``"mask"`` it is not.
        n_positions = position_y.shape[1]
        widened = sliding_windowed_max(
            position_y * beat_y.unsqueeze(1), tolerance_frames
        )
        total = widened.sum(dim=1, keepdim=True)

        position_y = torch.where(
            total > 1e-6, widened / total.clamp(min=1e-6), 1.0 / n_positions
        )

    # Soft-target cross-entropy over the position axis, per frame.
    log_q = F.log_softmax(position_logits, dim=1)
    position_ce = -(position_y * log_q).sum(dim=1)  # (B, T)

    if position_norm == "global":
        n_weighted = phase_w.sum().clamp(min=1.0)

        position_term = (position_ce * phase_w).sum() / n_weighted

    else:
        # Divide each clip by its own weight before averaging, so tempo stops
        # buying influence. Clips with no position annotation have `den == 0`
        # and therefore `num == 0` too — both are sums of `phase_w`-weighted
        # terms — so the clamp lets them contribute exactly zero with a
        # well-defined gradient. That keeps a fully unannotated batch finite
        # (it reduces to the beat term) without boolean indexing, which would
        # make the shape depend on the data.
        num = (position_ce * phase_w).sum(dim=-1)  # (B,)
        den = phase_w.sum(dim=-1)  # (B,)
        n_valid = (den > 1e-6).sum().clamp(min=1)

        position_term = (num / den.clamp(min=1e-6)).sum() / n_valid

    if return_terms:
        return beat_term, position_term

    return beat_term + position_term
