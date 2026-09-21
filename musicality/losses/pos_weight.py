"""Positive-class weighting for the frame-wise ``beat`` BCE term.

Beat frames are a small fraction of all frames, so the beat head needs a
``pos_weight`` to stop it collapsing to "never a beat". How small a fraction
depends on the tempo, which a single configured number cannot track — hence
``pos_weight="auto"``, which derives the weight per sample from the target
already in hand. Shared by every loss here that carries a beat head:
:func:`~musicality.losses.beat_only.beat_only_loss` and
:func:`~musicality.losses.beat_position.beat_position_loss`.

See ``docs/beat_phase_pos_weight_notes.md`` for the measurements.
"""

import torch

# Bounds on a *derived* pos_weight (see :func:`beat_pos_weight`). Neither end
# binds on real music: the slowest corpus in the collection sits at 12.5 and
# stays under 15 even after a 0.85 time-stretch. They exist for degenerate
# crops — a window holding a single beat derives ~200, and one holding none is
# bounded only by the epsilon in the denominator.
#
# The ceiling is 60 rather than 20 to leave room for sharp targets
# (``sigma_frames: 0``, paired with :mod:`musicality.losses.shift_tolerance`).
# Removing the Gaussian cuts the positive mass ~3.75x, so the derived ratio
# climbs to 13.9 on jtd, 22.1 on ballroom and 49.9 at rwc_classical's 10th
# percentile — a ceiling of 20 would bind on every corpus below ~135 BPM and
# silently undo the self-calibration. It stays a no-op under the smeared
# default, whose 12.5 maximum is the figure quoted just above.
AUTO_POS_WEIGHT_RANGE = (1.0, 60.0)

# Reproduces the hand-tuned pos_weight of 5 at ballroom's median tempo, where
# the derived neg:pos ratio is 4.51. Anchoring there makes self-calibration a
# pure *cross-tempo* change — neutral on the corpus every previous measurement
# was taken on, differing only where the tempo differs.
AUTO_POS_WEIGHT_ALPHA = 1.11


def beat_pos_weight(
    beat_y: torch.Tensor,
    pos_weight: torch.Tensor | float | str,
    alpha: float = AUTO_POS_WEIGHT_ALPHA,
    neg_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Positive-class weight for a beat BCE term — passed through, or derived
    per sample from the target when ``pos_weight`` is ``"auto"``.

    A fixed ``pos_weight`` is only correct at one tempo. The beat target is a
    Gaussian smeared to peak 1.0, so its mass per beat is a constant ~3.75
    frames regardless of tempo, while the beat *period* is not: 20.7 frames at
    ballroom's 125 BPM, 13.4 at jtd's 193, 46.1 at classical's 10th percentile
    of 56. The true neg:pos ratio therefore spans 2.6–11.3 across the corpora
    we train on, against a single configured 5.

    Since the ratio is just a function of the target already in hand, derive it
    rather than tune it:

    .. math::

        w_i = \alpha \, \frac{\bar{n}_i}{\bar{b}_i},
        \qquad \bar{b}_i = \frac{1}{T} \sum_t b_{i,t},
        \qquad \bar{n}_i = 1 - \bar{b}_i

    Deriving it per sample also means it tracks time-stretch augmentation,
    which silently invalidates a hand-tuned value on every augmented clip.

    :param beat_y: Beat target channel, shape ``(B, T)``, values in ``[0, 1]``.
    :param pos_weight: A number (or tensor) to pass through unchanged, or the
        string ``"auto"`` to derive one per sample.
    :param alpha: Scale on the derived ratio. Defaults to
        :data:`AUTO_POS_WEIGHT_ALPHA`; ``1.0`` is exact inverse-frequency
        weighting.
    :param neg_weight: Per-frame weight the *negative* term actually carries,
        shape ``(B, T)``, replacing :math:`\bar{n}_i` above. Only
        :func:`~musicality.losses.shift_tolerance.shift_tolerant_bce` passes
        it, because that loss ignores negatives near each annotation and the
        ratio has to count the frames that survive rather than every non-beat
        frame. On ballroom with sharp targets that is 13.2 against the 22.1 the
        raw target implies — a 1.7x difference the beat head would otherwise
        absorb as over-weighted positives. ``None`` uses ``1 - mean(beat_y)``,
        which is the plain BCE's ``(1 - y)``.
    :returns: Scalar tensor when passed through, shape ``(B, 1)`` when derived
        — which broadcasts against ``(B, T)`` inside
        :func:`~torch.nn.functional.binary_cross_entropy_with_logits`.
    """

    if not isinstance(pos_weight, str):
        return torch.as_tensor(pos_weight, device=beat_y.device, dtype=beat_y.dtype)

    if pos_weight != "auto":
        raise ValueError(
            f"Unknown pos_weight {pos_weight!r} — expected a number or 'auto'"
        )

    pos_frac = beat_y.mean(dim=-1, keepdim=True)  # (B, 1)
    neg_frac = (
        1.0 - pos_frac if neg_weight is None else neg_weight.mean(dim=-1, keepdim=True)
    )
    weight = alpha * neg_frac / pos_frac.clamp(min=1e-6)

    return weight.clamp(*AUTO_POS_WEIGHT_RANGE)
