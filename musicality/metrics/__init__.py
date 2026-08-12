"""Event-level evaluation for beat-phase detection: beat / "1" / "4" F-measure
and a "1-vs-3" phase-confusion rate.

Consumes the labeled beat list produced by :func:`musicality.postprocess.readout`
(or raw reference annotations) — this is the "does the final output line up with
the ground truth" evaluation, distinct from :func:`musicality.trainers.beat_phase_module.frame_accuracy`,
which is a cheap per-epoch training signal on raw frame probabilities.
"""

from .confusion import confusion_half_cycle_rate
from .f_measure import beat_f_measure, downbeat_f_measures

__all__ = ["beat_f_measure", "downbeat_f_measures", "confusion_half_cycle_rate"]
