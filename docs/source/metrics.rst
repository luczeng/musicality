Metrics
=======

Two families live here and they do not agree with each other:
:func:`~musicality.metrics.frame_accuracy.frame_accuracy` scores the raw
per-frame probability curve, while everything else scores a *decoded event
list*. The disagreement is large, runs in both directions, and has four
independent causes — ``docs/frame_vs_event_metrics.md`` is the measured
account, and the reference for which numbers to quote.

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.metrics.f_measure
   ~musicality.metrics.continuity
   ~musicality.metrics.confusion
   ~musicality.metrics.position_accuracy
   ~musicality.metrics.frame_accuracy
   ~musicality.metrics.tempo_acc1
