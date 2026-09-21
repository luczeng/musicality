Losses
======

One module per objective, named after the task it trains. Three families live
here: tempo as a number (regression) or a distribution over bins
(classification), and the frame-level beat objectives — beat alone, or beat
plus bar position.

The two bar-position losses differ in how they model position:
:func:`~musicality.losses.beat_phase.beat_phase_loss` uses two independent
sigmoids (``one``/``last``), which never asks the discriminative "1 or 3?"
question, and :func:`~musicality.losses.beat_position.beat_position_loss`
replaces them with a softmax over all ``G`` positions. The latter is what the
project trains with; the former is kept so existing checkpoints stay readable.

The two modules that are not losses themselves hold the knobs both beat
objectives share — which frames the position term is supervised on
(:mod:`~musicality.losses.phase_conditioning`) and how the beat term's
positive class is weighted (:mod:`~musicality.losses.pos_weight`). They are
coupled: conditioning on beats removes most of the imbalance ``pos_weight``
exists to correct.

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.losses.tempo_regression
   ~musicality.losses.tempo_classification
   ~musicality.losses.beat_only
   ~musicality.losses.beat_phase
   ~musicality.losses.beat_position
   ~musicality.losses.phase_conditioning
   ~musicality.losses.pos_weight
