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

The three modules that are not losses themselves hold the knobs both beat
objectives share: which frames the position term is supervised on
(:mod:`~musicality.losses.phase_conditioning`), how the beat term's positive
class is weighted (:mod:`~musicality.losses.pos_weight`), and how much timing
error the beat term forgives (:mod:`~musicality.losses.shift_tolerance`).

All three are coupled, and in a chain. Conditioning on beats removes most of
the imbalance ``pos_weight`` exists to correct. Shift tolerance then replaces
the target smearing that defines where "a beat" is at all, which changes the
gate ``phase_conditioning`` builds *and* the imbalance ``pos_weight`` measures
— a sharp target cuts the positive mass ~3.75x, and the ignore band removes
negatives that the weight would otherwise count. Change one and re-read the
other two.

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
   ~musicality.losses.shift_tolerance
