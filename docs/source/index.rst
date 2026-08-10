musicality
==========

API reference for the ``musicality`` tempo-estimation library.

Losses
------

.. autosummary::
   :toctree: generated
   :recursive:

   musicality.losses

Metrics
-------

.. autosummary::
   :toctree: generated
   :recursive:

   musicality.metrics

Loaders
-------

.. autosummary::
   :toctree: generated
   :recursive:

   musicality.loaders.tempo_dataset
   musicality.loaders.beat_dataset

Models
------

.. autosummary::
   :toctree: generated
   :recursive:

   musicality.models.tcn
   musicality.models.tempo_net
   musicality.models.huggingface
   musicality.models.torch_audio

Trainers
--------

.. autosummary::
   :toctree: generated
   :recursive:

   musicality.trainers.tempo_module
   musicality.trainers.beat_phase_module
   musicality.trainers.train
   musicality.trainers.train_beat_phase

Callbacks
---------

.. autosummary::
   :toctree: generated
   :recursive:

   musicality.callbacks.error_plot
   musicality.callbacks.metrics_logger

Data formats & splits
----------------------

.. autosummary::
   :toctree: generated
   :recursive:

   musicality.dataformats
   musicality.splits.splitter

Other
-----

.. autosummary::
   :toctree: generated
   :recursive:

   musicality.augmentations
   musicality.postprocess
