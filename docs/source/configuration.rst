Training configuration
=======================

Hydra config files for the two training entry points (``tools/train.py`` and
``tools/train_beat.py``). Every key is commented in place below — that's the
source of truth, not this page — and any of them can be overridden on the
command line, e.g. ``uv run python tools/train.py lr=3e-4``.

Tempo training (``configs/train.yaml``)
-----------------------------------------

.. literalinclude:: ../../configs/train.yaml
   :language: yaml

Beat-only training (``configs/beat_only_train.yaml``)
-----------------------------------------------------

.. literalinclude:: ../../configs/beat_only_train.yaml
   :language: yaml

Beat-phase training (``configs/beat_train.yaml``)
-----------------------------------------------------

.. literalinclude:: ../../configs/beat_train.yaml
   :language: yaml
