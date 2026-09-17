Baselines
=========

Third-party beat trackers, run over our own splits and scored by our own
metrics — the experiment ``plans/08_rethinking_the_approach.md`` §6.1 asks for.

Our downbeat F-measure sits roughly 20 points below what the beat-tracking
literature reports, and that gap has two explanations no amount of tuning can
separate: our *model* is behind the field, or our *data and annotations* are
harder than theirs. Running a published tracker over the same held-out tracks
decides it in one pass. If a state-of-the-art system also struggles on a
corpus, the corpus is hard; if it sails through, the model is the problem.

Everything after the prediction is shared with ``tools/eval_beat.py``: the
split comes from :func:`~musicality.evaluation.build_eval_dataset`, the scoring
from :func:`~musicality.evaluation.score_events`, the report from
:func:`~musicality.evaluation.summary_block`. A baseline row and a checkpoint
row in the same table differ in exactly one thing — who produced the beats.

.. warning::

   **Most of our validation split is in both trackers' training data.** Beat
   This!'s ``final*`` checkpoints held out GTZAN and nothing else; madmom's
   shipped weights were fitted on Ballroom and RWC Popular among others. On
   those corpora a baseline is recalling tracks it was trained on while our
   model is generalising, and averaging the two into a single "baseline beats
   us by N" headline is the easiest wrong conclusion available here.

   ``gtzan`` (held out by both), ``rwc_genre`` and ``jtd`` are the rows that
   mean something. :data:`~musicality.baselines.base.CORPUS_EXPOSURE` is the
   table, and ``tools/eval_baseline.py`` prints the affected corpora before the
   numbers.

.. note::

   madmom's *code* is BSD-licensed, but the pretrained model files it downloads
   with are **CC BY-NC-SA 4.0 — non-commercial use only**. That is fine for
   benchmarking and for research, and it is a constraint to know about before
   madmom ends up anywhere near a shipped product.

Installation
------------

Neither tracker is a default dependency::

    uv sync --extra baselines

madmom is pinned to its git ``main``: the PyPI release (0.16.1, 2018) does not
import on Python 3.10+, and its Cython extensions must be built against the
same numpy the project runs, which ``[tool.uv.extra-build-dependencies]`` in
``pyproject.toml`` arranges. Beat This! downloads its checkpoint (~78 MB) on
first use and caches it under ``torch.hub``.

Running one
-----------

.. code-block:: bash

    # the canonical report, per genre
    uv run python tools/eval_baseline.py --baseline madmom \
        --dataset merge --split val --binary-only

    # Beat This!, rows saved for later comparison
    uv run python tools/eval_baseline.py --baseline beat_this \
        --dataset merge --split val --binary-only \
        --output outputs/baselines/beat_this.csv

    # the clean comparison: a corpus neither tracker was trained on
    uv run python tools/eval_baseline.py --baseline beat_this --dataset gtzan

    # same frame probabilities, madmom's DBN instead of peak-picking
    uv run python tools/eval_baseline.py --baseline beat_this --dbn

    # fill the cache now, score later — or on another machine entirely
    uv run python tools/eval_baseline.py --baseline madmom --predict-only
    uv run python tools/eval_baseline.py --baseline madmom --from-cache

Predictions are cached per track under ``outputs/baselines/``, so the slow half
— running the tracker — happens once and the report can be re-cut at a
different tolerance, ``group_size`` or genre breakdown for free. The cache
header records the baseline's configuration and a mismatched one is ignored
rather than silently reused.

Because a cache holds nothing but times in seconds, the tracker does not have to
run where it is scored. Run it wherever is convenient — a GPU box, a machine
whose environment this project would rather not carry — copy the JSON in, and
score it with ``--from-cache``, which adopts the file's own header and never
constructs the tracker.

API
---

.. autosummary::
   :toctree: generated
   :recursive:

   ~musicality.baselines
   ~musicality.baselines.base
   ~musicality.baselines.cache
   ~musicality.baselines.evaluator
   ~musicality.baselines.madmom_baseline
   ~musicality.baselines.beat_this_baseline
