"""Callback that scores a fixed slice of the validation set on *events* —
beats and bar positions on full-length tracks — during training.

Every number the training log has ever shown is a **frame** metric: how well
the model labels 23 ms frames of a 16-second clip taken from the middle of a
track. What the model is judged on is **events**: beats recovered from a
full-length track, and bars numbered across it. The two do not track each
other closely, and the gap runs in both directions —
``plans/06_metric_calibration_and_eval_consolidation.md`` measured frame
``acc_position`` at 0.661 against an event ``position_acc`` of 0.581 (the clip
is cut from the middle, deliberately avoiding intros, so the frame number is
measured on the easiest 16 seconds of every track), while frame ``acc_beat``
at a 70 ms window reads 0.959 against an ``f_beat`` of 0.845 (balanced
accuracy averages in a true-negative rate that is pinned near 1.0 on any sane
model).

So the event numbers had to be reconstructed after the fact, from a
checkpoint, with ``tools/eval_beat.py``. This logs them as the run goes, from
:meth:`~musicality.evaluation.BeatEvaluator.score` — the same scoring path the
CLI uses, so a number seen in W&B during training and the same number
recomputed afterwards cannot disagree.

Two things make it affordable. It runs every ``every_n_epochs`` epochs rather
than every epoch, and it runs on a fixed subsample of the validation split
rather than all of it. The subsample is **stratified across corpora**
(:func:`stratified_sample`): a split file is written corpus by corpus, so
taking the first N tracks of a merged split yields N tracks of whichever
corpus was written first.
"""

import math
import random

import lightning as L

from musicality.evaluation import BeatEvaluator, summarize
from musicality.loaders.beat_dataset import BeatDataset


PREFIX = "val_event"

# Logged as ``{PREFIX}/{key}``, a deliberate subset of
# :data:`musicality.evaluation.SCORE_KEYS`: the two beat numbers, the two
# position numbers, and nothing that is a difference of others
# (``anchor_error``) or kept only for continuity with the numbers recorded in
# plans/04 and plans/05 (``f_one``/``f_last``/``confusion``).
#
# Every key here must be higher-is-better, because
# :data:`musicality.callbacks.metrics_logger._LOWER_BETTER` decides direction
# by substring match and none of these contain "loss" or "mae". Adding
# ``confusion`` (an error rate) means adding it there too, or its "best" will
# be its worst.
LOGGED_KEYS = (
    "f_beat",
    "cmlt",
    "amlt",
    "position_acc",
    "position_acc_best_offset",
)


def stratified_sample(refs: list, n_tracks: int | None, seed: int = 0) -> list:
    """Pick *n_tracks* refs spread as evenly as possible across the corpora present.

    Round-robin over corpora — one track from each in turn — so a corpus
    contributing 7 tracks out of 249 is represented on the same footing as one
    contributing 104, and a small corpus running out simply drops out of later
    rounds instead of truncating the sample.

    Within a corpus the order is a fixed-seed shuffle rather than the split
    file's own order. Split files are grouped by corpus *and* often ordered
    within it (ballroom is written genre by genre), so taking the first few of
    each corpus can quietly mean taking one dance style.

    Deterministic in *seed*: the same refs and seed give the same tracks on
    every epoch, every restart and every machine, which is what makes the
    logged curve a curve rather than a walk over different test sets.

    :param refs: ``TrackRef`` entries, typically a split's validation half.
    :param n_tracks: How many to keep. ``None`` (or a count at or above
        ``len(refs)``) keeps all of them.
    :param seed: Seed for the per-corpus shuffle.
    :returns: The selected refs, interleaved by corpus.
    """

    if not refs:
        return []

    by_corpus: dict[str, list] = {}
    for ref in refs:
        by_corpus.setdefault(ref.dataset_name, []).append(ref)

    rng = random.Random(seed)
    pools = [rng.sample(group, len(group)) for group in by_corpus.values()]

    limit = len(refs) if n_tracks is None else min(n_tracks, len(refs))

    selected = []
    for depth in range(max(len(pool) for pool in pools)):
        for pool in pools:
            if depth < len(pool):
                selected.append(pool[depth])

                if len(selected) >= limit:
                    return selected

    return selected


class EventMetricsLogger(L.Callback):
    """Logs ``val_event/*`` metrics for a fixed validation subsample.

    Postprocessing knobs are deliberately *not* parameters: leaving them unset
    makes :meth:`~musicality.evaluation.BeatEvaluator.resolve_postprocess` fall
    back to the tuned defaults for the detected task in
    ``configs/eval_beat.yaml``, which is the same thing ``tools/eval_beat.py``
    does. Two config files disagreeing about the decoder is exactly how the
    numbers drifted apart before.

    :param refs: Validation ``TrackRef`` entries to sample from — the same
        refs the validation dataloader is built over (see
        :func:`~musicality.trainers.common.resolve_beat_split_refs`).
    :param n_tracks: Size of the fixed subsample; ``None`` scores every ref.
        Cost is roughly one full-track model pass per track, so this trades
        directly against how often it can run.
    :param every_n_epochs: Score every N epochs. Values below 1 are treated as
        1. The final epoch is always scored regardless, so the run ends with a
        fresh event number for the model it finished with.
    :param sample_rate: Audio sample rate; must match training.
    :param hop_length: Frame hop, in samples; must match training.
    :param group_size: Beats per group the position head predicts over.
    :param binary_only: Passed to :class:`~musicality.loaders.beat_dataset.BeatDataset`;
        must match how the split was built, or tracks are dropped here that the
        dataloader keeps.
    :param tolerance: Event matching window, in seconds.
    :param seed: Seed for :func:`stratified_sample`.
    :param name: Label for the dataset in report lines.
    """

    def __init__(
        self,
        refs: list,
        *,
        n_tracks: int | None = 50,
        every_n_epochs: int = 5,
        sample_rate: int = 22050,
        hop_length: int = 512,
        group_size: int = 4,
        binary_only: bool = False,
        tolerance: float = 0.07,
        seed: int = 0,
        name: str = "val",
    ):
        self.refs = stratified_sample(refs, n_tracks, seed=seed)
        self.every_n_epochs = max(1, every_n_epochs)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.group_size = group_size
        self.binary_only = binary_only
        self.tolerance = tolerance
        self.name = name

        self._dataset = None

    @property
    def dataset(self) -> BeatDataset:
        """The subsample as a dataset, built on first use.

        Built lazily so that constructing the callback — which happens before
        the trainer exists — does no annotation I/O, and so a run that never
        reaches a scoring epoch never pays for it.
        """

        if self._dataset is None:
            self._dataset = BeatDataset(
                refs=self.refs,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                group_size=self.group_size,
                binary_only=self.binary_only,
            )

        return self._dataset

    def should_run(self, trainer) -> bool:
        """Whether this epoch is a scoring epoch.

        Skips Lightning's pre-training sanity check (an untrained model, and
        the resulting point would sit on the curve as if it were epoch 0's
        real score) and every rank but zero, since the pass is duplicated work
        on every rank and :meth:`log` is rank-zero-only anyway.
        """

        if trainer.sanity_checking or not trainer.is_global_zero:
            return False

        max_epochs = getattr(trainer, "max_epochs", None)
        is_last = max_epochs is not None and trainer.current_epoch + 1 >= max_epochs

        return is_last or trainer.current_epoch % self.every_n_epochs == 0

    def score(self, pl_module) -> list[dict]:
        """Decode and score the subsample with the model as it stands now.

        A fresh :class:`~musicality.evaluation.BeatEvaluator` per call: its
        probability cache is per-instance and exists to make *several decoder
        settings* share one model pass, which is the opposite of what is needed
        here — the weights are different every epoch, so a reused evaluator
        would report epoch 0's numbers forever.
        """

        evaluator = BeatEvaluator.from_module(
            pl_module,
            self.dataset,
            name=self.name,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            group_size=self.group_size,
            tolerance=self.tolerance,
            device=pl_module.device,
            verbose=False,
        )

        # Lightning has already switched to eval mode for the validation loop,
        # but this is cheap and the failure it prevents (dropout active during
        # a full-track decode) is silent.
        was_training = pl_module.training
        pl_module.eval()
        try:
            return evaluator.score()
        finally:
            if was_training:
                pl_module.train()

    def on_validation_epoch_end(self, trainer, pl_module):

        if not self.should_run(trainer):
            return

        print(
            f"[event metrics] epoch {trainer.current_epoch}: "
            f"decoding {len(self.dataset)} full track(s)..."
        )

        summary = summarize(self.score(pl_module))

        for key in LOGGED_KEYS:
            value = summary.get(key)

            # NaN means nothing was scorable — a beat-only split has no bar
            # positions to get right. Logging it would put a gap in the W&B
            # chart that reads as a failed epoch rather than an absent metric.
            if value is None or math.isnan(value):
                continue

            pl_module.log(f"{PREFIX}/{key}", float(value), rank_zero_only=True)
