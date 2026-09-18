#!/usr/bin/env python3
"""Score trained runs on full tracks and keep the comparison as one running board.

Training already reports on itself — every run writes a `training_report.json`
beside its checkpoints. What it cannot do is compare runs: each report scores
its own run, on whatever split and postprocessing that run was configured with,
at whatever epoch it happened to stop. Ranking four architectures against each
other means re-scoring all of them the same way, afterwards.

That is this tool. Point it at the checkpoint directories of whatever is new;
every run it finds is evaluated on one common split, with its postprocessing
knobs swept first (the shipped `beat_phase` knobs are marked UNVERIFIED in
`configs/eval_beat.yaml`, and a sweep has been worth more than a retrain), and
added to the board beside the runs already measured. The board is one JSON file
holding every number — per-run metrics, the knobs each was scored at, the
per-corpus breakdown, and a rendered table under `readable` — so a whole
comparison travels as a single attachment.

The sweep runs on the **train** split by default and the report on val, so the
knobs are not chosen on the tracks they are then scored on. `--sweep-split val`
restores the older behaviour, which is what `tools/eval_beat.py --sweep` does
and what makes its numbers optimistic.

Usage
-----
    # first time: two experiment folders, swept, on the split
    # configs/eval_beat.yaml names (merge, binary meter only, val)
    uv run python tools/leaderboard.py checkpoints_deeper checkpoints_norm

    # every time after: name only what is new
    uv run python tools/leaderboard.py checkpoints_new

    # a single checkpoint, no sweep (score at the config's shipped knobs)
    uv run python tools/leaderboard.py checkpoints/merge_v5.ckpt --no-sweep

    # tune the knobs on val too — faster (no second model pass), but the
    # reported numbers are then no longer held out
    uv run python tools/leaderboard.py checkpoints_deeper --sweep-split val

    # a one-off board of exactly these runs, extending and pushing nothing
    uv run python tools/leaderboard.py checkpoints_deeper --no-append

The running board
-----------------
The board lives in the DVC-tracked data repo (`musicality_db/leaderboard/`),
beside the splits. It is pulled before reading and pushed after writing, so a
run on a rented instance sees every experiment measured so far and hands its own
result back — the board outlives the machine, which is the whole point of it
being a running one. Only the `.dvc` pointer needs committing, and that is left
to whoever is at the keyboard.

Rows for runs not named on the command line are carried over; a run that *is*
named is re-measured and replaces its old row. Identity is the run label, not
the checkpoint filename, since more training on the same folder is the same
experiment with a better number.

Rows measured under different conditions are refused rather than merged (see
`COMPARABLE_KEYS`), before any model pass. Re-measuring every run in the old
board lifts that, which is how the split or the tolerance gets changed: name all
the folders once.

Which checkpoint per run
------------------------
A directory whose checkpoints carry a `valloss` in their filename is one run's
`save_top_k` group, and the best-scoring file represents it. A directory of
hand-named checkpoints (`merge_v5.ckpt`, `checkpoint_v6.ckpt`, ...) is instead
one entry per file — those are different models that happen to share a folder.
"""

import argparse
import itertools
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import musicality.dataformats as dataformats
from musicality.callbacks.event_metrics import stratified_sample
from musicality.callbacks.training_report import git_commit, jsonable
from musicality.evaluation import (
    DATA_DIR,
    SCORE_KEYS,
    BeatEvaluator,
    _fmt,
    group_by_corpus,
    summarize,
)
from musicality.evaluation import DEFAULTS as EVAL_DEFAULTS
from tools.eval_beat import (
    _BETTER,
    _LABELS,
    _RANK_CHOICES,
    rank_key,
    resolve_group_size,
    sweep_grid,
)

SCHEMA = 1
FILENAME = "leaderboard.json"

# The running board, kept between invocations unless --append names another or
# --no-append asks for a standalone one. A default rather than a flag to
# remember: a board is only useful once it has more than one row on it.
#
# It lives in the DVC-tracked data repo, beside the splits, rather than in this
# checkout. A board that accumulates over months has to outlive the machine
# that wrote it, and training runs on rented instances that are torn down; a
# path under `musicality_db` is pulled onto a fresh instance and pushed back
# the same way the splits and datasets already are.
DEFAULT_BOARD = dataformats.LEADERBOARD_DIR / FILENAME

# Where a --no-append one-off lands. Local and gitignored: a standalone board is
# a scratch comparison, not something to publish.
SCRATCH_DIR = Path("leaderboards")

SWEEP_DEFAULTS = EVAL_DEFAULTS["sweep"]

# The postprocessing knobs carried on every row, so a board says which decode
# produced its numbers — a swept row and a default row are not comparable
# otherwise.
KNOB_KEYS = (
    "beat_threshold",
    "min_distance_frames",
    "gate_tolerance",
    "decoder",
    "switch_penalty",
    "anchor_threshold",
)

# What has to match for a row measured by an earlier invocation to sit beside
# one measured now: the tracks it was scored on, and how a match was counted.
# Deliberately not the sweep settings — those are recorded per row (`swept_on`,
# plus the knobs themselves), so a swept and an unswept row are told apart by
# reading them rather than by forbidding the combination.
COMPARABLE_KEYS = (
    "dataset",
    "split",
    "val_split",
    "sample_rate",
    "hop_length",
    "tolerance",
    "binary_only",
    "group_size",
    "limit",
)

# Columns of the printed board. Narrow on purpose: the file and the W&B table
# carry every metric, this has to stay readable in a terminal.
BOARD_COLUMNS = (
    "f_beat",
    "cmlt",
    "amlt",
    "position_acc",
    "position_acc_best_offset",
    "confusion",
)

_VALLOSS = re.compile(r"valloss([0-9]*\.?[0-9]+)")


def run_checkpoints(run_dir: Path) -> list[tuple[str, Path]]:
    """The ``(label, checkpoint)`` pairs one directory contributes.

    ``save_top_k`` leaves several checkpoints of the *same* run side by side,
    each named with the ``val/loss`` it scored; there, the best-scoring file is
    the run and the other two are history. A directory whose checkpoint names
    carry no loss is read the other way — one entry per file — because that is
    how hand-named checkpoints (`merge_v5.ckpt`) sit in `checkpoints/`, and
    collapsing those to one would silently drop five models.
    """

    checkpoints = sorted(run_dir.glob("*.ckpt"))
    scored = [(match, c) for c in checkpoints if (match := _VALLOSS.search(c.name))]

    if scored:
        best = min(scored, key=lambda pair: float(pair[0].group(1)))[1]

        return [(str(run_dir), best)]

    return [(str(c.with_suffix("")), c) for c in checkpoints]


def find_runs(paths: list[Path]) -> list[tuple[str, Path]]:
    """Every run to score, as ``(label, checkpoint)``, in a stable order.

    A ``.ckpt`` path is taken as given. A directory is walked recursively, and
    each directory holding checkpoints resolved by :func:`run_checkpoints` —
    so a sweep directory (one subdirectory per learning rate) expands to one
    entry per learning rate without needing to be named individually.
    """

    runs = []
    for path in paths:
        if path.suffix == ".ckpt":
            runs.append((str(path.with_suffix("")), path))
            continue

        for run_dir in sorted({c.parent for c in path.rglob("*.ckpt")}):
            runs.extend(run_checkpoints(run_dir))

    return runs


def sweep_evaluator(checkpoint: Path, args, group_size: int) -> BeatEvaluator:
    """A second evaluator, over the split the knobs are tuned on.

    Separate from the reporting one on purpose. Knobs chosen on the same tracks
    they are then scored on are chosen partly for the noise in those tracks,
    and the board comes back optimistic — by an amount that differs per row,
    since 60-odd grid points against ~50 tracks give more room to whichever
    checkpoint's probability curves happen to suit some threshold. Tuning on
    train and reporting on val keeps every reported number held out.

    The tracks are a corpus-stratified subsample (:func:`stratified_sample`,
    sized by ``--sweep-tracks``): the train split is several times the size of
    the val one, and the sweep's model pass would otherwise cost more than the
    evaluation it exists to configure. Stratified rather than the first N,
    because a split file is written corpus by corpus — ``--limit`` on a merged
    split yields N tracks of whichever corpus was written first.
    """

    evaluator = BeatEvaluator(
        checkpoint=checkpoint,
        dataset=args.dataset,
        data_home=args.data_home,
        split=args.sweep_split,
        val_split=args.val_split,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        group_size=group_size,
        binary_only=args.binary_only,
        tolerance=args.tolerance,
        device=args.device,
        verbose=False,
    )

    module, task, dataset, indices = evaluator.load()

    keep = {
        (ref.dataset_name, ref.track_id)
        for ref in stratified_sample(
            [dataset.refs[i] for i in indices], args.sweep_tracks
        )
    }

    # Narrowing the loaded indices rather than the dataset: `BeatEvaluator`
    # memoizes both together, and everything downstream walks `indices`.
    evaluator._loaded = (
        module,
        task,
        dataset,
        [
            i
            for i in indices
            if (dataset.refs[i].dataset_name, dataset.refs[i].track_id) in keep
        ],
    )

    return evaluator


def sweep_knobs(evaluator: BeatEvaluator, task: str, group_size: int, args) -> dict:
    """Best postprocessing knobs for one checkpoint.

    The same two stages ``tools/eval_beat.py --sweep`` runs, and through the
    same :func:`~tools.eval_beat.sweep_grid`: beat detection first, ranked by
    ``f_beat`` because that is all those knobs can move, then the one
    bar-position knob the resolved decoder actually reads, on top of the
    winner. Both stages re-use the cached frame probabilities, so a sweep costs
    one model pass per checkpoint, not one per grid point.

    **The two stages rank by different metrics, and must.** A bar-position
    decoder relabels beats; it cannot move them. Every stage-2 candidate
    therefore scores the *same* ``f_beat`` to the last decimal, so ranking
    stage 2 by it is a tie that the sort resolves by candidate order — which
    silently pins ``switch_penalty`` to the first value in the list. Stage 2
    ranks by ``--sweep-rank-metric`` (a bar-position metric) instead; only the
    board's own ordering reads ``--rank-metric``.
    """

    beat_grid = [
        {"beat_threshold": bt, "min_distance_frames": md, "gate_tolerance": gt}
        for bt, md, gt in itertools.product(
            SWEEP_DEFAULTS["beat_thresholds"],
            SWEEP_DEFAULTS["min_distance_frames"],
            SWEEP_DEFAULTS["gate_tolerances"],
        )
    ]

    ranked = sweep_grid(
        evaluator,
        beat_grid,
        group_size=group_size,
        metric="f_beat",
        rank_by=args.rank_by,
    )
    best = {key: ranked[0][key] for key in beat_grid[0]}

    if task != "beat_phase":
        return best

    decoder = evaluator.resolve_postprocess()["decoder"]

    if decoder == "greedy":
        knob, values = "anchor_threshold", SWEEP_DEFAULTS["anchor_thresholds"]
    else:
        # `None` is a real switch_penalty — the exact single-offset decode, the
        # penalty -> infinity limit that no finite value reaches.
        knob, values = "switch_penalty", [None, *SWEEP_DEFAULTS["switch_penalties"]]

    ranked = sweep_grid(
        evaluator,
        [{**best, knob: value} for value in values],
        group_size=group_size,
        metric=args.sweep_rank_metric,
        rank_by=args.rank_by,
    )

    return {**best, knob: ranked[0][knob]}


def evaluate_run(label: str, checkpoint: Path, args) -> dict:
    """Score one checkpoint, returning its board row and per-corpus breakdown."""

    evaluator = BeatEvaluator(
        checkpoint=checkpoint,
        dataset=args.dataset,
        data_home=args.data_home,
        split=args.split,
        val_split=args.val_split,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        group_size=args.group_size,
        binary_only=args.binary_only,
        tolerance=args.tolerance,
        limit=args.limit,
        device=args.device,
        verbose=False,
    )

    _module, task, _dataset, indices = evaluator.load()
    group_size = resolve_group_size(
        evaluator, args.group_size if args.group_size is not None else 4
    )

    print(f"    task={task}  group_size={group_size}  tracks={len(indices)}")

    evaluator.compute_track_probs()

    knobs, swept_on, n_sweep = {}, None, 0
    if args.sweep:
        # Same split means the same tracks, so reuse the evaluator whose
        # probabilities are already cached rather than paying for them twice.
        tuner = (
            evaluator
            if args.sweep_split == args.split
            else sweep_evaluator(checkpoint, args, group_size)
        )
        swept_on = args.sweep_split
        n_sweep = len(tuner.load()[3])

        print(f"    sweeping on {swept_on} ({n_sweep} track(s))")

        knobs = sweep_knobs(tuner, task, group_size, args)

    rows = evaluator.score(group_size=group_size, **knobs)
    summary = summarize(rows)
    resolved = evaluator.resolve_postprocess(group_size=group_size, **knobs)

    row = {
        "run": label,
        "checkpoint": str(checkpoint),
        "task": task,
        # Per row, not just per board: a running board carries rows measured on
        # different days by different code, and the row has to say which.
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": args.commit,
        "swept_on": swept_on,
        "sweep_n_tracks": n_sweep,
        "n_tracks": summary["n_tracks"],
        **{key: summary[key] for key in SCORE_KEYS},
        **{f"macro_{key}": summary[f"macro_{key}"] for key in SCORE_KEYS},
        "worst_corpus": summary["worst_corpus"],
        "worst_position_acc": summary["worst_position_acc"],
        **{key: resolved[key] for key in KNOB_KEYS},
    }

    per_corpus = {
        corpus: {"n_tracks": len(group), **{k: summarize(group)[k] for k in SCORE_KEYS}}
        for corpus, group in group_by_corpus(rows).items()
    }

    return {"row": row, "per_corpus": per_corpus}


def board_name(path: Path) -> str:
    """The DVC target for a board at *path* — the folder holding it.

    DVC tracks the directory (``leaderboard.dvc``, beside ``splits.dvc``), not
    the file, so both pull and add name the folder.
    """

    return path.parent.name


def dvc(command: list[str], path: Path) -> bool:
    """Run one ``dvc`` command in the data repo holding *path*.

    Never fatal. The remote needs credentials this checkout may not have, and a
    board that cannot be synced is still a board — the run reports what failed
    and carries on with what is on disk.
    """

    cwd = path.parent.parent.resolve()
    if not (cwd / ".dvc").is_dir():
        print(f"[leaderboard] {cwd} is not a DVC repo — skipping dvc {command[0]}")
        return False

    print(f"[leaderboard] dvc {' '.join(command)} (in {cwd})")
    result = subprocess.run(["dvc", *command], cwd=cwd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[leaderboard] dvc {command[0]} failed: {result.stderr.strip()}")
        return False

    return True


def pull_board(path: Path) -> None:
    """Fetch the board from the DVC remote before reading it.

    Skipped when the pointer does not exist yet — the first invocation is what
    creates the board, and running a doomed ``dvc pull`` only to print its error
    reads like a failure when nothing is wrong.
    """

    name = board_name(path)

    if not (path.parent.parent / f"{name}.dvc").exists():
        print(f"[leaderboard] no {name}.dvc yet — starting a new board")
        return

    dvc(["pull", name], path)


def publish(path: Path) -> None:
    """Re-hash the board folder and upload it, then say what to commit.

    ``dvc add`` rewrites the ``.dvc`` pointer in the data repo and ``dvc push``
    uploads the content, but the pointer only becomes the shared truth once it
    is committed — and committing in someone else's repo is not this tool's
    call to make.
    """

    name = board_name(path)
    repo = path.parent.parent.resolve()

    if not dvc(["add", name], path):
        return
    if not dvc(["push", name], path):
        return

    print(
        f"\n[leaderboard] pushed. To share it:\n"
        f"    cd {repo} && git add {name}.dvc && git commit -m 'Update leaderboard'"
    )


def load_board(path: Path) -> dict:
    """The board already at *path*, or an empty one when there is none yet.

    A missing file is not an error: it makes the first invocation of a running
    board the same command as every later one.
    """

    if path is None or not path.exists():
        return {"leaderboard": [], "per_corpus": {}, "eval": {}}

    previous = json.loads(path.read_text())
    print(
        f"[leaderboard] {len(previous.get('leaderboard', []))} row(s) loaded "
        f"from {path}"
    )

    return previous


def carried_rows(previous: dict, rows: list[dict]) -> list[dict]:
    """Rows of *previous* that this invocation did not re-measure.

    Identity is the ``run`` label, not the checkpoint path: re-running a folder
    after more training picks a different epoch's file, and that is the same
    experiment with a better number, not a second entry on the board.
    """

    measured = {row["run"] for row in rows}

    return [
        row for row in previous.get("leaderboard", []) if row["run"] not in measured
    ]


def check_comparable(previous: dict, current: dict, carried: list[dict]) -> None:
    """Refuse to put rows measured under different conditions on one board.

    A leaderboard is a claim that its rows can be read against each other, and
    a row scored on a different split or at a different tolerance cannot. Only
    *carried* rows are at stake — re-measuring every run in the old board makes
    the old settings irrelevant, which is what lets the split be changed without
    a flag to override this.
    """

    if not carried or not previous:
        return

    differing = {
        key: (previous.get(key), current.get(key))
        for key in COMPARABLE_KEYS
        if previous.get(key) != current.get(key)
    }
    if not differing:
        return

    changes = "\n".join(
        f"    {key}: {was!r} -> {now!r}" for key, (was, now) in differing.items()
    )
    raise SystemExit(
        f"[leaderboard] cannot append: {len(carried)} row(s) in the existing "
        f"board were measured under different settings:\n{changes}\n"
        "    Re-measure those runs too (name their folders as well), or append "
        "to a different file."
    )


def warn_on_code_drift(carried: list[dict], commit: str | None) -> None:
    """Note carried rows produced by a different commit.

    Not fatal — most commits do not touch scoring — but a metric that moved
    between them would show up on the board as a model improvement.
    """

    stale = {row.get("git_commit") for row in carried} - {commit, None}
    if stale:
        print(
            f"[leaderboard] note: {len(carried)} carried row(s) were measured at "
            f"{', '.join(sorted(c[:8] for c in stale))}, this invocation is at "
            f"{(commit or 'unknown')[:8]}"
        )


def rank_rows(rows: list[dict], metric: str, rank_by: str) -> list[dict]:
    """Best first, in *metric*'s own better-is direction.

    A row that could not be scored under this metric (``None`` after
    :func:`~musicality.callbacks.training_report.jsonable` turned its NaN into
    one) sorts last rather than winning.
    """

    key = rank_key(metric, rank_by)
    higher_is_better = _BETTER[metric] is max

    def _rank(row: dict) -> float:
        value = row.get(key)
        if value is None:
            return math.inf

        return -value if higher_is_better else value

    return sorted(rows, key=_rank)


def render_board(rows: list[dict]) -> str:
    """The ranked table, as printed and as stored in the file's ``readable``."""

    width = max((len(row["run"]) for row in rows), default=3)
    header = f"{'run':<{width}}  {'n':>4}" + "".join(
        f" {_LABELS[c]:>9}" for c in BOARD_COLUMNS
    )

    lines = [header]
    for row in rows:
        lines.append(
            f"{row['run']:<{width}}  {row.get('n_tracks', 0):>4}"
            + "".join(f" {_fmt(row.get(c)):>9}" for c in BOARD_COLUMNS)
        )

    return "\n".join(lines)


def board_settings(args) -> dict:
    """The conditions every row on this board was measured under.

    One function because two callers must agree on it: it is stored in the file
    as ``eval``, and :func:`check_comparable` reads that back to decide whether
    an older board's rows may sit beside today's.
    """

    return {
        "dataset": args.dataset,
        "split": args.split,
        "val_split": args.val_split,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "tolerance": args.tolerance,
        "binary_only": args.binary_only,
        "group_size": args.group_size,
        "limit": args.limit,
        "swept": args.sweep,
        "sweep_split": args.sweep_split if args.sweep else None,
        "sweep_tracks": args.sweep_tracks
        if args.sweep and args.sweep_split != args.split
        else None,
        "ranked_by": rank_key(args.rank_metric, args.rank_by),
        "sweep_ranked_by": rank_key(args.sweep_rank_metric, args.rank_by)
        if args.sweep
        else None,
    }


def build_payload(rows: list[dict], per_corpus: dict, failed: dict, args) -> dict:
    """The whole leaderboard as one JSON-safe dict.

    Deliberately machine-readable, the way ``training_report.json`` is: NaN
    written as ``null`` so a strict parser accepts it, with a rendered board in
    ``readable`` so opening the file still shows something skimmable.
    """

    return jsonable(
        {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": args.commit,
            "eval": board_settings(args),
            "readable": render_board(rows) if rows else "nothing scored",
            "leaderboard": rows,
            "per_corpus": per_corpus,
            "failed": failed,
        }
    )


def write_payload(payload: dict, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

    print(f"\n[leaderboard] written to {path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every trained run in one or more checkpoint directories "
            "and publish the ranked comparison to W&B as a leaderboard."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Checkpoint directories (walked recursively) and/or .ckpt files",
    )

    data = parser.add_argument_group("evaluation split")
    data.add_argument(
        "--dataset",
        default=EVAL_DEFAULTS["dataset"],
        help="Split to evaluate on, default from configs/eval_beat.yaml",
    )
    data.add_argument(
        "--data-home", default=None, help=f"Defaults to {DATA_DIR}/<dataset>"
    )
    data.add_argument(
        "--split", choices=["train", "val", "all"], default=EVAL_DEFAULTS["split"]
    )
    data.add_argument("--val-split", type=float, default=EVAL_DEFAULTS["val_split"])
    data.add_argument("--sample-rate", type=int, default=EVAL_DEFAULTS["sample_rate"])
    data.add_argument("--hop-length", type=int, default=EVAL_DEFAULTS["hop_length"])
    data.add_argument("--group-size", type=int, default=None)
    data.add_argument(
        "--binary-only",
        action=argparse.BooleanOptionalAction,
        default=EVAL_DEFAULTS["binary_only"],
        help="Must match how the split was created",
    )
    data.add_argument("--tolerance", type=float, default=EVAL_DEFAULTS["tolerance"])
    data.add_argument(
        "--limit", type=int, default=None, help="Score only the first N tracks"
    )
    data.add_argument("--device", default=EVAL_DEFAULTS["device"])

    ranking = parser.add_argument_group("ranking")
    ranking.add_argument(
        "--no-sweep",
        dest="sweep",
        action="store_false",
        help=(
            "Score at the knobs configs/eval_beat.yaml ships instead of "
            "sweeping each checkpoint's own. Faster, and comparable to whatever "
            "was reported at training time — but the beat_phase knobs there are "
            "marked UNVERIFIED, so it usually understates every row."
        ),
    )
    ranking.add_argument(
        "--sweep-split",
        choices=["train", "val", "all"],
        default="train",
        help=(
            "Which split the knobs are tuned on. Defaults to train, so the "
            "reported val numbers stay held out — knobs chosen on the tracks "
            "they are then scored on make every row optimistic. Set it equal "
            "to --split to tune and report on the same tracks (what "
            "tools/eval_beat.py --sweep does), which skips the second model "
            "pass."
        ),
    )
    ranking.add_argument(
        "--sweep-tracks",
        type=int,
        default=50,
        help=(
            "Corpus-stratified subsample of the sweep split to tune against. "
            "The train split is much larger than val, and this pass is pure "
            "overhead on top of the evaluation."
        ),
    )
    ranking.add_argument(
        "--rank-metric",
        choices=SCORE_KEYS,
        default="f_beat",
        help="Which metric orders the board",
    )
    ranking.add_argument(
        "--sweep-rank-metric",
        choices=_RANK_CHOICES,
        default="position_acc",
        help=(
            "Which metric picks the winner of the sweep's bar-position stage. "
            "Must be a bar-position metric: a decoder relabels beats without "
            "moving them, so every candidate ties on f_beat and ranking by it "
            "would pick whichever came first in the list. Beat detection "
            "(stage 1) always ranks by f_beat, the only thing its knobs move."
        ),
    )
    ranking.add_argument(
        "--rank-by",
        choices=["macro", "micro"],
        default="macro",
        help=(
            "Which mean ranks on a multi-corpus split: 'macro' weights each "
            "corpus equally, 'micro' each track. Identical on one corpus."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--append",
        type=Path,
        default=DEFAULT_BOARD,
        help=(
            f"The running board to extend, default {DEFAULT_BOARD.name} in the "
            "DVC-tracked leaderboard folder. Rows for runs not named on this "
            "command line are carried over from it, and the merged board is "
            "written back there. A path that does not exist yet starts a new "
            "board, so the first invocation is the same command as every later "
            "one."
        ),
    )
    output.add_argument(
        "--no-append",
        dest="append",
        action="store_const",
        const=None,
        help=(
            "Score into a standalone board instead of extending the running "
            "one — a one-off comparison of exactly the runs named here, written "
            f"under {SCRATCH_DIR}/ and never pushed."
        ),
    )
    output.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write elsewhere than the board being extended",
    )
    output.add_argument(
        "--no-pull",
        dest="pull",
        action="store_false",
        help=(
            "Skip the `dvc pull` of the leaderboard folder. Without it a fresh "
            "machine starts an empty board and silently drops every run already "
            "measured."
        ),
    )
    output.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help=(
            "Write the board locally without `dvc add` + `dvc push`. The board "
            "then exists only on this machine until pushed by hand."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.commit = git_commit()  # once per invocation, stamped onto every row

    if args.pull and args.append is not None:
        pull_board(args.append)

    previous = load_board(args.append)

    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"[leaderboard] no .ckpt found under {args.runs}")

    if not args.sweep:
        sweeping = "off"
    elif args.sweep_split == args.split:
        # Same split, same tracks: the subsample never applies, and saying
        # otherwise would advertise a hold-out that is not there.
        sweeping = f"on {args.sweep_split} — the reported tracks, not held out"
    else:
        sweeping = f"on {args.sweep_split} ({args.sweep_tracks} tracks)"

    print(
        f"[leaderboard] {len(runs)} run(s)  dataset={args.dataset}  "
        f"split={args.split}  sweep={sweeping}"
    )

    # Before the model passes, not after: an incomparable append should cost
    # nothing. Checked again below against the rows actually measured, since a
    # checkpoint that fails to load leaves its old row standing.
    check_comparable(
        previous.get("eval") or {},
        board_settings(args),
        carried_rows(previous, [{"run": label} for label, _ckpt in runs]),
    )

    rows, per_corpus, failed = [], {}, {}
    for i, (label, checkpoint) in enumerate(runs, start=1):
        print(f"\n[{i}/{len(runs)}] {label}\n    {checkpoint.name}")

        try:
            result = evaluate_run(label, checkpoint, args)
        except Exception as error:
            # One unloadable checkpoint must not cost the other seven their
            # evaluation — the model pass is the expensive part here.
            print(f"    !! skipped: {type(error).__name__}: {error}")
            failed[label] = f"{type(error).__name__}: {error}"
            continue

        rows.append(result["row"])
        per_corpus[label] = result["per_corpus"]

    rows = jsonable(rows)

    carried = carried_rows(previous, rows)
    check_comparable(previous.get("eval") or {}, board_settings(args), carried)
    warn_on_code_drift(carried, args.commit)

    if carried:
        print(f"\n[leaderboard] {len(carried)} row(s) carried over unchanged")
        per_corpus = {
            **{
                k: v
                for k, v in (previous.get("per_corpus") or {}).items()
                if k in {r["run"] for r in carried}
            },
            **per_corpus,
        }

    rows = rank_rows(rows + carried, args.rank_metric, args.rank_by)
    ranked_by = rank_key(args.rank_metric, args.rank_by)

    print(f"\n{'=' * 78}\nLEADERBOARD  (ranked by {ranked_by})\n{'=' * 78}")
    print(render_board(rows) if rows else "nothing scored")

    payload = build_payload(rows, per_corpus, failed, args)
    path = args.output or args.append or SCRATCH_DIR / f"leaderboard-{args.stamp}.json"

    write_payload(payload, path)

    if args.push and args.append is not None:
        publish(path)


if __name__ == "__main__":
    main()
