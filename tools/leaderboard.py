#!/usr/bin/env python3
"""Score a set of trained runs on full tracks and publish the comparison as a
W&B leaderboard plus one shareable file.

Training already reports on itself — every run writes a `training_report.json`
beside its checkpoints. What it cannot do is compare runs: each report scores
its own run, on whatever split and postprocessing that run was configured with,
at whatever epoch it happened to stop. Ranking four architectures against each
other means re-scoring all of them the same way, afterwards.

That is this tool. Point it at the checkpoint directories, and for every run it
finds it re-runs the full-track evaluation on one common split, sweeps the
postprocessing knobs per checkpoint (the shipped `beat_phase` knobs are marked
UNVERIFIED in `configs/eval_beat.yaml`, and a sweep has been worth more than a
retrain), and writes the ranked board to its own W&B project — separate from the
training project, so a leaderboard is not buried among training runs. The same
board goes to a single JSON file, uploaded to that W&B run's Files tab: one
download, holding every number, to hand to someone else.

The sweep runs on the **train** split by default and the report on val, so the
knobs are not chosen on the tracks they are then scored on. `--sweep-split val`
restores the older behaviour, which is what `tools/eval_beat.py --sweep` does
and what makes its numbers optimistic.

Usage
-----
    # every run in two experiment folders, swept, on the merged val split
    uv run python tools/leaderboard.py checkpoints_deeper checkpoints_norm \\
        --dataset merge --split val

    # a single checkpoint, no sweep (score at the config's shipped knobs)
    uv run python tools/leaderboard.py checkpoints/merge_v5.ckpt --no-sweep

    # rank by bar-position accuracy instead of beat F-measure, no W&B
    uv run python tools/leaderboard.py checkpoints_deeper \\
        --rank-metric position_acc --no-wandb

    # tune the knobs on val too — faster (no second model pass), but the
    # reported numbers are then no longer held out
    uv run python tools/leaderboard.py checkpoints_deeper --sweep-split val

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
from datetime import datetime, timezone
from pathlib import Path

import wandb

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
from tools.eval_beat import _BETTER, _LABELS, rank_key, resolve_group_size, sweep_grid

SCHEMA = 1
FILENAME = "leaderboard.json"

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
        metric=args.rank_metric,
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
            f"{row['run']:<{width}}  {row['n_tracks']:>4}"
            + "".join(f" {_fmt(row[c]):>9}" for c in BOARD_COLUMNS)
        )

    return "\n".join(lines)


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
            "git_commit": git_commit(),
            "eval": {
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
            },
            "readable": render_board(rows) if rows else "nothing scored",
            "leaderboard": rows,
            "per_corpus": per_corpus,
            "failed": failed,
        }
    )


def start_wandb(args, config: dict):
    """Open the W&B run this board is published to.

    One run per invocation, in a project of its own: a leaderboard is a
    different kind of object from a training run, and mixing them makes both
    harder to find. Opened *before* the file is written so the file can carry
    the run's own URL — which is what makes an attachment stand alone.
    """

    return wandb.init(
        project=args.project,
        name=args.name,
        job_type="leaderboard",
        tags=args.tags,
        config=config,
    )


def log_board(run, rows: list[dict], path: Path) -> None:
    """Log the board as a sortable table, rank it in the run summary, and
    upload *path* to the run's Files tab.

    The attachment is the point: a W&B link needs an account and a download
    does not, so the whole board also leaves as one file.
    """

    columns = list(rows[0])
    table = wandb.Table(
        columns=columns, data=[[row[c] for c in columns] for row in rows]
    )

    run.log({"leaderboard": table})

    run.summary.update(
        {
            "n_runs": len(rows),
            "best/run": rows[0]["run"],
            **{f"best/{key}": rows[0][key] for key in SCORE_KEYS},
        }
    )

    run.save(str(path), base_path=str(path.parent), policy="now")


def write_payload(payload: dict, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

    print(f"\n[leaderboard] written to {path}")


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
    data.add_argument("--dataset", default=EVAL_DEFAULTS["dataset"])
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
        action="store_true",
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
        help="Which metric orders the board, and picks each sweep's winner",
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
        "--output",
        type=Path,
        default=None,
        help=f"Defaults to leaderboards/<name>/{FILENAME}",
    )
    output.add_argument("--project", default="musicality-leaderboard")
    output.add_argument(
        "--name",
        default=None,
        help="W&B run name and output subdirectory (default: a timestamp)",
    )
    output.add_argument("--tags", nargs="*", default=[], help="W&B tags")
    output.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Write the file only — no W&B run, no upload",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")

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

    rows = rank_rows(jsonable(rows), args.rank_metric, args.rank_by)
    ranked_by = rank_key(args.rank_metric, args.rank_by)

    print(f"\n{'=' * 78}\nLEADERBOARD  (ranked by {ranked_by})\n{'=' * 78}")
    print(render_board(rows) if rows else "nothing scored")

    payload = build_payload(rows, per_corpus, failed, args)
    path = args.output or Path("leaderboards") / args.name / FILENAME

    run = start_wandb(args, payload["eval"]) if args.wandb and rows else None
    if run is not None:
        payload["wandb"] = {
            "project": args.project,
            "name": run.name,
            "id": run.id,
            "url": run.url,
        }

    write_payload(payload, path)

    if run is not None:
        log_board(run, rows, path)
        wandb.finish()
        # No URL offline: the run is sitting unsynced in ./wandb/, which is
        # worth saying rather than printing `None`.
        print(
            f"[leaderboard] {payload['wandb']['url'] or 'offline — `wandb sync` to upload'}"
        )


if __name__ == "__main__":
    main()
