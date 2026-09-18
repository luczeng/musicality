#!/usr/bin/env python3
"""Score trained runs on full tracks and keep the comparison as one running board.

`tools/eval_beat.py` says how good one checkpoint is. This ranks several against
each other, which their `training_report.json` files cannot: each was written
under that run's own split and postprocessing, so comparing reports compares the
settings as much as the models.

Name whatever is new; every run found is evaluated on the split
`configs/eval_beat.yaml` names, with its postprocessing swept first, and merged
into the board beside the runs already measured. The board is one JSON file in
the DVC-tracked data repo, pulled before reading and pushed after writing, so it
outlives the instance that produced it.

    uv run python tools/leaderboard.py checkpoints_deeper checkpoints_norm
    uv run python tools/leaderboard.py checkpoints_new

See `docs/source/workflows.rst` ("Comparing runs") for why the sweep runs on a
different split from the report, which metric decides what, and what makes two
boards refuse to merge.
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
    SCORE_KEYS,
    BeatEvaluator,
    _fmt,
    group_by_corpus,
    summarize,
)
from musicality.evaluation import DEFAULTS as EVAL_DEFAULTS
from tools.eval_beat import _BETTER, _LABELS, rank_key, resolve_group_size, sweep_grid

SCHEMA = 1

# The board lives in the DVC-tracked data repo beside the splits, not in this
# checkout: one that accumulates over months has to outlive the rented instance
# that wrote it.
DEFAULT_BOARD = dataformats.LEADERBOARD_DIR / "leaderboard.json"

# What every row on a board was measured under, straight from the one config
# that defines it. Also what has to match for an older row to sit beside a new
# one — a row scored on another split or at another tolerance cannot.
RUN = {
    key: EVAL_DEFAULTS[key]
    for key in (
        "dataset",
        "split",
        "val_split",
        "binary_only",
        "sample_rate",
        "hop_length",
        "tolerance",
    )
}
SWEEP = EVAL_DEFAULTS["sweep"]

# The postprocessing knobs carried on every row, so a board says which decode
# produced its numbers.
KNOB_KEYS = (
    "beat_threshold",
    "min_distance_frames",
    "gate_tolerance",
    "decoder",
    "switch_penalty",
    "anchor_threshold",
)

# Columns of the printed board. Narrow on purpose: the file carries every
# metric, this has to stay readable in a terminal.
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
    each named with the ``val/loss`` it scored; there the best-scoring file is
    the run and the others are history. A directory whose names carry no loss is
    read the other way — one entry per file — because that is how hand-named
    checkpoints (`merge_v5.ckpt`) sit in `checkpoints/`, and collapsing those to
    one would silently drop five models.
    """

    checkpoints = sorted(run_dir.glob("*.ckpt"))
    scored = [(match, c) for c in checkpoints if (match := _VALLOSS.search(c.name))]

    if scored:
        return [(str(run_dir), min(scored, key=lambda p: float(p[0].group(1)))[1])]

    return [(str(c.with_suffix("")), c) for c in checkpoints]


def find_runs(paths: list[Path]) -> list[tuple[str, Path]]:
    """Every run to score, as ``(label, checkpoint)``, in a stable order.

    Directories are walked recursively, so a sweep directory (one subdirectory
    per learning rate) expands without being named run by run.
    """

    runs = []
    for path in paths:
        if path.suffix == ".ckpt":
            runs.append((str(path.with_suffix("")), path))
            continue

        for run_dir in sorted({c.parent for c in path.rglob("*.ckpt")}):
            runs.extend(run_checkpoints(run_dir))

    return runs


def evaluator_for(checkpoint: Path, split: str, device: str, **overrides):
    """A :class:`BeatEvaluator` over *split*, under the configured run settings."""

    return BeatEvaluator(
        checkpoint=checkpoint,
        **{**RUN, "split": split},
        device=device,
        verbose=False,
        **overrides,
    )


def sweep_evaluator(checkpoint: Path, group_size: int, device: str) -> BeatEvaluator:
    """A second evaluator, over the split the knobs are tuned on.

    Separate from the reporting one on purpose: knobs chosen on the same tracks
    they are then scored on are chosen partly for the noise in those tracks, and
    by an amount that differs per row. Subsampled stratified across corpora
    because the train split is much larger than val, and because a split file is
    written corpus by corpus — the first N tracks would tune one genre's knobs.
    """

    evaluator = evaluator_for(checkpoint, SWEEP["split"], device, group_size=group_size)
    module, task, dataset, indices = evaluator.load()

    keep = {
        (ref.dataset_name, ref.track_id)
        for ref in stratified_sample(
            [dataset.refs[i] for i in indices], SWEEP["tracks"]
        )
    }

    # Narrow the loaded indices rather than the dataset: the evaluator memoizes
    # both together, and everything downstream walks `indices`.
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


def sweep_knobs(evaluator: BeatEvaluator, task: str, group_size: int) -> dict:
    """Best postprocessing knobs for one checkpoint, in two stages.

    Beat detection first, ranked by ``f_beat`` because that is all those knobs
    can move; then the one bar-position knob the resolved decoder reads, ranked
    by ``position_acc``. **The two metrics must differ**: a bar-position decoder
    relabels beats without moving them, so every stage-2 candidate scores an
    identical ``f_beat`` and ranking by it is a tie the sort breaks by candidate
    order — silently pinning ``switch_penalty`` to the first value in the list.

    Both stages re-use the cached frame probabilities, so a sweep costs one
    model pass per checkpoint, not one per grid point.
    """

    beat_grid = [
        {"beat_threshold": bt, "min_distance_frames": md, "gate_tolerance": gt}
        for bt, md, gt in itertools.product(
            SWEEP["beat_thresholds"],
            SWEEP["min_distance_frames"],
            SWEEP["gate_tolerances"],
        )
    ]

    ranked = sweep_grid(
        evaluator, beat_grid, group_size=group_size, metric="f_beat", rank_by="macro"
    )
    best = {key: ranked[0][key] for key in beat_grid[0]}

    if task != "beat_phase":
        return best

    if evaluator.resolve_postprocess()["decoder"] == "greedy":
        knob, values = "anchor_threshold", SWEEP["anchor_thresholds"]
    else:
        # `None` is a real switch_penalty — the exact single-offset decode, the
        # penalty -> infinity limit no finite value reaches.
        knob, values = "switch_penalty", [None, *SWEEP["switch_penalties"]]

    ranked = sweep_grid(
        evaluator,
        [{**best, knob: value} for value in values],
        group_size=group_size,
        metric="position_acc",
        rank_by="macro",
    )

    return {**best, knob: ranked[0][knob]}


def evaluate_run(label: str, checkpoint: Path, args) -> dict:
    """Score one checkpoint, returning its board row and per-corpus breakdown."""

    evaluator = evaluator_for(checkpoint, RUN["split"], args.device, limit=args.limit)
    _module, task, _dataset, indices = evaluator.load()
    group_size = resolve_group_size(evaluator, 4)

    print(f"    task={task}  group_size={group_size}  tracks={len(indices)}")
    evaluator.compute_track_probs()

    knobs, n_sweep = {}, 0
    if args.sweep:
        tuner = sweep_evaluator(checkpoint, group_size, args.device)
        n_sweep = len(tuner.load()[3])

        print(f"    sweeping on {SWEEP['split']} ({n_sweep} track(s))")
        knobs = sweep_knobs(tuner, task, group_size)

    rows = evaluator.score(group_size=group_size, **knobs)
    summary = summarize(rows)

    row = {
        "run": label,
        "checkpoint": str(checkpoint),
        "task": task,
        # Per row, not just per board: a running board carries rows measured on
        # different days by different code, and the row has to say which.
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": args.commit,
        "swept_on": SWEEP["split"] if knobs else None,
        "sweep_n_tracks": n_sweep,
        "n_tracks": summary["n_tracks"],
        **{key: summary[key] for key in SCORE_KEYS},
        **{f"macro_{key}": summary[f"macro_{key}"] for key in SCORE_KEYS},
        "worst_corpus": summary["worst_corpus"],
        "worst_position_acc": summary["worst_position_acc"],
        **{
            key: evaluator.resolve_postprocess(group_size=group_size, **knobs)[key]
            for key in KNOB_KEYS
        },
    }

    per_corpus = {
        corpus: {"n_tracks": len(group), **{k: summarize(group)[k] for k in SCORE_KEYS}}
        for corpus, group in group_by_corpus(rows).items()
    }

    return {"row": row, "per_corpus": per_corpus}


def dvc(command: list[str], board: Path) -> bool:
    """Run one ``dvc`` command in the data repo holding *board*.

    Never fatal. The remote needs credentials this checkout may not have, and a
    board that cannot be synced is still a board.
    """

    repo = board.parent.parent.resolve()
    if not (repo / ".dvc").is_dir():
        print(f"[leaderboard] {repo} is not a DVC repo — skipping dvc {command[0]}")
        return False

    print(f"[leaderboard] dvc {' '.join(command)} (in {repo})")
    result = subprocess.run(["dvc", *command], cwd=repo, capture_output=True, text=True)

    if result.returncode:
        print(f"[leaderboard] dvc {command[0]} failed: {result.stderr.strip()}")

    return not result.returncode


def load_board(board: Path, pull: bool) -> dict:
    """The board already at *board*, pulled from the DVC remote first.

    A missing board is not an error — the first invocation is what creates it,
    and a doomed `dvc pull` printing an error would read like a failure when
    nothing is wrong. DVC tracks the *folder* (`leaderboard.dvc`, beside
    `splits.dvc`), so that is what pull and add name.
    """

    if pull and (board.parent.parent / f"{board.parent.name}.dvc").exists():
        dvc(["pull", board.parent.name], board)

    if not board.exists():
        print(f"[leaderboard] no board at {board} yet — starting one")
        return {}

    previous = json.loads(board.read_text())
    print(f"[leaderboard] {len(previous['leaderboard'])} row(s) loaded from {board}")

    return previous


def publish(board: Path) -> None:
    """Re-hash the board folder and upload it, then say what to commit.

    ``dvc add`` rewrites the pointer and ``dvc push`` uploads the content, but
    the pointer only becomes the shared truth once committed — and committing in
    someone else's repo is not this tool's call to make.
    """

    name = board.parent.name

    if dvc(["add", name], board) and dvc(["push", name], board):
        print(
            f"\n[leaderboard] pushed. To share it:\n    cd {board.parent.parent.resolve()}"
            f" && git add {name}.dvc && git commit -m 'Update leaderboard'"
        )


def merge(previous: dict, rows: list[dict], settings: dict) -> list[dict]:
    """Rows of *previous* this invocation did not re-measure, ready to be kept.

    Identity is the ``run`` label, not the checkpoint path: re-running a folder
    after more training picks a different epoch's file, and that is the same
    experiment with a better number rather than a second entry.

    Raises when a surviving row was measured under different settings — a
    leaderboard is a claim that its rows can be read against each other. Only
    survivors are at stake, so re-measuring every run in the old board is what
    lets the split or the tolerance change without a flag to override this.
    """

    measured = {row["run"] for row in rows}
    carried = [r for r in previous.get("leaderboard", []) if r["run"] not in measured]

    differing = {
        key: (previous.get("eval", {}).get(key), value)
        for key, value in settings.items()
        if carried and previous.get("eval", {}).get(key) != value
    }
    if differing:
        changes = "\n".join(
            f"    {k}: {a!r} -> {b!r}" for k, (a, b) in differing.items()
        )
        raise SystemExit(
            f"[leaderboard] cannot extend: {len(carried)} row(s) already on the board "
            f"were measured under different settings:\n{changes}\n"
            "    Re-measure those runs too (name their folders as well), or use "
            "--board to start a separate one."
        )

    return carried


def rank_rows(rows: list[dict], metric: str) -> list[dict]:
    """Best first, by *metric*'s macro mean, in its own better-is direction.

    Macro weights each corpus equally; micro lets the largest corpus decide for
    all of them. A row that could not be scored (``None`` once
    :func:`jsonable` has turned its NaN into one) sorts last rather than winning.
    """

    key = rank_key(metric, "macro")
    sign = -1 if _BETTER[metric] is max else 1

    return sorted(rows, key=lambda r: math.inf if r.get(key) is None else sign * r[key])


def render_board(rows: list[dict]) -> str:
    """The ranked table, as printed and as stored in the file's ``readable``."""

    width = max((len(row["run"]) for row in rows), default=3)
    lines = [
        f"{'run':<{width}}  {'n':>4}"
        + "".join(f" {_LABELS[c]:>9}" for c in BOARD_COLUMNS)
    ]

    for row in rows:
        lines.append(
            f"{row['run']:<{width}}  {row.get('n_tracks', 0):>4}"
            + "".join(f" {_fmt(row.get(c)):>9}" for c in BOARD_COLUMNS)
        )

    return "\n".join(lines)


def build_payload(rows: list[dict], per_corpus: dict, failed: dict, args) -> dict:
    """The whole board as one JSON-safe dict.

    Machine-readable the way ``training_report.json`` is — NaN written as
    ``null`` so a strict parser accepts it — with a rendered board in
    ``readable`` so opening the file still shows something skimmable.
    """

    return jsonable(
        {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": args.commit,
            "eval": settings_for(args),
            "ranked_by": rank_key(args.rank_metric, "macro"),
            "readable": render_board(rows) if rows else "nothing scored",
            "leaderboard": rows,
            "per_corpus": per_corpus,
            "failed": failed,
        }
    )


def settings_for(args) -> dict:
    """The conditions this invocation measures under, and the ones an existing
    board's rows must have been measured under to be kept."""

    return {**RUN, "limit": args.limit, "swept": args.sweep}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained runs on the split configs/eval_beat.yaml names "
            "and merge them into one running leaderboard."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Checkpoint directories (walked recursively) and/or .ckpt files",
    )
    parser.add_argument(
        "--board",
        type=Path,
        default=DEFAULT_BOARD,
        help=(
            f"The running board to extend, default {DEFAULT_BOARD}. Rows for "
            "runs not named here are carried over from it and the merged board "
            "is written back. A path that does not exist yet starts a new "
            "board, so the first invocation is the same command as every later "
            "one."
        ),
    )
    parser.add_argument(
        "--rank-metric",
        choices=SCORE_KEYS,
        default="f_beat",
        help="Which metric orders the board (macro mean)",
    )
    parser.add_argument(
        "--no-sweep",
        dest="sweep",
        action="store_false",
        help=(
            "Score at the knobs configs/eval_beat.yaml ships instead of "
            "sweeping each checkpoint's own. Faster, but those beat_phase "
            "values are marked UNVERIFIED, so it usually understates every row."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Score only the first N tracks"
    )
    parser.add_argument("--device", default=EVAL_DEFAULTS["device"])
    parser.add_argument(
        "--no-pull",
        dest="pull",
        action="store_false",
        help="Skip the `dvc pull`. A stale board then loses whatever it misses.",
    )
    parser.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help="Write locally without `dvc add` + `dvc push`",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.commit = git_commit()  # once per invocation, stamped onto every row

    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"[leaderboard] no .ckpt found under {args.runs}")

    previous = load_board(args.board, args.pull)

    print(
        f"[leaderboard] {len(runs)} run(s)  dataset={RUN['dataset']}  "
        f"split={RUN['split']}  sweep={'on ' + SWEEP['split'] if args.sweep else 'off'}"
    )

    # Before the model passes, not after: an incompatible board should cost
    # nothing. Re-checked below against the rows actually measured, since a
    # checkpoint that fails to load leaves its old row standing.
    merge(previous, [{"run": label} for label, _ckpt in runs], settings_for(args))

    rows, per_corpus, failed = [], {}, {}
    for i, (label, checkpoint) in enumerate(runs, start=1):
        print(f"\n[{i}/{len(runs)}] {label}\n    {checkpoint.name}")

        try:
            result = evaluate_run(label, checkpoint, args)
        except Exception as error:
            # One unloadable checkpoint must not cost the others their
            # evaluation — the model pass is the expensive part here.
            print(f"    !! skipped: {type(error).__name__}: {error}")
            failed[label] = f"{type(error).__name__}: {error}"
            continue

        rows.append(result["row"])
        per_corpus[label] = result["per_corpus"]

    rows = jsonable(rows)
    carried = merge(previous, rows, settings_for(args))

    if carried:
        print(f"\n[leaderboard] {len(carried)} row(s) carried over unchanged")
        kept = {row["run"] for row in carried}
        per_corpus = {
            **{k: v for k, v in previous["per_corpus"].items() if k in kept},
            **per_corpus,
        }

    rows = rank_rows(rows + carried, args.rank_metric)

    print(
        f"\n{'=' * 78}\nLEADERBOARD (by {rank_key(args.rank_metric, 'macro')})\n{'=' * 78}"
    )
    print(render_board(rows) if rows else "nothing scored")

    args.board.parent.mkdir(parents=True, exist_ok=True)
    args.board.write_text(
        json.dumps(build_payload(rows, per_corpus, failed, args), indent=2)
    )
    print(f"\n[leaderboard] written to {args.board.resolve()}")

    if args.push:
        publish(args.board)


if __name__ == "__main__":
    main()
