#!/usr/bin/env python3
"""Score trained runs on full tracks and keep the comparison as one running board.

`eval_beat.py` says how good one checkpoint is; this ranks several against each
other, which their `training_report.json` files cannot — each was written under
its own run's settings. Name whatever is new and it joins the board beside the
runs already measured.

    uv run python tools/leaderboard.py checkpoints_deeper checkpoints_norm
    uv run python tools/leaderboard.py checkpoints_new

The board lives on W&B, in a project of its own. Every invocation fetches the
published board, merges its own rows into it, and publishes it back: as a
sortable table, which is the board to look at, and as one `leaderboard.json`
artifact holding every number, which is the board to hand to someone else. The
copy in this checkout is a working file — training runs on rented instances,
so W&B is what a board survives in. Nothing about publishing needs a model
pass, so a board in hand goes up on its own:

    uv run python tools/leaderboard.py --publish-only

See `docs/source/workflows.rst` ("Comparing runs") for the design.
"""

import argparse
import itertools
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import wandb

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
from tools.eval_beat import (
    _BETTER,
    _LABELS,
    rank_key,
    resolve_group_size,
    sweep_grid,
)

SCHEMA = 1
FILENAME = "leaderboard.json"

# Its own W&B project: a leaderboard is a different kind of object from a
# training run, and mixing them makes both harder to find.
PROJECT = "musicality-leaderboard"

# The artifact every board version is published under. `:latest` is what the
# next invocation starts from, on whatever machine it runs.
ARTIFACT = "leaderboard"

# A working copy, not the record — W&B holds that, so this one is gitignored
# and a fresh clone fetches the board rather than carrying it.
DEFAULT_BOARD = dataformats.ROOT / "leaderboard" / FILENAME

# What every row was measured under, and what an older row must match to be kept.
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

# Carried on every row, so a board says which decode produced its numbers.
KNOB_KEYS = (
    "beat_threshold",
    "min_distance_frames",
    "gate_tolerance",
    "decoder",
    "switch_penalty",
    "anchor_threshold",
)

# Narrow on purpose: the file carries every metric, this has to fit a terminal.
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

    A ``save_top_k`` group (``val/loss`` in every name) is one run, represented
    by its best file. Names without a loss are hand-named checkpoints sharing a
    folder — one run each, since collapsing those would drop five models.
    """

    checkpoints = sorted(run_dir.glob("*.ckpt"))
    scored = [(match, c) for c in checkpoints if (match := _VALLOSS.search(c.name))]

    if scored:
        return [(str(run_dir), min(scored, key=lambda p: float(p[0].group(1)))[1])]

    return [(str(c.with_suffix("")), c) for c in checkpoints]


def find_runs(paths: list[Path]) -> list[tuple[str, Path]]:
    """Every run to score, as ``(label, checkpoint)``, in a stable order.

    Walked recursively, so naming a sweep directory covers every learning rate.
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

    Separate from the reporting one so the board's numbers stay held out.
    Stratified rather than the first N tracks: a split file is written corpus by
    corpus, so the head of one is a single genre.
    """

    evaluator = evaluator_for(checkpoint, SWEEP["split"], device, group_size=group_size)
    module, task, dataset, indices = evaluator.load()

    keep = {
        (ref.dataset_name, ref.track_id)
        for ref in stratified_sample(
            [dataset.refs[i] for i in indices], SWEEP["tracks"]
        )
    }

    # Indices rather than the dataset: everything downstream walks `indices`.
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

    Beat detection ranked by ``f_beat``, then the resolved decoder's one
    bar-position knob ranked by ``position_acc``. **The metrics must differ**: a
    decoder relabels beats without moving them, so every stage-2 candidate ties
    on ``f_beat`` and the sort would pick by list order.

    Both stages re-use the cached probabilities — one model pass per checkpoint.
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
        # `None` is a real value: the exact single-offset decode.
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
        # Per row: a board carries rows measured on different days by different code.
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


def fetch_board(board: Path, project: str) -> None:
    """Replace *board* with the published one, if there is one.

    What this invocation extends is whatever was published last, not whatever
    this machine happens to hold: training runs on rented instances that start
    empty, and a stale local copy would silently drop every row measured
    elsewhere since. Never fatal — the first invocation ever finds nothing
    published, and an unreachable W&B is a reason to score locally rather than
    to refuse.
    """

    try:
        artifact = wandb.Api().artifact(f"{project}/{ARTIFACT}:latest")

        # Downloaded to a scratch directory rather than wandb's default
        # ./artifacts/, which would keep a second copy of every version ever
        # fetched in the checkout. The board itself is the only file in there.
        with tempfile.TemporaryDirectory() as scratch:
            published = Path(artifact.download(root=scratch)) / FILENAME

            board.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(published, board)
    except Exception as error:
        print(f"[leaderboard] nothing fetched ({type(error).__name__}: {error})")
        return

    print(f"[leaderboard] {artifact.name} fetched to {board}")


def load_board(board: Path, project: str, fetch: bool) -> dict:
    """The board this invocation extends, fetched from W&B first.

    A missing board is not an error: the first invocation starts one.
    """

    if fetch:
        fetch_board(board, project)

    if not board.exists():
        print(f"[leaderboard] no board at {board} yet — starting one")
        return {}

    previous = json.loads(board.read_text())
    print(f"[leaderboard] {len(previous['leaderboard'])} row(s) loaded from {board}")

    return previous


def publish(payload: dict, board: Path, project: str) -> None:
    """Publish the board to W&B: a table to look at, a file to keep.

    The table is the board — every column of every row, sortable and filterable
    in the browser, which is the only place a board of this width reads well.
    The artifact is the same board as one file, versioned: it is what the next
    invocation fetches, wherever it runs, and what travels to someone who wants
    the numbers rather than a browser tab.

    One run per invocation, in a project of its own, so a board is not buried
    among training runs.
    """

    rows = payload["leaderboard"]
    columns = list(rows[0])

    run = wandb.init(project=project, job_type="leaderboard", config=payload["eval"])

    run.log(
        {
            "leaderboard": wandb.Table(
                columns=columns,
                data=[[row.get(key) for key in columns] for row in rows],
            )
        }
    )
    run.summary.update(
        {
            "n_runs": len(rows),
            "best/run": rows[0]["run"],
            **{f"best/{key}": rows[0].get(key) for key in SCORE_KEYS},
        }
    )

    artifact = wandb.Artifact(ARTIFACT, type="leaderboard")
    artifact.add_file(str(board), name=FILENAME)
    run.log_artifact(artifact)

    url = run.url
    wandb.finish()

    # No URL offline: the run is sitting unsynced in ./wandb/, which is worth
    # saying rather than printing `None`.
    print(f"\n[leaderboard] published to {url or 'offline — `wandb sync` to upload'}")


def merge(previous: dict, rows: list[dict], settings: dict) -> list[dict]:
    """Rows of *previous* this invocation did not re-measure, ready to be kept.

    Identity is the ``run`` label, not the checkpoint path: more training on a
    folder is the same experiment with a better number, not a second entry.

    Raises when a *surviving* row was measured under different settings — so
    re-measuring the whole board is what lets those settings change, no override
    flag needed.
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

    Macro so the largest corpus does not decide for all of them. An unscorable
    row (``None``, once :func:`jsonable` has seen its NaN) sorts last.
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

    NaN as ``null`` so a strict parser accepts it, the way
    ``training_report.json`` does, plus a rendered table in ``readable``.
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


def write_board(payload: dict, board: Path) -> None:
    """Write the board out, creating its folder on the first run."""

    board.parent.mkdir(parents=True, exist_ok=True)
    board.write_text(json.dumps(payload, indent=2))

    print(f"\n[leaderboard] written to {board.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained runs on the split configs/eval_beat.yaml names "
            "and merge them into one running leaderboard on W&B."
        )
    )
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Checkpoint directories (walked recursively) and/or .ckpt files",
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
    parser.add_argument("--project", default=PROJECT, help="W&B project to publish to")
    parser.add_argument(
        "--board",
        type=Path,
        default=DEFAULT_BOARD,
        help=(
            f"Where the board is kept locally, default {DEFAULT_BOARD}. Any "
            "other path is a throwaway comparison: it is neither fetched from "
            "nor published to W&B."
        ),
    )
    parser.add_argument(
        "--no-fetch",
        dest="fetch",
        action="store_false",
        help="Extend the local board as it is, without fetching the published one",
    )
    parser.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        help="Score and write locally, without a W&B run",
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help=(
            "Publish the board already on disk and stop. Scores nothing, so it "
            "costs no model pass — for putting an existing board on W&B, or "
            "for retrying a publish that failed."
        ),
    )

    args = parser.parse_args()

    if not args.runs and not args.publish_only:
        parser.error("name at least one checkpoint directory, or --publish-only")

    if args.board.resolve() != DEFAULT_BOARD.resolve():
        # A board somewhere else is a throwaway by definition: it must not
        # start from the shared one, and must certainly not be published over
        # it — the next run anywhere would merge into numbers meant for one.
        if args.publish_only:
            parser.error("--publish-only publishes the shared board, not a --board")

        args.fetch = args.publish = False
        print(f"[leaderboard] {args.board} is a local board — W&B untouched")

    return args


def main():
    args = parse_args()

    if args.publish_only:
        # Not fetched first: publishing is for the board in hand, and fetching
        # would replace it with the one already up there.
        payload = load_board(args.board, args.project, fetch=False)

        if not payload.get("leaderboard"):
            raise SystemExit(f"[leaderboard] no board at {args.board} to publish")

        publish(payload, args.board, args.project)

        return

    args.commit = git_commit()  # once per invocation, stamped onto every row

    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"[leaderboard] no .ckpt found under {args.runs}")

    previous = load_board(args.board, args.project, args.fetch)

    print(
        f"[leaderboard] {len(runs)} run(s)  dataset={RUN['dataset']}  "
        f"split={RUN['split']}  sweep={'on ' + SWEEP['split'] if args.sweep else 'off'}"
    )

    # Before the model passes, so an incompatible board costs nothing. Re-checked
    # below: a checkpoint that fails to load leaves its old row standing.
    merge(previous, [{"run": label} for label, _ckpt in runs], settings_for(args))

    rows, per_corpus, failed = [], {}, {}
    for i, (label, checkpoint) in enumerate(runs, start=1):
        print(f"\n[{i}/{len(runs)}] {label}\n    {checkpoint.name}")

        try:
            result = evaluate_run(label, checkpoint, args)
        except Exception as error:
            # One bad checkpoint must not cost the others their model pass.
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

    payload = build_payload(rows, per_corpus, failed, args)

    write_board(payload, args.board)

    if args.publish and rows:
        publish(payload, args.board, args.project)


if __name__ == "__main__":
    main()
