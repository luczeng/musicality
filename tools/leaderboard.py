#!/usr/bin/env python3
"""Score trained runs on full tracks and keep the comparison as one running board.

`eval_beat.py` says how good one checkpoint is; this ranks several against each
other, which their `training_report.json` files cannot — each was written under
its own run's settings. Name whatever is new and it joins the board beside the
runs already measured.

    uv run python tools/leaderboard.py checkpoints_deeper checkpoints_norm
    uv run python tools/leaderboard.py checkpoints_new

The board is written twice, to the two places it is read from.
`leaderboard.json` is the record, DVC-tracked in the data repo beside the
splits. `leaderboard/LEADERBOARD.md` is the page, git-tracked in *this*
checkout, so the standings render on GitHub and show up in a diff — no
`dvc pull` to read them. The page is derived from the board and can be rebuilt
from it alone, scoring nothing:

    uv run python tools/leaderboard.py --render-only

See `docs/source/workflows.rst` ("Comparing runs") for the design.
"""

import argparse
import itertools
import json
import math
import os
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
from tools.eval_beat import (
    _BETTER,
    _KNOB_LABELS,
    _LABELS,
    rank_key,
    resolve_group_size,
    sweep_grid,
)

SCHEMA = 1

# In the data repo, not this checkout: a board has to outlive the rented
# instance that wrote it.
DEFAULT_BOARD = dataformats.LEADERBOARD_DIR / "leaderboard.json"

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


def dvc(command: list[str], board: Path) -> bool:
    """Run one ``dvc`` command in the data repo holding *board*.

    Never fatal: a board that cannot be synced is still a board.
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

    A missing board is not an error — the first invocation creates it, and
    pulling before the pointer exists would only print a scary non-failure. DVC
    tracks the *folder*, so that is what pull names.
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

    The pointer only becomes the shared truth once committed, and committing in
    someone else's repo is not this tool's call.
    """

    name = board.parent.name

    if dvc(["add", name], board) and dvc(["push", name], board):
        print(
            f"\n[leaderboard] pushed. To share it:\n    cd {board.parent.parent.resolve()}"
            f" && git add {name}.dvc && git commit -m 'Update leaderboard'"
        )


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
    """Write the board by replacing the path, not by writing into it.

    ``setup_remote.sh`` sets ``cache.type symlink``, so a pulled board is a
    symlink into DVC's read-only cache: writing to it raises ``PermissionError``,
    and would corrupt the cache if it did not. Replacing is also atomic.
    """

    board.parent.mkdir(parents=True, exist_ok=True)

    scratch = board.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(payload, indent=2))
    os.replace(scratch, board)

    print(f"\n[leaderboard] written to {board.resolve()}")


# The human-readable twin of the board, in *this* checkout rather than beside
# the JSON: the data repo is behind a `dvc pull` and renders nowhere, so the
# standings live in git, where GitHub renders them and a diff shows what moved.
# What the page holds, and why, is docs/source/workflows.rst ("Comparing runs").
PAGE_PATH = dataformats.ROOT / "leaderboard" / "LEADERBOARD.md"


def ranked_metric(payload: dict) -> str:
    """The metric a board is ordered by, without its macro/micro prefix.

    Read off the payload, not off ``args``: a board written by an older
    invocation names it under ``eval``, and must still re-render.
    """

    key = payload.get("ranked_by") or payload.get("eval", {}).get("ranked_by", "")

    return key.removeprefix("macro_").removeprefix("micro_") or "f_beat"


def _when(stamp: str | None) -> str:
    """An ISO timestamp as minutes UTC — the precision a reader uses."""

    return f"{(stamp or '?')[:16].replace('T', ' ')} UTC"


def _cell(value: float | None) -> str:
    """One metric as a table cell: :func:`_fmt` without its terminal padding."""

    return _fmt(value).strip()


def _knob(row: dict, key: str) -> str:
    """One decode knob. ``None`` is a value — on ``switch_penalty`` it is the
    exact single-offset decode — so only a key the row never carried is a gap."""

    if key not in row:
        return "—"

    return "none" if row[key] is None else str(row[key])


def section(
    title: str,
    header: list[str],
    body: list[list[str]],
    note: str = "",
    labels: int = 1,
) -> list[str]:
    """One section of the page: a heading, a table, an optional note under it.

    The first *labels* columns are left-aligned and the rest right-aligned, the
    way numbers read.
    """

    align = ["---"] * labels + ["---:"] * (len(header) - labels)

    return [
        f"## {title}",
        "",
        *["| " + " | ".join(row) + " |" for row in (header, align, *body)],
        *(["", note] if note else []),
    ]


def page_header(payload: dict, rows: list[dict], metric: str, total: int) -> list[str]:
    """Title, who leads, the conditions every row shares, and how to rebuild
    the page — the question it is most often opened with."""

    settings = payload.get("eval", {})
    swept = [row for row in rows if row.get("swept_on")]
    shown = f", best {len(rows)} of {total} shown" if total > len(rows) else ""

    return [
        "# Beat leaderboard",
        "",
        f"**{rows[0]['run']}** leads {total} run(s) on macro `{metric}` "
        f"({_cell(rows[0].get(f'macro_{metric}'))}){shown}.",
        "",
        f"- Scored on `{settings.get('dataset', '?')}` / `{settings.get('split', '?')}`"
        f"{', binary-only' if settings.get('binary_only') else ''}, full tracks, "
        f"tolerance {settings.get('tolerance', '?')}s",
        "- Postprocessing "
        + (
            f"swept per checkpoint on `{swept[0]['swept_on']}` (see *Decode*)"
            if swept
            else "as shipped in `configs/eval_beat.yaml`, not swept"
        ),
        f"- Measured {_when(payload.get('generated_utc'))} at commit "
        f"`{(payload.get('git_commit') or '?')[:7]}`",
        "",
        "Written by `tools/leaderboard.py` from `leaderboard.json` in the data "
        "repo, which holds every metric, per track and per corpus. Rebuild this "
        "page from it, scoring nothing: `uv run python tools/leaderboard.py "
        "--render-only`.",
    ]


def page_ranking(rows: list[dict], metric: str) -> list[str]:
    """The board itself, as macro means — the numbers the ranking is made of.

    Macro rather than the per-track means the terminal prints: every corpus
    counts once, so the largest one does not decide the order.
    """

    def label(column: str) -> str:
        text = f"{_LABELS[column]} {'↑' if _BETTER[column] is max else '↓'}"

        return f"**{text}**" if column == metric else text

    body = [
        [str(i), f"`{row['run']}`", str(row.get("n_tracks", 0))]
        + [_cell(row.get(f"macro_{c}")) for c in BOARD_COLUMNS]
        for i, row in enumerate(rows, start=1)
    ]

    return section(
        "Ranking",
        ["#", "run", "n", *map(label, BOARD_COLUMNS)],
        body,
        f"Macro means, each corpus weighted once; ranked by `{metric}`. The "
        "per-track (micro) means are in the JSON.",
    )


def page_per_corpus(rows: list[dict], per_corpus: dict, metric: str) -> list[str]:
    """The ranking metric per corpus, one column per run, best in bold.

    What the macro mean averaged away: the weakest corpus is what gates "works
    everywhere", and it is rarely the same corpus for every run.
    """

    if not per_corpus:
        return []

    counts: dict[str, int] = {}
    for row in rows:
        for corpus, stats in per_corpus.get(row["run"], {}).items():
            counts.setdefault(corpus, stats.get("n_tracks") or 0)

    body = []
    for corpus, n_tracks in sorted(counts.items(), key=lambda kv: -kv[1]):
        values = [
            per_corpus.get(row["run"], {}).get(corpus, {}).get(metric) for row in rows
        ]
        scored = [v for v in values if v is not None and not math.isnan(v)]
        best = _BETTER[metric](scored) if scored else None

        body.append(
            [f"`{corpus or '<unknown>'}`", str(n_tracks)]
            + [
                f"**{_cell(v)}**" if v is not None and v == best else _cell(v)
                for v in values
            ]
        )

    return section(
        f"`{metric}` per corpus",
        ["corpus", "n", *[f"#{i}" for i in range(1, len(rows) + 1)]],
        body,
        "Columns are the ranking positions above; best per corpus in bold.",
    )


def page_decode(rows: list[dict]) -> list[str]:
    """What each row was decoded with, and where those knobs came from.

    A swept board ranks models *at their own best decode*, so the decode is
    part of the result rather than a footnote to it.
    """

    body = []
    for i, row in enumerate(rows, start=1):
        swept_on = row.get("swept_on")
        tuned = (
            f"`{swept_on}` ({row.get('sweep_n_tracks', 0)} tracks)"
            if swept_on
            else "config"
        )

        body.append(
            [str(i), row.get("task", "?"), *[_knob(row, k) for k in KNOB_KEYS], tuned]
        )

    return section(
        "Decode",
        ["#", "task", *[_KNOB_LABELS.get(k, k) for k in KNOB_KEYS], "tuned on"],
        body,
        "Tuned on a split the board does not report, so these numbers stay held "
        "out. `none` on `switch_pen` is the exact single-offset decode.",
        labels=2,
    )


def page_provenance(rows: list[dict]) -> list[str]:
    """Each row's checkpoint, when it was measured, and at what commit.

    A running board carries rows measured on different days by different code,
    so "which of these is stale" has to be answerable from the page — and the
    path leads straight to the model behind a number.
    """

    body = [
        [
            str(i),
            f"`{row.get('checkpoint', '?')}`",
            _when(row.get("measured_utc")),
            f"`{(row.get('git_commit') or '?')[:7]}`",
        ]
        for i, row in enumerate(rows, start=1)
    ]

    return section(
        "Provenance", ["#", "checkpoint", "measured", "commit"], body, labels=4
    )


def page_failed(failed: dict) -> list[str]:
    """Runs that were named but produced no row — the page is the only place
    their absence is explained."""

    if not failed:
        return []

    return section(
        "Not scored",
        ["run", "error"],
        [[f"`{run}`", error] for run, error in sorted(failed.items())],
        labels=2,
    )


def render_page(payload: dict, top: int = 0) -> str:
    """The whole board as one Markdown page, or its best *top* rows.

    The JSON is the record; this is the thing a person opens. *top* of 0 is
    every run, and it cuts the whole page rather than one table, so the
    per-corpus columns and the decode beside a row stay that row's own.
    """

    all_rows = payload.get("leaderboard", [])
    rows = all_rows[:top] if top else all_rows
    metric = ranked_metric(payload)

    if rows:
        blocks = [
            page_header(payload, rows, metric, len(all_rows)),
            page_ranking(rows, metric),
            page_per_corpus(rows, payload.get("per_corpus", {}), metric),
            page_decode(rows),
            page_provenance(rows),
        ]
    else:
        blocks = [["# Beat leaderboard", "", "Nothing scored yet."]]

    blocks.append(page_failed(payload.get("failed", {})))

    return "\n\n".join("\n".join(block) for block in blocks if block) + "\n"


def write_page(payload: dict, board: Path, top: int = 0) -> None:
    """Write :data:`PAGE_PATH`, if *board* is the running one.

    Only the running board earns the committed page: a ``--board`` elsewhere is
    a throwaway comparison by definition, and overwriting the repo's page with
    one would put numbers nobody can reproduce under version control. The
    folder is created here — the page is the only thing in it, so a fresh clone
    that has never scored anything has neither. Written in place rather than
    through a scratch file the way :func:`write_board` is: this one is in git,
    never a symlink into DVC's read-only cache.
    """

    if board.resolve() != DEFAULT_BOARD.resolve():
        return

    PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGE_PATH.write_text(render_page(payload, top))

    print(f"[leaderboard] page written to {PAGE_PATH} — commit it to share it")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained runs on the split configs/eval_beat.yaml names "
            "and merge them into one running leaderboard."
        )
    )
    parser.add_argument(
        "runs",
        nargs="*",
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
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help=(
            "List only the best N runs on the page "
            f"({PAGE_PATH.parent.name}/{PAGE_PATH.name}); default 0, every "
            "run. Only the default --board has a page."
        ),
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            f"Re-write {PAGE_PATH.name} from the board already on disk, and "
            "stop. Scores nothing, so it costs no model pass — for reading a "
            "pulled board, or re-rendering one after the page changed. Add "
            "--no-pull --no-push to touch no remote."
        ),
    )

    args = parser.parse_args()

    if not args.runs and not args.render_only:
        parser.error("name at least one checkpoint directory, or --render-only")

    return args


def main():
    args = parse_args()

    if args.render_only:
        payload = load_board(args.board, args.pull)

        if not payload:
            raise SystemExit(f"[leaderboard] no board at {args.board} to render")

        write_page(payload, args.board, args.top)

        if args.push:
            publish(args.board)

        return

    args.commit = git_commit()  # once per invocation, stamped onto every row

    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"[leaderboard] no .ckpt found under {args.runs}")

    previous = load_board(args.board, args.pull)

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
    write_page(payload, args.board, args.top)

    if args.push:
        publish(args.board)


if __name__ == "__main__":
    main()
