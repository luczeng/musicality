#!/usr/bin/env python3
"""Score a public beat tracker on our own held-out tracks.

The experiment ``plans/08_rethinking_the_approach.md`` §6.1 asks for: run
madmom or Beat This! over the same split ``tools/eval_beat.py`` scores a
checkpoint on, through the same scorer, and print the same report. What comes
back says whether our 20-point downbeat deficit is our model or our data.

Predictions are cached per track under ``outputs/baselines/``, so the slow half
(running the tracker — madmom is minutes per track on CPU) happens once and the
report can be re-cut for free.

Read this before quoting anything it prints
-------------------------------------------

Both trackers were trained on public beat corpora, and most of our validation
split *is* those corpora. On ballroom and the RWC sets a baseline is recalling
tracks it was fitted on; only gtzan (held out by both), rwc_genre and jtd are
clean comparisons. The run prints which corpora are affected before the
numbers; :data:`musicality.baselines.base.CORPUS_EXPOSURE` is the table.

Usage
-----
    # the canonical report, per genre
    uv run python tools/eval_baseline.py --baseline madmom \\
        --dataset merge --split val --binary-only

    # Beat This!, current state of the art, rows saved for later comparison
    uv run python tools/eval_baseline.py --baseline beat_this \\
        --dataset merge --split val --binary-only --output outputs/beat_this.csv

    # a clean comparison: the corpus neither tracker was trained on
    uv run python tools/eval_baseline.py --baseline beat_this \\
        --dataset gtzan --split val

    # same frame probabilities, madmom's DBN instead of peak-picking
    uv run python tools/eval_baseline.py --baseline beat_this --dbn

    # fill the cache overnight, score later
    uv run python tools/eval_baseline.py --baseline madmom --predict-only

``--binary-only`` must match how the checkpoint's split was created, exactly as
in ``tools/eval_beat.py`` — without it, "val" on a merged split is a different
set of tracks.
"""

import argparse
from pathlib import Path

import yaml

import musicality.dataformats as dataformats
from musicality.baselines import (
    BASELINES,
    BaselineEvaluator,
    CachedBaseline,
    build_baseline,
)
from musicality.baselines.evaluator import default_cache_path
from tools.eval_beat import print_genre_breakdown, print_offset_profile, write_rows

DEFAULTS = yaml.safe_load(
    (dataformats.ROOT / "configs" / "eval_baseline.yaml").read_text()
)


def baseline_options(args: argparse.Namespace) -> dict:
    """Constructor arguments for the selected baseline.

    Each tracker takes a disjoint set of knobs, and passing one tracker's knob
    to the other is a ``TypeError`` rather than a silent no-op — so the CLI
    accepts them all and this decides which ones are real for this run.
    """

    if args.baseline == "madmom":
        return {
            "variant": args.variant,
            "beats_per_bar": tuple(args.beats_per_bar),
            "min_bpm": args.min_bpm,
            "max_bpm": args.max_bpm,
            "transition_lambda": args.transition_lambda,
        }

    return {
        "checkpoint": args.model,
        "dbn": args.dbn,
        "device": args.device,
        "float16": args.float16,
    }


def resolve_cache_path(args: argparse.Namespace) -> Path | bool:
    """Where this run's predictions are cached — or ``False`` for no cache."""

    if args.no_cache:
        return False

    if args.cache is not None:
        return Path(args.cache)

    return default_cache_path(
        args.baseline,
        args.dataset,
        args.split,
        args.binary_only,
        cache_dir=DEFAULTS["cache_dir"],
    )


def warn_about_split(args: argparse.Namespace) -> None:
    """Flag the one mistake that silently invalidates the whole run.

    A merged split exists in two versions, ``beat_phase-merge`` and
    ``beat_phase-merge-binary``, and they hold different tracks. Scoring a
    baseline against one while the checkpoint it is being compared to was
    validated on the other produces two numbers that look comparable and are
    not.
    """

    if args.dataset.startswith("merge") and not args.binary_only:
        print(
            "[warning] --dataset merge without --binary-only reads the "
            "'beat_phase-merge' split, not 'beat_phase-merge-binary'. Our "
            "beat-phase checkpoints are trained and validated on the binary "
            "one — pass --binary-only to compare against them.\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a public beat tracker (madmom, Beat This!) on our splits, "
            "with our metrics."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=list(BASELINES),
        help="Which tracker to run",
    )
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--data-home", default=None)
    parser.add_argument(
        "--split", choices=["train", "val", "all"], default=DEFAULTS["split"]
    )
    parser.add_argument("--val-split", type=float, default=DEFAULTS["val_split"])
    parser.add_argument("--sample-rate", type=int, default=DEFAULTS["sample_rate"])
    parser.add_argument("--hop-length", type=int, default=DEFAULTS["hop_length"])
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULTS["group_size"],
        help="Beats per group: 4 for bar position, 8 for phrase position",
    )
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help=(
            "Drop tracks whose beats-per-bar isn't a multiple of 2. Must match "
            "how the split was created."
        ),
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULTS["tolerance"])
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable mir_eval's standard 5s warm-up trim",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Score only the first N tracks"
    )

    cache = parser.add_argument_group(
        "prediction cache",
        "Running a tracker is minutes per track; scoring it is seconds.",
    )
    cache.add_argument(
        "--cache", default=None, help=f"Defaults to {DEFAULTS['cache_dir']}/<run>.json"
    )
    cache.add_argument(
        "--no-cache", action="store_true", help="Neither read nor write a cache"
    )
    cache.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run the tracker even where a cached prediction exists",
    )
    cache.add_argument(
        "--cache-every",
        type=int,
        default=20,
        help="Flush the cache every N newly predicted tracks (default 20)",
    )
    cache.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Score an existing cache without constructing the tracker — for "
            "predictions produced on another machine. Fails if any selected "
            "track is missing from it."
        ),
    )
    cache.add_argument(
        "--predict-only",
        action="store_true",
        help="Fill the cache and stop, without scoring",
    )

    madmom = parser.add_argument_group("madmom")
    madmom.add_argument(
        "--variant",
        choices=["downbeat", "beat"],
        default=DEFAULTS["madmom"]["variant"],
        help="downbeat: RNN+DBN downbeat tracker. beat: beat tracker, no positions",
    )
    madmom.add_argument(
        "--beats-per-bar",
        type=int,
        nargs="+",
        default=DEFAULTS["madmom"]["beats_per_bar"],
        help="Bar lengths the DBN may choose between",
    )
    madmom.add_argument("--min-bpm", type=float, default=DEFAULTS["madmom"]["min_bpm"])
    madmom.add_argument("--max-bpm", type=float, default=DEFAULTS["madmom"]["max_bpm"])
    madmom.add_argument(
        "--transition-lambda",
        type=float,
        default=DEFAULTS["madmom"]["transition_lambda"],
    )

    beat_this = parser.add_argument_group("beat_this")
    beat_this.add_argument(
        "--model",
        default=DEFAULTS["beat_this"]["model"],
        help=(
            "Checkpoint name, path or URL. final0/1/2 and small0/1/2 held out "
            "only GTZAN; single_final0/1/2 and fold0..7 hold out a documented "
            "validation split"
        ),
    )
    beat_this.add_argument(
        "--dbn",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["beat_this"]["dbn"],
        help="Postprocess with madmom's DBN instead of peak-picking",
    )
    beat_this.add_argument("--device", default=DEFAULTS["beat_this"]["device"])
    beat_this.add_argument(
        "--float16",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["beat_this"]["float16"],
    )

    report = parser.add_argument_group("report sections")
    report.add_argument(
        "--per-genre",
        action="store_true",
        help="Per-corpus breakdown (automatic when the split spans several)",
    )
    report.add_argument(
        "--profile",
        action="store_true",
        help="Phase-offset profile: where the bar numbering lands when it is wrong",
    )
    report.add_argument("--output", type=Path, default=None, help="Write rows to CSV")

    return parser.parse_args()


def main():
    args = parse_args()

    warn_about_split(args)

    cache_path = resolve_cache_path(args)

    if args.from_cache:
        if cache_path is False:
            raise SystemExit("--from-cache and --no-cache are contradictory.")
        baseline = CachedBaseline.from_cache_file(cache_path)
    else:
        baseline = build_baseline(args.baseline, **baseline_options(args))

    evaluator = BaselineEvaluator(
        baseline,
        dataset=args.dataset,
        data_home=args.data_home,
        split=args.split,
        val_split=args.val_split,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        group_size=args.group_size,
        binary_only=args.binary_only,
        tolerance=args.tolerance,
        trim=not args.no_trim,
        limit=args.limit,
        cache_path=cache_path,
        refresh=args.refresh,
        cache_every=args.cache_every,
    )

    if args.predict_only:
        predictions = evaluator.predict_all()
        print(f"[baseline] {len(predictions)} track(s) predicted")
        return

    rows = evaluator.run()

    if len(set(evaluator.track_corpora())) > 1:
        args.per_genre = True

    if args.profile:
        print("\n" + "=" * 78)
        print(f"PHASE-OFFSET PROFILE  (split={args.split})")
        print("=" * 78)
        print_offset_profile(rows, args.group_size)

    if args.per_genre:
        print("\n" + "=" * 78)
        print(f"PER-GENRE BREAKDOWN  ({baseline.describe()}, split={args.split})")
        print("=" * 78)
        print_genre_breakdown(rows)

        note = evaluator.exposure_note()
        if note:
            print(f"\n  {note}")

    if args.output is not None:
        write_rows(args.output, rows)


if __name__ == "__main__":
    main()
