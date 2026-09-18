# Beat leaderboard

**checkpoints_change_tcn_stem/lr_0.0008/lr_sweep-0.0008** leads 7 run(s) on macro `f_beat` (0.763).

- Scored on `merge` / `val`, binary-only, full tracks, tolerance 0.07s
- Postprocessing swept per checkpoint on `train` (see *Decode*)
- Measured 2026-09-18 18:08 UTC at commit `f8a702c`

Written by `tools/leaderboard.py` and committed here, so the standings read on GitHub without a `dvc pull`. The board behind it — every metric, per track and per corpus — is `leaderboard.json` under `../musicality_db/leaderboard` in the data repo. Rebuild this page from it alone, scoring nothing, with `uv run python tools/leaderboard.py --render-only` (add `--no-pull --no-push` to touch no remote).

## Ranking

| # | run | n | **f_beat ↑** | cmlt ↑ | amlt ↑ | pos_acc ↑ | best_off ↑ | confuse ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `checkpoints_change_tcn_stem/lr_0.0008/lr_sweep-0.0008` | 277 | 0.763 | 0.514 | 0.746 | 0.594 | 0.630 | 0.262 |
| 2 | `checkpoints_deeper/20260917-194554` | 277 | 0.761 | 0.578 | 0.726 | 0.640 | 0.707 | 0.239 |
| 3 | `checkpoints_change_tcn_stem/lr_0.002/lr_sweep-0.002` | 277 | 0.756 | 0.506 | 0.721 | 0.574 | 0.616 | 0.287 |
| 4 | `checkpoints_deeper/20260917-204539` | 277 | 0.755 | 0.608 | 0.710 | 0.661 | 0.703 | 0.231 |
| 5 | `checkpoints_norm/20260917-211603` | 277 | 0.750 | 0.565 | 0.713 | 0.565 | 0.635 | 0.296 |
| 6 | `checkpoints_deeper/20260917-191728` | 277 | 0.746 | 0.495 | 0.719 | 0.588 | 0.634 | 0.274 |
| 7 | `checkpoints_norm/20260917-212927` | 277 | 0.673 | 0.515 | 0.655 | 0.612 | 0.676 | 0.278 |

Macro means, each corpus weighted once; ranked by `f_beat`. The per-track (micro) means are in the JSON, under each row's bare metric keys.

## `f_beat` per corpus

| corpus | n | #1 | #2 | #3 | #4 | #5 | #6 | #7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ballroom` | 104 | 0.834 | 0.860 | 0.819 | **0.863** | 0.832 | 0.779 | 0.844 |
| `jtd` | 96 | **0.957** | 0.955 | 0.950 | 0.936 | 0.917 | 0.951 | 0.913 |
| `gtzan` | 28 | 0.853 | 0.923 | 0.824 | **0.939** | 0.867 | 0.841 | 0.809 |
| `rwc_popular` | 19 | 0.811 | 0.784 | 0.782 | 0.828 | 0.808 | 0.772 | **0.884** |
| `rwc_genre` | 16 | **0.634** | 0.615 | 0.580 | 0.612 | 0.578 | 0.590 | 0.566 |
| `rwc_classical` | 7 | **0.555** | 0.456 | 0.541 | 0.403 | 0.490 | 0.538 | 0.184 |
| `rwc_jazz` | 7 | 0.699 | 0.736 | **0.795** | 0.703 | 0.757 | 0.747 | 0.511 |

Columns are the ranking positions above. Best per corpus in bold.

## Decode

| # | task | beat_thr | min_dist | gate_tol | decoder | switch_pen | anchor_thr | tuned on |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | beat_phase | 0.6 | 4 | 0.15 | global | 0.25 | 0.8 | `train` (50 tracks) |
| 2 | beat_phase | 0.8 | 1 | 0.3 | global | 1.0 | 0.8 | `train` (50 tracks) |
| 3 | beat_phase | 0.6 | 1 | 0.2 | global | 0.25 | 0.8 | `train` (50 tracks) |
| 4 | beat_phase | 0.8 | 4 | 0.15 | global | 0.25 | 0.8 | `train` (50 tracks) |
| 5 | beat_phase | 0.8 | 4 | 0.3 | global | 5.0 | 0.8 | `train` (50 tracks) |
| 6 | beat_phase | 0.6 | 4 | 0.15 | global | 0.25 | 0.8 | `train` (50 tracks) |
| 7 | beat_phase | 0.8 | 1 | 0.3 | global | 2.0 | 0.8 | `train` (50 tracks) |

Knobs tuned on a split the board does not report, so these numbers stay held out. `none` on `switch_pen` is the exact single-offset decode.

## Provenance

| # | checkpoint | measured | commit |
| --- | --- | --- | --- |
| 1 | `checkpoints_change_tcn_stem/lr_0.0008/lr_sweep-0.0008/beat-phase-epoch91-valloss1.4794.ckpt` | 2026-09-18 18:05 UTC | `f8a702c` |
| 2 | `checkpoints_deeper/20260917-194554/beat-phase-epoch119-valloss1.3981.ckpt` | 2026-09-18 17:57 UTC | `f8a702c` |
| 3 | `checkpoints_change_tcn_stem/lr_0.002/lr_sweep-0.002/beat-phase-epoch95-valloss1.4757.ckpt` | 2026-09-18 18:08 UTC | `f8a702c` |
| 4 | `checkpoints_deeper/20260917-204539/beat-phase-epoch103-valloss1.3894.ckpt` | 2026-09-18 17:59 UTC | `f8a702c` |
| 5 | `checkpoints_norm/20260917-211603/beat-phase-epoch27-valloss1.6707.ckpt` | 2026-09-18 18:01 UTC | `f8a702c` |
| 6 | `checkpoints_deeper/20260917-191728/beat-phase-epoch115-valloss1.5100.ckpt` | 2026-09-18 17:54 UTC | `f8a702c` |
| 7 | `checkpoints_norm/20260917-212927/beat-phase-epoch75-valloss1.6898.ckpt` | 2026-09-18 18:03 UTC | `f8a702c` |

The path is where that run's checkpoint was when it was scored, so a row leads straight to the model behind it.
