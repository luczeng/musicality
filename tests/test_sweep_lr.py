"""Where `tools/sweep_lr.py` puts each run of a sweep.

The sweep identifier is what keeps two sweeps on the same day from writing into
one directory — where `save_top_k` cannot prune the other sweep's checkpoints
and the second `training_report.json` overwrites the first. These pin the paths,
not anything about training.
"""

from pathlib import Path

from omegaconf import OmegaConf

from musicality.trainers.common import build_checkpoint_callback
from tools.sweep_lr import _compose, run_overrides, sweep_directory


def test_sweep_directory_sits_below_the_config_checkpoint_dir():
    cfg = OmegaConf.create({"checkpoint_dir": "checkpoints_beat/"})

    assert sweep_directory(cfg, "20260918-141530") == Path(
        "checkpoints_beat/lr_sweep-20260918-141530"
    )


def test_sweep_directory_follows_a_checkpoint_dir_override():
    """Derived from the config, so `--overrides checkpoint_dir=...` still wins."""

    cfg = OmegaConf.create({"checkpoint_dir": "/scratch/runs"})

    assert sweep_directory(cfg, "deeper-trunk") == Path(
        "/scratch/runs/lr_sweep-deeper-trunk"
    )


def test_two_sweeps_the_same_day_get_different_directories():
    cfg = OmegaConf.create({"checkpoint_dir": "checkpoints_beat/"})

    assert sweep_directory(cfg, "20260918-091500") != sweep_directory(
        cfg, "20260918-214500"
    )


def test_run_overrides_survive_hydra_and_land_in_the_sweep_directory():
    """Composed rather than string-compared: the quoted tag is the part that
    could fail in Hydra's override grammar."""

    sweep_dir = Path("checkpoints_beat/lr_sweep-20260918-141530")

    cfg = _compose(run_overrides(sweep_dir, "20260918-141530", 0.002))

    assert cfg.lr == 0.002
    assert Path(cfg.checkpoint_dir) == sweep_dir
    assert cfg.wandb.run_name == "lr_0.002"
    assert list(cfg.wandb.tags) == ["lr_sweep", "sweep-20260918-141530"]


def test_each_learning_rate_checkpoints_into_its_own_leaf():
    """End of the chain that matters: `build_checkpoint_callback` appends
    `wandb.run_name` to `checkpoint_dir`, so this is the directory the
    checkpoints and the run's `training_report.json` actually land in."""

    sweep_dir = Path("checkpoints_beat/lr_sweep-20260918-141530")

    dirpaths = [
        build_checkpoint_callback(
            _compose(run_overrides(sweep_dir, "20260918-141530", lr)), "beat-phase"
        ).dirpath
        for lr in (0.0008, 0.002)
    ]

    # Lightning absolutises `dirpath`; the last two segments are what this
    # test is about — the sweep's own folder, then one leaf per lr.
    tails = [Path(*Path(d).parts[-2:]) for d in dirpaths]

    assert tails == [
        Path("lr_sweep-20260918-141530/lr_0.0008"),
        Path("lr_sweep-20260918-141530/lr_0.002"),
    ]
