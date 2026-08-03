#!/usr/bin/env python3
"""Launch beat-phase (beat/one/last) training with Hydra config."""

import musicality._hydra_compat  # noqa: F401

import hydra
from omegaconf import DictConfig

from musicality.trainers.train_beat_phase import train


@hydra.main(config_path="../configs", config_name="beat_train", version_base="1.3")
def launch_training(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    launch_training()
