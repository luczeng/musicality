#!/usr/bin/env python3
"""Launch tempo estimation training with Hydra config."""

import hydra
from omegaconf import DictConfig

from musicality.trainers.train import train


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def launch_training(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    launch_training()
