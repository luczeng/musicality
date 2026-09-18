"""Loads dataformat.yaml and exposes hardcoded directory names as a typed object."""

from dataclasses import dataclass
from pathlib import Path

import yaml


_YAML_PATH = Path(__file__).parent / "dataformat.yaml"

ROOT = Path(__file__).parent.parent.parent


@dataclass
class DataFormat:
    """Hardcoded directory names loaded from dataformat.yaml.

    :param data_dir: Root data directory name.
    :param splits_dir: Splits subdirectory name.
    :param leaderboard_dir: Subdirectory holding the running leaderboard. DVC
        tracked like ``splits``, so a board survives the machine that wrote it.
    :param tracks_dirname: Per-dataset subfolder holding custom-recorded audio.
    :param annotations_dirname: Per-dataset subfolder holding annotation sidecars.
    :param beats_suffix: File suffix for beat-time annotation files.
    :param metadata_suffix: File suffix for track metadata sidecar files.
    """

    data_dir: str
    splits_dir: str
    leaderboard_dir: str
    tracks_dirname: str
    annotations_dirname: str
    beats_suffix: str
    metadata_suffix: str


def load() -> DataFormat:
    """Load and return the DataFormat from dataformat.yaml.

    :returns: Populated DataFormat instance.
    :rtype: DataFormat
    """

    with _YAML_PATH.open() as f:
        raw = yaml.safe_load(f)

    return DataFormat(**raw)


# The canonical, load-once config. Other modules should read
# FORMAT/DATA_DIR/SPLITS_DIR/LEADERBOARD_DIR directly rather than calling load()
# again or re-exporting their own copies of individual fields, so there's
# exactly one place the on-disk layout is defined.
FORMAT = load()
DATA_DIR = ROOT / FORMAT.data_dir
SPLITS_DIR = ROOT / FORMAT.splits_dir
LEADERBOARD_DIR = ROOT / FORMAT.leaderboard_dir
