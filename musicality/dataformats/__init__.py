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
    :param brid_dir: BRID dataset subdirectory name.
    :param splits_dir: Splits subdirectory name.
    """

    data_dir: str
    brid_dir: str
    splits_dir: str


def load() -> DataFormat:
    """Load and return the DataFormat from dataformat.yaml.

    :returns: Populated DataFormat instance.
    :rtype: DataFormat
    """

    with _YAML_PATH.open() as f:
        raw = yaml.safe_load(f)

    return DataFormat(**raw)
