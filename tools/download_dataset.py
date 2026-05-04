#!/usr/bin/env python3
"""Download an audio dataset using mirdata."""

from pathlib import Path

import click
import mirdata
import yaml

import musicality.dataformats as dataformats

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "download.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _is_present(dataset_dir: Path) -> bool:
    return dataset_dir.exists() and any(dataset_dir.iterdir())


def _download_one(dataset: str, data_home: Path) -> bool:
    """Download one dataset. Returns True on success, False on failure."""
    dataset_dir = data_home / dataset
    if _is_present(dataset_dir):
        click.echo(f"Skipping '{dataset}' — already present at {dataset_dir}")
        return True
    click.echo(f"Downloading '{dataset}' to {dataset_dir} ...")
    ds = mirdata.initialize(dataset, data_home=str(dataset_dir))
    try:
        ds.download()
    except Exception as e:
        click.echo(f"WARNING: Failed to download '{dataset}': {e}", err=True)
        return False
    for archive in list(dataset_dir.rglob("*.tar.gz")) + list(dataset_dir.rglob("*.zip")):
        archive.unlink()
        click.echo(f"Removed {archive}")
    click.echo(f"Done: {dataset}")
    return True


@click.command()
@click.argument("dataset", required=False)
@click.option(
    "--all", "download_all", is_flag=True,
    help="Download all datasets listed in configs/download.yaml.",
)
@click.option(
    "--list", "list_datasets", is_flag=True, help="List available datasets."
)
@click.option(
    "--data-home", default=None,
    help="Directory to download data into. Defaults to data_home in configs/download.yaml.",
)
def main(dataset, download_all, list_datasets, data_home):
    """Download a mirdata DATASET to the data folder.

    Examples:
      python tools/download_dataset.py ballroom
      python tools/download_dataset.py --all
    """

    if list_datasets:
        click.echo("Available datasets:")
        for name in sorted(mirdata.list_datasets()):
            click.echo(f"  {name}")
        return

    cfg = _load_config()
    resolved_data_home = Path(data_home) if data_home else Path(cfg["data_home"])

    if download_all:
        failed = [n for n in cfg["datasets"] if not _download_one(n, resolved_data_home)]
        if failed:
            click.echo(f"\nFailed datasets: {', '.join(failed)}", err=True)
            raise SystemExit(1)
        return

    if not dataset:
        click.echo(
            "Provide a dataset name, use --all, or use --list to see available datasets.",
            err=True,
        )
        raise SystemExit(1)

    if dataset not in mirdata.list_datasets():
        click.echo(f"Unknown dataset: {dataset}", err=True)
        click.echo("Run with --list to see available datasets.", err=True)
        raise SystemExit(1)

    if not _download_one(dataset, resolved_data_home):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
