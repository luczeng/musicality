#!/usr/bin/env python3
"""Download an audio dataset using mirdata."""

from pathlib import Path

import click
import mirdata
import yaml

import musicality.dataformats as dataformats

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "download.yaml"


def _download_one(dataset: str, data_home: Path) -> None:
    dataset_dir = data_home / dataset
    click.echo(f"Downloading '{dataset}' to {dataset_dir} ...")
    ds = mirdata.initialize(dataset, data_home=str(dataset_dir))
    ds.download()
    for archive in list(dataset_dir.rglob("*.tar.gz")) + list(dataset_dir.rglob("*.zip")):
        archive.unlink()
        click.echo(f"Removed {archive}")
    click.echo(f"Done: {dataset}")


@click.command()
@click.argument("dataset", required=False)
@click.option(
    "--all", "download_all", is_flag=True,
    help=f"Download all datasets listed in configs/download.yaml.",
)
@click.option(
    "--list", "list_datasets", is_flag=True, help="List available datasets."
)
@click.option(
    "--data-home",
    default=DATA_DIR,
    show_default=True,
    help="Directory to download data into.",
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

    if download_all:
        with open(CONFIG_PATH) as f:
            names = yaml.safe_load(f)["datasets"]
        for name in names:
            _download_one(name, Path(data_home))
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

    _download_one(dataset, Path(data_home))


if __name__ == "__main__":
    main()
