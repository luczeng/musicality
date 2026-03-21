#!/usr/bin/env python3
"""Download an audio dataset using mirdata."""

from pathlib import Path

import click
import mirdata

import musicality.dataformats as dataformats

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir


@click.command()
@click.argument("dataset", required=False)
@click.option(
    "--list", "list_datasets", is_flag=True, help="List available datasets."
)
@click.option(
    "--data-home",
    default=DATA_DIR,
    show_default=True,
    help="Directory to download data into.",
)
def main(dataset, list_datasets, data_home):
    """Download a mirdata DATASET to the data folder.

    Example: python tools/download_dataset.py gtzan_genre
    """

    if list_datasets:
        click.echo("Available datasets:")
        for name in sorted(mirdata.list_datasets()):
            click.echo(f"  {name}")
        return

    if not dataset:
        click.echo(
            "Provide a dataset name or use --list to see available datasets.",
            err=True,
        )
        raise SystemExit(1)

    if dataset not in mirdata.list_datasets():
        click.echo(f"Unknown dataset: {dataset}", err=True)
        click.echo("Run with --list to see available datasets.", err=True)
        raise SystemExit(1)

    dataset_dir = Path(data_home) / dataset
    click.echo(f"Downloading '{dataset}' to {dataset_dir} ...")

    ds = mirdata.initialize(dataset, data_home=str(dataset_dir))
    ds.download()

    for archive in list(dataset_dir.rglob("*.tar.gz")) + list(dataset_dir.rglob("*.zip")):
        archive.unlink()
        click.echo(f"Removed {archive}")

    click.echo("Done.")


if __name__ == "__main__":
    main()
