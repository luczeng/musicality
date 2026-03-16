#!/usr/bin/env python3
"""Download an audio dataset using mirdata."""

import os
import click
import mirdata

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@click.command()
@click.argument("dataset", required=False)
@click.option("--list", "list_datasets", is_flag=True, help="List available datasets.")
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
        click.echo("Provide a dataset name or use --list to see available datasets.", err=True)
        raise SystemExit(1)

    if dataset not in mirdata.list_datasets():
        click.echo(f"Unknown dataset: {dataset}", err=True)
        click.echo("Run with --list to see available datasets.", err=True)
        raise SystemExit(1)

    click.echo(f"Downloading '{dataset}' to {data_home} ...")
    ds = mirdata.initialize(dataset, data_home=data_home)
    ds.download()
    click.echo("Done.")


if __name__ == "__main__":
    main()
