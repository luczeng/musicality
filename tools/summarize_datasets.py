#!/usr/bin/env python3
"""List datasets present in the data/ directory with their mirdata annotation types."""

from pathlib import Path

import mirdata

import musicality.dataformats as dataformats

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir

EXCLUDED = {"audio", "get_path"}


def get_annotations(mirdata_name):

    ds = mirdata.initialize(mirdata_name)

    return [a for a in dir(ds._track_class) if not a.startswith("_") and a not in EXCLUDED]


def count_songs(name, path):

    available = mirdata.list_datasets()

    if name in available:
        try:
            ds = mirdata.initialize(name, data_home=str(path))
            return len(ds.track_ids)
        except Exception:
            pass

    # fallback: count files recursively
    total = sum(1 for f in Path(path).rglob("*") if f.is_file())

    return total


def count_annotated(name, path, annotation_fields):
    """Count tracks that have at least one non-None value for the given annotation fields.

    :param name: mirdata dataset name.
    :param path: Path to the dataset directory.
    :param annotation_fields: List of annotation attribute names to check.
    :returns: Number of annotated tracks, or None if the dataset can't be loaded.
    :rtype: int or None
    """

    try:
        ds = mirdata.initialize(name, data_home=str(path))
    except Exception:
        return None

    count = 0

    for tid in ds.track_ids:
        track = ds.track(tid)
        if any(getattr(track, field, None) is not None for field in annotation_fields):
            count += 1

    return count


def main():

    available = mirdata.list_datasets()
    entries = sorted(e.name for e in DATA_DIR.iterdir() if e.is_dir())

    rows = []
    for name in entries:
        path = DATA_DIR / name
        songs = count_songs(name, path)

        if name in available:
            annotation_fields = get_annotations(name)
            annotations = ", ".join(annotation_fields)
            annotated = count_annotated(name, path, annotation_fields)
            annotated_str = f"{annotated}/{songs}" if annotated is not None else "?"
        else:
            annotations = "unknown"
            annotated_str = "?"

        rows.append((name, songs, annotated_str, annotations))

    col1 = max(len("Dataset"), max(len(r[0]) for r in rows))
    col2 = max(len("Tracks"), max(len(str(r[1])) for r in rows))
    col3 = max(len("Annotated"), max(len(r[2]) for r in rows))
    col4 = max(len("Annotations"), max(len(r[3]) for r in rows))

    header = f"{'Dataset':<{col1}}  {'Tracks':>{col2}}  {'Annotated':>{col3}}  {'Annotations':<{col4}}"
    print(header)
    print("-" * len(header))

    for name, songs, annotated_str, annotations in rows:
        print(f"{name:<{col1}}  {songs:>{col2}}  {annotated_str:>{col3}}  {annotations:<{col4}}")

    print(f"\n{len(entries)} dataset(s) found.")


if __name__ == "__main__":
    main()
