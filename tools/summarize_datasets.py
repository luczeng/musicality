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
            ds = mirdata.initialize(name, data_home=path)
            return len(ds.track_ids)
        except Exception:
            pass

    # fallback: count files recursively
    total = sum(1 for f in Path(path).rglob("*") if f.is_file())

    return total


def main():

    available = mirdata.list_datasets()
    entries = sorted(e.name for e in DATA_DIR.iterdir() if e.is_dir())

    rows = []
    for name in entries:
        path = DATA_DIR / name
        annotations = ", ".join(get_annotations(name)) if name in available else "unknown"
        songs = count_songs(name, path)
        rows.append((name, songs, annotations))

    col1 = max(len("Dataset"), max(len(r[0]) for r in rows))
    col2 = max(len("Songs"), max(len(str(r[1])) for r in rows))
    col3 = max(len("Annotations"), max(len(r[2]) for r in rows))

    header = f"{'Dataset':<{col1}}  {'Songs':>{col2}}  {'Annotations':<{col3}}"
    print(header)
    print("-" * len(header))

    for name, songs, annotations in rows:
        print(f"{name:<{col1}}  {songs:>{col2}}  {annotations:<{col3}}")

    print(f"\n{len(entries)} dataset(s) found.")


if __name__ == "__main__":
    main()
