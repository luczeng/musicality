#!/usr/bin/env python3
"""List datasets present in the data/ directory with their mirdata annotation types."""

import os
import mirdata

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

EXCLUDED = {"audio", "get_path"}


def get_annotations(mirdata_name):
    ds = mirdata.initialize(mirdata_name)
    return [a for a in dir(ds._track_class) if not a.startswith("_") and a not in EXCLUDED]


def main():
    available = mirdata.list_datasets()
    entries = sorted(e for e in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, e)))

    rows = []
    for name in entries:
        if name in available:
            annotations = ", ".join(get_annotations(name))
        else:
            annotations = "unknown"
        rows.append((name, annotations))

    col1 = max(len("Dataset"), max(len(r[0]) for r in rows))
    col2 = max(len("Annotations"), max(len(r[1]) for r in rows))

    header = f"{'Dataset':<{col1}}  {'Annotations':<{col2}}"
    print(header)
    print("-" * len(header))
    for name, annotations in rows:
        print(f"{name:<{col1}}  {annotations:<{col2}}")

    print(f"\n{len(entries)} dataset(s) found.")


if __name__ == "__main__":
    main()
