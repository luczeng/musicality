#!/usr/bin/env python3
"""List datasets present in the data/ directory with their mirdata annotation types."""

from pathlib import Path

import mirdata

import musicality.dataformats as dataformats

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir

EXCLUDED = {"audio", "get_path"}


def get_annotations(mirdata_name):

    ds = mirdata.initialize(mirdata_name)

    return [
        a for a in dir(ds._track_class) if not a.startswith("_") and a not in EXCLUDED
    ]


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


def max_beat_phase(track):
    """Get the highest bar-position label in a track's beat annotation, if any.

    Bar-position labels count a beat's place within the bar (1, 2, 3, 4, 1, ...),
    so the max value indicates how finely the dataset marks metrical position
    (e.g. 2 for downbeat/backbeat only, 4 for a full quarter-note grid).

    :param track: A mirdata track instance.
    :returns: The highest position label found, or None if the track has no positions.
    :rtype: int or None
    """

    beats = getattr(track, "beats", None)
    positions = getattr(beats, "positions", None)

    if positions is None or len(positions) == 0:
        return None

    return int(max(positions))


def count_annotated(name, path, annotation_fields):
    """Count tracks that have at least one non-None value for the given annotation fields,
    and find the finest bar-phase granularity present in the beat annotations.

    :param name: mirdata dataset name.
    :param path: Path to the dataset directory.
    :param annotation_fields: List of annotation attribute names to check.
    :returns: (annotated count, max bar-phase label seen across all tracks, or None),
        or (None, None) if the dataset can't be loaded.
    :rtype: tuple[int | None, int | None]
    """

    try:
        ds = mirdata.initialize(name, data_home=str(path))

        count = 0
        max_phase = None

        for tid in ds.track_ids:
            track = ds.track(tid)

            if any(
                getattr(track, field, None) is not None for field in annotation_fields
            ):
                count += 1

            track_max_phase = max_beat_phase(track)
            if track_max_phase is not None:
                max_phase = max(max_phase or 0, track_max_phase)

        return count, max_phase
    except Exception:
        return None, None


def main():

    available = mirdata.list_datasets()
    entries = sorted(e.name for e in DATA_DIR.iterdir() if e.is_dir())

    rows = []
    for name in entries:
        path = DATA_DIR / name
        songs = count_songs(name, path)

        if name in available:
            annotation_fields = get_annotations(name)
            annotated, max_phase = count_annotated(name, path, annotation_fields)

            display_fields = [
                f"{f} (bar-phase 1-{max_phase})" if f == "beats" and max_phase else f
                for f in annotation_fields
            ]
            annotations = ", ".join(display_fields)

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
        print(
            f"{name:<{col1}}  {songs:>{col2}}  {annotated_str:>{col3}}  {annotations:<{col4}}"
        )

    print(f"\n{len(entries)} dataset(s) found.")


if __name__ == "__main__":
    main()
