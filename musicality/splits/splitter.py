from pathlib import Path

from torch.utils.data import Dataset, Subset, random_split

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef


def _write_refs(path: Path, refs: list[TrackRef]) -> None:

    path.write_text("\n".join(f"{r.dataset_name}/{r.track_id}" for r in refs))


def _read_refs(path: Path) -> list[TrackRef]:

    refs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        dataset_name, track_id = line.split("/", 1)
        refs.append(
            TrackRef(dataset_name, track_id, dataformats.DATA_DIR / dataset_name)
        )

    return refs


class Splitter:
    """Loads persistent train/val splits for a dataset.

    Splits are stored under ``splits_dir/<name>/train.txt`` and ``val.txt``,
    one ``dataset_name/track_id`` line per track (see
    :class:`~musicality.dataformats.track_io.TrackRef`) rather than
    positional indices — so a split's contents can be resolved, concatenated
    with another dataset's split, or loaded standalone (see
    :meth:`load_refs`) independently of any one dataset instance's ordering.
    This is what lets ``tools/merge_datasets.py`` combine several datasets'
    existing splits into one.

    ``run()`` only ever reads these files — it never generates a split, so the
    same files (typically version-controlled via DVC) produce the same split
    on every machine. Use ``create()`` (or ``tools/create_splits.py``) to
    generate the files in the first place.

    :param dataset: The dataset to split. Must expose ``.refs`` — a
        ``list[TrackRef]`` parallel to its samples (both ``TempoDataset``
        and ``BeatDataset`` do).
    :param splits_dir: Root directory where split subfolders are stored.
    :param name: Dataset name, used as the subfolder under ``splits_dir``.
    :param val_split: Fraction of the dataset to use for validation.
    """

    def __init__(self, dataset: Dataset, splits_dir: Path, name: str, val_split: float):

        self.dataset = dataset
        self.splits_dir = splits_dir
        self.name = name
        self.val_split = val_split

    def run(self) -> tuple[Subset, Subset]:
        """Return (train_ds, val_ds) loaded from disk.

        :raises FileNotFoundError: If no split has been generated for ``name`` yet.
        :returns: Tuple of (train_ds, val_ds).
        :rtype: tuple[Subset, Subset]
        """

        existing = self._load()

        if existing is None:
            raise FileNotFoundError(
                f"No split found for '{self.name}' in {self.splits_dir}. "
                f"Run `uv run python tools/create_splits.py` to generate it."
            )

        train_indices, val_indices = existing
        print(
            f"[Splitter] Loaded existing split '{self.name}' ({len(train_indices)} train, {len(val_indices)} val)"
        )
        return Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)

    def create(self) -> tuple[Subset, Subset]:
        """Generate a new split and persist it to disk, overwriting any existing one.

        :returns: Tuple of (train_ds, val_ds).
        :rtype: tuple[Subset, Subset]
        """

        n_val = int(len(self.dataset) * self.val_split)
        n_train = len(self.dataset) - n_val
        train_ds, val_ds = random_split(self.dataset, [n_train, n_val])

        train_refs = [self.dataset.refs[i] for i in train_ds.indices]
        val_refs = [self.dataset.refs[i] for i in val_ds.indices]

        Splitter.save_refs(self.splits_dir, self.name, train_refs, val_refs)

        print(
            f"[Splitter] Split saved to {self.splits_dir / self.name} ({n_train} train, {n_val} val)"
        )

        return train_ds, val_ds

    def _load(self) -> tuple[list[int], list[int]] | None:
        """Return (train_indices, val_indices) into ``self.dataset``, or None
        if no split file exists.

        Each saved track not found in ``self.dataset.refs`` (e.g. removed
        since the split was created) is dropped, with a printed warning,
        rather than failing.

        :returns: Tuple of index lists, or None if no split file exists.
        :rtype: tuple[list, list] or None
        """

        split_dir = self.splits_dir / self.name
        train_file = split_dir / "train.txt"
        val_file = split_dir / "val.txt"

        if not (train_file.exists() and val_file.exists()):
            return None

        ref_index = {
            (ref.dataset_name, ref.track_id): i
            for i, ref in enumerate(self.dataset.refs)
        }

        def _indices(refs: list[TrackRef]) -> list[int]:

            indices = []
            n_missing = 0

            for ref in refs:
                idx = ref_index.get((ref.dataset_name, ref.track_id))
                if idx is None:
                    n_missing += 1
                    continue
                indices.append(idx)

            if n_missing:
                print(
                    f"[Splitter] '{self.name}': {n_missing} saved track(s) not found "
                    f"in the current dataset, dropped"
                )

            return indices

        return _indices(_read_refs(train_file)), _indices(_read_refs(val_file))

    @staticmethod
    def load_refs(splits_dir: Path, name: str) -> tuple[list[TrackRef], list[TrackRef]]:
        """Return a split's ``(train_refs, val_refs)`` directly — no parent
        dataset needed. The read counterpart to :meth:`save_refs`, and what
        lets ``TempoDataset``/``BeatDataset`` be built straight from a split
        via their ``refs=`` argument, for a plain or a merged name alike.

        :raises FileNotFoundError: If no split has been generated for ``name`` yet.
        """

        try:
            return Splitter.load_refs_from_dir(splits_dir / name)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No split found for '{name}' in {splits_dir}. Run "
                f"`uv run python tools/create_splits.py` (or "
                f"`tools/merge_datasets.py` for a merged name) to generate it."
            ) from None

    @staticmethod
    def load_refs_from_dir(split_dir: Path) -> tuple[list[TrackRef], list[TrackRef]]:
        """Return ``(train_refs, val_refs)`` read straight from *split_dir*'s
        ``train.txt``/``val.txt`` — the same format :meth:`load_refs` reads,
        but for a folder anywhere on disk rather than one registered under a
        canonical ``splits_dir`` by name. Lets a training config point
        directly at a folder of lists (``data.split_dir``) without going
        through ``splits_dir``/``data.name`` lookup at all.

        :raises FileNotFoundError: If *split_dir* has no ``train.txt``/``val.txt``.
        """

        train_file = split_dir / "train.txt"
        val_file = split_dir / "val.txt"

        if not (train_file.exists() and val_file.exists()):
            raise FileNotFoundError(f"No split found at {split_dir}.")

        return _read_refs(train_file), _read_refs(val_file)

    @staticmethod
    def save_refs(
        splits_dir: Path,
        name: str,
        train_refs: list[TrackRef],
        val_refs: list[TrackRef],
    ) -> None:
        """Persist ``(train_refs, val_refs)`` to ``splits_dir/<name>/``. The
        write counterpart to :meth:`load_refs`; also what :meth:`create`
        uses internally.
        """

        split_dir = splits_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)

        _write_refs(split_dir / "train.txt", train_refs)
        _write_refs(split_dir / "val.txt", val_refs)
