from pathlib import Path

from torch.utils.data import Dataset, Subset, random_split


class Splitter:
    """Loads persistent train/val splits for a dataset.

    Splits are stored under ``splits_dir/<name>/train.txt`` and ``val.txt``.
    ``run()`` only ever reads these files — it never generates a split, so the
    same files (typically version-controlled via DVC) produce the same split
    on every machine. Use ``create()`` (or ``tools/create_splits.py``) to
    generate the files in the first place.

    :param dataset: The dataset to split.
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

        self._save(list(train_ds.indices), list(val_ds.indices))

        print(
            f"[Splitter] Split saved to {self.splits_dir / self.name} ({n_train} train, {n_val} val)"
        )

        return train_ds, val_ds

    def _load(self) -> tuple[list, list] | None:
        """Return (train_indices, val_indices) from disk, or None if not found.

        :returns: Tuple of index lists, or None if no split file exists.
        :rtype: tuple[list, list] or None
        """

        split_dir = self.splits_dir / self.name

        train_file = split_dir / "train.txt"
        val_file = split_dir / "val.txt"

        if train_file.exists() and val_file.exists():
            train_indices = list(map(int, train_file.read_text().splitlines()))
            val_indices = list(map(int, val_file.read_text().splitlines()))
            return train_indices, val_indices

        return None

    def _save(self, train_indices: list, val_indices: list) -> None:
        """Persist train and val indices to disk as plain text files.

        :param train_indices: List of training sample indices.
        :param val_indices: List of validation sample indices.
        """

        split_dir = self.splits_dir / self.name
        split_dir.mkdir(parents=True, exist_ok=True)

        (split_dir / "train.txt").write_text("\n".join(map(str, train_indices)))
        (split_dir / "val.txt").write_text("\n".join(map(str, val_indices)))
