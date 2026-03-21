from pathlib import Path

from torch.utils.data import Dataset, Subset, random_split


class Splitter:
    """Manages persistent train/val splits for a dataset.

    On the first run, randomly splits the dataset and saves the indices to
    `splits_dir` as `<key>_train.txt` and `<key>_val.txt`. On subsequent runs
    with the same key, the saved indices are reloaded so the split is identical.

    :param dataset: The dataset to split.
    :param splits_dir: Directory where split files are stored.
    :param key: Unique identifier for this split (e.g. dataset name + size + ratio).
    :param val_split: Fraction of the dataset to use for validation.
    """

    def __init__(self, dataset: Dataset, splits_dir: Path, key: str, val_split: float):

        self.dataset = dataset
        self.splits_dir = splits_dir
        self.key = key
        self.val_split = val_split

    def run(self) -> tuple[Subset, Subset]:
        """Return (train_ds, val_ds), loading from disk or creating a new split.

        :returns: Tuple of (train_ds, val_ds).
        :rtype: tuple[Subset, Subset]
        """

        existing = self._load()

        if existing is not None:
            train_indices, val_indices = existing
            return Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)

        n_val = int(len(self.dataset) * self.val_split)
        n_train = len(self.dataset) - n_val
        train_ds, val_ds = random_split(self.dataset, [n_train, n_val])

        self._save(list(train_ds.indices), list(val_ds.indices))

        return train_ds, val_ds

    def _load(self) -> tuple[list, list] | None:
        """Return (train_indices, val_indices) from disk, or None if not found.

        :returns: Tuple of index lists, or None if no split file exists.
        :rtype: tuple[list, list] or None
        """

        train_file = self.splits_dir / f"{self.key}_train.txt"
        val_file = self.splits_dir / f"{self.key}_val.txt"

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

        self.splits_dir.mkdir(exist_ok=True)

        (self.splits_dir / f"{self.key}_train.txt").write_text("\n".join(map(str, train_indices)))
        (self.splits_dir / f"{self.key}_val.txt").write_text("\n".join(map(str, val_indices)))
