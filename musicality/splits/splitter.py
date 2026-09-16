from pathlib import Path

from torch.utils.data import Dataset, Subset, random_split

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import TrackRef, load_metadata, resolve_track_audio


class MissingTrackDataError(FileNotFoundError):
    """A split lists tracks whose files are not on this machine.

    Raised by :func:`verify_refs_present`, and therefore by every path that
    reads a ``train.txt``/``val.txt`` (see :func:`_read_refs`). The failure it
    exists to stop is a partial data pull — e.g. ``dvc pull`` fetching some
    corpora but not others on a fresh remote instance. Before this check, a
    corpus missing from disk was dropped track by track with nothing but a
    printed line, so a run could train on a strictly smaller dataset than its
    split describes and report metrics as if nothing had happened.
    """


def missing_files(ref: TrackRef) -> list[Path]:
    """Return the files *ref* needs but doesn't have on disk.

    A track is loadable when both halves of this project's format are
    present: its audio under ``tracks/`` and its default-slot annotation
    under ``annotations/`` (see docs/source/data.rst). Either one absent
    makes the track unusable, so both are reported.

    :returns: The absent paths — empty when the track is fully present.
    """

    missing = []

    if resolve_track_audio(ref.dataset_name, ref.track_id, ref.data_home) is None:
        missing.append(
            ref.data_home / dataformats.FORMAT.tracks_dirname / f"{ref.track_id}.wav"
        )

    beats_path = (
        ref.data_home
        / dataformats.FORMAT.annotations_dirname
        / f"{ref.track_id}{dataformats.FORMAT.beats_suffix}"
    )
    if not beats_path.exists():
        missing.append(beats_path)

    return missing


def verify_refs_present(refs: list[TrackRef], context: str = "") -> None:
    """Raise unless every ref in *refs* resolves to files on disk.

    :param refs: Refs to check, typically one side of a split.
    :param context: Where the refs came from (a split file), named in the
        error so a failure points at the list to fix.
    :raises MissingTrackDataError: If any ref is missing its audio or its
        annotation, with a per-corpus breakdown — a partial pull usually
        takes out whole corpora, and the count per corpus is what tells that
        apart from a handful of individually broken tracks.
    """

    missing = [(ref, paths) for ref in refs if (paths := missing_files(ref))]

    if not missing:
        return

    per_dataset: dict[str, int] = {}
    for ref, _ in missing:
        per_dataset[ref.dataset_name] = per_dataset.get(ref.dataset_name, 0) + 1

    breakdown = "\n".join(
        f"  {name}: {count} of {sum(r.dataset_name == name for r in refs)} track(s)"
        for name, count in sorted(per_dataset.items(), key=lambda kv: -kv[1])
    )
    examples = "\n".join(f"  {path}" for _, paths in missing[:5] for path in paths[:2])

    where = f" in {context}" if context else ""

    raise MissingTrackDataError(
        f"{len(missing)} of {len(refs)} track(s){where} are not on disk:\n"
        f"{breakdown}\n"
        f"Missing files (first few):\n{examples}\n"
        "The split lists data this machine does not have — the usual cause is an "
        "incomplete `dvc pull` in the data directory (see tools/setup_remote.sh). "
        "Pull the missing corpora, or regenerate the split from the data you do have."
    )


def is_flagged(ref: TrackRef) -> bool:
    """Return whether *ref*'s annotation is flagged as not fit for training.

    Reads the track's default-slot metadata (the same slot
    ``TempoDataset``/``BeatDataset`` read) and reports True if either
    annotator flag is set:

    - ``warning`` — "this annotation looks wrong" (the annotator's ⚠ Flag).
    - ``needs_review`` — "take another look" (the annotator's ❓ To Review).

    A track with no metadata file at all is not flagged.
    """

    metadata = load_metadata(ref.dataset_name, ref.track_id, data_home=ref.data_home)

    if metadata is None:
        return False

    return bool(metadata.warning) or bool(metadata.needs_review)


def drop_flagged_refs(refs: list[TrackRef], context: str = "") -> list[TrackRef]:
    """Return *refs* without the tracks :func:`is_flagged` rejects, printing
    a one-line count when anything was dropped.

    :param refs: Refs to filter.
    :param context: Short label for the printed message (e.g. the split file
        the refs came from), so several calls in one run stay tellable apart.
    """

    kept = [ref for ref in refs if not is_flagged(ref)]

    n_flagged = len(refs) - len(kept)
    if n_flagged:
        where = f" in {context}" if context else ""
        print(
            f"[Splitter] {n_flagged} flagged track(s){where} skipped "
            f"(warning / needs_review)"
        )

    return kept


def _write_refs(path: Path, refs: list[TrackRef]) -> None:

    path.write_text("\n".join(f"{r.dataset_name}/{r.track_id}" for r in refs))


def _read_refs(path: Path, verify: bool = True) -> list[TrackRef]:
    """Read one side of a split file into refs.

    :param verify: Fail (:class:`MissingTrackDataError`) if any listed track
        is not on disk. On by default: a split is a claim about which tracks
        a run uses, so silently loading fewer of them makes every number that
        run reports a number about a different dataset. Turn it off only
        where the refs are read as labels rather than as data to load (the
        annotator's split badges).
    """

    refs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        dataset_name, track_id = line.split("/", 1)
        refs.append(
            TrackRef(dataset_name, track_id, dataformats.DATA_DIR / dataset_name)
        )

    refs = drop_flagged_refs(refs, context=f"{path.parent.name}/{path.name}")

    if verify:
        verify_refs_present(refs, context=f"{path.parent.name}/{path.name}")

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

    Every read path also checks that the tracks a split lists are actually
    on disk, and raises :class:`MissingTrackDataError` if they aren't (see
    :func:`verify_refs_present`) — a split is a claim about which tracks a
    run uses, and a partial data pull must stop the run rather than quietly
    shrink it.

    Tracks their annotator flagged (``warning`` or ``needs_review`` in the
    track metadata — see :func:`is_flagged`) are skipped on both sides:
    ``create()`` never writes them into a new split, and every read path
    (:meth:`run`, :meth:`load_refs`, :meth:`load_refs_from_dir`) drops them
    from splits saved before the flag was set. So flagging a track in the
    annotator takes it out of training and validation immediately, with no
    need to regenerate any split file.

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
        :raises MissingTrackDataError: If the split lists tracks that are not
            on disk.
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

        Flagged tracks (see :func:`is_flagged`) are excluded from the pool
        before splitting, so they land in neither side and ``val_split`` is
        taken as a fraction of the eligible tracks only.

        :returns: Tuple of (train_ds, val_ds). Their ``.indices`` index into
            ``self.dataset``, not into the filtered pool.
        :rtype: tuple[Subset, Subset]
        """

        eligible = [i for i, ref in enumerate(self.dataset.refs) if not is_flagged(ref)]

        n_val = int(len(eligible) * self.val_split)
        n_train = len(eligible) - n_val

        # Split positions within `eligible`, then map back, so the returned
        # Subsets stay indexed against the full dataset.
        train_pos, val_pos = random_split(range(len(eligible)), [n_train, n_val])
        train_indices = [eligible[i] for i in train_pos.indices]
        val_indices = [eligible[i] for i in val_pos.indices]

        train_refs = [self.dataset.refs[i] for i in train_indices]
        val_refs = [self.dataset.refs[i] for i in val_indices]

        Splitter.save_refs(self.splits_dir, self.name, train_refs, val_refs)

        print(
            f"[Splitter] Split saved to {self.splits_dir / self.name} ({n_train} train, {n_val} val)"
        )

        return Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)

    def _load(self) -> tuple[list[int], list[int]] | None:
        """Return (train_indices, val_indices) into ``self.dataset``, or None
        if no split file exists.

        Each saved track not found in ``self.dataset.refs`` is dropped, with
        a printed warning, rather than failing. By the time this runs the
        split's tracks are known to be on disk (:func:`_read_refs` verifies
        that), so a drop here means the dataset itself filtered the track out
        — e.g. ``binary_only`` rejecting a waltz — not missing data.

        :returns: Tuple of index lists, or None if no split file exists.
        :rtype: tuple[list, list] or None
        """

        split_path = self.splits_dir / self.name
        train_file = split_path / "train.txt"
        val_file = split_path / "val.txt"

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
    def load_refs(
        splits_dir: Path, name: str, verify: bool = True
    ) -> tuple[list[TrackRef], list[TrackRef]]:
        """Return a split's ``(train_refs, val_refs)`` directly — no parent
        dataset needed. The read counterpart to :meth:`save_refs`, and what
        lets ``TempoDataset``/``BeatDataset`` be built straight from a split
        via their ``refs=`` argument, for a plain or a merged name alike.

        :param verify: See :meth:`load_refs_from_dir`.
        :raises FileNotFoundError: If no split has been generated for ``name`` yet.
        :raises MissingTrackDataError: If ``verify`` and any listed track is
            not on disk.
        """

        try:
            return Splitter.load_refs_from_dir(splits_dir / name, verify=verify)
        except MissingTrackDataError:
            # A FileNotFoundError subclass, so it would otherwise be rewritten
            # below into "no split found" — the opposite of what happened: the
            # split is right there, it's the data it names that is missing.
            raise
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No split found for '{name}' in {splits_dir}. Run "
                f"`uv run python tools/create_splits.py` (or "
                f"`tools/merge_datasets.py` for a merged name) to generate it."
            ) from None

    @staticmethod
    def load_refs_from_dir(
        split_path: Path, verify: bool = True
    ) -> tuple[list[TrackRef], list[TrackRef]]:
        """Return ``(train_refs, val_refs)`` read straight from *split_path*'s
        ``train.txt``/``val.txt`` — the same format :meth:`load_refs` reads,
        but for a folder anywhere on disk rather than one registered under a
        canonical ``splits_dir`` by name. Lets a training config point
        directly at a folder of lists (``data.input``, when it contains a
        ``/``) without going through ``splits_dir`` lookup at all.

        :param verify: Fail if a listed track is not on disk — see
            :func:`_read_refs`. Leave it on for anything that will load the
            audio; pass ``False`` only to read a split as a list of names.
        :raises FileNotFoundError: If *split_path* has no ``train.txt``/``val.txt``.
        :raises MissingTrackDataError: If ``verify`` and any listed track is
            not on disk.
        """

        train_file = split_path / "train.txt"
        val_file = split_path / "val.txt"

        if not (train_file.exists() and val_file.exists()):
            raise FileNotFoundError(f"No split found at {split_path}.")

        return _read_refs(train_file, verify), _read_refs(val_file, verify)

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

        split_path = splits_dir / name
        split_path.mkdir(parents=True, exist_ok=True)

        _write_refs(split_path / "train.txt", train_refs)
        _write_refs(split_path / "val.txt", val_refs)
