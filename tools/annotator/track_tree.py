"""The dataset → track tree the annotator picks tracks from.

Its own widget so that another tool can offer exactly the same selection —
``tools/plot_probabilities.py`` does. Datasets are listed with their track and
annotation counts and expand, lazily, into their tracks; every track row
carries the annotator's indicator columns: annotation saved (●/✕), annotated
meter, train/val split, and the warning/review flags.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .data import (
    annotation_meter_label,
    has_annotation,
    list_datasets,
    load_dataset_tracks,
    track_needs_review,
    track_split,
    track_warning,
)

# Child item standing in for a dataset's tracks until it is first expanded.
_PLACEHOLDER = "__loading__"


class TrackTree(QWidget):
    """A sort selector over a tree of datasets, each expanding into its tracks.

    Clicking a track emits :attr:`track_clicked` with ``(dataset_name,
    track_id)``, and clicking a dataset header emits :attr:`dataset_clicked`
    — what a click does is up to the owner. ``tree`` is the underlying
    :class:`QTreeWidget`, for an owner that adds a context menu of its own.
    """

    track_clicked = Signal(str, str)
    dataset_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._sort_combo = QComboBox()
        self._sort_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._sort_combo.addItem("Alphabetical", "alphabetical")
        self._sort_combo.addItem("Recording date (newest)", "recording_date")
        self._sort_combo.currentIndexChanged.connect(lambda _i: self.populate())

        sort_bar = QHBoxLayout()
        sort_bar.addWidget(QLabel("Sort:"))
        sort_bar.addWidget(self._sort_combo)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.header().hide()
        self.tree.setColumnWidth(0, 320)
        self.tree.setColumnWidth(1, 18)
        self.tree.setColumnWidth(2, 36)
        self.tree.setColumnWidth(3, 24)
        self.tree.setColumnWidth(4, 24)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tree.itemExpanded.connect(self._on_item_expanded)
        self.tree.itemClicked.connect(self._on_item_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        layout.addLayout(sort_bar)
        layout.addWidget(self.tree)

        self.populate()

    def populate(
        self,
        selected_dataset: str | None = None,
        selected_track: str | None = None,
    ) -> None:
        """(Re)build the dataset list from disk, expanding *selected_dataset*
        and selecting *selected_track* inside it."""

        self.tree.clear()
        bold = QFont()
        bold.setBold(True)
        infos = list_datasets()
        if self._sort_combo.currentData() == "recording_date":
            infos.sort(key=lambda info: info.mtime, reverse=True)
        else:
            infos.sort(key=lambda info: info.name.lower())
        for info in infos:
            suffix = f"  ({info.n_tracks} · {info.n_annotations} ann)"
            ds_item = QTreeWidgetItem([info.name + suffix])
            ds_item.setFont(0, bold)
            ds_item.setData(0, Qt.ItemDataRole.UserRole, info.name)
            ds_item.setFirstColumnSpanned(True)
            # Placeholder keeps the expand arrow visible until the user expands
            placeholder = QTreeWidgetItem([""])
            placeholder.setData(0, Qt.ItemDataRole.UserRole, _PLACEHOLDER)
            ds_item.addChild(placeholder)
            self.tree.addTopLevelItem(ds_item)

            if info.name == selected_dataset:
                ds_item.setExpanded(True)  # triggers lazy-load of children
                if selected_track:
                    self._select(ds_item, selected_track)

    def reload_dataset(self, dataset_name: str, selected_track: str) -> None:
        """Re-read an expanded dataset's tracks from disk and select
        *selected_track* — after a rename, say. A collapsed dataset is left
        alone: it reads its tracks afresh when next expanded anyway."""

        ds_item = self._dataset_item(dataset_name)
        if ds_item is None or not ds_item.isExpanded():
            return

        self._load_children(ds_item)
        self._select(ds_item, selected_track)

    def refresh_row(self, dataset_name: str, track_id: str) -> None:
        """Refresh one track's indicator columns without rebuilding the tree."""

        ds_item = self._dataset_item(dataset_name)
        if ds_item is None:
            return

        for j in range(ds_item.childCount()):
            child = ds_item.child(j)
            if child.data(0, Qt.ItemDataRole.UserRole) == track_id:
                self._set_row_indicators(child, dataset_name, track_id)
                return

    def _dataset_item(self, dataset_name: str) -> QTreeWidgetItem | None:

        for i in range(self.tree.topLevelItemCount()):
            ds_item = self.tree.topLevelItem(i)
            if ds_item.data(0, Qt.ItemDataRole.UserRole) == dataset_name:
                return ds_item

        return None

    def _select(self, ds_item: QTreeWidgetItem, track_id: str) -> None:

        for j in range(ds_item.childCount()):
            child = ds_item.child(j)
            if child.data(0, Qt.ItemDataRole.UserRole) == track_id:
                self.tree.setCurrentItem(child)
                self.tree.scrollToItem(child)
                return

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() != 1:
            return
        if item.child(0).data(0, Qt.ItemDataRole.UserRole) != _PLACEHOLDER:
            return
        self._load_children(item)

    def _load_children(self, item: QTreeWidgetItem) -> None:
        """(Re)populate *item*'s track children from disk, replacing whatever is there."""
        item.takeChildren()
        dataset_name = item.data(0, Qt.ItemDataRole.UserRole)
        for track_id in load_dataset_tracks(dataset_name):
            track_item = QTreeWidgetItem()
            track_item.setText(0, track_id)
            track_item.setData(0, Qt.ItemDataRole.UserRole, track_id)
            self._set_row_indicators(track_item, dataset_name, track_id)
            item.addChild(track_item)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        parent = item.parent()
        if parent is None:
            self.dataset_clicked.emit(item.data(0, Qt.ItemDataRole.UserRole))
            return

        track_id = item.data(0, Qt.ItemDataRole.UserRole)
        if track_id != _PLACEHOLDER:
            self.track_clicked.emit(parent.data(0, Qt.ItemDataRole.UserRole), track_id)

    @staticmethod
    def _set_row_indicators(
        item: QTreeWidgetItem, dataset_name: str, track_id: str
    ) -> None:
        item.setToolTip(1, "Has a saved annotation from this app")
        if has_annotation(dataset_name, track_id):
            item.setText(1, "●")
            item.setForeground(1, QColor("#44cc44"))
        else:
            item.setText(1, "✕")
            item.setForeground(1, QColor("#cc4444"))
        item.setToolTip(
            2,
            "Annotated meter (1..N bar positions) — "
            "• for beats with no position data, blank for no annotation",
        )
        item.setText(2, annotation_meter_label(dataset_name, track_id))

        item.setToolTip(
            3,
            "Train/val split (musicality_db/splits) — "
            "blank if no split has been generated for this dataset/track",
        )
        split = track_split(dataset_name, track_id)
        if split == "train":
            item.setText(3, "T")
            item.setForeground(3, QColor("#4488cc"))
        elif split == "val":
            item.setText(3, "V")
            item.setForeground(3, QColor("#cc8844"))
        else:
            item.setText(3, "")

        item.setToolTip(
            4,
            "⚠ Flagged as a suspicious annotation (toggle with the Flag button) — "
            "takes priority in this column over ❓ needs review (toggle with the "
            "To Review button)",
        )
        if track_warning(dataset_name, track_id):
            item.setText(4, "⚠")
            item.setForeground(4, QColor("#ffcc00"))
        elif track_needs_review(dataset_name, track_id):
            item.setText(4, "❓")
            item.setForeground(4, QColor("#aa66ff"))
        else:
            item.setText(4, "")
