"""Main application window for track inspection and beat annotation."""

from __future__ import annotations

import datetime
import platform
import time
import librosa
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import musicality.dataformats as dataformats

from .audio import AudioEngine
from .inference import (
    EVAL_DEFAULTS,
    checkpoint_label,
    infer_beats,
    list_checkpoints,
    load_module,
)
from .recorder import Recorder, _SR as _REC_SR
from .data import (
    DEFAULT_N_BEATS,
    TrackData,
    TrackMetadata,
    active_bar_index,
    active_beat_position,
    add_beat,
    annotation_meter_label,
    annotation_path,
    bar_indices,
    beats_per_bar,
    cycle_positions,
    has_annotation,
    is_accent_beat,
    list_datasets,
    load_dataset_tracks,
    load_metadata,
    load_track,
    delete_track,
    remove_beat,
    rename_track,
    save_annotations,
    save_metadata,
    tempo_from_beats,
)
from .metronome_widget import MetronomeWidget
from .tap_tempo_widget import TapTempoWidget
from .waveform_widget import WaveformWidget

_TICK_MS = 30  # ~33 fps refresh rate


class MainWindow(QMainWindow):
    """Main window assembling waveform view, metronome, and audio engine.

    Layout
    ------
    Top-left  : controls — playback, recording, speed, count, accent, structure,
                beat inference, delete/rename
    Top-right : read-only track info — beat analytics + captured metadata
    Center    : inferred-beats WaveformWidget strip (hidden until "Infer Beats"
                is run), then the main WaveformWidget (resizable, takes
                remaining space)
    Bottom    : MetronomeWidget, then TapTempoWidget (fixed height)

    Beat inference
    --------------
    "Infer Beats" runs a trained beat-phase checkpoint (picked from the
    dropdown, scanned from ``checkpoints_beat/``) on the full currently-loaded
    track and shows the predicted beat times on a second waveform strip above
    the main one, so they never overlap the manually annotated beats below.
    Switching "Clicks" from Manual to Inferred plays the audible click track
    against the inferred beats instead, so the prediction can be heard
    against the audio.

    Keyboard shortcuts
    ------------------
    Space      : tap tempo
    P          : play / pause
    Left/Right : previous / next track
    Ctrl+S     : save annotations
    Ctrl+click : add beat at clicked position on waveform
    Ctrl+right-click : remove beat nearest to clicked position
    """

    _playback_finished = Signal()

    def __init__(
        self,
        dataset_name: str,
        track_ids: list[str],
        index: int = 0,
    ) -> None:
        super().__init__()
        self._dataset_name = dataset_name
        self._track_ids = track_ids
        self._index = index
        self._track: TrackData | None = None
        self._track_audio: np.ndarray | None = None
        self._track_sr: int = 44100
        self._inferred_beat_times: np.ndarray = np.array([])
        self._inferred_beat_positions: np.ndarray | None = None
        self._beat_module = None
        self._beat_module_path = None
        self._beat_module_task = None
        self._engine = AudioEngine()
        self._recorder = Recorder()
        self._timer = QTimer(self)
        self._n_beats = DEFAULT_N_BEATS
        self._accent_bars: float = 1.0
        self._record_start: float = 0.0
        self._record_tick: int = 0

        self._setup_ui()

        self._playback_finished.connect(self._on_playback_finished)
        self._engine.on_finished(self._playback_finished.emit)

        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._load_track(self._index)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self._prev_btn = QPushButton("◀  Prev")
        self._prev_btn.setFixedWidth(90)
        self._prev_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._prev_btn.clicked.connect(self._on_prev)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedWidth(90)
        self._play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_btn.clicked.connect(self._on_play)

        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.setFixedWidth(90)
        self._pause_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._pause_btn.clicked.connect(self._on_pause)

        self._next_btn = QPushButton("Next  ▶")
        self._next_btn.setFixedWidth(90)
        self._next_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._next_btn.clicked.connect(self._on_next)

        self._save_btn = QPushButton("💾  Save")
        self._save_btn.setFixedWidth(90)
        self._save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._save_btn.clicked.connect(self._on_save)

        self._delete_btn = QPushButton("🗑  Annotation")
        self._delete_btn.setFixedWidth(105)
        self._delete_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._delete_btn.clicked.connect(self._on_delete)

        self._delete_track_btn = QPushButton("🗑  Track")
        self._delete_track_btn.setFixedWidth(90)
        self._delete_track_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._delete_track_btn.clicked.connect(self._on_delete_track)

        self._rename_btn = QPushButton("✏  Rename track")
        self._rename_btn.setFixedWidth(120)
        self._rename_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._rename_btn.clicked.connect(self._on_rename)

        self._record_dataset_edit = QLineEdit("swing")
        self._record_dataset_edit.setFixedWidth(110)
        self._record_dataset_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._record_dataset_edit.setPlaceholderText("dataset")

        # Optional — who annotated this track. Purely descriptive metadata,
        # like Device/Location: doesn't affect where the annotation is saved
        # (that's TrackData.annotator_id, the multi-annotator slot selector,
        # left alone here).
        self._author_edit = QLineEdit()
        self._author_edit.setFixedWidth(140)
        self._author_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._author_edit.setPlaceholderText("e.g. luc")

        self._record_btn = QPushButton("⏺  Record new track")
        self._record_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self._on_record_toggle)

        self._elapsed_label = QLabel("")
        self._elapsed_label.setStyleSheet("color: #cc4444; font-weight: bold;")
        self._elapsed_label.setVisible(False)

        self._restart_btn = QPushButton("⏮  Restart")
        self._restart_btn.setFixedWidth(95)
        self._restart_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._restart_btn.clicked.connect(self._on_restart)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(80)
        self._volume_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._volume_slider.valueChanged.connect(
            lambda v: self._engine.set_volume(v / 100)
        )

        self._click_btn = QPushButton("🥁  Click")
        self._click_btn.setFixedWidth(80)
        self._click_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._click_btn.setCheckable(True)
        self._click_btn.clicked.connect(
            lambda checked: self._engine.set_click_enabled(checked)
        )

        self._click_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._click_volume_slider.setRange(0, 100)
        self._click_volume_slider.setValue(70)
        self._click_volume_slider.setFixedWidth(80)
        self._click_volume_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._click_volume_slider.valueChanged.connect(
            lambda v: self._engine.set_click_volume(v / 100)
        )

        self._click_source_group = QButtonGroup(self)
        self._click_source_group.setExclusive(True)
        self._click_source_buttons: list[QPushButton] = []
        for label in ("Manual", "Inferred"):
            btn = QPushButton(label)
            btn.setFixedWidth(70)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setChecked(label == "Manual")
            btn.clicked.connect(lambda _checked: self._update_engine_clicks())
            self._click_source_group.addButton(btn)
            self._click_source_buttons.append(btn)

        self._checkpoint_combo = QComboBox()
        self._checkpoint_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._checkpoint_combo.setFixedWidth(260)
        checkpoints = list_checkpoints()
        for path in checkpoints:
            self._checkpoint_combo.addItem(checkpoint_label(path), path)
        if not checkpoints:
            self._checkpoint_combo.addItem("(no checkpoints found)", None)
            self._checkpoint_combo.setEnabled(False)

        self._infer_btn = QPushButton("🔮  Infer Beats")
        self._infer_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._infer_btn.setEnabled(bool(checkpoints))
        self._infer_btn.clicked.connect(self._on_infer_beats)

        self._track_label = QLabel()
        self._track_label.setStyleSheet("font-weight: bold;")

        self._stats_label = QLabel()
        self._stats_label.setStyleSheet("color: #aaaaaa;")

        record_bar = QHBoxLayout()
        record_bar.addWidget(QLabel("Dataset:"))
        record_bar.addWidget(self._record_dataset_edit)
        record_bar.addWidget(self._record_btn)
        record_bar.addWidget(self._elapsed_label)
        record_bar.addStretch()

        play_bar = QHBoxLayout()
        play_bar.addWidget(self._prev_btn)
        play_bar.addWidget(self._restart_btn)
        play_bar.addWidget(self._play_btn)
        play_bar.addWidget(self._pause_btn)
        play_bar.addWidget(self._next_btn)
        play_bar.addWidget(self._save_btn)
        play_bar.addStretch()

        rename_bar = QHBoxLayout()
        rename_bar.addWidget(self._rename_btn)
        rename_bar.addStretch()

        sound_bar = QHBoxLayout()
        sound_bar.addWidget(QLabel("🔊"))
        sound_bar.addWidget(self._volume_slider)
        sound_bar.addSpacing(16)
        sound_bar.addWidget(self._click_btn)
        sound_bar.addSpacing(4)
        sound_bar.addWidget(QLabel("🔊"))
        sound_bar.addWidget(self._click_volume_slider)
        sound_bar.addSpacing(16)
        sound_bar.addWidget(QLabel("Clicks:"))
        for btn in self._click_source_buttons:
            sound_bar.addWidget(btn)
        sound_bar.addStretch()

        infer_bar = QHBoxLayout()
        infer_bar.addWidget(QLabel("Beat model:"))
        infer_bar.addWidget(self._checkpoint_combo)
        infer_bar.addWidget(self._infer_btn)
        infer_bar.addStretch()

        delete_bar = QHBoxLayout()
        delete_bar.addWidget(self._delete_btn)
        delete_bar.addWidget(self._delete_track_btn)
        delete_bar.addStretch()

        self._waveform = WaveformWidget()
        self._waveform.seek_requested.connect(self._on_seek)
        self._waveform.beat_added.connect(self._on_beat_added)
        self._waveform.beat_removed.connect(self._on_beat_removed)

        # Inferred-beats strip: a second waveform view stacked above the
        # main one, so model predictions never overlap the manual beat
        # markers — only visible once "Infer Beats" has produced a result.
        self._inferred_label = QLabel("🔮  Inferred beats")
        self._inferred_label.setStyleSheet("color: #ff44cc; font-weight: bold;")
        self._inferred_waveform = WaveformWidget()
        self._inferred_waveform.setMaximumHeight(45)
        self._inferred_waveform.seek_requested.connect(self._on_seek)

        self._inferred_container = QWidget()
        inferred_layout = QVBoxLayout(self._inferred_container)
        inferred_layout.setContentsMargins(0, 0, 0, 0)
        inferred_layout.setSpacing(2)
        inferred_layout.addWidget(self._inferred_label)
        inferred_layout.addWidget(self._inferred_waveform)
        self._inferred_container.setVisible(False)

        self._metronome = MetronomeWidget()
        self._metronome.set_state(DEFAULT_N_BEATS, None)

        self._speed = 1.0
        self._speed_combo = QComboBox()
        self._speed_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for pct in range(100, 45, -5):
            self._speed_combo.addItem(f"{pct}%", pct / 100)
        self._speed_combo.currentIndexChanged.connect(
            lambda i: self._on_speed_changed(self._speed_combo.itemData(i))
        )
        speed_bar = QHBoxLayout()
        speed_bar.addWidget(QLabel("Speed:"))
        speed_bar.addWidget(self._speed_combo)
        speed_bar.addStretch()

        # Count: length of the tap phrase in beats. Tapping always starts on
        # beat 1 (e.g. an 8-count "sentence" in swing dancing).
        self._count_group = QButtonGroup(self)
        self._count_group.setExclusive(True)
        count_bar = QHBoxLayout()
        count_bar.addWidget(QLabel("Count:"))
        for n_beats in (4, 8):
            btn = QPushButton(str(n_beats))
            btn.setCheckable(True)
            btn.setFixedWidth(40)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setChecked(n_beats == self._n_beats)
            btn.clicked.connect(lambda _checked, n=n_beats: self._on_count_changed(n))
            self._count_group.addButton(btn)
            count_bar.addWidget(btn)
        count_bar.addStretch()

        self._accent_group = QButtonGroup(self)
        self._accent_group.setExclusive(True)
        accent_bar = QHBoxLayout()
        accent_bar.addWidget(QLabel("Accent:"))
        for label, accent_bars in (
            ("Half Bar", 0.5),
            ("Every Bar", 1),
            ("Every 2 Bars", 2),
            ("Every 8 Bars", 8),
            ("Every 32 Bars", 32),
        ):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setChecked(accent_bars == self._accent_bars)
            btn.clicked.connect(
                lambda _checked, n=accent_bars: self._on_accent_mode_changed(n)
            )
            self._accent_group.addButton(btn)
            accent_bar.addWidget(btn)
        accent_bar.addStretch()

        self._structure_group = QButtonGroup(self)
        self._structure_group.setExclusive(True)
        structure_bar = QHBoxLayout()
        structure_bar.addWidget(QLabel("Structure:"))
        for label in ("Swing", "Blues"):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setChecked(label == "Swing")
            self._structure_group.addButton(btn)
            structure_bar.addWidget(btn)
        structure_bar.addStretch()

        # Tapping always starts on count position 1 — that's guaranteed, not
        # something to confirm. This instead tracks whether that first tap
        # also happens to be the true start of a section, vs. landing
        # mid-section — mirrors the mobile companion's "Section alignment"
        # toggle exactly (same two labels).
        self._section_group = QButtonGroup(self)
        self._section_group.setExclusive(True)
        section_bar = QHBoxLayout()
        section_bar.addWidget(QLabel("Section:"))
        for label in ("Section start", "Mid-section"):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setChecked(label == "Section start")
            self._section_group.addButton(btn)
            section_bar.addWidget(btn)
        section_bar.addStretch()

        author_bar = QHBoxLayout()
        author_bar.addWidget(QLabel("Author:"))
        author_bar.addWidget(self._author_edit)
        author_bar.addStretch()

        self._tap_widget = TapTempoWidget()
        self._tap_widget.reset_requested.connect(self._on_reset_beats)

        self._metadata_label = QLabel()
        self._metadata_label.setStyleSheet("color: #aaaaaa;")
        self._metadata_label.setWordWrap(True)

        # Left: every button/slider/combo that changes something (playback,
        # recording, speed, accents, structure, delete/rename). Right:
        # read-only track info — beat-derived analytics (_stats_label) and
        # captured metadata (_metadata_label, e.g. phone-recorded duration
        # and tap-BPM stats). A vertical divider keeps the two visually apart.
        controls_column = QVBoxLayout()
        controls_column.setSpacing(4)
        controls_column.addLayout(record_bar)
        controls_column.addLayout(play_bar)
        controls_column.addLayout(rename_bar)
        controls_column.addLayout(sound_bar)
        controls_column.addLayout(infer_bar)
        controls_column.addLayout(speed_bar)
        controls_column.addLayout(count_bar)
        controls_column.addLayout(accent_bar)
        controls_column.addLayout(structure_bar)
        controls_column.addLayout(section_bar)
        controls_column.addLayout(author_bar)
        controls_column.addLayout(delete_bar)
        controls_column.addStretch()

        info_column = QVBoxLayout()
        info_column.setSpacing(4)
        info_column.addWidget(self._track_label)
        info_column.addWidget(self._stats_label)
        info_column.addWidget(self._metadata_label)
        info_column.addStretch()

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)

        top_row = QHBoxLayout()
        top_row.addLayout(controls_column)
        top_row.addWidget(divider)
        top_row.addLayout(info_column, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(4)
        right_layout.addLayout(top_row)
        right_layout.addWidget(self._inferred_container)
        right_layout.addWidget(self._waveform, stretch=1)
        right_layout.addWidget(self._metronome)
        right_layout.addWidget(self._tap_widget)

        # Left panel: dataset tree
        self._dataset_sort_combo = QComboBox()
        self._dataset_sort_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._dataset_sort_combo.addItem("Alphabetical", "alphabetical")
        self._dataset_sort_combo.addItem("Recording date (newest)", "recording_date")
        self._dataset_sort_combo.currentIndexChanged.connect(
            lambda _i: self._populate_dataset_list()
        )
        sort_bar = QHBoxLayout()
        sort_bar.addWidget(QLabel("Sort:"))
        sort_bar.addWidget(self._dataset_sort_combo)

        self._dataset_tree = QTreeWidget()
        self._dataset_tree.setColumnCount(3)
        self._dataset_tree.header().hide()
        self._dataset_tree.setColumnWidth(0, 320)
        self._dataset_tree.setColumnWidth(1, 18)
        self._dataset_tree.setColumnWidth(2, 36)
        self._dataset_tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._dataset_tree.itemExpanded.connect(self._on_item_expanded)
        self._dataset_tree.itemClicked.connect(self._on_item_clicked)
        self._dataset_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._dataset_tree.customContextMenuRequested.connect(
            self._on_tree_context_menu
        )
        self._populate_dataset_list()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(0)
        left_layout.addLayout(sort_bar)
        left_layout.addWidget(self._dataset_tree)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 800])

        self.setCentralWidget(splitter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Track loading / navigation
    # ------------------------------------------------------------------

    def _load_track(self, index: int) -> None:
        """Load the track at *index*, replacing any currently loaded track."""
        self._engine.stop()

        self._index = index
        track_id = self._track_ids[index]
        self._track = load_track(self._dataset_name, track_id)
        self._n_beats = beats_per_bar(self._track.beat_positions, default=self._n_beats)
        self._sync_count_buttons()

        # Inferred beats belong to the previous track's audio — clear them.
        self._inferred_beat_times = np.array([])
        self._inferred_beat_positions = None
        self._inferred_waveform.set_beats(np.array([]), None)
        self._inferred_container.setVisible(False)

        # Swap in this track's audio/waveform/beats before any code below that
        # reads per-track display fields (info label, metadata panel) — so a
        # bad field on this particular track (e.g. a non-numeric tempo from
        # mirdata) can't leave playback silently stuck on the previous track.
        # Resample to a fixed rate rather than preserving each track's native
        # sr: some datasets' native rate (e.g. gtzan's 22050 Hz) doesn't
        # match the audio device's own rate, and letting the OS resample
        # on-the-fly during playback (rather than once here, in software)
        # produces audible noise on some hardware. _REC_SR (44100) matches
        # what every other dataset here already uses natively.
        audio, sr = librosa.load(self._track.audio_path, sr=_REC_SR, mono=True)
        self._track_audio = audio
        self._track_sr = sr
        self._engine.load(audio, sr)
        self._waveform.set_waveform(audio, sr)
        self._waveform.set_beats(self._track.beat_times, self._track.beat_positions)
        self._inferred_waveform.set_waveform(audio, sr)
        self._update_engine_clicks()

        self._metronome.set_state(self._n_beats, None)
        self._tap_widget.reset()
        metadata = load_metadata(self._dataset_name, track_id) or TrackMetadata()
        self._set_structure(metadata.structure)
        self._set_section_aligned(metadata.section_aligned)
        self._author_edit.setText(metadata.annotator_id or "")
        self._update_metadata_label()

        self._prev_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < len(self._track_ids) - 1)

        self._update_info_label()
        self.setWindowTitle(f"{self._dataset_name}  /  {track_id}")

    def _populate_dataset_list(self, *, keep_selection: bool = False) -> None:
        selected_dataset = self._dataset_name if keep_selection else None
        selected_track = (
            self._track.track_id if keep_selection and self._track else None
        )

        self._dataset_tree.clear()
        bold = QFont()
        bold.setBold(True)
        infos = list_datasets()
        if self._dataset_sort_combo.currentData() == "recording_date":
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
            placeholder.setData(0, Qt.ItemDataRole.UserRole, "__loading__")
            ds_item.addChild(placeholder)
            self._dataset_tree.addTopLevelItem(ds_item)

            if info.name == selected_dataset:
                ds_item.setExpanded(True)  # triggers lazy-load of children
                if selected_track:
                    for j in range(ds_item.childCount()):
                        child = ds_item.child(j)
                        if child.data(0, Qt.ItemDataRole.UserRole) == selected_track:
                            self._dataset_tree.setCurrentItem(child)
                            self._dataset_tree.scrollToItem(child)
                            break

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() != 1:
            return
        if item.child(0).data(0, Qt.ItemDataRole.UserRole) != "__loading__":
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
            self._set_annotation_indicator(track_item, dataset_name, track_id)
            item.addChild(track_item)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        parent = item.parent()
        if parent is None:
            return  # dataset header — expand/collapse handled by Qt
        dataset_name = parent.data(0, Qt.ItemDataRole.UserRole)
        track_id = item.data(0, Qt.ItemDataRole.UserRole)
        if dataset_name != self._dataset_name:
            self._engine.stop()
            self._dataset_name = dataset_name
            self._track_ids = load_dataset_tracks(dataset_name)
        elif track_id not in self._track_ids:
            self._track_ids = load_dataset_tracks(dataset_name)
        if track_id in self._track_ids:
            self._load_track(self._track_ids.index(track_id))

    def _on_tree_context_menu(self, pos) -> None:
        item = self._dataset_tree.itemAt(pos)
        if item is None or item.parent() is None:
            return  # empty space or a dataset header — no menu

        track_id = item.data(0, Qt.ItemDataRole.UserRole)
        if track_id == "__loading__":
            return

        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("🗑  Delete track")
        action = menu.exec(self._dataset_tree.viewport().mapToGlobal(pos))
        if action is None:
            return

        # Rename/delete act on self._track, so make sure the right-clicked
        # track is the one currently loaded before invoking them.
        if self._track is None or track_id != self._track.track_id:
            self._on_item_clicked(item, 0)

        if action is rename_action:
            self._on_rename()
        elif action is delete_action:
            self._on_delete_track()

    @staticmethod
    def _set_annotation_indicator(
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

    def _update_annotation_indicator(self) -> None:
        """Refresh the ●/✕ for the currently loaded track without rebuilding the tree."""
        if not self._track:
            return
        for i in range(self._dataset_tree.topLevelItemCount()):
            ds_item = self._dataset_tree.topLevelItem(i)
            if ds_item.data(0, Qt.ItemDataRole.UserRole) != self._dataset_name:
                continue
            for j in range(ds_item.childCount()):
                child = ds_item.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == self._track.track_id:
                    self._set_annotation_indicator(
                        child, self._dataset_name, self._track.track_id
                    )
                    return

    def _on_prev(self) -> None:
        if self._index > 0:
            self._load_track(self._index - 1)

    def _on_next(self) -> None:
        if self._index < len(self._track_ids) - 1:
            self._load_track(self._index + 1)

    def _on_accent_mode_changed(self, accent_bars: float) -> None:
        self._accent_bars = accent_bars
        self._metronome.set_accent_bars(accent_bars)
        self._waveform.set_accent_bars(accent_bars)
        self._inferred_waveform.set_accent_bars(accent_bars)
        self._update_engine_clicks()

    def _sync_count_buttons(self) -> None:
        """Check the Count button matching ``self._n_beats``, if any (e.g. a
        loaded mirdata meter like 3/4 time has no matching quick-select button)."""
        for btn in self._count_group.buttons():
            btn.setChecked(btn.text() == str(self._n_beats))

    def _on_count_changed(self, n_beats: int) -> None:
        """Change the tap phrase length, re-tagging any already-tapped beats.

        Existing taps are recycled 1..n_beats from the first beat, so
        switching mid-annotation still guarantees tap 1 lands on position 1.
        """
        self._n_beats = n_beats
        if self._track is not None and len(self._track.beat_times) > 0:
            positions = cycle_positions(len(self._track.beat_times), n_beats)
            self._track = TrackData(
                dataset_name=self._track.dataset_name,
                track_id=self._track.track_id,
                audio_path=self._track.audio_path,
                tempo=self._track.tempo,
                beat_times=self._track.beat_times,
                beat_positions=positions,
            )
        self._refresh_beats()
        self._metronome.set_state(self._n_beats, None)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _on_speed_changed(self, speed: float) -> None:
        self._speed = speed
        self._engine.set_speed(speed)

    def _on_play(self) -> None:
        if not self._engine.is_playing:
            self._engine.play()

    def _on_pause(self) -> None:
        if self._engine.is_playing:
            self._engine.pause()

    def _on_play_pause(self) -> None:
        if self._engine.is_playing:
            self._on_pause()
        else:
            self._on_play()

    def _on_infer_beats(self) -> None:
        if self._track_audio is None:
            return
        checkpoint_path = self._checkpoint_combo.currentData()
        if checkpoint_path is None:
            self.statusBar().showMessage(
                "No checkpoint found under checkpoints_beat/.", 4000
            )
            return

        self._infer_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Running beat inference…")
        QApplication.processEvents()
        try:
            if checkpoint_path != self._beat_module_path:
                self._beat_module, self._beat_module_task = load_module(checkpoint_path)
                self._beat_module_path = checkpoint_path
            task_defaults = EVAL_DEFAULTS[self._beat_module_task]
            beat_times, beat_positions = infer_beats(
                self._beat_module,
                self._beat_module_task,
                self._track_audio,
                self._track_sr,
                group_size=task_defaults.get("group_size", 4),
                beat_threshold=task_defaults["beat_threshold"],
                min_distance_frames=task_defaults["min_distance_frames"],
                gate_tolerance=task_defaults["gate_tolerance"],
                anchor_threshold=task_defaults.get("anchor_threshold", 0.5),
            )
        finally:
            QApplication.restoreOverrideCursor()
            self._infer_btn.setEnabled(True)

        self._inferred_beat_times = beat_times
        self._inferred_beat_positions = beat_positions
        self._inferred_waveform.set_accent_bars(self._accent_bars)
        self._inferred_waveform.set_beats(beat_times, beat_positions)
        self._inferred_waveform.set_position(self._engine.position)
        self._inferred_container.setVisible(True)
        self._update_engine_clicks()
        self.statusBar().showMessage(
            f"Inferred {len(beat_times)} beats ({checkpoint_label(checkpoint_path)}).",
            5000,
        )

    def _on_record_toggle(self, checked: bool) -> None:
        if checked:
            self._recorder.start()
            self._record_btn.setText("⏹  Stop rec")
            self._record_start = time.monotonic()
            self._record_tick = 0
            self._elapsed_label.setText("● 0:00")
            self._elapsed_label.setVisible(True)
            self._waveform.set_beats(np.array([]), None)
            self._inferred_container.setVisible(False)
        else:
            dataset = self._record_dataset_edit.text().strip() or "swing"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = dataformats.DATA_DIR / dataset / "tracks"
            path = self._recorder.stop(save_dir, f"recording_{timestamp}")
            self._record_btn.setText("⏺  Record")
            self._elapsed_label.setVisible(False)
            if self._track_audio is not None:
                self._waveform.set_waveform(self._track_audio, self._track_sr)
            if self._track is not None:
                self._waveform.set_beats(
                    self._track.beat_times, self._track.beat_positions
                )
            if len(self._inferred_beat_times) > 0:
                self._inferred_waveform.set_waveform(self._track_audio, self._track_sr)
                self._inferred_container.setVisible(True)
            self.statusBar().showMessage(f"Recording saved → {path}", 4000)
            self._populate_dataset_list(keep_selection=True)
            if dataset == self._dataset_name:
                self._track_ids = load_dataset_tracks(dataset)

    def _on_restart(self) -> None:
        self._engine.seek(0.0)
        self._waveform.set_position(0.0)
        self._inferred_waveform.set_position(0.0)
        if not self._engine.is_playing:
            self._engine.play()

    def _on_seek(self, t: float) -> None:
        self._engine.seek(t)
        self._waveform.set_position(t)
        self._inferred_waveform.set_position(t)

    def _on_playback_finished(self) -> None:
        """Called on the main thread when playback reaches the end."""

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _on_beat_added(self, t: float) -> None:
        self._track = add_beat(self._track, t, n_beats=self._n_beats)
        self._refresh_beats()
        if self._click_btn.isChecked() and self._engine.is_playing:
            idx = int(np.searchsorted(self._track.beat_times, t))
            if self._track.beat_positions is not None:
                is_down = bool(self._track.beat_positions[idx] == 1)
            else:
                is_down = (idx % self._n_beats) == 0
            self._engine.trigger_click_now(is_down)

    def _on_beat_removed(self, t: float) -> None:
        self._track = remove_beat(self._track, t, n_beats=self._n_beats)
        self._refresh_beats()

    def _on_reset_beats(self) -> None:
        if self._track is None:
            return
        self._track = TrackData(
            dataset_name=self._track.dataset_name,
            track_id=self._track.track_id,
            audio_path=self._track.audio_path,
            tempo=self._track.tempo,
            beat_times=np.array([]),
            beat_positions=None,
        )
        self._refresh_beats()

    def _set_structure(self, structure: str | None) -> None:
        """Check the Swing/Blues button matching *structure* (Swing if unset/unknown)."""
        for btn in self._structure_group.buttons():
            btn.setChecked(btn.text() == (structure or "Swing").capitalize())

    def _current_structure(self) -> str:
        checked = self._structure_group.checkedButton()
        return checked.text().lower() if checked is not None else "swing"

    def _set_section_aligned(self, section_aligned: bool | None) -> None:
        """Check the Section-start/Mid-section button matching *section_aligned*
        (Section start if unset)."""
        value = True if section_aligned is None else section_aligned
        for btn in self._section_group.buttons():
            btn.setChecked(btn.text() == ("Section start" if value else "Mid-section"))

    def _current_section_aligned(self) -> bool:
        checked = self._section_group.checkedButton()
        return checked is None or checked.text() == "Section start"

    def _on_save(self) -> None:
        path = annotation_path(self._track)
        save_annotations(self._track, path)

        metadata = (
            load_metadata(self._dataset_name, self._track.track_id) or TrackMetadata()
        )
        metadata.structure = self._current_structure()
        metadata.section_aligned = self._current_section_aligned()
        metadata.annotator_id = self._author_edit.text().strip() or None
        # Only fill in if unset — a track captured on the phone should keep
        # reporting its actual recording device, not the laptop it happens
        # to be annotated on.
        if not metadata.device:
            metadata.device = platform.node()
        save_metadata(self._dataset_name, self._track.track_id, metadata)
        self._update_metadata_label()

        self.statusBar().showMessage(f"Saved → {path}", 3000)
        self._update_annotation_indicator()

    def _on_delete(self) -> None:
        path = annotation_path(self._track)
        if not path.exists():
            self.statusBar().showMessage("No manual annotation to delete.", 3000)
            return
        reply = QMessageBox.question(
            self,
            "Delete annotation",
            f"Delete manual annotation for {self._track.track_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        path.unlink()
        self._track = load_track(self._dataset_name, self._track.track_id)
        self._refresh_beats()
        self.statusBar().showMessage(f"Deleted → {path}", 3000)
        self._update_annotation_indicator()

    def _on_delete_track(self) -> None:
        if self._track is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete track",
            f"Permanently delete '{self._track.track_id}' (audio + annotation)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_track(self._track)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot delete", str(exc))
            return
        self._track_ids.pop(self._index)
        if self._track_ids:
            self._index = min(self._index, len(self._track_ids) - 1)
            self._load_track(self._index)
        else:
            self._track = None
            self._engine.stop()
            self._waveform.set_waveform(np.array([]), 44100)
            self._waveform.set_beats(np.array([]), None)
            self._inferred_beat_times = np.array([])
            self._inferred_beat_positions = None
            self._inferred_waveform.set_beats(np.array([]), None)
            self._inferred_container.setVisible(False)
            self._track_label.setText("")
            self._stats_label.setText("")
        self._populate_dataset_list(keep_selection=True)
        self.statusBar().showMessage(f"Track deleted.", 3000)

    def _on_rename(self) -> None:
        if self._track is None:
            return
        new_id, ok = QInputDialog.getText(
            self,
            "Rename track",
            "New name:",
            text=self._track.track_id,
        )
        if not ok or not new_id.strip():
            return
        new_id = new_id.strip()
        try:
            self._track = rename_track(self._track, new_id)
        except ValueError as exc:
            QMessageBox.warning(self, "Rename failed", str(exc))
            return
        self._track_ids[self._index] = new_id
        self.setWindowTitle(f"{self._dataset_name}  /  {new_id}")
        self._update_info_label()
        # Refresh tree row if the dataset is currently expanded, keeping it
        # expanded and re-selecting the renamed track instead of collapsing.
        for i in range(self._dataset_tree.topLevelItemCount()):
            ds_item = self._dataset_tree.topLevelItem(i)
            if ds_item.data(0, Qt.ItemDataRole.UserRole) != self._dataset_name:
                continue
            if ds_item.isExpanded():
                self._load_children(ds_item)
                for j in range(ds_item.childCount()):
                    child = ds_item.child(j)
                    if child.data(0, Qt.ItemDataRole.UserRole) == new_id:
                        self._dataset_tree.setCurrentItem(child)
                        self._dataset_tree.scrollToItem(child)
                        break
            break
        self.statusBar().showMessage(f"Renamed → {new_id}", 3000)

    def _click_source(self) -> str:
        checked = self._click_source_group.checkedButton()
        return checked.text().lower() if checked is not None else "manual"

    def _update_engine_clicks(self) -> None:
        if self._track is None:
            return
        if self._click_source() == "inferred":
            beat_times = self._inferred_beat_times
            beat_positions = self._inferred_beat_positions
        else:
            beat_times = self._track.beat_times
            beat_positions = self._track.beat_positions
        beat_frames = (beat_times * self._track_sr).astype(int)
        n = len(beat_times)
        if beat_positions is not None:
            positions = beat_positions
        else:
            positions = np.array([(i % self._n_beats) + 1 for i in range(n)])
        bars = bar_indices(beat_positions, n)
        beat_is_down = np.array(
            [
                is_accent_beat(positions[i], bars[i], self._n_beats, self._accent_bars)
                for i in range(n)
            ],
            dtype=bool,
        )
        self._engine.set_clicks(beat_frames, beat_is_down, self._track_sr)

    def _refresh_beats(self) -> None:
        """Refresh derived views after beats change. Does not touch
        ``self._n_beats`` — that's owned by the Count selector (see
        :meth:`_on_count_changed`), not re-derived from partial tap data."""
        self._waveform.set_beats(self._track.beat_times, self._track.beat_positions)
        self._update_engine_clicks()
        self._update_info_label()

    def _update_info_label(self) -> None:
        if self._track is None:
            return
        track_id = self._track_ids[self._index]
        self._track_label.setText(
            f"[{self._index + 1}/{len(self._track_ids)}]  {track_id}"
        )

        dur = self._engine.duration
        m, s = divmod(int(dur), 60)
        dur_str = f"{m}:{s:02d}"

        beat_times = self._track.beat_times
        n = len(beat_times)
        if n >= 2:
            intervals = np.diff(beat_times)
            mean_bpm = 60.0 / np.mean(intervals)
            median_bpm = 60.0 / np.median(intervals)
            beat_str = f"{n} beats  •  mean {mean_bpm:.1f}  •  med {median_bpm:.1f} BPM"
        elif n == 1:
            beat_str = "1 beat"
        else:
            beat_str = "no annotations"

        parts = [f"duration {dur_str}"]
        if self._track.tempo:
            parts.append(f"ref {self._track.tempo:.1f} BPM")
        parts.append(beat_str)
        self._stats_label.setText("  •  ".join(parts))

    def _update_metadata_label(self) -> None:
        """Refresh the read-only metadata panel from the track's .meta.json, if any."""
        if self._track is None:
            self._metadata_label.setText("")
            return
        metadata = load_metadata(self._dataset_name, self._track.track_id)
        if metadata is None:
            self._metadata_label.setText("No metadata")
            return

        parts = []
        if metadata.annotator_id:
            parts.append(f"Author: {metadata.annotator_id}")
        if metadata.device:
            parts.append(f"Device: {metadata.device}")
        if metadata.location:
            parts.append(f"Location: {metadata.location}")
        if metadata.structure:
            parts.append(f"Structure: {metadata.structure}")
        if metadata.section_aligned is not None:
            parts.append(
                f"Section: {'aligned' if metadata.section_aligned else 'mid-section'}"
            )
        if metadata.duration_s is not None:
            m, s = divmod(int(metadata.duration_s), 60)
            parts.append(f"Rec. duration: {m}:{s:02d}")
        if metadata.bpm_mean is not None:
            parts.append(f"Tap BPM mean: {metadata.bpm_mean:.1f}")
        if metadata.bpm_median is not None:
            parts.append(f"Tap BPM median: {metadata.bpm_median:.1f}")
        if metadata.bpm_std is not None:
            parts.append(f"Tap BPM std: {metadata.bpm_std:.2f}")
        self._metadata_label.setText("\n".join(parts) if parts else "No metadata")

    # ------------------------------------------------------------------
    # Timer tick
    # ------------------------------------------------------------------

    _RECORD_WAVEFORM_EVERY = 17  # ~510 ms at 30 ms/tick

    def _tick(self) -> None:
        if self._recorder.is_recording:
            elapsed = time.monotonic() - self._record_start
            m, s = divmod(int(elapsed), 60)
            self._elapsed_label.setText(f"● {m}:{s:02d}")
            self._record_tick = (self._record_tick + 1) % self._RECORD_WAVEFORM_EVERY
            if self._record_tick == 0:
                audio = self._recorder.current_audio
                if audio is not None and len(audio) > 0:
                    self._waveform.set_waveform(audio, _REC_SR)
            return

        t = self._engine.position
        self._waveform.set_position(t)
        self._inferred_waveform.set_position(t)
        pos = active_beat_position(
            self._track.beat_times, self._track.beat_positions, t, self._n_beats
        )
        bar_index = active_bar_index(
            self._track.beat_times, self._track.beat_positions, t, self._n_beats
        )
        self._metronome.set_state(self._n_beats, pos, bar_index)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Space:
            if (
                not self._recorder.is_recording
                and self._track is not None
                and self._engine.position < self._waveform._duration
            ):
                self._on_beat_added(self._engine.position)
                self._tap_widget.tap()
        elif key == Qt.Key.Key_P:
            self._on_play_pause()
        elif key == Qt.Key.Key_Left:
            self._on_prev()
        elif key == Qt.Key.Key_Right:
            self._on_next()
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            if self._track is not None:
                self._on_save()
        elif (
            key == Qt.Key.Key_S
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._on_save()
        else:
            super().keyPressEvent(event)
