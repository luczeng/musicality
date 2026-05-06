"""Main application window for track inspection and beat annotation."""

from __future__ import annotations

import datetime
import time
import librosa
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .audio import AudioEngine
from .recorder import Recorder, _SR as _REC_SR
from .data import (
    DATA_DIR,
    TrackData,
    active_beat_position,
    add_beat,
    annotation_path,
    beats_per_bar,
    has_annotation,
    list_datasets,
    load_dataset_tracks,
    load_track,
    remove_beat,
    save_annotations,
    tempo_from_beats,
)
from .metronome_widget import MetronomeWidget
from .tap_tempo_widget import TapTempoWidget
from .waveform_widget import WaveformWidget

_TICK_MS = 30   # ~33 fps refresh rate


class MainWindow(QMainWindow):
    """Main window assembling waveform view, metronome, and audio engine.

    Layout
    ------
    Toolbar : [◀ Prev]  [▶ Play/Pause]  [Next ▶]  [💾 Save]  [track info]
    Center  : WaveformWidget  (resizable, takes remaining space)
    Bottom  : MetronomeWidget (fixed height)

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
        self._engine = AudioEngine()
        self._recorder = Recorder()
        self._timer = QTimer(self)
        self._n_beats = 4
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
        self._play_btn.clicked.connect(self._on_play_pause)

        self._next_btn = QPushButton("Next  ▶")
        self._next_btn.setFixedWidth(90)
        self._next_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._next_btn.clicked.connect(self._on_next)

        self._save_btn = QPushButton("💾  Save")
        self._save_btn.setFixedWidth(90)
        self._save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._save_btn.clicked.connect(self._on_save)

        self._record_dataset_edit = QLineEdit("swing")
        self._record_dataset_edit.setFixedWidth(110)
        self._record_dataset_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._record_dataset_edit.setPlaceholderText("dataset")

        self._record_btn = QPushButton("⏺  Record")
        self._record_btn.setFixedWidth(100)
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

        self._info_label = QLabel()

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._prev_btn)
        toolbar.addWidget(self._restart_btn)
        toolbar.addWidget(self._play_btn)
        toolbar.addWidget(self._next_btn)
        toolbar.addWidget(self._save_btn)
        toolbar.addWidget(QLabel("🔊"))
        toolbar.addWidget(self._volume_slider)
        toolbar.addWidget(QLabel("Dataset:"))
        toolbar.addWidget(self._record_dataset_edit)
        toolbar.addWidget(self._record_btn)
        toolbar.addWidget(self._elapsed_label)
        toolbar.addWidget(self._info_label)
        toolbar.addStretch()

        self._waveform = WaveformWidget()
        self._waveform.seek_requested.connect(self._on_seek)
        self._waveform.beat_added.connect(self._on_beat_added)
        self._waveform.beat_removed.connect(self._on_beat_removed)

        self._metronome = MetronomeWidget()
        self._metronome.set_state(4, None)

        self._tap_widget = TapTempoWidget()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(4)
        right_layout.addLayout(toolbar)
        right_layout.addWidget(self._waveform, stretch=1)
        right_layout.addWidget(self._metronome)
        right_layout.addWidget(self._tap_widget)

        # Left panel: dataset tree
        self._dataset_tree = QTreeWidget()
        self._dataset_tree.setColumnCount(2)
        self._dataset_tree.header().hide()
        self._dataset_tree.setColumnWidth(0, 355)
        self._dataset_tree.setColumnWidth(1, 18)
        self._dataset_tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._dataset_tree.itemExpanded.connect(self._on_item_expanded)
        self._dataset_tree.itemClicked.connect(self._on_item_clicked)
        self._populate_dataset_list()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(0)
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
        self._play_btn.setText("▶  Play")

        self._index = index
        track_id = self._track_ids[index]
        self._track = load_track(self._dataset_name, track_id)
        self._n_beats = beats_per_bar(self._track.beat_positions)

        self._update_info_label()
        self.setWindowTitle(f"{self._dataset_name}  /  {track_id}")

        self._waveform.set_beats(self._track.beat_times, self._track.beat_positions)
        self._metronome.set_state(self._n_beats, None)
        self._tap_widget.reset()

        self._prev_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < len(self._track_ids) - 1)

        audio, sr = librosa.load(self._track.audio_path, sr=None, mono=True)
        self._track_audio = audio
        self._track_sr = sr
        self._engine.load(audio, sr)
        self._waveform.set_waveform(audio, sr)

    def _populate_dataset_list(self) -> None:
        self._dataset_tree.clear()
        bold = QFont()
        bold.setBold(True)
        for info in list_datasets():
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

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() != 1:
            return
        if item.child(0).data(0, Qt.ItemDataRole.UserRole) != "__loading__":
            return
        item.takeChild(0)
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
            self._play_btn.setText("▶  Play")
            self._dataset_name = dataset_name
            self._track_ids = load_dataset_tracks(dataset_name)
        if track_id in self._track_ids:
            self._load_track(self._track_ids.index(track_id))

    @staticmethod
    def _set_annotation_indicator(
        item: QTreeWidgetItem, dataset_name: str, track_id: str
    ) -> None:
        if has_annotation(dataset_name, track_id):
            item.setText(1, "●")
            item.setForeground(1, QColor("#44cc44"))
        else:
            item.setText(1, "✕")
            item.setForeground(1, QColor("#cc4444"))

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

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _on_play_pause(self) -> None:
        if self._engine.is_playing:
            self._engine.pause()
            self._play_btn.setText("▶  Play")
        else:
            self._engine.play()
            self._play_btn.setText("⏸  Pause")

    def _on_record_toggle(self, checked: bool) -> None:
        if checked:
            self._recorder.start()
            self._record_btn.setText("⏹  Stop rec")
            self._record_start = time.monotonic()
            self._record_tick = 0
            self._elapsed_label.setText("● 0:00")
            self._elapsed_label.setVisible(True)
            self._waveform.set_beats(np.array([]), None)
        else:
            dataset = self._record_dataset_edit.text().strip() or "swing"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = DATA_DIR / dataset / "tracks"
            path = self._recorder.stop(save_dir, f"recording_{timestamp}")
            self._record_btn.setText("⏺  Record")
            self._elapsed_label.setVisible(False)
            if self._track_audio is not None:
                self._waveform.set_waveform(self._track_audio, self._track_sr)
            if self._track is not None:
                self._waveform.set_beats(self._track.beat_times, self._track.beat_positions)
            self.statusBar().showMessage(f"Recording saved → {path}", 4000)
            self._populate_dataset_list()

    def _on_restart(self) -> None:
        self._engine.seek(0.0)
        self._waveform.set_position(0.0)
        if not self._engine.is_playing:
            self._engine.play()
            self._play_btn.setText("⏸  Pause")

    def _on_seek(self, t: float) -> None:
        self._engine.seek(t)
        self._waveform.set_position(t)

    def _on_playback_finished(self) -> None:
        """Called on the main thread when playback reaches the end."""
        self._play_btn.setText("▶  Play")

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _on_beat_added(self, t: float) -> None:
        self._track = add_beat(self._track, t)
        self._refresh_beats()

    def _on_beat_removed(self, t: float) -> None:
        self._track = remove_beat(self._track, t)
        self._refresh_beats()

    def _on_save(self) -> None:
        path = annotation_path(self._track)
        save_annotations(self._track, path)
        self.statusBar().showMessage(f"Saved → {path}", 3000)
        self._update_annotation_indicator()

    def _refresh_beats(self) -> None:
        self._n_beats = beats_per_bar(self._track.beat_positions)
        self._waveform.set_beats(self._track.beat_times, self._track.beat_positions)
        self._update_info_label()

    def _update_info_label(self) -> None:
        if self._track is None:
            return
        track_id = self._track_ids[self._index]
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
        if self._track.tempo:
            label = (
                f"[{self._index + 1}/{len(self._track_ids)}]  {track_id}"
                f"   —   ref {self._track.tempo:.1f} BPM   —   {beat_str}"
            )
        else:
            label = f"[{self._index + 1}/{len(self._track_ids)}]  {track_id}   —   {beat_str}"
        self._info_label.setText(label)

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
        pos = active_beat_position(
            self._track.beat_times, self._track.beat_positions, t
        )
        self._metronome.set_state(self._n_beats, pos)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Space:
            if not self._recorder.is_recording and self._track is not None:
                self._on_beat_added(self._engine.position)
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
