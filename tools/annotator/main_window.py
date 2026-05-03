"""Main application window for track inspection and beat annotation."""

from __future__ import annotations

import librosa
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .audio import AudioEngine
from .data import (
    TrackData,
    active_beat_position,
    add_beat,
    annotation_path,
    beats_per_bar,
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
        self._engine = AudioEngine()
        self._timer = QTimer(self)
        self._n_beats = 4

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
        self.resize(1200, 280)

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

        self._info_label = QLabel()

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._prev_btn)
        toolbar.addWidget(self._play_btn)
        toolbar.addWidget(self._next_btn)
        toolbar.addWidget(self._save_btn)
        toolbar.addWidget(self._info_label)
        toolbar.addStretch()

        self._waveform = WaveformWidget()
        self._waveform.seek_requested.connect(self._on_seek)
        self._waveform.beat_added.connect(self._on_beat_added)
        self._waveform.beat_removed.connect(self._on_beat_removed)

        self._metronome = MetronomeWidget()
        self._metronome.set_state(4, None)

        self._tap_widget = TapTempoWidget()

        root = QVBoxLayout()
        root.setSpacing(4)
        root.addLayout(toolbar)
        root.addWidget(self._waveform, stretch=1)
        root.addWidget(self._metronome)
        root.addWidget(self._tap_widget)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)
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

        computed = tempo_from_beats(self._track.beat_times)
        if computed is not None:
            tempo_str = f"{computed:.1f} BPM"
            if self._track.tempo:
                tempo_str += f" (ann: {self._track.tempo:.1f})"
        elif self._track.tempo:
            tempo_str = f"{self._track.tempo:.1f} BPM"
        else:
            tempo_str = "? BPM"
        self._info_label.setText(
            f"[{index + 1}/{len(self._track_ids)}]  {track_id}"
            f"   —   {tempo_str}   —   {self._n_beats}/4"
        )
        self.setWindowTitle(f"{self._dataset_name}  /  {track_id}")

        self._waveform.set_beats(self._track.beat_times, self._track.beat_positions)
        self._metronome.set_state(self._n_beats, None)
        self._tap_widget.reset()

        self._prev_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < len(self._track_ids) - 1)

        audio, sr = librosa.load(self._track.audio_path, sr=None, mono=True)
        self._engine.load(audio, sr)
        self._waveform.set_waveform(audio, sr)

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

    def _refresh_beats(self) -> None:
        self._n_beats = beats_per_bar(self._track.beat_positions)
        self._waveform.set_beats(self._track.beat_times, self._track.beat_positions)

    # ------------------------------------------------------------------
    # Timer tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
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
            self._tap_widget.tap()
        elif key == Qt.Key.Key_P:
            self._on_play_pause()
        elif key == Qt.Key.Key_Left:
            self._on_prev()
        elif key == Qt.Key.Key_Right:
            self._on_next()
        elif (
            key == Qt.Key.Key_S
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._on_save()
        else:
            super().keyPressEvent(event)
