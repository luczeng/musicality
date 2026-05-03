"""Main application window for track inspection and beat annotation."""

from __future__ import annotations

import librosa
from PySide6.QtCore import Qt, QTimer
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
    remove_beat,
    save_annotations,
)
from .metronome_widget import MetronomeWidget
from .waveform_widget import WaveformWidget

_TICK_MS = 30   # ~33 fps refresh rate


class MainWindow(QMainWindow):
    """Main window assembling waveform view, metronome, and audio engine.

    Layout
    ------
    Toolbar : [▶ Play/Pause]  [💾 Save]  [track info]
    Center  : WaveformWidget  (resizable, takes remaining space)
    Bottom  : MetronomeWidget (fixed height)

    Keyboard shortcuts
    ------------------
    Space      : play / pause
    Ctrl+S     : save annotations
    Ctrl+click : add beat at clicked position on waveform
    Ctrl+right-click : remove beat nearest to clicked position
    """

    def __init__(self, track: TrackData) -> None:
        super().__init__()
        self._track = track
        self._engine = AudioEngine()
        self._timer = QTimer(self)
        self._n_beats = beats_per_bar(track.beat_positions)

        self._setup_ui()
        self._load_audio()

        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle(f"{self._track.dataset_name}  /  {self._track.track_id}")
        self.resize(1200, 280)

        # Toolbar widgets
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedWidth(90)
        self._play_btn.clicked.connect(self._on_play_pause)

        self._save_btn = QPushButton("💾  Save")
        self._save_btn.setFixedWidth(90)
        self._save_btn.clicked.connect(self._on_save)

        tempo_str = f"{self._track.tempo:.1f} BPM" if self._track.tempo else "? BPM"
        self._info_label = QLabel(
            f"{self._track.track_id}   —   {tempo_str}   —   {self._n_beats}/4"
        )

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._play_btn)
        toolbar.addWidget(self._save_btn)
        toolbar.addWidget(self._info_label)
        toolbar.addStretch()

        # Central widgets
        self._waveform = WaveformWidget()
        self._waveform.set_beats(self._track.beat_times, self._track.beat_positions)
        self._waveform.seek_requested.connect(self._on_seek)
        self._waveform.beat_added.connect(self._on_beat_added)
        self._waveform.beat_removed.connect(self._on_beat_removed)

        self._metronome = MetronomeWidget()
        self._metronome.set_state(self._n_beats, None)

        # Root layout
        root = QVBoxLayout()
        root.setSpacing(4)
        root.addLayout(toolbar)
        root.addWidget(self._waveform, stretch=1)
        root.addWidget(self._metronome)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _load_audio(self) -> None:
        audio, sr = librosa.load(self._track.audio_path, sr=None, mono=True)
        self._engine.load(audio, sr)
        self._waveform.set_waveform(audio, sr)
        self._engine.on_finished(self._on_playback_finished)

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
        if event.key() == Qt.Key.Key_Space:
            self._on_play_pause()
        elif (
            event.key() == Qt.Key.Key_S
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._on_save()
        else:
            super().keyPressEvent(event)
