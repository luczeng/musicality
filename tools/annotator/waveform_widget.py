"""Waveform display widget with beat markers and a playback cursor.

The audio envelope is pre-computed once on load (``compute_envelope``) and
the widget re-uses it on every ``paintEvent``, mapping the fixed-length
envelope array to the current widget width on the fly.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .data import bar_indices


# ---------------------------------------------------------------------------
# Pure helper — separated for testability
# ---------------------------------------------------------------------------


def compute_envelope(
    audio: np.ndarray, n_points: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample *audio* to *n_points* min/max pairs for waveform display.

    Splits the signal into exactly *n_points* equal-width segments and
    returns the (min, max) amplitude in each segment.

    :param audio: 1-D float array.
    :param n_points: Number of output points.
    :returns: ``(min_envelope, max_envelope)``, each of length *n_points*.
    """
    n = len(audio)
    if n == 0:
        return np.zeros(n_points), np.zeros(n_points)

    edges = np.linspace(0, n, n_points + 1, dtype=int)
    min_env = np.array(
        [
            audio[edges[i] : edges[i + 1]].min() if edges[i] < edges[i + 1] else 0.0
            for i in range(n_points)
        ]
    )
    max_env = np.array(
        [
            audio[edges[i] : edges[i + 1]].max() if edges[i] < edges[i + 1] else 0.0
            for i in range(n_points)
        ]
    )
    peak = max(np.abs(min_env).max(), np.abs(max_env).max())
    if peak > 0:
        min_env /= peak
        max_env /= peak
    return min_env, max_env


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class WaveformWidget(QWidget):
    """Displays an audio waveform with beat markers and a playback cursor.

    Signals
    -------
    seek_requested(float):
        Emitted on left-click with the target time in seconds.
    beat_added(float):
        Emitted on Ctrl+left-click with the clicked time in seconds.
    beat_removed(float):
        Emitted on Ctrl+right-click with the clicked time in seconds.
        :func:`~tools.annotator.data.remove_beat` applies a tolerance
        window, so the exact time need not land on an existing beat.
    """

    seek_requested = Signal(float)
    beat_added = Signal(float)
    beat_removed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._env_min: np.ndarray | None = None
        self._env_max: np.ndarray | None = None
        self._beat_times: np.ndarray = np.array([])
        self._beat_positions: np.ndarray | None = None
        self._group_bars: int = 1
        self._position: float = 0.0
        self._duration: float = 0.0
        self.setMinimumHeight(40)
        self.setMaximumHeight(60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_waveform(self, audio: np.ndarray, sr: int) -> None:
        """Pre-compute the display envelope from *audio* sampled at *sr* Hz."""
        self._duration = len(audio) / sr
        self._env_min, self._env_max = compute_envelope(audio)
        self.update()

    def set_beats(
        self,
        beat_times: np.ndarray,
        beat_positions: np.ndarray | None,
    ) -> None:
        """Update the beat markers."""
        self._beat_times = beat_times
        self._beat_positions = beat_positions
        self.update()

    def set_accent_group(self, group_bars: int) -> None:
        """Set how many bars form one accent group (see MetronomeWidget docstring)."""
        self._group_bars = max(1, group_bars)
        self.update()

    def set_position(self, seconds: float) -> None:
        """Move the playback cursor to *seconds*."""
        self._position = seconds
        self.update()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()
        mid = h // 2

        painter.fillRect(self.rect(), QColor("#1a1a1a"))

        # Waveform envelope
        if self._env_min is not None:
            n = len(self._env_min)
            pen = QPen(QColor("#4488cc"))
            pen.setWidth(1)
            painter.setPen(pen)
            for i in range(n):
                x = int(i / n * w)
                y_top = int(mid - self._env_max[i] * mid * 0.9)
                y_bot = int(mid - self._env_min[i] * mid * 0.9)
                painter.drawLine(x, y_top, x, y_bot)

        if self._duration == 0:
            return

        # Beat markers
        for i, t in enumerate(self._beat_times):
            x = int(t / self._duration * w)
            pos = (
                int(self._beat_positions[i])
                if self._beat_positions is not None
                else (i % 4) + 1
            )
            color = "#44cc44" if pos == 1 else "#cc7700"
            painter.setPen(QPen(QColor(color), 1))
            painter.drawLine(x, 0, x, h)

        # Playback cursor
        x = int(self._position / self._duration * w)
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawLine(x, 0, x, h)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._duration == 0:
            return
        t = event.position().x() / self.width() * self._duration
        t = float(np.clip(t, 0.0, self._duration))
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if event.button() == Qt.MouseButton.LeftButton:
            if ctrl:
                self.beat_added.emit(t)
            else:
                self.seek_requested.emit(t)
        elif event.button() == Qt.MouseButton.RightButton and ctrl:
            self.beat_removed.emit(t)
