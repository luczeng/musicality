"""Live metronome dot display.

Shows one dot per beat in the bar.  The active dot lights up green on a
downbeat (position 1) and yellow on any other beat.  All other dots are
dark grey.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget


class MetronomeWidget(QWidget):
    """A row of dots that flash in sync with beat annotations during playback.

    Colours
    -------
    - Inactive : dark grey  (#444444)
    - Beat     : yellow     (#FFD700)
    - Downbeat : green      (#44CC44)  — beat position 1
    """

    _COLOR_INACTIVE = QColor("#444444")
    _COLOR_BEAT     = QColor("#FFD700")
    _COLOR_DOWNBEAT = QColor("#44CC44")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._n_beats: int = 4
        self._active: int | None = None   # 1-indexed active position, or None
        self.setFixedHeight(56)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, n_beats: int, active_position: int | None) -> None:
        """Refresh the display.

        :param n_beats: Total number of dots (beats per bar).
        :param active_position: 1-indexed position of the lit dot, or
            ``None`` when no beat is active (e.g. before playback starts).
        """
        self._n_beats = max(1, n_beats)
        self._active = active_position
        self.update()

    # ------------------------------------------------------------------
    # Qt override
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(self.rect(), QColor("#1a1a1a"))

        r = min(h // 2 - 6, w // (self._n_beats * 2 + 1))
        r = max(r, 4)
        spacing = w / self._n_beats

        for i in range(self._n_beats):
            cx = int(spacing * (i + 0.5))
            cy = h // 2
            pos = i + 1  # 1-indexed
            if pos == self._active:
                color = self._COLOR_DOWNBEAT if pos == 1 else self._COLOR_BEAT
            else:
                color = self._COLOR_INACTIVE
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
