"""Widget for live tap-tempo estimation."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

_RECENT_N = 8   # intervals used for the "recent" estimate
_WARMUP = 4     # initial intervals discarded to let tapping stabilise


class TapTempoWidget(QWidget):
    """Estimates BPM from user taps and shows live statistics.

    Call :meth:`tap` each time the user presses the tap key.  The widget
    displays the instantaneous tempo (last interval), a recent mean (last
    ``_RECENT_N`` intervals), and full-history mean / median / variance.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._timestamps: list[float] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tap(self) -> None:
        """Record a tap at the current wall-clock time and refresh."""
        self._timestamps.append(time.perf_counter())
        self._refresh()

    def reset(self) -> None:
        """Clear all tap history."""
        self._timestamps.clear()
        self._refresh()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setFixedHeight(40)

        self._last_label = QLabel("Last: —")
        self._recent_label = QLabel(f"Recent ({_RECENT_N}): —")
        self._all_label = QLabel("All — Mean: —   Median: —   Std: —")
        self._n_label = QLabel("N: 0")

        reset_btn = QPushButton("Reset taps")
        reset_btn.setFixedWidth(90)
        reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset_btn.clicked.connect(self.reset)

        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 4, 0)
        layout.addWidget(QLabel("Tap (Space):"))
        layout.addSpacing(8)
        layout.addWidget(self._last_label)
        layout.addSpacing(16)
        layout.addWidget(self._recent_label)
        layout.addSpacing(16)
        layout.addWidget(self._all_label)
        layout.addStretch()
        layout.addWidget(self._n_label)
        layout.addWidget(reset_btn)
        self.setLayout(layout)

    def _refresh(self) -> None:
        n = len(self._timestamps)
        self._n_label.setText(f"N: {n}")

        tempos = 60.0 / np.diff(self._timestamps) if n >= 2 else np.array([])
        valid = tempos[_WARMUP:]   # discard first _WARMUP intervals

        if len(valid) == 0:
            self._last_label.setText("Last: —")
            self._recent_label.setText(f"Recent ({_RECENT_N}): —")
            self._all_label.setText("All — Mean: —   Median: —   Std: —")
            return

        self._last_label.setText(f"Last: {valid[-1]:.1f} BPM")

        recent = valid[-_RECENT_N:]
        self._recent_label.setText(f"Recent ({_RECENT_N}): {np.mean(recent):.1f} BPM")

        self._all_label.setText(
            f"All — Mean: {np.mean(valid):.1f}   "
            f"Median: {np.median(valid):.1f}   "
            f"Std: {np.std(valid):.2f}"
        )
