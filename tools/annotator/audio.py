"""Audio playback engine backed by sounddevice.

The frame counter is updated inside the sounddevice callback (audio thread)
and read from the main thread via the ``position`` property.  Writing a
Python int is atomic under the GIL, so no explicit lock is needed for this
single-writer / single-reader pattern.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd


def _make_click(sr: int, freq: float, duration: float = 0.02) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * np.exp(-200 * t)).astype(np.float32)


class AudioEngine:
    """Plays a mono float32 audio buffer with precise position tracking.

    Usage::

        engine = AudioEngine()
        engine.load(audio_array, sample_rate)
        engine.play()
        print(engine.position)  # seconds
        engine.pause()
        engine.seek(5.0)
        engine.play()
        engine.stop()
    """

    def __init__(self) -> None:
        self._audio: np.ndarray | None = None
        self._sr: int = 22050
        self._frame: float = 0.0
        self._volume: float = 1.0
        self._speed: float = 1.0
        self._stream: sd.OutputStream | None = None
        self._finished_cb = None
        self._beats_data: tuple[np.ndarray, np.ndarray] = (
            np.array([], dtype=int),
            np.array([], dtype=bool),
        )
        self._high_click: np.ndarray = _make_click(self._sr, 1000.0)
        self._low_click: np.ndarray = _make_click(self._sr, 600.0)
        self._click_enabled: bool = False
        self._click_volume: float = 0.7
        self._immediate_click: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, audio: np.ndarray, sr: int) -> None:
        """Load a mono audio array and reset position to zero."""
        self.stop()
        self._audio = np.asarray(audio, dtype=np.float32)
        self._sr = sr
        self._frame = 0

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        return self._frame / self._sr

    @property
    def duration(self) -> float:
        """Total audio duration in seconds, or 0 if no audio is loaded."""
        if self._audio is None:
            return 0.0
        return len(self._audio) / self._sr

    @property
    def is_playing(self) -> bool:
        """True while the audio stream is actively outputting samples."""
        return self._stream is not None and self._stream.active

    def play(self) -> None:
        """Start or resume playback from the current position.

        If playback has already reached the end, rewinds to the beginning first.
        """
        if self._audio is None or self.is_playing:
            return
        if self._frame >= len(self._audio):
            self._frame = 0
        self._start_stream(self._frame)

    def pause(self) -> None:
        """Pause playback, preserving the current position."""
        if self._stream is not None:
            self._stream.stop()

    def stop(self) -> None:
        """Stop playback and reset position to zero."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._frame = 0

    def seek(self, seconds: float) -> None:
        """Jump to *seconds* and resume if currently playing."""
        if self._audio is None:
            return
        was_playing = self.is_playing
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._frame = max(0, min(int(seconds * self._sr), len(self._audio) - 1))
        if was_playing:
            self._start_stream(self._frame)

    def set_volume(self, level: float) -> None:
        """Set playback volume. *level* is 0.0 (silent) to 1.0 (full)."""
        self._volume = max(0.0, min(1.0, level))

    def set_clicks(
        self,
        beat_frames: np.ndarray,
        beat_is_down: np.ndarray,
        sr: int,
    ) -> None:
        """Update beat click info. Regenerates click sounds if sample rate changed."""
        if sr != self._sr:
            self._high_click = _make_click(sr, 1000.0)
            self._low_click = _make_click(sr, 600.0)
        self._beats_data = (beat_frames, beat_is_down)  # single atomic assignment

    def set_click_enabled(self, enabled: bool) -> None:
        self._click_enabled = enabled

    def set_click_volume(self, level: float) -> None:
        self._click_volume = max(0.0, min(1.0, level))

    def trigger_click_now(self, is_down: bool) -> None:
        """Fire a click at the next callback invocation (annotation feedback)."""
        self._immediate_click = self._high_click if is_down else self._low_click

    def on_finished(self, callback) -> None:
        """Register *callback* to be called when playback reaches the end."""
        self._finished_cb = callback

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_stream(self, start_frame: int) -> None:
        frame_ref = [start_frame]
        pending: list[np.ndarray] = []  # click samples that spill into the next buffer

        def _mix(
            outdata: np.ndarray, frames: int, click: np.ndarray, offset: int
        ) -> None:
            scaled = click * self._click_volume
            n = min(len(scaled), frames - offset)
            outdata[offset : offset + n, 0] += scaled[:n]
            if n < len(scaled):
                pending.append(scaled[n:])

        def _callback(outdata, frames, time_info, status):
            pos = frame_ref[0]
            end = pos + frames
            chunk = self._audio[pos:end]
            actual = len(chunk)
            outdata[:actual, 0] = chunk * self._volume
            if actual < frames:
                outdata[actual:] = 0

            # Carry over click samples that spilled from the previous buffer
            carried = pending.copy()
            pending.clear()
            offset = 0
            for tail in carried:
                n = min(len(tail), frames - offset)
                outdata[offset : offset + n, 0] += tail[:n]
                if n < len(tail):
                    pending.append(tail[n:])
                offset += n

            # Immediate click (annotation feedback — beat already in the past)
            immediate = self._immediate_click
            if immediate is not None:
                self._immediate_click = None
                _mix(outdata, frames, immediate, 0)

            if self._click_enabled:
                beat_frames, beat_is_down = (
                    self._beats_data
                )  # single read — always consistent
                for i, bf in enumerate(beat_frames):
                    if pos <= bf < end:
                        click = self._high_click if beat_is_down[i] else self._low_click
                        _mix(outdata, frames, click, bf - pos)

            frame_ref[0] = pos + actual
            self._frame = frame_ref[0]
            if actual < frames:
                raise sd.CallbackStop()

        def _finished():
            if self._finished_cb is not None:
                self._finished_cb()

        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            callback=_callback,
            finished_callback=_finished,
        )
        self._stream.start()
