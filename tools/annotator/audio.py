"""Audio playback engine backed by sounddevice.

The frame counter is updated inside the sounddevice callback (audio thread)
and read from the main thread via the ``position`` property.  Writing a
Python int is atomic under the GIL, so no explicit lock is needed for this
single-writer / single-reader pattern.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd


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
        self._frame: int = 0
        self._stream: sd.OutputStream | None = None
        self._finished_cb = None

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
        """Start or resume playback from the current position."""
        if self._audio is None or self.is_playing:
            return
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

    def on_finished(self, callback) -> None:
        """Register *callback* to be called when playback reaches the end."""
        self._finished_cb = callback

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_stream(self, start_frame: int) -> None:
        frame_ref = [start_frame]

        def _callback(outdata, frames, time_info, status):
            pos = frame_ref[0]
            end = pos + frames
            chunk = self._audio[pos:end]
            actual = len(chunk)
            outdata[:actual, 0] = chunk
            if actual < frames:
                outdata[actual:] = 0
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
