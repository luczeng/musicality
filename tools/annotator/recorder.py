"""Microphone recorder backed by sounddevice."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

_SR = 44100


def _find_input_device() -> int | None:
    """Return the device index of the first device with at least one input channel.

    Prefers the system default input if it actually has input channels; otherwise
    falls back to the first device that does.
    """
    devices = sd.query_devices()
    default_input, _ = sd.default.device

    if default_input is not None and devices[default_input]["max_input_channels"] > 0:
        return default_input

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(
                f"[recorder] Default input device has no input channels; "
                f"using '{dev['name']}' (device {i}) instead."
            )
            return i

    return None


class Recorder:
    """Records microphone input to a WAV file.

    Usage::

        rec = Recorder()
        rec.start()
        audio = rec.current_audio   # live numpy array while recording
        path = rec.stop(save_dir, name)
    """

    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def current_audio(self) -> np.ndarray | None:
        """Return accumulated recording as a 1-D float32 array, or None if empty."""
        with self._lock:
            if not self._chunks:
                return None
            return np.concatenate(self._chunks).squeeze()

    def start(self) -> None:
        if self.is_recording:
            return

        device = _find_input_device()
        if device is None:
            print("WARNING: no input device with microphone channels found.")
            return

        with self._lock:
            self._chunks = []

        def _callback(indata, frames, time_info, status):
            if status:
                print(f"[recorder] {status}")
            with self._lock:
                self._chunks.append(indata.copy())

        self._stream = sd.InputStream(
            device=device,
            samplerate=_SR,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def stop(self, save_dir: Path, name: str) -> Path:
        """Stop recording and save to *save_dir*/*name*.wav. Returns the path."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            chunks = list(self._chunks)
            self._chunks = []

        safe_name = re.sub(r"[^\w\-]", "_", name.strip()) or "recording"
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{safe_name}.wav"

        if chunks:
            audio = np.concatenate(chunks)
            if np.abs(audio).max() == 0.0:
                print(
                    "WARNING: recording is silent — check microphone permissions in "
                    "System Settings → Privacy & Security → Microphone."
                )
            sf.write(str(out_path), audio, _SR)
        else:
            sf.write(str(out_path), np.zeros((1, 1), dtype="float32"), _SR)

        return out_path
