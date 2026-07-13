"""Tests for tools.annotator.audio.AudioEngine.

sounddevice is mocked so no real audio device is required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from tools.annotator.audio import AudioEngine


AUDIO = np.zeros(22050, dtype=np.float32)  # 1 second of silence
SR = 22050


class TestAudioEngineInitialState:
    def test_position_is_zero(self):
        assert AudioEngine().position == 0.0

    def test_duration_is_zero(self):
        assert AudioEngine().duration == 0.0

    def test_not_playing(self):
        assert not AudioEngine().is_playing


class TestAudioEngineLoad:
    def test_duration_after_load(self):
        e = AudioEngine()
        e.load(AUDIO, SR)
        assert e.duration == pytest.approx(1.0)

    def test_position_reset_on_load(self):
        e = AudioEngine()
        e.load(AUDIO, SR)
        e._frame = 5000
        e.load(AUDIO, SR)
        assert e.position == 0.0


class TestAudioEngineSeek:
    def test_seek_updates_position(self):
        e = AudioEngine()
        e.load(AUDIO, SR)
        e.seek(0.5)
        assert e.position == pytest.approx(0.5, abs=1 / SR)

    def test_seek_clamps_below_zero(self):
        e = AudioEngine()
        e.load(AUDIO, SR)
        e.seek(-1.0)
        assert e.position == 0.0

    def test_seek_clamps_above_duration(self):
        e = AudioEngine()
        e.load(AUDIO, SR)
        e.seek(100.0)
        assert e.position <= e.duration


class TestAudioEngineStop:
    def test_stop_resets_position(self):
        e = AudioEngine()
        e.load(AUDIO, SR)
        e._frame = 5000
        e.stop()
        assert e.position == 0.0


class TestAudioEnginePlay:
    @patch("tools.annotator.audio.sd.OutputStream")
    def test_play_starts_stream(self, mock_cls):
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_cls.return_value = mock_stream

        e = AudioEngine()
        e.load(AUDIO, SR)
        e.play()

        mock_cls.assert_called_once()
        mock_stream.start.assert_called_once()

    @patch("tools.annotator.audio.sd.OutputStream")
    def test_play_noop_when_already_playing(self, mock_cls):
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_cls.return_value = mock_stream

        e = AudioEngine()
        e.load(AUDIO, SR)
        e.play()
        e.play()  # second call should be a no-op

        assert mock_cls.call_count == 1

    @patch("tools.annotator.audio.sd.OutputStream")
    def test_pause_stops_stream(self, mock_cls):
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_cls.return_value = mock_stream

        e = AudioEngine()
        e.load(AUDIO, SR)
        e.play()
        e.pause()

        mock_stream.stop.assert_called_once()


class TestAudioEngineSpeed:
    def test_default_speed_is_full(self):
        assert AudioEngine()._speed == 1.0

    def test_set_speed_clamps_to_valid_range(self):
        e = AudioEngine()
        e.set_speed(0.9)
        assert e._speed == pytest.approx(0.9)
        e.set_speed(2.0)
        assert e._speed == 1.0
        e.set_speed(-1.0)
        assert e._speed == 0.1

    @patch("tools.annotator.audio.sd.OutputStream")
    def test_slower_speed_advances_position_more_slowly(self, mock_cls):
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_cls.return_value = mock_stream

        e = AudioEngine()
        e.load(AUDIO, SR)
        e.set_speed(0.9)
        e.play()

        callback = mock_cls.call_args.kwargs["callback"]
        frames = 512
        outdata = np.zeros((frames, 1), dtype=np.float32)
        callback(outdata, frames, None, None)

        assert e.position == pytest.approx(frames * 0.9 / SR)

    @patch("tools.annotator.audio.sd.OutputStream")
    def test_full_speed_advances_position_exactly(self, mock_cls):
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_cls.return_value = mock_stream

        e = AudioEngine()
        e.load(AUDIO, SR)
        e.play()

        callback = mock_cls.call_args.kwargs["callback"]
        frames = 512
        outdata = np.zeros((frames, 1), dtype=np.float32)
        callback(outdata, frames, None, None)

        assert e.position == pytest.approx(frames / SR)
