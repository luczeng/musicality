"""Tests for tools.mobile_companion.server."""

import io
import re

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import tools.annotator.data as annotator_data
from tools.mobile_companion.server import app

client = TestClient(app)


def _wav_bytes(duration_s: float = 0.5, sr: int = 22050) -> bytes:
    """Synthetic sine-wave WAV, at a different sample rate than the server's target."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestHealth:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_status_ok(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


class TestDatasets:
    def test_returns_200(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        response = client.get("/datasets")
        assert response.status_code == 200

    def test_reflects_custom_dataset(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)

        dataset_dir = tmp_path / "field_recordings"
        tracks_dir = dataset_dir / "tracks"
        tracks_dir.mkdir(parents=True)
        (tracks_dir / "track1.wav").touch()
        (tracks_dir / "track2.wav").touch()

        ann_dir = dataset_dir / "annotations"
        ann_dir.mkdir()
        (ann_dir / "track1.beats").touch()

        response = client.get("/datasets")
        assert response.json() == [
            {"name": "field_recordings", "n_tracks": 2, "n_annotations": 1}
        ]

    def test_empty_data_dir_returns_empty_list(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        response = client.get("/datasets")
        assert response.json() == []


class TestUploadTrack:
    def test_returns_200_with_track_id(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        response = client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "sound check"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "dataset": "field_recordings",
            "track_id": "sound_check",
        }

    def test_writes_wav_at_expected_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "take1"},
        )
        out_path = tmp_path / "field_recordings" / "tracks" / "take1.wav"
        assert out_path.exists()

    def test_written_wav_is_resampled_and_redecodable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={
                "file": ("clip.wav", _wav_bytes(duration_s=0.5, sr=22050), "audio/wav")
            },
            data={"name": "take1"},
        )
        out_path = tmp_path / "field_recordings" / "tracks" / "take1.wav"
        audio, sr = sf.read(str(out_path))
        assert sr == 44100
        assert len(audio) == pytest.approx(0.5 * 44100, abs=100)

    def test_missing_name_generates_track_id(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        response = client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        )
        assert response.status_code == 200
        track_id = response.json()["track_id"]
        assert re.fullmatch(r"field_\d{8}_\d{6}", track_id)
        assert (tmp_path / "field_recordings" / "tracks" / f"{track_id}.wav").exists()

    def test_creates_tracks_dir_for_new_dataset(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        assert not (tmp_path / "new_dataset").exists()
        response = client.post(
            "/datasets/new_dataset/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "first"},
        )
        assert response.status_code == 200
        assert (tmp_path / "new_dataset" / "tracks" / "first.wav").exists()

    def test_invalid_audio_returns_400(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        response = client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", b"not audio data", "audio/wav")},
        )
        assert response.status_code == 400
