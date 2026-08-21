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


def _stereo_wav_bytes(duration_s: float = 0.5, sr: int = 44100) -> bytes:
    """Synthetic 2-channel WAV, to exercise the mono mixdown branch."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    audio = np.stack([left, right], axis=1)
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

    def test_mixes_down_stereo_to_mono(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _stereo_wav_bytes(), "audio/wav")},
            data={"name": "take1"},
        )
        out_path = tmp_path / "field_recordings" / "tracks" / "take1.wav"
        audio, _ = sf.read(str(out_path))
        assert audio.ndim == 1

    def test_no_resample_when_already_target_rate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={
                "file": ("clip.wav", _wav_bytes(duration_s=0.5, sr=44100), "audio/wav")
            },
            data={"name": "take1"},
        )
        out_path = tmp_path / "field_recordings" / "tracks" / "take1.wav"
        audio, sr = sf.read(str(out_path))
        assert sr == 44100
        assert len(audio) == int(0.5 * 44100)

    def test_missing_name_generates_track_id(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        response = client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        )
        assert response.status_code == 200
        track_id = response.json()["track_id"]
        assert re.fullmatch(r"field_\d{8}_\d{6}_\d{6}", track_id)
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


class TestUploadAnnotation:
    def _upload_clip(self, dataset: str, track_id: str) -> None:
        client.post(
            f"/datasets/{dataset}/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": track_id},
        )

    def test_returns_200(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        response = client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0]},
        )
        assert response.status_code == 200

    def test_beat_times_round_trip_via_load_track(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0]},
        )
        track = annotator_data.load_track("field_recordings", "take1")
        np.testing.assert_allclose(track.beat_times, [0.5, 1.0, 1.5, 2.0])

    def test_sorts_out_of_order_tap_times(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [1.5, 0.5, 1.0]},
        )
        track = annotator_data.load_track("field_recordings", "take1")
        np.testing.assert_allclose(track.beat_times, [0.5, 1.0, 1.5])

    def test_returns_tempo_estimated_from_taps(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        response = client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.0, 0.5, 1.0, 1.5]},
        )
        assert response.json()["tempo"] == pytest.approx(120.0)

    def test_saves_structure_and_device(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={
                "tap_times": [0.5, 1.0, 1.5, 2.0],
                "structure": "blues",
                "device": "iPhone 13 mini",
            },
        )
        metadata = annotator_data.load_metadata("field_recordings", "take1")
        assert metadata.structure == "blues"
        assert metadata.device == "iPhone 13 mini"

    def test_omitting_structure_and_device_saves_no_metadata(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0]},
        )
        assert annotator_data.load_metadata("field_recordings", "take1") is None

    def test_preserves_existing_field_when_only_one_is_sent(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        annotator_data.save_metadata(
            "field_recordings", "take1", annotator_data.TrackMetadata(device="iPhone")
        )
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0], "structure": "swing"},
        )
        metadata = annotator_data.load_metadata("field_recordings", "take1")
        assert metadata.structure == "swing"
        assert metadata.device == "iPhone"

    def test_saves_section_aligned_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0], "section_aligned": True},
        )
        metadata = annotator_data.load_metadata("field_recordings", "take1")
        assert metadata.section_aligned is True

    def test_saves_section_aligned_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0], "section_aligned": False},
        )
        metadata = annotator_data.load_metadata("field_recordings", "take1")
        assert metadata.section_aligned is False

    def test_omitting_section_aligned_saves_no_metadata(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0]},
        )
        assert annotator_data.load_metadata("field_recordings", "take1") is None


class TestListTracks:
    def test_empty_dataset_returns_empty_list(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        response = client.get("/datasets/field_recordings/tracks")
        assert response.status_code == 200
        assert response.json() == []

    def test_lists_uploaded_tracks_sorted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "take2"},
        )
        client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "take1"},
        )
        response = client.get("/datasets/field_recordings/tracks")
        track_ids = [t["track_id"] for t in response.json()]
        assert track_ids == ["take1", "take2"]

    def test_has_annotation_reflects_saved_beats(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "take1"},
        )
        client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "take2"},
        )
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0]},
        )
        response = client.get("/datasets/field_recordings/tracks")
        by_id = {t["track_id"]: t for t in response.json()}
        assert by_id["take1"]["has_annotation"] is True
        assert by_id["take2"]["has_annotation"] is False

    def test_meter_reflects_beat_positions(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "take1"},
        )
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]},
        )
        response = client.get("/datasets/field_recordings/tracks")
        assert response.json()[0]["meter"] == "1..8"


class TestGetAnnotation:
    def _upload_clip(self, dataset: str, track_id: str) -> None:
        client.post(
            f"/datasets/{dataset}/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": track_id},
        )

    def test_returns_200(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        response = client.get("/datasets/field_recordings/tracks/take1/annotations")
        assert response.status_code == 200

    def test_round_trips_saved_tap_times(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={"tap_times": [0.5, 1.0, 1.5, 2.0]},
        )
        response = client.get("/datasets/field_recordings/tracks/take1/annotations")
        assert response.json()["tap_times"] == [0.5, 1.0, 1.5, 2.0]

    def test_round_trips_metadata(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        client.post(
            "/datasets/field_recordings/tracks/take1/annotations",
            json={
                "tap_times": [0.0, 0.5, 1.0, 1.5],
                "structure": "blues",
                "device": "iPhone 13 mini",
                "section_aligned": True,
            },
        )
        response = client.get("/datasets/field_recordings/tracks/take1/annotations")
        body = response.json()
        assert body["structure"] == "blues"
        assert body["device"] == "iPhone 13 mini"
        assert body["section_aligned"] is True
        assert body["tempo"] == pytest.approx(120.0)

    def test_track_with_no_saved_annotation_returns_empty_tap_times(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        self._upload_clip("field_recordings", "take1")
        response = client.get("/datasets/field_recordings/tracks/take1/annotations")
        assert response.status_code == 200
        assert response.json()["tap_times"] == []


class TestGetAudio:
    def test_returns_200_with_wav_content_type(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"name": "take1"},
        )
        response = client.get("/datasets/field_recordings/tracks/take1/audio")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_returned_bytes_are_redecodable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        client.post(
            "/datasets/field_recordings/tracks",
            files={
                "file": ("clip.wav", _wav_bytes(duration_s=0.5, sr=22050), "audio/wav")
            },
            data={"name": "take1"},
        )
        response = client.get("/datasets/field_recordings/tracks/take1/audio")
        audio, sr = sf.read(io.BytesIO(response.content))
        assert sr == 44100
        assert len(audio) == pytest.approx(0.5 * 44100, abs=100)

    def test_missing_track_returns_404(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        response = client.get("/datasets/field_recordings/tracks/nope/audio")
        assert response.status_code == 404
