"""Tests for tools.mobile_companion.server."""

from fastapi.testclient import TestClient

import tools.annotator.data as annotator_data
from tools.mobile_companion.server import app

client = TestClient(app)


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
