"""Tests for tools.mobile_companion.server."""

from fastapi.testclient import TestClient

from tools.mobile_companion.server import app

client = TestClient(app)


class TestHealth:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_status_ok(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}
