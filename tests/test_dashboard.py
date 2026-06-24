"""Tests for the Flask dashboard."""
import os

import pytest

from vigilo.dashboard import app


@pytest.fixture
def client():
    demo_log = "data/demo/sample.conn.log"
    demo_ckpt = "checkpoints/demo/vigilo.pt"
    if not os.path.exists(demo_log) or not os.path.exists(demo_ckpt):
        pytest.skip("demo bundle missing — run: python scripts/generate_demo_data.py")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_healthz(client):
    rv = client.get("/healthz")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["status"] == "ok"


def test_api_results(client):
    rv = client.get("/api/results")
    assert rv.status_code == 200
    data = rv.get_json()
    assert "results" in data
    assert data["devices"] >= 1


def test_index_renders(client):
    rv = client.get("/")
    assert rv.status_code == 200
    assert b"Vigilo" in rv.data
