"""Integration tests using bundled demo data."""
import os
from pathlib import Path

import pytest

from vigilo.ensemble import analyze

ROOT = Path(__file__).resolve().parents[1]
DEMO_LOG = ROOT / "data" / "demo" / "sample.conn.log"
DEMO_CKPT = ROOT / "checkpoints" / "demo" / "vigilo.pt"


@pytest.fixture(scope="module")
def demo_paths():
    if not DEMO_LOG.exists() or not DEMO_CKPT.exists():
        pytest.skip("demo bundle missing — run: python scripts/generate_demo_data.py")
    return str(DEMO_LOG), str(DEMO_CKPT)


def test_analyze_finds_beacon_device(demo_paths):
    log, ckpt = demo_paths
    results = analyze(log, ckpt)
    by_device = {r["device"]: r for r in results}
    assert "192.168.1.99" in by_device
    alert_devices = [r for r in results if r["verdict"] == "ALERT"]
    assert alert_devices, "expected at least one ALERT on sample demo log"


def test_analyze_healthy_devices_ok(demo_paths):
    log, ckpt = demo_paths
    results = analyze(log, ckpt)
    by_device = {r["device"]: r for r in results}
    assert by_device.get("192.168.1.10", {}).get("verdict") == "ok"
