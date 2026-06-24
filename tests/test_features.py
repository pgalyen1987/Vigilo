"""Tests for vigilo.features behavioral windows."""
import numpy as np

from vigilo.features import N_FEATURES, device_windows
from vigilo.zeek import parse_conn_log
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "mini.conn.log"


def test_device_windows_shape():
    conns = parse_conn_log(FIXTURE)
    w = device_windows(conns, window_s=300.0)
    assert w.ndim == 2
    assert w.shape[1] == N_FEATURES
    assert w.shape[0] >= 1
    assert np.isfinite(w).all()
