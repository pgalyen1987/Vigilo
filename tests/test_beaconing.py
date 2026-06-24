"""Tests for vigilo.beaconing periodic C2 detection."""
from vigilo.beaconing import beacon_pairs
from vigilo.zeek import parse_conn_log
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "beacon.conn.log"


def test_beacon_pairs_detects_regular_external_traffic():
    conns = parse_conn_log(FIXTURE)
    hits = beacon_pairs(conns, min_hits=6)
    assert hits
    src, dst, port, n, cv, mean, score = hits[0]
    assert src == "192.168.1.99"
    assert dst == "203.0.113.50"
    assert port == 443
    assert n >= 6
    assert score > 1.0


def test_beacon_pairs_ignores_irregular_traffic():
    conns = parse_conn_log(Path(__file__).parent / "fixtures" / "mini.conn.log")
    hits = [h for h in beacon_pairs(conns) if h[0] == "192.168.1.10"]
    assert hits == []
