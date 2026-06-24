"""Tests for vigilo.zeek conn.log parsing."""
from pathlib import Path

from vigilo.zeek import Conn, group_by_device, is_local_ip, parse_conn_log

FIXTURE = Path(__file__).parent / "fixtures" / "mini.conn.log"


def test_parse_conn_log_reads_fields():
    rows = parse_conn_log(FIXTURE)
    assert len(rows) == 4
    assert all(isinstance(r, Conn) for r in rows)
    assert rows[0].src == "192.168.1.10"
    assert rows[0].proto == "udp"


def test_parse_conn_log_sorted_by_time():
    rows = parse_conn_log(FIXTURE)
    ts = [r.ts for r in rows]
    assert ts == sorted(ts)


def test_group_by_device_local_only():
    rows = parse_conn_log(FIXTURE)
    devs = group_by_device(rows)
    assert "192.168.1.10" in devs
    assert "8.8.8.8" not in devs


def test_is_local_ip():
    assert is_local_ip("192.168.1.1")
    assert is_local_ip("10.0.0.5")
    assert is_local_ip("172.16.0.1")
    assert not is_local_ip("8.8.8.8")
