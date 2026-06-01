"""
Parse Zeek conn.log (and IoT-23 conn.log.labeled) into per-connection records.

Zeek logs are tab-separated with a `#fields` header naming the columns. We read
that header so we are robust to column ordering and to the extra label columns
present in IoT-23's `.labeled` files (tunnel_parents, label, detailed-label).

This same parser handles our proof data (IoT-23) and real deployments
(pfSense/OPNsense/Zeek conn.log), which is the whole point.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MISSING = {"-", "(empty)", ""}


def _f(v: str) -> float:
    return 0.0 if v in _MISSING else float(v)


@dataclass
class Conn:
    ts: float
    src: str
    dst: str
    dst_port: int
    proto: str
    service: str
    duration: float
    orig_bytes: float
    resp_bytes: float
    orig_pkts: float
    resp_pkts: float
    conn_state: str
    label: str          # "Benign" / "Malicious" / "" (unknown)


def parse_conn_log(path: str | Path, max_lines: int | None = None,
                   stride: int = 1) -> list[Conn]:
    """Stream a Zeek conn.log[.labeled] into Conn records, sorted by time.

    max_lines caps how many data rows are kept — a memory safety valve.
    stride keeps only every Nth data row, so a capped read SAMPLES ACROSS the
    whole capture (preserving its full time span) instead of taking just the
    front — important for huge captures where the front is time-compressed.
    """
    path = Path(path)
    fields: list[str] | None = None
    idx: dict[str, int] = {}
    rows: list[Conn] = []
    n_seen = 0      # data rows encountered
    n_kept = 0      # data rows kept

    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
                idx = {name: i for i, name in enumerate(fields)}
                continue
            if line.startswith("#") or not line.strip():
                continue
            if fields is None:
                continue
            if max_lines is not None and n_kept >= max_lines:
                break
            if stride > 1 and (n_seen % stride) != 0:
                n_seen += 1
                continue
            n_seen += 1
            n_kept += 1
            p = line.rstrip("\n").split("\t")
            if len(p) < len(fields):
                continue

            def g(name, default="-"):
                return p[idx[name]] if name in idx and idx[name] < len(p) else default

            # IoT-23 stores the label in trailing column(s) whose exact tab/space
            # layout varies between dataset versions (separate `label` column in
            # some, space-packed `tunnel_parents label detailed-label` in others).
            # The tokens "Malicious"/"Benign" only appear in the label, so scan
            # the row robustly instead of relying on a fixed column index.
            label = "Malicious" if "Malicious" in line else ("Benign" if "Benign" in line else "")

            rows.append(Conn(
                ts=_f(g("ts")),
                src=g("id.orig_h"),
                dst=g("id.resp_h"),
                dst_port=int(_f(g("id.resp_p"))),
                proto=g("proto"),
                service=g("service"),
                duration=_f(g("duration")),
                orig_bytes=_f(g("orig_bytes")),
                resp_bytes=_f(g("resp_bytes")),
                orig_pkts=_f(g("orig_pkts")),
                resp_pkts=_f(g("resp_pkts")),
                conn_state=g("conn_state"),
                label=label,
            ))

    rows.sort(key=lambda c: c.ts)
    return rows


def is_local_ip(ip: str) -> bool:
    """RFC1918 private address — i.e., a device on the monitored network.

    External IPs appearing as sources are response traffic, not devices we
    monitor, so the engine scores only local devices (matches deployment)."""
    return (ip.startswith("192.168.")
            or ip.startswith("10.")
            or any(ip.startswith(f"172.{o}.") for o in range(16, 32)))


def group_by_device(conns: list[Conn], local_only: bool = True) -> dict[str, list[Conn]]:
    """Group connections by source device (originating host).

    local_only restricts to RFC1918 source IPs (the devices on the network)."""
    out: dict[str, list[Conn]] = {}
    for c in conns:
        if local_only and not is_local_ip(c.src):
            continue
        out.setdefault(c.src, []).append(c)
    return out
