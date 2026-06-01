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


def parse_conn_log(path: str | Path) -> list[Conn]:
    """Stream a Zeek conn.log[.labeled] into Conn records, sorted by time."""
    path = Path(path)
    fields: list[str] | None = None
    idx: dict[str, int] = {}
    rows: list[Conn] = []

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
            p = line.rstrip("\n").split("\t")
            if len(p) < len(fields):
                continue

            def g(name, default="-"):
                return p[idx[name]] if name in idx and idx[name] < len(p) else default

            # IoT-23 label lives in the trailing columns; Zeek base logs have none.
            label = ""
            if "label" in idx:
                label = g("label")
            elif len(p) > len(fields):       # labels appended past #fields
                label = p[-2] if len(p) >= 2 else ""
            label = "Malicious" if "alicious" in label else ("Benign" if "enign" in label else "")

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


def group_by_device(conns: list[Conn]) -> dict[str, list[Conn]]:
    """Group connections by source device (originating host)."""
    out: dict[str, list[Conn]] = {}
    for c in conns:
        out.setdefault(c.src, []).append(c)
    return out
