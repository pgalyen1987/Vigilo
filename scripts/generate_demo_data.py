#!/usr/bin/env python3
"""
Generate bundled demo Zeek conn.log files and a trained checkpoint.

Creates:
  data/demo/benign.conn.log   — synthetic home-network traffic for training
  data/demo/sample.conn.log   — benign traffic + one beaconing device (for alerts)
  checkpoints/demo/vigilo.pt  — model trained on benign.conn.log

Run from repo root:
    python scripts/generate_demo_data.py
"""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"
CKPT_DIR = ROOT / "checkpoints" / "demo"

FIELD_NAMES = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
    "service", "duration", "orig_bytes", "resp_bytes", "conn_state",
    "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts",
    "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "tunnel_parents",
]

DEVICES = {
    "192.168.1.10": "laptop",
    "192.168.1.20": "phone",
    "192.168.1.30": "camera",
    "192.168.1.40": "plug",
}

EXTERNAL = [
    ("8.8.8.8", 53, "udp", "dns", "SF"),
    ("1.1.1.1", 53, "udp", "dns", "SF"),
    ("142.250.80.46", 443, "tcp", "ssl", "SF"),
    ("151.101.1.140", 443, "tcp", "ssl", "SF"),
    ("52.84.0.0", 443, "tcp", "ssl", "SF"),
    ("34.117.59.81", 443, "tcp", "ssl", "SF"),
]


def _row(ts, src, dst, port, proto, service, state, ob, rb, op, rp):
    uid = f"C{random.randint(0, 0xFFFFFF):06x}"
    dur = round(random.uniform(0.01, 2.5), 3)
    return (
        f"{ts:.6f}\t{uid}\t{src}\t{random.randint(49152, 65535)}\t{dst}\t{port}\t"
        f"{proto}\t{service}\t{dur}\t{ob}\t{rb}\t{state}\t"
        f"T\tF\t0\t^-\t{op}\t{rp}\t{ob + 54}\t{rb + 54}\t-"
    )


def _benign_flows(start_ts: float, duration_s: float, rng: random.Random) -> list[str]:
    rows: list[str] = []
    window_s = 300.0
    n_windows = int(duration_s // window_s)
    for win in range(n_windows):
        t_base = start_ts + win * window_s
        for src in DEVICES:
            for _ in range(rng.randint(3, 8)):
                dst, port, proto, service, state = rng.choice(EXTERNAL)
                t = t_base + rng.uniform(1.0, window_s - 1.0)
                ob = rng.randint(60, 4000)
                rb = rng.randint(60, 8000)
                op = rng.randint(3, 20)
                rp = rng.randint(3, 20)
                rows.append(_row(t, src, dst, port, proto, service, state, ob, rb, op, rp))
    return rows


def _beacon_flows(start_ts: float, duration_s: float, src: str, dst: str,
                  port: int, interval: float) -> list[str]:
    rows: list[str] = []
    t = start_ts
    end = start_ts + duration_s
    i = 0
    while t < end:
        rows.append(_row(t, src, dst, port, "tcp", "ssl", "SF", 120, 80, 4, 3))
        i += 1
        t += interval + random.uniform(-0.05, 0.05)
    return rows


def _write_conn(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: float(r.split("\t", 1)[0]))
    with open(path, "w") as f:
        f.write("#fields\t" + "\t".join(FIELD_NAMES) + "\n")
        f.write("#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\t"
                "count\tcount\tstring\tbool\tbool\tcount\tstring\tcount\tcount\t"
                "count\tcount\tset[string]\n")
        for row in rows:
            f.write(row + "\n")
    print(f"[demo] wrote {path} ({len(rows)} flows)")


def main() -> int:
    rng = random.Random(42)
    start = 1_700_000_000.0
    duration = 7200.0  # 2 hours → enough 5-min windows per device

    benign_rows = _benign_flows(start, duration, rng)
    _write_conn(DEMO_DIR / "benign.conn.log", benign_rows)

    sample_rows = list(benign_rows)
    sample_rows.extend(_beacon_flows(start, duration, "192.168.1.99", "203.0.113.50", 443, 60.0))
    _write_conn(DEMO_DIR / "sample.conn.log", sample_rows)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / "vigilo.pt"
    if not ckpt.exists():
        print("[demo] training demo checkpoint (5 epochs, ~30s)...")
        subprocess.run(
            [
                sys.executable, "-m", "vigilo.train",
                "--logs", str(DEMO_DIR / "benign.conn.log"),
                "--output-dir", str(CKPT_DIR),
                "--epochs", "5",
                "--device", "cpu",
            ],
            cwd=ROOT,
            check=True,
        )
    else:
        print(f"[demo] checkpoint already exists: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
