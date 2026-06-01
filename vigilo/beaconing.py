"""
Beaconing detector — catches stealthy C&C that the volume model misses.

Malware command-and-control calls home on a clock (every 30s, 60s, ...). That
periodicity is invisible to per-window volume features (a handful of beacons get
averaged away), but it's a strong signal on its own: a real device's chatter to
a host is irregular; a beacon is metronome-steady.

For each (local device -> destination) pair we measure the regularity of the
inter-connection intervals. Very low interval variation + enough repetitions =
a beacon, regardless of how few flows it is.

This complements the forecast-error detector (which catches loud attacks):
ensemble = volume model (loud) + beaconing (periodic stealth).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from vigilo.zeek import parse_conn_log, is_local_ip


def beacon_pairs(conns, min_hits=8, min_interval=0.5, max_interval=3600.0):
    """Return per (device, dst, port) beaconing records, strongest first.

    Each record: (src, dst, port, n, cv, mean_interval, score).
    score = regularity (1-CV, clipped) weighted by log(repetitions); high = beacon.
    """
    pairs: dict[tuple, list[float]] = defaultdict(list)
    for c in conns:
        if not is_local_ip(c.src):
            continue
        pairs[(c.src, c.dst, c.dst_port)].append(c.ts)

    out = []
    for (src, dst, port), ts in pairs.items():
        if len(ts) < min_hits:
            continue
        t = np.sort(np.asarray(ts, dtype=np.float64))
        iv = np.diff(t)
        iv = iv[iv > 0]
        if len(iv) < min_hits - 1:
            continue
        mean_iv = float(iv.mean())
        if mean_iv < min_interval or mean_iv > max_interval:
            continue
        cv = float(iv.std() / mean_iv) if mean_iv > 0 else 9.0
        regularity = max(0.0, 1.0 - cv)          # 1.0 = perfectly periodic
        score = regularity * float(np.log1p(len(t)))
        out.append((src, dst, port, len(t), cv, mean_iv, score))

    out.sort(key=lambda r: -r[6])
    return out


def device_beacon_score(conns, **kw) -> dict[str, float]:
    """Per-device beaconing score = its strongest beacon pair."""
    best: dict[str, float] = {}
    for src, dst, port, n, cv, m, score in beacon_pairs(conns, **kw):
        if score > best.get(src, 0.0):
            best[src] = score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=1.5,
                    help="device flagged if beacon score exceeds this")
    ap.add_argument("--max-flows", type=int, default=2_000_000)
    args = ap.parse_args()

    conns = parse_conn_log(args.log, max_lines=args.max_flows)
    pairs = beacon_pairs(conns, min_hits=args.min_hits)
    print(f"[BEACON] top periodic (device -> dst:port) relationships:", flush=True)
    print(f"{'device':<16}{'destination':<18}{'port':>6}{'hits':>7}{'CV':>7}"
          f"{'interval_s':>12}{'score':>8}", flush=True)
    for src, dst, port, n, cv, m, score in pairs[:15]:
        mark = "  <== BEACON" if score > args.threshold else ""
        print(f"{src:<16}{dst:<18}{port:>6}{n:>7}{cv:>7.3f}{m:>12.1f}{score:>8.2f}{mark}",
              flush=True)
    flagged = [s for s, v in device_beacon_score(conns, min_hits=args.min_hits).items()
               if v > args.threshold]
    print(f"\n[BEACON] devices flagged as beaconing: {flagged}", flush=True)


if __name__ == "__main__":
    main()
