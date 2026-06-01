"""
Per-device behavioral feature windows.

For each device we bin its connections into fixed time windows and summarize each
window into a vector capturing the behaviors that distinguish normal from
malicious activity:

  - volume      : #connections, bytes, packets
  - fan-out     : #distinct destination IPs / ports
  - failure     : fraction of S0 / REJ states (scanning signature)
  - regularity  : destination-IP entropy (low = beaconing to one host)
  - spread      : destination-port entropy (high = port scanning)

The model forecasts the next window's vector from a device's recent history;
a large forecast error means the device is behaving unlike its own normal.
"""
from __future__ import annotations

import numpy as np

from vigilo.zeek import Conn

FEATURE_NAMES = [
    "log_n_conns", "log_distinct_dst_ip", "log_distinct_dst_port",
    "log_orig_bytes", "log_resp_bytes", "log_orig_pkts", "log_resp_pkts",
    "frac_tcp", "frac_udp", "frac_s0", "frac_rej", "frac_sf",
    "dst_ip_entropy", "dst_port_entropy", "log_mean_duration",
]
N_FEATURES = len(FEATURE_NAMES)


def _entropy(counts: list[int]) -> float:
    c = np.array(counts, dtype=np.float64)
    if c.sum() <= 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def _window_vec(conns: list[Conn]) -> np.ndarray:
    n = len(conns)
    dst_ips: dict[str, int] = {}
    dst_ports: dict[int, int] = {}
    tcp = udp = s0 = rej = sf = 0
    ob = rb = op = rp = dur = 0.0
    for c in conns:
        dst_ips[c.dst] = dst_ips.get(c.dst, 0) + 1
        dst_ports[c.dst_port] = dst_ports.get(c.dst_port, 0) + 1
        tcp += c.proto == "tcp"
        udp += c.proto == "udp"
        s0 += c.conn_state == "S0"
        rej += c.conn_state == "REJ"
        sf += c.conn_state == "SF"
        ob += c.orig_bytes; rb += c.resp_bytes
        op += c.orig_pkts; rp += c.resp_pkts
        dur += c.duration
    inv = 1.0 / max(n, 1)
    return np.array([
        np.log1p(n),
        np.log1p(len(dst_ips)),
        np.log1p(len(dst_ports)),
        np.log1p(ob), np.log1p(rb), np.log1p(op), np.log1p(rp),
        tcp * inv, udp * inv, s0 * inv, rej * inv, sf * inv,
        _entropy(list(dst_ips.values())),
        _entropy(list(dst_ports.values())),
        np.log1p(dur * inv),
    ], dtype=np.float32)


def device_windows(conns: list[Conn], window_s: float = 300.0) -> np.ndarray:
    """Return (T, N_FEATURES) feature windows for one device, time-ordered."""
    if not conns:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    t0 = conns[0].ts
    buckets: dict[int, list[Conn]] = {}
    for c in conns:
        b = int((c.ts - t0) // window_s)
        buckets.setdefault(b, []).append(c)
    bmax = max(buckets)
    rows = [_window_vec(buckets.get(b, [])) for b in range(bmax + 1)]
    return np.stack(rows, axis=0)


def per_asset_normalize(windows: np.ndarray, baseline: int = 10):
    """Normalize a device's windows by its own early-life (healthy) baseline.

    Use when a device has a trusted healthy history (detect when IT changes)."""
    if windows.shape[0] == 0:
        return windows
    base = windows[:baseline] if windows.shape[0] >= baseline else windows
    mu = base.mean(axis=0)
    sd = base.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (windows - mu) / sd


def fit_global(window_list: list[np.ndarray]):
    """Fit population mean/std over all benign training windows.

    Use for cold-start cross-device scoring (does this device look like a
    normal device?) when a per-device baseline isn't yet available."""
    allw = np.concatenate([w for w in window_list if len(w)], axis=0)
    mu = allw.mean(axis=0)
    sd = allw.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_global(windows: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (windows - mu) / sd
