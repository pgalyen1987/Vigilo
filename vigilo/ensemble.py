"""
Ensemble verdict: combine the two detectors into one per-device decision.

  - volume model  (forecaster): catches loud attacks (scans, floods, noisy C&C)
  - beaconing      (periodicity): catches stealthy periodic C&C the volume model misses

A device is ALERTed if either fires, with a human-readable reason — the basis
for the dashboard and for explainable alerts.
"""
from __future__ import annotations

import numpy as np
import torch

from arch.config import ModelConfig
from vigilo.zeek import parse_conn_log, group_by_device
from vigilo.features import device_windows, apply_global, per_asset_normalize
from vigilo.beaconing import beacon_pairs
from vigilo.forecaster import PdMForecaster


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = PdMForecaster(ModelConfig(**ck["model_config"]), n_features=ck["n_features"])
    m.load_state_dict(ck["model_state"]); m.eval()
    gmu = np.array(ck["global_mu"], dtype=np.float32) if ck.get("global_mu") else None
    gsd = np.array(ck["global_sd"], dtype=np.float32) if ck.get("global_sd") else None
    return m, ck, gmu, gsd


@torch.no_grad()
def _volume_score(model, ck, gmu, gsd, dconns) -> float:
    w = device_windows(dconns, window_s=ck["window_s"])
    if w.shape[0] < ck.get("min_windows", 12):
        return 0.0
    if ck.get("normalize") == "global" and gmu is not None:
        wn = apply_global(w, gmu, gsd)
    else:
        wn = per_asset_normalize(w, baseline=ck.get("baseline", 10))
    x = torch.from_numpy(wn.astype(np.float32)).unsqueeze(0)
    pred = model(x)[0][0].numpy()
    err = ((pred[:-1] - wn[1:]) ** 2).mean(axis=1)
    return float(np.percentile(err, 95)) if len(err) else 0.0


def analyze(log_path: str, ckpt_path: str = "checkpoints/vigilo/vigilo.pt",
            vol_thresh: float = 2.0, beacon_thresh: float = 1.5,
            max_flows: int = 2_000_000) -> list[dict]:
    """Return per-device verdicts: score, beaconing, ALERT/ok, reasons."""
    model, ck, gmu, gsd = load_model(ckpt_path)
    conns = parse_conn_log(log_path, max_lines=max_flows)

    # strongest beacon per device (with detail for the reason string)
    top_beacon: dict[str, tuple] = {}
    for src, dst, port, n, cv, m, score in beacon_pairs(conns):
        if src not in top_beacon:
            top_beacon[src] = (dst, port, cv, m, score)

    results = []
    for src, dconns in group_by_device(conns).items():
        vol = _volume_score(model, ck, gmu, gsd, dconns)
        b = top_beacon.get(src)
        bscore = b[4] if b else 0.0
        reasons = []
        if vol > vol_thresh:
            reasons.append(f"abnormal traffic pattern (anomaly score {vol:.1f})")
        if bscore > beacon_thresh:
            reasons.append(f"periodic beaconing to {b[0]}:{b[1]} every ~{b[3]:.0f}s "
                           f"(regularity {1-b[2]:.2f})")
        results.append({
            "device": src,
            "flows": len(dconns),
            "volume_score": round(vol, 2),
            "beacon_score": round(bscore, 2),
            "verdict": "ALERT" if reasons else "ok",
            "reasons": reasons,
        })
    results.sort(key=lambda r: (r["verdict"] != "ALERT", -max(r["volume_score"], r["beacon_score"])))
    return results


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt", default="checkpoints/vigilo/vigilo.pt")
    a = ap.parse_args()
    for r in analyze(a.log, a.ckpt):
        mark = "🚨" if r["verdict"] == "ALERT" else "  "
        print(f"{mark} {r['device']:<16} vol={r['volume_score']:<6} beacon={r['beacon_score']:<6} "
              f"{'; '.join(r['reasons'])}", flush=True)
