"""
Evaluate detector tightness: separate benign windows from malicious windows.

Loads a trained checkpoint, scores per-window forecast error on a held-out
benign log and a malware log, and reports:
  - mean score for each class
  - ROC-AUC (rank statistic, no sklearn dependency)
  - detection rate at benign-calibrated false-positive rates (1% and 5%)

"Tightening" = pushing benign scores down and AUC / detection@1%FPR up.

Usage:
    python -m vigilo.evaluate --benign data/iot23/benign5.labeled \
        --malware data/iot23/cap3-1.labeled --ckpt checkpoints/vigilo/vigilo.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from vigilo.zeek import parse_conn_log, group_by_device
from vigilo.features import device_windows, per_asset_normalize, apply_global
from arch.config import ModelConfig
from vigilo.forecaster import PdMForecaster


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = PdMForecaster(ModelConfig(**ck["model_config"]), n_features=ck["n_features"])
    m.load_state_dict(ck["model_state"]); m.eval()
    gmu = np.array(ck["global_mu"], dtype=np.float32) if ck.get("global_mu") else None
    gsd = np.array(ck["global_sd"], dtype=np.float32) if ck.get("global_sd") else None
    return m, ck, gmu, gsd


@torch.no_grad()
def score_log(model, ck, gmu, gsd, log_path):
    """Return (all_window_errors, per_device_peak_scores) for a log."""
    windows, peaks = [], []
    for src, dc in group_by_device(parse_conn_log(log_path)).items():
        w = device_windows(dc, window_s=ck["window_s"])
        if w.shape[0] < ck.get("min_windows", 12):
            continue
        wn = apply_global(w, gmu, gsd) if (ck.get("normalize") == "global" and gmu is not None) \
            else per_asset_normalize(w, baseline=ck.get("baseline", 10))
        x = torch.from_numpy(wn.astype(np.float32)).unsqueeze(0)
        pred = model(x)[0][0].numpy()
        err = ((pred[:-1] - wn[1:]) ** 2).mean(axis=1)
        windows.extend(err.tolist())
        peaks.append(float(np.percentile(err, 95)))   # device decision = peak window
    return np.array(windows), np.array(peaks)


def auc(pos, neg):
    """ROC-AUC via Mann-Whitney rank statistic."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", required=True)
    ap.add_argument("--malware", required=True)
    ap.add_argument("--ckpt", default="checkpoints/vigilo/vigilo.pt")
    args = ap.parse_args()

    model, ck, gmu, gsd = load(args.ckpt)
    bw, bp = score_log(model, ck, gmu, gsd, args.benign)
    mw, mp = score_log(model, ck, gmu, gsd, args.malware)
    if len(bw) == 0 or len(mw) == 0:
        raise SystemExit("Not enough windows to evaluate.")

    print("== window-level (every 5-min window) ==", flush=True)
    print(f"benign windows:  {len(bw):5d}  mean={bw.mean():.3f}", flush=True)
    print(f"malware windows: {len(mw):5d}  mean={mw.mean():.3f}", flush=True)
    print(f"ROC-AUC: {auc(mw, bw):.3f}", flush=True)
    for fpr in (0.01, 0.05):
        tau = np.quantile(bw, 1 - fpr)
        print(f"  @ {fpr:.0%} FPR (τ={tau:.3f}): detection = {(mw > tau).mean():.1%}", flush=True)

    print("\n== device-level (the product's actual decision: peak window) ==", flush=True)
    print(f"benign device peaks:  {np.round(np.sort(bp), 2).tolist()}", flush=True)
    print(f"malware device peaks: {np.round(np.sort(mp), 2).tolist()}", flush=True)
    if len(bp):
        tau = bp.max()
        print(f"threshold = max benign peak = {tau:.2f}  →  "
              f"malware devices flagged: {(mp > tau).sum()}/{len(mp)}", flush=True)


if __name__ == "__main__":
    main()
