"""
Score devices in a conn.log for anomalous behavior.

Per device: forecast each window from history, take the forecast error as the
anomaly score. A device is flagged if its peak score exceeds a threshold
calibrated on benign traffic at a target false-positive rate.

If the log carries IoT-23 labels, we also report detection rate vs. FPR so the
engine can be evaluated honestly.

Usage:
    python -m vigilo.detect --log path/to/conn.log.labeled \
        --ckpt checkpoints/vigilo/vigilo.pt --target-fpr 0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vigilo.zeek import parse_conn_log, group_by_device
from vigilo.features import (
    device_windows, per_asset_normalize, apply_global, N_FEATURES,
)
from arch.config import ModelConfig
from vigilo.forecaster import PdMForecaster


@torch.no_grad()
def device_scores(model, windows: np.ndarray, device) -> np.ndarray:
    """Per-window forecast error (T,) for one normalized device sequence."""
    if windows.shape[0] < 2:
        return np.zeros(windows.shape[0])
    x = torch.from_numpy(windows.astype(np.float32)).unsqueeze(0).to(device)
    pred, _ = model(x)
    pred = pred[0].cpu().numpy()
    err = np.zeros(windows.shape[0])
    err[1:] = ((pred[:-1] - windows[1:]) ** 2).mean(axis=1)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt", default="checkpoints/vigilo/vigilo.pt")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PdMForecaster(ModelConfig(**ckpt["model_config"]), n_features=ckpt["n_features"])
    model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
    window_s = ckpt.get("window_s", 300.0)
    baseline = ckpt.get("baseline", 10)
    min_windows = ckpt.get("min_windows", 12)
    normalize = ckpt.get("normalize", "asset")
    gmu = np.array(ckpt["global_mu"], dtype=np.float32) if ckpt.get("global_mu") else None
    gsd = np.array(ckpt["global_sd"], dtype=np.float32) if ckpt.get("global_sd") else None

    def normalize_windows(w):
        if normalize == "global" and gmu is not None:
            return apply_global(w, gmu, gsd)
        return per_asset_normalize(w, baseline=baseline)

    conns = parse_conn_log(args.log)
    devices = group_by_device(conns)

    rows = []   # (src, peak_score, label_is_malicious)
    for src, dconns in devices.items():
        w = device_windows(dconns, window_s=window_s)
        if w.shape[0] < min_windows:
            continue
        wn = normalize_windows(w)
        scores = device_scores(model, wn, device)
        peak = float(np.percentile(scores, 95)) if len(scores) else 0.0
        mal = sum(c.label == "Malicious" for c in dconns)
        rows.append((src, peak, mal > 0, mal, len(dconns)))

    if not rows:
        raise SystemExit("No devices with enough windows to score.")

    peaks = np.array([r[1] for r in rows])
    benign = np.array([r[1] for r in rows if not r[2]])
    has_labels = any(r[2] for r in rows) or any(r[3] == 0 for r in rows)

    # Calibrate threshold on benign devices (or all devices if unlabeled).
    ref = benign if len(benign) else peaks
    tau = float(np.quantile(ref, 1.0 - args.target_fpr)) if len(ref) else 0.0

    print(f"[vigilo] {len(rows)} devices scored  (window={window_s:.0f}s)", flush=True)
    print(f"[vigilo] threshold τ={tau:.3f} @ target FPR {args.target_fpr:.0%}\n", flush=True)
    print(f"{'device':<18}{'score':>10}  {'flag':>6}  label", flush=True)
    for src, peak, mal, ncm, ncn in sorted(rows, key=lambda r: -r[1]):
        flag = "ALERT" if peak > tau else ""
        lab = "malicious" if mal else "benign"
        print(f"{src:<18}{peak:>10.3f}  {flag:>6}  {lab}", flush=True)

    if any(r[2] for r in rows):     # we have malicious labels → evaluate
        mal_rows = [r for r in rows if r[2]]
        ben_rows = [r for r in rows if not r[2]]
        det = sum(r[1] > tau for r in mal_rows)
        fp = sum(r[1] > tau for r in ben_rows)
        print(f"\n[EVAL] malicious devices detected: {det}/{len(mal_rows)} "
              f"({det/max(len(mal_rows),1):.0%})", flush=True)
        print(f"[EVAL] benign devices false-flagged: {fp}/{max(len(ben_rows),1)} "
              f"({fp/max(len(ben_rows),1):.0%})", flush=True)


if __name__ == "__main__":
    main()
