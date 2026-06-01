"""
Cross-capture benchmark: the real reliability number.

Scores every device in a set of benign and malware captures, then reports
device-level detection (the product's actual decision) across many infected
devices and malware families — not n=1.

Ground truth: a device is "infected" if it originates any Malicious-labeled
flow; "benign" otherwise. We calibrate the alert threshold on benign device
peak-scores at a target FPR, then measure detection over infected devices.

Usage:
    python -m vigilo.benchmark --ckpt checkpoints/vigilo/vigilo.pt \
        --benign "data/iot23/benign-*.labeled" \
        --malware "data/iot23/malware-*.labeled"
"""
from __future__ import annotations

import argparse
import glob

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
def device_peaks(model, ck, gmu, gsd, log_path, max_flows=2_000_000):
    """Yield (src, peak_score, infected_bool, n_flows) per scorable device."""
    # For huge captures, sample evenly across the file (stride) so the kept rows
    # span the full time range — otherwise the time-compressed front yields too
    # few windows and the (loud) infected device gets skipped.
    import os
    size = os.path.getsize(log_path)
    est_lines = size / 200          # ~200 bytes/line in IoT-23 conn logs
    stride = max(1, int(est_lines / max_flows))
    out = []
    for src, dc in group_by_device(
            parse_conn_log(log_path, max_lines=max_flows, stride=stride)).items():
        w = device_windows(dc, window_s=ck["window_s"])
        if w.shape[0] < ck.get("min_windows", 12):
            continue
        wn = apply_global(w, gmu, gsd) if (ck.get("normalize") == "global" and gmu is not None) \
            else per_asset_normalize(w, baseline=ck.get("baseline", 10))
        x = torch.from_numpy(wn.astype(np.float32)).unsqueeze(0)
        pred = model(x)[0][0].numpy()
        err = ((pred[:-1] - wn[1:]) ** 2).mean(axis=1)
        peak = float(np.percentile(err, 95)) if len(err) else 0.0
        infected = any(c.label == "Malicious" for c in dc)
        out.append((src, peak, infected, len(dc)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vigilo/vigilo.pt")
    ap.add_argument("--benign", required=True, help="glob for benign logs")
    ap.add_argument("--malware", required=True, help="glob for malware logs")
    args = ap.parse_args()

    model, ck, gmu, gsd = load(args.ckpt)

    benign_logs = sorted(glob.glob(args.benign))
    malware_logs = sorted(glob.glob(args.malware))
    print(f"[BENCH] {len(benign_logs)} benign + {len(malware_logs)} malware captures", flush=True)

    # Benign-device peaks (from benign captures) → calibrate threshold + FPR.
    benign_peaks = []
    for lp in benign_logs:
        for src, peak, inf, n in device_peaks(model, ck, gmu, gsd, lp):
            if not inf:
                benign_peaks.append(peak)
    benign_peaks = np.array(benign_peaks)

    # Infected-device peaks (from malware captures) → detection.
    infected, fp_in_malware = [], []
    per_capture = []
    for lp in malware_logs:
        rows = device_peaks(model, ck, gmu, gsd, lp)
        inf_rows = [r for r in rows if r[2]]
        for src, peak, _, n in inf_rows:
            infected.append(peak)
        for src, peak, isinf, n in rows:
            if not isinf:
                fp_in_malware.append(peak)
        name = lp.split("/")[-1].replace(".labeled", "")
        per_capture.append((name, max((r[1] for r in inf_rows), default=0.0), len(inf_rows)))
    infected = np.array(infected)

    all_benign = np.concatenate([benign_peaks, np.array(fp_in_malware)]) if len(fp_in_malware) \
        else benign_peaks
    if len(all_benign) == 0 or len(infected) == 0:
        raise SystemExit("Not enough labeled devices to benchmark.")

    print(f"\n[BENCH] benign devices: {len(all_benign)}  infected devices: {len(infected)}",
          flush=True)
    print(f"[BENCH] infected peak: median={np.median(infected):.2f} "
          f"min={infected.min():.2f} max={infected.max():.2f}", flush=True)
    print(f"[BENCH] benign  peak: median={np.median(all_benign):.2f} "
          f"p95={np.percentile(all_benign,95):.2f} max={all_benign.max():.2f}", flush=True)

    print("\n[BENCH] detection at calibrated FPR:", flush=True)
    for fpr in (0.01, 0.05, 0.10):
        tau = np.quantile(all_benign, 1 - fpr)
        det = (infected > tau).mean()
        print(f"  FPR≈{fpr:.0%}  τ={tau:6.2f}  →  detection {det:.1%} "
              f"({int((infected>tau).sum())}/{len(infected)})", flush=True)

    print("\n[BENCH] per-capture (infected-device peak):", flush=True)
    for name, peak, ndev in sorted(per_capture, key=lambda r: -r[1]):
        print(f"  {name:<14} peak={peak:7.2f}  infected_devices={ndev}", flush=True)


if __name__ == "__main__":
    main()
