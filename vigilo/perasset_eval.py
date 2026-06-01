"""
Per-asset baselining evaluation — the deployment-realistic test.

Each device is normalized by ITS OWN early-life baseline and judged against its
own threshold (no global averaging). Two experiments:

  1. FP test  — on purely-benign CICIoT devices: baseline on the first half,
     measure false positives on the second half. (Does per-asset stay quiet on
     a device's own normal?)

  2. Splice test (simulated deployment) — concatenate a benign device's timeline
     (healthy) with an infected device's traffic (compromised), baseline on the
     healthy part, and check we DETECT the compromised part while staying quiet
     on the benign part. Synthetic (sources differ) but directly simulates
     "device was healthy, then got hit" — which neither dataset has natively.

Usage:
    python -m vigilo.perasset_eval --ckpt checkpoints/vigilo_asset/vigilo.pt
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from arch.config import ModelConfig
from vigilo.zeek import parse_conn_log, group_by_device
from vigilo.features import device_windows, per_asset_normalize
from vigilo.forecaster import PdMForecaster


def load(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = PdMForecaster(ModelConfig(**ck["model_config"]), n_features=ck["n_features"])
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, ck


def device_table(log, window_s, min_windows, max_flows=2_000_000):
    size = os.path.getsize(log)
    stride = max(1, int((size / 200) / max_flows))
    conns = parse_conn_log(log, max_lines=max_flows, stride=stride)
    out = {}
    for src, dc in group_by_device(conns).items():
        w = device_windows(dc, window_s=window_s)
        if w.shape[0] >= min_windows:
            out[src] = (w, any(c.label == "Malicious" for c in dc))
    return out


@torch.no_grad()
def win_errors(model, wn):
    x = torch.from_numpy(wn.astype(np.float32)).unsqueeze(0)
    pred = model(x)[0][0].numpy()
    return ((pred[:-1] - wn[1:]) ** 2).mean(axis=1)   # length T-1; err[i] predicts win i+1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vigilo_asset/vigilo.pt")
    ap.add_argument("--cic", default="data/cic/*.conn")
    ap.add_argument("--malware", default="data/iot23/malware-*.labeled")
    ap.add_argument("--n-splice", type=int, default=10)
    args = ap.parse_args()

    model, ck = load(args.ckpt)
    ws, mw = ck["window_s"], ck.get("min_windows", 12)

    # ---- gather benign (CICIoT) and infected (IoT-23) device window-sets ----
    benign = []
    for log in sorted(glob.glob(args.cic)):
        for src, (w, inf) in device_table(log, ws, mw).items():
            if not inf:
                benign.append(w)
    infected = []
    for log in sorted(glob.glob(args.malware)):
        for src, (w, inf) in device_table(log, ws, mw).items():
            if inf:
                infected.append((os.path.basename(log).replace(".labeled", ""), w))
    print(f"[PA] {len(benign)} benign devices, {len(infected)} infected devices", flush=True)

    # ---- 1. FP test (per-asset, benign only) ----
    fps = []
    for w in benign:
        T = w.shape[0]
        if T < 24:
            continue
        B = T // 2
        err = win_errors(model, per_asset_normalize(w, baseline=B))
        base, test = err[:B - 1], err[B - 1:]
        if len(base) < 3 or len(test) < 1:
            continue
        tau = np.quantile(base, 0.99)
        fps.append(float((test > tau).mean()))
    fps = np.array(fps)
    print(f"\n[PA] === FP test (benign devices, baseline=first half) ===", flush=True)
    print(f"[PA] devices tested: {len(fps)}  mean false-positive rate: {fps.mean():.2%}  "
          f"median: {np.median(fps):.2%}", flush=True)

    # ---- 2. Splice test (healthy benign -> compromised) ----
    print(f"\n[PA] === Splice test (benign healthy → infected) ===", flush=True)
    usable_benign = [w for w in benign if w.shape[0] >= 24]
    detected = 0; n = 0; fp_acc = []
    for i, (name, wm) in enumerate(infected[:args.n_splice]):
        if not usable_benign:
            break
        wb = usable_benign[i % len(usable_benign)]
        Tb = wb.shape[0]
        w = np.concatenate([wb, wm], axis=0)
        B = Tb // 2
        err = win_errors(model, per_asset_normalize(w, baseline=B))
        tau = float(np.quantile(err[:B - 1], 0.99))
        benign_test = err[B - 1:Tb - 1]
        mal = err[Tb - 1:]
        det = bool((mal > tau).any())
        detected += det; n += 1
        fp = float((benign_test > tau).mean()) if len(benign_test) else 0.0
        fp_acc.append(fp)
        print(f"  {name:<14} detect={'YES' if det else 'no ':<3} "
              f"mal_windows_flagged={(mal>tau).mean():.0%}  benign_FP={fp:.0%}", flush=True)
    if n:
        print(f"\n[PA] splice detection: {detected}/{n} ({detected/n:.0%})  "
              f"mean benign FP in splice: {np.mean(fp_acc):.1%}", flush=True)


if __name__ == "__main__":
    main()
