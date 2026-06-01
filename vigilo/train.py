"""
Train Vigilo on BENIGN traffic only.

Learns to forecast each device's next behavioral window from its recent history.
Trained purely on normal traffic; malicious behavior is detected at runtime as
high forecast error (see detect.py). No malware samples needed for training.

Usage:
    python -m vigilo.train --logs path/to/benign/conn.log [more.log ...] \
        --output-dir checkpoints/vigilo --epochs 30 --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vigilo.zeek import parse_conn_log, group_by_device
from vigilo.features import (
    device_windows, per_asset_normalize, fit_global, apply_global, N_FEATURES,
)
from vigilo.model_config import ModelConfigPdM
from vigilo.forecaster import PdMForecaster


def device_window_list(log_paths, window_s, min_windows, max_flows=None):
    """Raw (unnormalized) per-device window matrices from benign logs."""
    devs = []
    for lp in log_paths:
        conns = parse_conn_log(lp, max_lines=max_flows)
        for src, dconns in group_by_device(conns).items():
            w = device_windows(dconns, window_s=window_s)
            if w.shape[0] >= min_windows:
                devs.append(w)
    return devs


def slice_seqs(window_list, seq_len):
    seqs = []
    for w in window_list:
        for s in range(0, max(1, w.shape[0] - 1), seq_len):
            chunk = w[s:s + seq_len + 1]
            if chunk.shape[0] >= 4:
                seqs.append(chunk.astype(np.float32))
    return seqs


def collate(batch):
    L = max(w.shape[0] for w in batch)
    x = np.zeros((len(batch), L, N_FEATURES), dtype=np.float32)
    m = np.zeros((len(batch), L), dtype=bool)
    for i, w in enumerate(batch):
        x[i, :w.shape[0]] = w
        m[i, :w.shape[0]] = True
    return torch.from_numpy(x), torch.from_numpy(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="Benign conn.log file(s)")
    ap.add_argument("--output-dir", default="checkpoints/vigilo")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--window-s", type=float, default=300.0)
    ap.add_argument("--baseline", type=int, default=10)
    ap.add_argument("--min-windows", type=int, default=12)
    ap.add_argument("--max-flows", type=int, default=2_000_000,
                    help="cap flows read per capture (memory safety on huge logs)")
    ap.add_argument("--normalize", choices=["global", "asset"], default="global",
                    help="global = cross-device cold-start; asset = per-device baseline")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    devs = device_window_list(args.logs, args.window_s, args.min_windows, args.max_flows)
    if not devs:
        raise SystemExit("No usable benign devices — check logs / thresholds.")

    if args.normalize == "global":
        gmu, gsd = fit_global(devs)
        normed = [apply_global(w, gmu, gsd) for w in devs]
    else:
        gmu = gsd = None
        normed = [per_asset_normalize(w, baseline=args.baseline) for w in devs]
    seqs = slice_seqs(normed, args.seq_len)
    if not seqs:
        raise SystemExit("No usable training sequences.")
    print(f"[vigilo] {len(devs)} benign devices → {len(seqs)} sequences "
          f"({N_FEATURES} features, {args.window_s:.0f}s windows, norm={args.normalize})",
          flush=True)

    cfg = ModelConfigPdM(max_seq_len=args.seq_len + 1)
    model = PdMForecaster(cfg, n_features=N_FEATURES).to(device)
    print(f"[vigilo] forecaster params: {model.num_parameters():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        order = rng.permutation(len(seqs)); model.train(); run = nb = 0.0
        for s in range(0, len(order), args.batch_size):
            batch = [seqs[i] for i in order[s:s + args.batch_size]]
            x, m = collate(batch); x, m = x.to(device), m.to(device)
            pred, aux = model(x)
            tgt, p, mm = x[:, 1:], pred[:, :-1], m[:, 1:].unsqueeze(-1)
            mse = (((p - tgt) ** 2) * mm).sum() / mm.sum().clamp(min=1) / N_FEATURES
            loss = mse + 0.01 * aux
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run += mse.item(); nb += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch={epoch+1}/{args.epochs} forecast_mse={run/max(nb,1):.4f}", flush=True)

    torch.save({"model_state": model.state_dict(), "model_config": cfg.__dict__,
                "n_features": N_FEATURES, "window_s": args.window_s,
                "baseline": args.baseline, "min_windows": args.min_windows,
                "normalize": args.normalize,
                "global_mu": None if gmu is None else gmu.tolist(),
                "global_sd": None if gsd is None else gsd.tolist()},
               out / "vigilo.pt")
    (out / "meta.json").write_text(json.dumps({"n_features": N_FEATURES,
                                               "window_s": args.window_s}))
    print(f"[vigilo] saved {out/'vigilo.pt'}", flush=True)


if __name__ == "__main__":
    main()
