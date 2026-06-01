# Vigilo — Benchmark Results

Local network anomaly detection. Trained on **benign traffic only** (no malware
samples), runs on **CPU**, ~1.3M parameters. Detection is device-level (a device
is flagged on its peak behavioral-forecast-error window); the threshold is
calibrated on benign devices at a target false-positive rate.

## IoT-23 (20 malware families, 8 benign captures)

Trained on benign honeypot captures (29 benign devices). Evaluated on every
malware family in IoT-23 (Mirai, Okiru, Torii, Gagfyt, Hajime, Kenjiro, etc.).

| Metric | Value |
|---|---|
| Detection @ ~1% false-positive rate | **75% (15/20 infected devices)** |
| Benign baseline (device peak) | p95 = 1.93, max = 1.97 — tight |
| Training data | benign traffic only, 29 devices |
| Model | ~1.3M params, CPU, fully local |

**What it catches well:** all loud attacks — port scans, DDoS/flooding, noisy
botnet C&C. These score far above the benign ceiling (e.g. 1444, 65, 5.3).

**What it misses (the 5):** stealthy, low-volume C&C — e.g. one capture has only
16 malicious flows among 3,193 benign. These sit right at the benign ceiling
and need per-device baselining or flow-level inspection (roadmap).

## Honesty notes

- IoT-23 is **lab data**, not real home/field traffic. Field validation pending.
- An earlier benchmark looked artificially perfect due to a label-parsing bug
  (infected devices silently counted as benign); fixing it produced the honest
  numbers above. The harness now also scores only local (RFC1918) devices and
  samples large captures across their full timespan.
- Real-traffic plumbing is proven: live `tshark` capture (no sudo) →
  `pcap_to_conn` → engine, verified on a real wlan0 capture.

## Reproduce

```bash
python -m vigilo.train     --logs data/iot23/benign-*.labeled --normalize global --window-s 300
python -m vigilo.benchmark --benign "data/iot23/benign-*.labeled" --malware "data/iot23/malware-*.labeled"
```

## Follow-up findings (data experiments)

- **Run-to-run variance is real.** Same IoT-only setup gave 75% @1% FPR one run,
  40% @1% / 75% @5% / 95% @10% another. Cause: ~1.3M-param model + threshold
  calibrated on only ~14 benign devices = noisy threshold. Stable underneath:
  loud attacks always caught, stealthy always missed.
- **General-host (PC) benign traffic HURTS.** Adding CTU-Normal (PC/laptop
  captures) to "normal" dropped detection 75% -> 20%: PCs are noisy/varied, so
  the model learned scanning-like behavior as normal. Train "normal" on
  IoT-like traffic only. (CTU-Normal removed.)
- **Device-rich IoT datasets are inaccessible/incompatible:** N-BaIoT and
  CICIoT2023-CSV are pre-aggregated classification features (no per-device flow
  sequences); CICIoT2023 pcaps are huge + non-scriptable; UNSW is gated. Data
  quantity is not the bottleneck — stealthy-attack sensitivity and threshold
  calibration are.
- **Implication:** per-asset baselining (per-device normal, learned in place) is
  the path to "works with any device" and to a stable per-device threshold —
  not more global training data.
