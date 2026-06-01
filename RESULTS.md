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
