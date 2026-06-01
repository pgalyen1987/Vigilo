# NetSentinel — local network anomaly detection

Learns each device's normal network behavior and flags deviations — beaconing,
scanning, exfiltration, new C&C destinations. Runs entirely on-device; **no
traffic data leaves the network.**

Ingests **Zeek `conn.log`** — the same format used by our proof data (IoT-23)
and by real deployments (pfSense/OPNsense/Zeek). One loader, demo to production.

Engine: the `PdMForecaster` state-space model (`pdm/forecaster.py`). Per-device
behavioral feature windows → forecast next window → anomaly = forecast-error
surprise. Trains on benign traffic only; **no malware samples needed.**

## Run

```bash
# Train on benign Zeek logs (cross-device cold-start model)
python -m netsentinel.train --logs benign1.log benign2.log \
    --output-dir checkpoints/netsentinel --normalize global --device cpu

# Score a log; flags devices behaving unlike normal (reports detection vs FPR
# if the log carries IoT-23 labels)
python -m netsentinel.detect --log suspect.conn.log \
    --ckpt checkpoints/netsentinel/netsentinel.pt --target-fpr 0.05
```

## Two detection modes

- `--normalize global` — score a device against the population of normal devices
  (cold start, day one, before a device has its own history).
- `--normalize asset` — score a device against **its own** learned baseline
  (detect when a known-good device changes).

## Proof (IoT-23)

Trained on one benign IoT device (CTU-Honeypot-Capture-4-1), then scored unseen
devices. The Mirai-infected port-scanner (CTU-IoT-Malware-Capture-3-1,
192.168.2.5) scored ~3–4× higher (95p error 31.5) than benign devices it never
saw (5–9), cleanly separable at a threshold near 12.

This is a small-scale proof (one benign training device), not a validated
benchmark — more benign devices tighten the normal baseline and lower the
false-positive rate. It demonstrates the engine end-to-end on real malware
traffic.

## Features (per device, per time window)

Volume (conns/bytes/packets), fan-out (distinct dst IPs/ports), failure rate
(S0/REJ — scanning), destination-IP entropy (low = beaconing), destination-port
entropy (high = scanning), mean duration. See `features.py`.
