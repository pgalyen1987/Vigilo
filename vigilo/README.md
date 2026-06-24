# Vigilo — core package

On-device network anomaly detection. Learns each device's normal network behavior
and flags deviations — beaconing, scanning, flooding, C2. Runs entirely on-device;
**no traffic data leaves the network.**

Ingests **Zeek `conn.log`** — the same format used by pfSense/OPNsense/Zeek
deployments and our IoT-23 benchmarks.

## Modules

| Module | Purpose |
|--------|---------|
| `cli.py` | Unified `vigilo` command entry point |
| `train.py` | Train forecaster on benign conn.logs |
| `detect.py` | Score devices for anomalies |
| `ensemble.py` | Volume model + beaconing combined verdicts |
| `dashboard.py` | Flask web UI + JSON API |
| `report.py` | Static HTML report generator |
| `beaconing.py` | Periodic C2 detector |
| `features.py` | 15-feature behavioral windows |
| `forecaster.py` | Mamba-2 forecaster model |
| `zeek.py` | conn.log parser |
| `pcap_to_conn.py` | pcap → conn.log via tshark |

## Run

```bash
# Demo (bundled synthetic data)
vigilo serve

# Train on your benign Zeek logs
vigilo train --logs data/home/home.conn.log --output-dir checkpoints/vigilo

# Score a log
vigilo detect --log suspect.conn.log --ckpt checkpoints/vigilo/vigilo.pt
```

See the [root README](../README.md) and [docs/DEPLOY-SPAN-GATEWAY.md](../docs/DEPLOY-SPAN-GATEWAY.md).
