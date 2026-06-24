# Vigilo

On-device network anomaly detection. Learns normal per-device behavior from benign traffic and flags deviations — beaconing, scanning, flooding, C2 — without sending any data off the network.

## How it works

1. **Ingest** Zeek `conn.log` (or convert pcap via tshark)
2. **Featurize** per-device 5-minute behavioral windows (15 features)
3. **Forecast** the next window with a ~1.3M-param Mamba-2 state-space model
4. **Score** anomaly as forecast error — high surprise = abnormal behavior
5. **Ensemble** with a beaconing detector for stealthy periodic C2

## Quick start (demo — works out of the box)

Bundled synthetic demo data and a pretrained checkpoint ship with the repo:

```bash
pip install -e .
python scripts/generate_demo_data.py   # skip if data/demo/ already present
vigilo serve
# open http://127.0.0.1:8088 — expect 1 ALERT (beaconing device 192.168.1.99)
```

Or with Docker:

```bash
cp .env.example .env
make build && make up
# open http://localhost:8088
```

Run tests:

```bash
pip install -e ".[dev]"
make test
```

## Production deployment

For real networks, use **passive capture** (switch SPAN or gateway Zeek) — not MITM:

```bash
cp .env.span.example .env
docker compose --profile span up -d
```

Full guide: [docs/DEPLOY-SPAN-GATEWAY.md](docs/DEPLOY-SPAN-GATEWAY.md)

Bettercap MITM scripts exist for lab use only (`scripts/bettercap_monitor.sh`).

## Train on your traffic

```bash
vigilo train --logs data/home/home.conn.log --output-dir checkpoints/vigilo
vigilo detect --log suspect.conn.log --ckpt checkpoints/vigilo/vigilo.pt
```

### Launch the dashboard

```bash
VIGILO_LOG=data/zeek/conn.log VIGILO_CKPT=checkpoints/vigilo/vigilo.pt vigilo serve
```

### Generate a static report

```bash
vigilo report --log data/home/home.conn.log --out reports/home.html
```

### Convert pcap to conn.log

```bash
vigilo ingest capture.pcap data/home/capture.conn.log
```

## Docker deployment

```bash
cp .env.example .env
make build
make up
make logs
```

### Train inside Docker

```bash
docker compose run --rm --profile tools train \
  --logs data/home/home.conn.log --output-dir checkpoints/vigilo
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `VIGILO_LOG` | `data/demo/sample.conn.log` | Path to the conn.log to analyze |
| `VIGILO_CKPT` | `checkpoints/demo/vigilo.pt` | Trained model checkpoint |
| `VIGILO_HOST` | `0.0.0.0` | Dashboard bind address |
| `VIGILO_PORT` | `8088` | Dashboard port |
| `CAPTURE_IFACE` | `eth1` | SPAN mirror NIC (span profile only) |
| `VIGILO_WORKERS` | `2` | Gunicorn worker count |
| `VIGILO_TIMEOUT` | `120` | Gunicorn request timeout (seconds) |

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard UI |
| `/healthz` | GET | Liveness probe (JSON) |
| `/api/results` | GET | Detection results as JSON |

## Project structure

```
vigilo/           Core Python package (CLI, train, detect, dashboard)
arch/             Mamba-2 / MoE model architecture
scripts/          Capture helpers (bettercap, demo generator)
docs/             Deployment guides
release/          Customer distribution bundle
tests/            pytest suite
```

## CLI reference

```
vigilo train    Train the forecaster on benign traffic
vigilo detect   Score devices in a conn.log for anomalies
vigilo serve    Launch the live web dashboard
vigilo report   Render a static HTML report
vigilo beacon   Run standalone beaconing detector
vigilo ingest   Convert a pcap to conn.log via tshark
vigilo version  Print version and exit
```

Run `vigilo <command> --help` for command-specific options.
