# Vigilo

On-device network anomaly detection. Learns normal per-device behavior from benign traffic and flags deviations — beaconing, scanning, flooding, C2 — without sending any data off the network.

## How it works

1. **Ingest** Zeek `conn.log` (or convert pcap via tshark)
2. **Featurize** per-device 5-minute behavioral windows (15 features)
3. **Forecast** the next window with a ~1.3M-param Mamba-2 state-space model
4. **Score** anomaly as forecast error — high surprise = abnormal behavior
5. **Ensemble** with a beaconing detector for stealthy periodic C2

## Quick start

### Local install

```bash
pip install -e .
```

### Train on benign traffic

```bash
vigilo train --logs data/home/home.conn.log --output-dir checkpoints/vigilo
```

### Detect anomalies

```bash
vigilo detect --log suspect.conn.log --ckpt checkpoints/vigilo/vigilo.pt
```

### Launch the dashboard

```bash
VIGILO_LOG=data/home/home.conn.log vigilo serve
# open http://127.0.0.1:8088
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
cp .env.example .env      # edit to point at your log + checkpoint
make build                 # build the image
make up                    # start the dashboard
make logs                  # tail output
```

Or directly:

```bash
docker compose build
docker compose up -d vigilo
```

### Train inside Docker

```bash
docker compose run --rm train --logs data/home/home.conn.log --output-dir checkpoints/vigilo
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `VIGILO_LOG` | *(none)* | Path to the conn.log to analyze |
| `VIGILO_CKPT` | `checkpoints/vigilo/vigilo.pt` | Trained model checkpoint |
| `VIGILO_HOST` | `0.0.0.0` | Dashboard bind address |
| `VIGILO_PORT` | `8088` | Dashboard port |
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
vigilo/
  cli.py          Unified CLI entry point
  train.py        Train on benign conn.logs
  detect.py       Score devices for anomalies
  ensemble.py     Volume + beaconing combined verdicts
  dashboard.py    Flask web UI + API
  report.py       Static HTML report generator
  beaconing.py    Periodic C2 detector
  features.py     15-feature behavioral windows
  forecaster.py   Mamba-2 forecaster model
  zeek.py         conn.log parser
  pcap_to_conn.py pcap converter (tshark)

arch/             Shared Mamba-2 / MoE architecture
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
