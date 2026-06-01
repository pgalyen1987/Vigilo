# Vigilo — Setup Guide

On-device network anomaly detection. Learns normal behavior, flags deviations.

## Prerequisites

- **Docker Desktop** — [download here](https://www.docker.com/products/docker-desktop)
  - Windows: requires WSL 2 (Docker Desktop installer handles this)
  - macOS / Linux: install and start Docker Desktop or Docker Engine

## Quick setup

### Linux / macOS

```bash
chmod +x setup.sh
./setup.sh
```

### Windows (PowerShell)

```powershell
.\setup.ps1
```

## Manual setup

If you prefer to do it step by step:

```bash
# 1. Load the image
docker load -i vigilo-0.1.0.tar.gz

# 2. Create folders
mkdir -p data checkpoints/vigilo reports

# 3. Configure
cp .env.example .env
# Edit .env — set VIGILO_LOG to your conn.log path

# 4. Start
docker compose up -d

# 5. Open
# http://localhost:8088
```

## Preparing your data

Vigilo analyzes Zeek-format `conn.log` files. Place them in the `data/` folder.

**If you have a pcap file**, convert it inside the container:

```bash
docker compose exec vigilo vigilo ingest /app/data/capture.pcap /app/data/capture.conn.log
```

**If you have Zeek running**, point your conn.log output to the `data/` folder.

## Training a model

Vigilo ships with no pretrained model — you train it on YOUR network's benign traffic.
This teaches it what "normal" looks like for your specific environment.

```bash
docker compose run --rm vigilo train --logs /app/data/your-benign.conn.log --output-dir /app/checkpoints/vigilo
```

Training takes 1–5 minutes on CPU depending on log size.

## Configuration

Edit `.env` to adjust settings:

| Variable | Default | Description |
|---|---|---|
| `VIGILO_LOG` | *(required)* | Path to conn.log to analyze |
| `VIGILO_CKPT` | `checkpoints/vigilo/vigilo.pt` | Trained model checkpoint |
| `VIGILO_PORT` | `8088` | Dashboard port |
| `VIGILO_WORKERS` | `2` | Server worker count |

## Stopping / restarting

```bash
docker compose down      # stop
docker compose up -d     # start
docker compose restart   # restart
docker compose logs -f   # view logs
```

## Troubleshooting

**"Cannot connect to the Docker daemon"**
Start Docker Desktop. On Linux, run `sudo systemctl start docker`.

**Dashboard shows no devices**
Check that `VIGILO_LOG` in `.env` points to a valid conn.log file inside `data/`.

**"No checkpoint found"**
You need to train first — see "Training a model" above.

**Port conflict**
Change `VIGILO_PORT` in `.env` to an available port.
