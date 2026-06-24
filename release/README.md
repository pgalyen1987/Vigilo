# Vigilo — Setup Guide

On-device network anomaly detection. Learns normal behavior, flags deviations.

## Prerequisites

- **Docker Desktop** — [download here](https://www.docker.com/products/docker-desktop)
  - Windows: requires WSL 2 (Docker Desktop installer handles this)
  - macOS / Linux: install and start Docker Desktop or Docker Engine

## Quick setup (demo dashboard)

The release bundle includes synthetic demo data and a pretrained checkpoint.

### Linux / macOS

```bash
chmod +x setup.sh
./setup.sh
docker compose up -d
# http://localhost:8088
```

### Windows (PowerShell)

```powershell
.\setup.ps1
docker compose up -d
```

## Manual setup

```bash
# 1. Load the image
docker load -i vigilo-0.1.0.tar.gz

# 2. Create folders (demo data is pre-bundled in data/demo/)
mkdir -p data/zeek checkpoints/vigilo reports

# 3. Configure
cp .env.example .env

# 4. Start
docker compose up -d

# 5. Open http://localhost:8088
```

## Passive SPAN / gateway deployment

For production, mirror LAN traffic to a capture host and run Zeek alongside Vigilo:

```bash
cp .env.span.example .env
# Edit CAPTURE_IFACE to your mirror NIC (e.g. eth1)
docker compose --profile span up -d
```

See `docs/DEPLOY-SPAN-GATEWAY.md` for switch SPAN setup, pfSense/OPNsense, and troubleshooting.

## Preparing your data

Vigilo analyzes Zeek-format `conn.log` files. Place them in the `data/` folder.

**If you have a pcap file**, convert it inside the container:

```bash
docker compose exec vigilo vigilo ingest /app/data/capture.pcap /app/data/capture.conn.log
```

**If you have Zeek running**, point your conn.log output to the `data/` folder.

## Training a model

The demo checkpoint works for verification. For real alerts, train on YOUR network's benign traffic:

```bash
docker compose run --rm --profile tools train \
  --logs /app/data/your-benign.conn.log \
  --output-dir /app/checkpoints/vigilo
```

Training takes 1–5 minutes on CPU depending on log size.

Update `.env`:

```ini
VIGILO_CKPT=checkpoints/vigilo/vigilo.pt
VIGILO_LOG=data/zeek/conn.log
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `VIGILO_LOG` | `data/demo/sample.conn.log` | Path to conn.log to analyze |
| `VIGILO_CKPT` | `checkpoints/demo/vigilo.pt` | Trained model checkpoint |
| `VIGILO_PORT` | `8088` | Dashboard port |
| `CAPTURE_IFACE` | `eth1` | Mirror NIC (span profile) |

## Stopping / restarting

```bash
docker compose down
docker compose up -d
docker compose logs -f
```

## Troubleshooting

**Dashboard shows no devices** — Check `VIGILO_LOG` points to a valid conn.log in `data/`.

**No checkpoint found** — Use the bundled demo checkpoint or train first.

**SPAN capture empty** — See `docs/DEPLOY-SPAN-GATEWAY.md` troubleshooting section.

**Port conflict** — Change `VIGILO_PORT` in `.env`.
