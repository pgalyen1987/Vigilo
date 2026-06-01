#!/usr/bin/env bash
set -euo pipefail

VERSION="0.1.0"
IMAGE="vigilo-${VERSION}.tar.gz"

echo "=============================="
echo "  Vigilo ${VERSION} — Setup"
echo "=============================="
echo ""

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed."
    echo "Install Docker Desktop from https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "[1/4] Creating directories..."
mkdir -p data checkpoints/vigilo reports

echo "[2/4] Loading Docker image..."
if [ -f "$IMAGE" ]; then
    docker load -i "$IMAGE"
else
    echo "WARNING: ${IMAGE} not found in current directory. Skipping image load."
    echo "         Place the image file here and re-run, or pull from your registry."
fi

echo "[3/4] Setting up configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from template — edit it to point at your conn.log"
else
    echo "  .env already exists, skipping"
fi

echo "[4/4] Done!"
echo ""
echo "Next steps:"
echo "  1. Place your conn.log in the data/ folder"
echo "  2. Edit .env and set VIGILO_LOG=data/your-file.conn.log"
echo "  3. Run: docker compose up -d"
echo "  4. Open: http://localhost:8088"
echo ""
