#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-0.1.0}"
IMAGE_NAME="vigilo"
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"
RELEASE_DIR="release"

echo "==> Building ${IMAGE_TAG}"
docker build -t "${IMAGE_TAG}" -t "${IMAGE_NAME}:latest" .

echo "==> Exporting image to ${RELEASE_DIR}/"
mkdir -p "${RELEASE_DIR}"
docker save "${IMAGE_TAG}" | gzip > "${RELEASE_DIR}/${IMAGE_NAME}-${VERSION}.tar.gz"

echo "==> Packaging customer files"
cp release/docker-compose.yml "${RELEASE_DIR}/docker-compose.yml" 2>/dev/null || true
cp release/README.md "${RELEASE_DIR}/README.md" 2>/dev/null || true
cp .env.example "${RELEASE_DIR}/.env.example"

IMAGE_SIZE=$(du -h "${RELEASE_DIR}/${IMAGE_NAME}-${VERSION}.tar.gz" | cut -f1)
echo ""
echo "==> Release built:"
echo "    Image:   ${RELEASE_DIR}/${IMAGE_NAME}-${VERSION}.tar.gz (${IMAGE_SIZE})"
echo "    Compose: ${RELEASE_DIR}/docker-compose.yml"
echo "    Docs:    ${RELEASE_DIR}/README.md"
echo ""
echo "Ship the release/ folder to customers."
echo "They run: docker load < vigilo-${VERSION}.tar.gz && docker compose up -d"
