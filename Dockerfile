# ── Stage 1: build & install ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY pyproject.toml .
COPY arch/ arch/
COPY vigilo/ vigilo/
RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: lean production image ──────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tshark curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false vigilo

COPY --from=builder /install /usr/local

WORKDIR /app

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /app/data /app/checkpoints/vigilo /app/reports \
    && chown -R vigilo:vigilo /app

ENV VIGILO_HOST=0.0.0.0
ENV VIGILO_PORT=8088
ENV VIGILO_LOG=""
ENV VIGILO_CKPT=checkpoints/vigilo/vigilo.pt
ENV VIGILO_WORKERS=2
ENV VIGILO_TIMEOUT=120

EXPOSE 8088

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8088/healthz || exit 1

USER vigilo

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["serve"]
