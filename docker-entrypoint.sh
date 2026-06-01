#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-serve}"
shift 2>/dev/null || true

case "$CMD" in
    serve)
        echo "[vigilo] starting dashboard on ${VIGILO_HOST:-0.0.0.0}:${VIGILO_PORT:-8088}"
        exec gunicorn vigilo.dashboard:app \
            --bind "${VIGILO_HOST:-0.0.0.0}:${VIGILO_PORT:-8088}" \
            --workers "${VIGILO_WORKERS:-2}" \
            --timeout "${VIGILO_TIMEOUT:-120}" \
            --access-logfile - \
            "$@"
        ;;
    train)
        echo "[vigilo] training forecaster"
        exec python -m vigilo.train "$@"
        ;;
    detect)
        echo "[vigilo] running detection"
        exec python -m vigilo.detect "$@"
        ;;
    report)
        echo "[vigilo] generating report"
        exec python -m vigilo.report "$@"
        ;;
    beacon)
        echo "[vigilo] running beaconing detector"
        exec python -m vigilo.beaconing "$@"
        ;;
    ingest)
        echo "[vigilo] converting pcap to conn.log"
        exec python -m vigilo.pcap_to_conn "$@"
        ;;
    shell)
        exec /bin/bash "$@"
        ;;
    *)
        exec "$CMD" "$@"
        ;;
esac
