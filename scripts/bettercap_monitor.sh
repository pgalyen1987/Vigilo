#!/usr/bin/env bash
# Vigilo live MITM tap  (YOUR OWN network only).
#
# bettercap ARP-spoofs the LAN so every device's traffic routes THROUGH this PC.
# Because we become the relay, we see plaintext L3 traffic -- no WPA handshake or
# decryption needed (this is why it works where WiFi monitor mode failed). We
# capture it in rotating time windows and feed each window to Zeek -> Vigilo,
# producing one anomaly report per window.
#
#   sudo bash scripts/bettercap_monitor.sh [window_seconds]     (default 300)
#
# Stop with Ctrl-C: the trap restores every device's ARP table and stops capture.
#
# ⚠ MITM is disruptive (your PC becomes a bottleneck for ALL LAN traffic and can
#   break flaky IoT devices) and intercepts EVERY device on the LAN, including
#   other people's. Only run on a network you own, for devices you're entitled to
#   monitor. A switch mirror/SPAN port or running this on the gateway is a
#   strictly better, non-disruptive, passive alternative if you have the option.

set -euo pipefail

# Config comes from .env (copy .env.example -> .env). No secret needed: MITM relays
# plaintext L3, so there's no WPA key to decrypt with.
VIGILO="/home/me/SAAS/Vigilo"
[ -f "$VIGILO/.env" ] && { set -a; . "$VIGILO/.env"; set +a; }
IFACE="${CAPTURE_IFACE:-wlan0}"            # your LAN interface
SUBNET="${CAPTURE_SUBNET:-192.168.1.0/24}" # whole LAN; narrow to one IP to scope it
WINDOW="${1:-300}"                         # seconds per analysis window
SPOOL="$VIGILO/data/home/live_$(date +%Y%m%d_%H%M%S)"
# run docker/output as the invoking user, not root, even under sudo
U="${SUDO_UID:-$(id -u)}"; G="${SUDO_GID:-$(id -g)}"
mkdir -p "$SPOOL/pcap" "$SPOOL/reports"

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run with sudo (bettercap + tcpdump need root)."; exit 1; }

cleanup() {
    echo ""
    echo "[*] stopping: restoring ARP tables and ending capture..."
    [ -n "${TCPDUMP_PID:-}" ] && kill "$TCPDUMP_PID" 2>/dev/null || true
    # SIGINT lets bettercap re-ARP every victim back to the real gateway
    [ -n "${BC_PID:-}" ] && kill -INT "$BC_PID" 2>/dev/null || true
    sleep 3
    [ -n "${BC_PID:-}" ] && kill "$BC_PID" 2>/dev/null || true
    chown -R "$U:$G" "$SPOOL" 2>/dev/null || true
    echo "[*] done. reports under: $SPOOL/reports"
}
trap cleanup EXIT INT TERM

echo "[1/3] starting bettercap MITM on $IFACE (targets $SUBNET)..."
# net.probe discovers hosts; arp.spoof fullduplex poisons both victims AND the
# gateway so we see traffic in both directions.
bettercap -iface "$IFACE" -no-colors -eval \
    "set arp.spoof.fullduplex true; set arp.spoof.targets $SUBNET; net.probe on; arp.spoof on" \
    >"$SPOOL/bettercap.log" 2>&1 &
BC_PID=$!
sleep 10   # give it time to ARP the LAN before we start counting on forwarded traffic

echo "[2/3] rotating capture every ${WINDOW}s -> $SPOOL/pcap/ ..."
# capture forwarded IP traffic only (skip ARP/our own spoof chatter)
tcpdump -i "$IFACE" -n -G "$WINDOW" -w "$SPOOL/pcap/win-%Y%m%d_%H%M%S.pcap" \
    'ip or ip6' >/dev/null 2>&1 &
TCPDUMP_PID=$!

echo "[3/3] processing loop (Ctrl-C to stop)..."
shopt -s nullglob
declare -A processed
while true; do
    newest="$(ls -t "$SPOOL"/pcap/win-*.pcap 2>/dev/null | head -1 || true)"
    for pcap in "$SPOOL"/pcap/win-*.pcap; do
        [ -n "${processed[$pcap]:-}" ] && continue
        [ "$pcap" = "$newest" ] && continue          # newest file is still being written
        base="$(basename "$pcap" .pcap)"
        # pcap -> Zeek conn.log
        if ! bash "$VIGILO/scripts/pcap_to_conn_zeek.sh" "$pcap" "$SPOOL/$base.conn" 2>/dev/null; then
            processed[$pcap]=1; continue
        fi
        # Zeek conn.log -> Vigilo anomaly report
        if docker run --rm --user "$U:$G" \
                -v "$VIGILO/checkpoints:/app/checkpoints" -v "$SPOOL:/work" \
                vigilo:latest report --log "/work/$base.conn" --out "/work/reports/$base.html" \
                >/dev/null 2>&1; then
            echo "    $base -> reports/$base.html"
        else
            echo "    $base (too little data for a report)"
        fi
        processed[$pcap]=1
    done
    sleep 5
done
