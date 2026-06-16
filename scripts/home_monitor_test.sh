#!/usr/bin/env bash
# Vigilo home-network test via WiFi monitor mode.
# Captures 20 min of all WiFi traffic, decrypts it, runs Vigilo, restores network.
#
# ⚠ This PC goes OFFLINE during the 20-min capture. The trap below ALWAYS
#   restores your network when the script ends (even on error/Ctrl-C).
#
#   1. Edit the four values below.
#   2. Run:  bash scripts/home_monitor_test.sh   (it will sudo-prompt once)

set -eu   # POSIX-portable (no pipefail bashism, no arrays — runs under sh or bash)

# Config comes from .env (copy .env.example -> .env and fill in YOUR network).
VIGILO="/home/me/SAAS/Vigilo"
[ -f "$VIGILO/.env" ] && { set -a; . "$VIGILO/.env"; set +a; }
SSID="${WIFI_SSID:?set WIFI_SSID in .env (copy .env.example -> .env)}"
WIFI_PASS="${WIFI_PASS:?set WIFI_PASS in .env}"
CHANNEL="${WIFI_CHANNEL:-1}"   # your WiFi channel (router admin; 2.4GHz often 1/6/11)
DURATION="${1:-1200}"          # capture seconds; pass as arg (e.g. 28800 = 8h overnight)
PCAP="$VIGILO/data/home/monitor_capture.pcap"
DEC="$VIGILO/data/home/monitor_dec.pcap"
CONN="$VIGILO/data/home/monitor.conn"
REPORT="reports/monitor_report.html"

restore_network() {
    echo "[*] restoring network..."
    local mon
    mon="$(iw dev 2>/dev/null | awk '/Interface/{i=$2} /type monitor/{print i}' | head -1)"
    [ -n "${mon:-}" ] && sudo airmon-ng stop "$mon" >/dev/null 2>&1 || true
    sudo systemctl restart NetworkManager >/dev/null 2>&1 || true
    echo "[*] network restored."
}
trap restore_network EXIT   # always runs, even on error/Ctrl-C

echo "[1/6] entering monitor mode — THIS PC GOES OFFLINE for ~$((DURATION/60)) min..."
sudo airmon-ng check kill
sudo airmon-ng start wlan0

MON="$(iw dev | awk '/Interface/{i=$2} /type monitor/{print i}' | head -1)"
[ -z "$MON" ] && { echo "ERROR: no monitor interface came up"; exit 1; }
sudo iw dev "$MON" set channel "$CHANNEL"
echo "    monitor iface=$MON  channel=$CHANNEL"

echo "[2/6] capturing $((DURATION/60)) min on channel $CHANNEL..."
# tcpdump -Z root: stay root and write the file directly (Debian's tcpdump/tshark
# otherwise DROP privileges and can't write here -> "Permission denied").
sudo timeout "$DURATION" tcpdump -Z root -i "$MON" -w "$PCAP" 2>/dev/null &
CAP_PID=$!

# Force WPA handshakes WITHOUT touching any device: deauth the 2.4GHz AP(s) so
# clients (incl. the TV) reconnect and re-do the 4-way handshake we need to
# decrypt their traffic. Done EARLY (not at 15 min) so the rest of the capture
# records the now-decryptable traffic. Broadcast deauth (all clients); if the TV
# still doesn't show, you'd need its MAC for a targeted deauth (-c <TV_MAC>).
BSSIDS="${CAPTURE_BSSIDS:?set CAPTURE_BSSIDS in .env (your 2.4GHz AP BSSIDs)}"
sleep 45
for round in 1 2 3; do
    echo "    [deauth round $round/3] forcing reconnects to capture handshakes..."
    for b in $BSSIDS; do
        sudo aireplay-ng -0 8 -a "$b" "$MON" >/dev/null 2>&1 || true
    done
    sleep 120
done

echo "    deauth rounds done; letting the capture finish..."
wait "$CAP_PID" 2>/dev/null || true      # timeout returns 124 when it stops tcpdump — expected
sudo chown "$(id -u):$(id -g)" "$PCAP" 2>/dev/null || true

echo "[3/6] capture done; restoring network early so the rest runs online..."
restore_network
trap - EXIT
sleep 8

echo "[4/6] decrypting WiFi capture with your PSK..."
tshark -r "$PCAP" -o wlan.enable_decryption:TRUE \
    -o "uat:80211_keys:\"wpa-pwd\",\"$WIFI_PASS:$SSID\"" -w "$DEC"

echo "[5/6] pcap -> conn.log via tshark adapter (Zeek can't parse 802.11 monitor pcaps)..."
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$VIGILO/data:/app/data" \
    vigilo:latest ingest "data/home/monitor_dec.pcap" "data/home/monitor.conn"

echo "[6/6] running Vigilo (Docker) on your network..."
cd "$VIGILO"
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$VIGILO/data:/app/data" \
    -v "$VIGILO/checkpoints:/app/checkpoints" \
    -v "$VIGILO/reports:/app/reports" \
    vigilo:latest report --log "data/home/monitor.conn" --out "$REPORT"

echo ""
echo "DONE ✔  open: $VIGILO/$REPORT"
