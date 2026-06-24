#!/usr/bin/env bash
# Vigilo: capture LAN traffic via bettercap MITM, then train on it (YOUR network).
#
# bettercap does the MITM (ARP-spoof) ONLY; tcpdump does the capture. Why split:
# bettercap's own sniffer buffers and only flushes on a clean exit (you can't watch
# progress or recover a partial capture), whereas tcpdump flushes incrementally so
# you can see it grow and never lose everything. tcpdump is AppArmor-confined on
# Kali, so we relax its profile to complain mode and stage the pcap in /var/tmp.
#
#   sudo bash scripts/bettercap_capture_train.sh [capture_seconds]   (default 3600)
#
# Prints a heartbeat every 30s so it doesn't look frozen. Ctrl-C stops early and
# still converts + trains on what was captured. ARP is ALWAYS restored on exit.
# MITM is disruptive and intercepts every device — own network only.

set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
IFACE="${CAPTURE_IFACE:-wlan0}"
SUBNET="${CAPTURE_SUBNET:-192.168.1.0/24}"
DURATION="${1:-3600}"
U="${SUDO_UID:-$(id -u)}"; G="${SUDO_GID:-$(id -g)}"

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run with sudo."; exit 1; }

OUT="$VIGILO/data/home/train_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
# tcpdump is AppArmor-confined and can't write under /home; stage in /var/tmp
# (disk-backed, same filesystem = instant move) and move into the project after.
PCAP_TMP="/var/tmp/vigilo_cap_$(date +%s)_$$.pcap"
PCAP="$OUT/capture.pcap"; CONN="$OUT/capture.conn"

exec > >(tee -a "$OUT/run.log") 2>&1

# Relax tcpdump's AppArmor confinement so it can write the pcap. Prefer aa-complain
# (apparmor-utils); if that's absent, unload the profile directly with apparmor_parser.
# REVERSIBLE: `sudo apparmor_parser -r /etc/apparmor.d/usr.bin.tcpdump` or just reboot.
aa-complain /usr/bin/tcpdump >/dev/null 2>&1 \
  || apparmor_parser -R /etc/apparmor.d/usr.bin.tcpdump >/dev/null 2>&1 \
  || true

echo "[1/5] bettercap MITM (ARP-spoof only) on $IFACE -> targets $SUBNET ..."
bettercap -iface "$IFACE" -no-colors -eval \
  "set arp.spoof.fullduplex true; set arp.spoof.targets $SUBNET; net.probe on; arp.spoof on" \
  >"$OUT/bettercap.log" 2>&1 &
BC_PID=$!
sleep 8   # let ARP poisoning take hold before capturing

cleanup() {
    echo "[*] stopping: ending capture, restoring ARP..."
    kill "${TD_PID:-0}" 2>/dev/null || true         # tcpdump flushes + closes pcap cleanly
    kill -INT "$BC_PID" 2>/dev/null || true; sleep 3 # bettercap re-ARPs everyone back
    kill "$BC_PID" 2>/dev/null || true
    chown -R "$U:$G" "$OUT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[2/5] capturing forwarded traffic with tcpdump -> $PCAP_TMP ..."
tcpdump -i "$IFACE" -n -w "$PCAP_TMP" 'ip or ip6' >"$OUT/tcpdump.log" 2>&1 &
TD_PID=$!
sleep 3
if ! kill -0 "$TD_PID" 2>/dev/null || [ ! -s "$PCAP_TMP" ]; then
    echo "ERROR: tcpdump isn't writing — AppArmor likely still blocking. Fix with:"
    echo "    sudo aa-complain /usr/bin/tcpdump      (then re-run)"
    echo "    # or fully: sudo systemctl stop apparmor"
    echo "  details: $OUT/tcpdump.log"; exit 1
fi

echo "[3/5] capturing ${DURATION}s — heartbeat every 30s (Ctrl-C to stop early & still train):"
end=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$end" ]; do
    sleep 30
    echo "    [$(date +%H:%M:%S)] capturing... $(du -h "$PCAP_TMP" 2>/dev/null | cut -f1)"
done

cleanup; trap - EXIT INT TERM
mv "$PCAP_TMP" "$PCAP" 2>/dev/null || true
chown "$U:$G" "$PCAP" 2>/dev/null || true
[ -s "$PCAP" ] || { echo "ERROR: no packets captured."; exit 1; }
echo "    captured $(du -h "$PCAP" | cut -f1)"

echo "[4/5] pcap -> Zeek conn.log..."
sudo -u "#$U" bash "$VIGILO/scripts/pcap_to_conn_zeek.sh" "$PCAP" "$CONN"

echo "[5/5] training forecaster on captured traffic..."
docker run --rm --user "$U:$G" \
  -e USER=vigilo -e HOME=/tmp -e TORCHINDUCTOR_CACHE_DIR=/tmp/ti -e TRITON_CACHE_DIR=/tmp/triton \
  -v "$VIGILO/data:/app/data" -v "$VIGILO/checkpoints:/app/checkpoints" \
  "$VIGILO_IMAGE" train --logs "data/home/$(basename "$OUT")/capture.conn" \
  --output-dir checkpoints/vigilo

echo "done. model -> checkpoints/vigilo/vigilo.pt   data -> $OUT"
