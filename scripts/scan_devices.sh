#!/usr/bin/env bash
# Quick scan: list connected client devices and which channel each is on.
# Channel-hops 2.4 + 5 GHz for ~75s. PC goes OFFLINE briefly; auto-restores.
#
#   bash scripts/scan_devices.sh
set -eu

DURATION="${1:-75}"

restore_network() {
    mon="$(iw dev 2>/dev/null | awk '/Interface/{i=$2} /type monitor/{print i}' | head -1)"
    [ -n "${mon:-}" ] && sudo airmon-ng stop "$mon" >/dev/null 2>&1 || true
    sudo systemctl restart NetworkManager >/dev/null 2>&1 || true
}
trap restore_network EXIT

echo "[*] monitor mode (PC offline ~$((DURATION))s)..."
sudo airmon-ng check kill >/dev/null 2>&1
sudo airmon-ng start wlan0 >/dev/null 2>&1
MON="$(iw dev | awk '/Interface/{i=$2} /type monitor/{print i}' | head -1)"
[ -z "$MON" ] && { echo "ERROR: no monitor interface"; exit 1; }

echo "[*] scanning 2.4 + 5 GHz for ${DURATION}s (channel hopping)..."
rm -f /tmp/vigscan*.csv 2>/dev/null || true
sudo timeout "$DURATION" airodump-ng --band abg -w /tmp/vigscan --output-format csv "$MON" >/dev/null 2>&1 || true

restore_network; trap - EXIT

echo ""
echo "=== connected clients -> channel ==="
printf "%-20s %-20s %-8s\n" "CLIENT MAC" "ASSOC. AP (BSSID)" "CHANNEL"
awk -F',' '
    # AP section: map BSSID -> channel (field 4)
    /^BSSID,/ {sec="ap"; next}
    /Station MAC,/ {sec="st"; next}
    sec=="ap" && $1 ~ /:/ {gsub(/ /,"",$1); gsub(/ /,"",$4); ch[$1]=$4}
    sec=="st" && $1 ~ /:/ {
        mac=$1; bss=$6; gsub(/ /,"",mac); gsub(/ /,"",bss);
        if (bss ~ /:/) printf "%-20s %-20s %-8s\n", mac, bss, (ch[bss]?ch[bss]:"?")
    }
' /tmp/vigscan*.csv 2>/dev/null | sort -u
echo ""
echo "(channels 1-11 = 2.4GHz, 36+ = 5GHz. Cross-ref MACs with your router's device list to name them.)"
