#!/usr/bin/env bash
# Per-device Vigilo capture (YOUR OWN network only).
# Deauths ONLY the phones you list (to force their handshakes), captures, decrypts,
# and splits traffic per device into its own folder (conn.log + report).
#
# Phones can't be auto-detected over the air (randomized MACs), so you list them.
# Get MACs from your router's device list (it shows names like "Patrick's A16").
# Do NOT include the A16 — leaving it out means it won't be deauthed.
#
#   bash scripts/perdevice_capture.sh [seconds]   (default 600 = 10 min)

set -eu

# Config comes from .env (copy .env.example -> .env and fill in YOUR network).
# iOS rotates its MAC per reconnect, so one phone shows as several "iPhone" entries
# in the router list — put every MAC you OWN in CAPTURE_PHONE_MACS. Each gets deauthed.
VIGILO="/home/me/SAAS/Vigilo"
[ -f "$VIGILO/.env" ] && { set -a; . "$VIGILO/.env"; set +a; }
SSID="${WIFI_SSID:?set WIFI_SSID in .env (copy .env.example -> .env)}"
WIFI_PASS="${WIFI_PASS:?set WIFI_PASS in .env}"
CHANNEL="${WIFI_CHANNEL:-48}"
BSSIDS="${CAPTURE_BSSIDS:?set CAPTURE_BSSIDS in .env (your AP BSSIDs)}"
PHONE_MACS="${CAPTURE_PHONE_MACS:?set CAPTURE_PHONE_MACS in .env (YOUR devices only)}"

DURATION="${1:-600}"
OUT="$VIGILO/data/home/perdevice_$(date +%Y%m%d_%H%M%S)"
PCAP="$OUT/capture.pcap"; DEC="$OUT/decrypted.pcap"; CONN="$OUT/all.conn"
mkdir -p "$OUT"

[ -z "$PHONE_MACS" ] && { echo "ERROR: set PHONE_MACS (phone MACs from your router list, excluding the A16)."; exit 1; }

restore_network() {
    echo "[*] restoring network..."
    mon="$(iw dev 2>/dev/null | awk '/Interface/{i=$2} /type monitor/{print i}' | head -1)"
    [ -n "${mon:-}" ] && sudo airmon-ng stop "$mon" >/dev/null 2>&1 || true
    sudo systemctl restart NetworkManager >/dev/null 2>&1 || true
    echo "[*] network restored."
}
trap restore_network EXIT

echo "[1/6] monitor mode (PC OFFLINE ~$((DURATION/60)) min)..."
sudo airmon-ng check kill
sudo airmon-ng start wlan0
MON="$(iw dev | awk '/Interface/{i=$2} /type monitor/{print i}' | head -1)"
[ -z "$MON" ] && { echo "ERROR: no monitor interface"; exit 1; }
sudo iw dev "$MON" set channel "$CHANNEL"

echo "[2/6] capturing $((DURATION/60)) min..."
sudo timeout "$DURATION" tcpdump -Z root -i "$MON" -w "$PCAP" 2>/dev/null &
CAP_PID=$!

echo "[3/6] deauthing ONLY listed phones: $PHONE_MACS"
sleep 20
for round in 1 2 3; do
    echo "    [deauth round $round/3]"
    for b in $BSSIDS; do
        for c in $PHONE_MACS; do
            sudo aireplay-ng -0 5 -a "$b" -c "$c" "$MON" >/dev/null 2>&1 || true
        done
    done
    sleep 60
done
wait "$CAP_PID" 2>/dev/null || true
sudo chown "$(id -u):$(id -g)" "$PCAP" 2>/dev/null || true

echo "[4/6] restoring network + decrypting..."
restore_network; trap - EXIT; sleep 8
tshark -r "$PCAP" -o wlan.enable_decryption:TRUE \
    -o "uat:80211_keys:\"wpa-pwd\",\"$WIFI_PASS:$SSID\"" -w "$DEC"

echo "[4.5/6] verifying decryption per phone (did we catch a USABLE handshake?)..."
# Knowing the PSK is NOT enough: WPA2 derives a fresh per-session key from each
# association's 4-way handshake. No complete handshake for a session => its unicast
# traffic stays encrypted even though we have the password. One pass over the
# decrypted pcap, bucketed by source MAC, tells us per phone whether it worked.
tshark -r "$DEC" -T fields -e wlan.sa -e wlan.fc.protected -e frame.protocols 2>/dev/null \
| awk -F'\t' -v macs="$PHONE_MACS" '
    BEGIN{ n=split(tolower(macs),a," "); for(i=1;i<=n;i++) want[a[i]]=1 }
    { sa=tolower($1); if(!(sa in want)) next
      if(index($3,"eapol")){ eap[sa]++; next }
      if(index($3,"tcp")||index($3,"tls")||index($3,"quic")){ dec[sa]++; next }
      if($2=="1") enc[sa]++ }
    END{
      for(m in want){
        e=eap[m]+0; d=dec[m]+0; x=enc[m]+0
        if(d>0)      v="DECRYPTED \xe2\x9c\x94  ("d" app pkts readable)"
        else if(x>0) v="FAILED \xe2\x9c\x98    (no usable handshake; "x" frames still encrypted)"
        else         v="not seen on ch '"$CHANNEL"' (phone was on another AP/channel?)"
        printf("    %-17s  eapol=%-3d  %s\n", m, e, v)
      }
    }'

echo "[5/6] pcap -> conn.log + split per device..."
docker run --rm --user "$(id -u):$(id -g)" -v "$OUT:/work" \
    vigilo:latest ingest /work/decrypted.pcap /work/all.conn
HDR="$(grep -m1 '^#fields' "$CONN" || echo '')"
grep -v '^#' "$CONN" | awk -v out="$OUT" -v hdr="$HDR" -F'\t' '
    $3 ~ /^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)/ {
        d=out"/dev_"$3; f=d"/conn.log";
        if(!(d in seen)){ system("mkdir -p \"" d "\""); print hdr > f; seen[d]=1 }
        print >> f
    }'

echo "[6/6] per-device Vigilo reports..."
for d in "$OUT"/dev_*; do
    [ -d "$d" ] || continue
    ip="$(basename "$d" | sed 's/^dev_//')"
    docker run --rm --user "$(id -u):$(id -g)" \
        -v "$VIGILO/checkpoints:/app/checkpoints" -v "$d:/work" \
        vigilo:latest report --log /work/conn.log --out /work/report.html >/dev/null 2>&1 \
        && echo "    $ip -> $d/report.html" || echo "    $ip -> (too little data)"
done

echo ""
echo "DONE ✔  per-device folders under: $OUT"
ls -1 "$OUT" 2>/dev/null | grep '^dev_' || true
