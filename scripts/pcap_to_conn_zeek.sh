#!/usr/bin/env bash
# pcap -> native Zeek conn.log via Docker (no Zeek install needed, no libc issue).
# Higher fidelity than the tshark adapter (real conn_state/service) and identical
# format to the IoT-23 training data.
#
# Usage: scripts/pcap_to_conn_zeek.sh <input.pcap> <output.conn.log>
set -e
pcap="$(realpath "$1")"; out="$2"
pdir="$(dirname "$pcap")"; pbase="$(basename "$pcap")"
odir="$(dirname "$(realpath -m "$out")")"; obase="$(basename "$out")"
mkdir -p "$odir"
tmp="$(mktemp -d "$odir/zeek.XXXXXX")"
docker run --rm -v "$pdir":/pcap:ro -v "$tmp":/out -w /out --entrypoint zeek \
  zeek/zeek -r "/pcap/$pbase" -C
mv "$tmp/conn.log" "$out"
rm -rf "$tmp"
echo "[zeek] wrote $out ($(grep -vc '^#' "$out") flows)"
