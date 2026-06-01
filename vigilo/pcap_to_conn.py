"""
Convert a pcap to a Zeek-style conn.log using tshark (no Zeek install needed).

Aggregates packets into bidirectional flows (5-tuple, split on idle gaps) and
writes the conn.log fields Vigilo's parser/featurizer use. Approximate vs. real
Zeek (conn_state is heuristic), but sufficient for behavioral anomaly features.

Usage:
    # capture ~1 hour of your machine's traffic:
    sudo tcpdump -i any -w /tmp/home.pcap        # or: tshark -i <iface> -w ...
    # convert + (no Zeek needed):
    python -m vigilo.pcap_to_conn /tmp/home.pcap data/home/home.conn.log
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

IDLE_SPLIT = 120.0   # seconds of inactivity that ends a flow

_SVC = {443: "ssl", 80: "http", 53: "dns", 123: "ntp", 22: "ssh",
        25: "smtp", 993: "ssl", 587: "smtp", 8883: "ssl", 1883: "mqtt"}


def run_tshark(pcap: str) -> list[str]:
    cmd = [
        "tshark", "-r", pcap, "-T", "fields", "-E", "separator=\t", "-E", "occurrence=f",
        "-e", "frame.time_epoch", "-e", "ip.proto", "-e", "ip.src", "-e", "ip.dst",
        "-e", "tcp.srcport", "-e", "tcp.dstport", "-e", "udp.srcport", "-e", "udp.dstport",
        "-e", "frame.len", "-e", "tcp.flags",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"tshark failed: {out.stderr[:300]}")
    return out.stdout.splitlines()


class Flow:
    __slots__ = ("ts", "last", "src", "dst", "sport", "dport", "proto",
                 "ob", "rb", "op", "rp", "rst", "resp_seen")

    def __init__(self, ts, src, dst, sport, dport, proto):
        self.ts = ts; self.last = ts
        self.src = src; self.dst = dst; self.sport = sport; self.dport = dport
        self.proto = proto
        self.ob = self.rb = self.op = self.rp = 0
        self.rst = False; self.resp_seen = False

    def state(self) -> str:
        if not self.resp_seen:
            return "S0"               # no reply — scanning / dead host signal
        if self.rst:
            return "REJ"
        return "SF"


def to_flows(lines):
    flows: dict[tuple, Flow] = {}
    done: list[Flow] = []
    for ln in lines:
        p = ln.split("\t")
        if len(p) < 10:
            continue
        try:
            ts = float(p[0]); proto = p[1]; src = p[2]; dst = p[3]
        except ValueError:
            continue
        if not src or not dst:
            continue
        sport = p[4] or p[6] or "0"
        dport = p[5] or p[7] or "0"
        length = int(p[8] or 0)
        flags = p[9]
        proto_name = {"6": "tcp", "17": "udp", "1": "icmp"}.get(proto, proto or "-")
        try:
            sp, dp = int(sport), int(dport)
        except ValueError:
            sp = dp = 0

        # canonical bidirectional key (lower endpoint first) + initiator memory
        a, b = (src, sp), (dst, dp)
        key = (proto_name,) + (a + b if a <= b else b + a)
        f = flows.get(key)
        if f is None or (ts - f.last) > IDLE_SPLIT:
            if f is not None:
                done.append(f)
            f = Flow(ts, src, dst, sp, dp, proto_name)
            flows[key] = f
        f.last = ts
        forward = (src == f.src)
        if forward:
            f.ob += length; f.op += 1
        else:
            f.rb += length; f.rp += 1; f.resp_seen = True
        if flags and ("0x004" in flags or flags.endswith("4")):   # RST bit
            f.rst = True
    done.extend(flows.values())
    return done


_FIELDS = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
           "proto", "service", "duration", "orig_bytes", "resp_bytes",
           "conn_state", "orig_pkts", "resp_pkts"]


def write_conn_log(flows, out_path):
    flows.sort(key=lambda f: f.ts)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as w:
        w.write("#fields\t" + "\t".join(_FIELDS) + "\n")
        for i, f in enumerate(flows):
            dur = max(0.0, f.last - f.ts)
            svc = _SVC.get(f.dport, "-")
            w.write("\t".join(str(x) for x in [
                f"{f.ts:.6f}", f"C{i}", f.src, f.sport, f.dst, f.dport,
                f.proto, svc, f"{dur:.3f}", f.ob, f.rb, f.state(), f.op, f.rp,
            ]) + "\n")
    return len(flows)


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python -m vigilo.pcap_to_conn <in.pcap> <out.conn.log>")
    pcap, out = sys.argv[1], sys.argv[2]
    print(f"[pcap2conn] reading {pcap} via tshark ...", flush=True)
    flows = to_flows(run_tshark(pcap))
    n = write_conn_log(flows, out)
    print(f"[pcap2conn] wrote {n} flows → {out}", flush=True)


if __name__ == "__main__":
    main()
