"""
Render the dashboard to a STATIC HTML report (no server required).

Same UI as the live dashboard, but written to a .html file you can open
directly in a browser, screenshot, or share. Sidesteps needing a running server.

Usage:
    python -m vigilo.report --log data/iot23/malware-3-1.labeled \
        --out reports/malware-3-1.html
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from jinja2 import Template

from vigilo.ensemble import analyze
from vigilo.dashboard import TEMPLATE


def render(log: str, ckpt: str, out: str):
    results = analyze(log, ckpt)
    n_alert = sum(r["verdict"] == "ALERT" for r in results)
    html = Template(TEMPLATE).render(
        results=results, n=len(results), n_alert=n_alert, log=os.path.basename(log))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html)
    print(f"[report] {out}  ({len(results)} devices, {n_alert} alerts)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt", default="checkpoints/vigilo/vigilo.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    render(a.log, a.ckpt, a.out)


if __name__ == "__main__":
    main()
