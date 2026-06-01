"""
Vigilo dashboard — a local web UI showing per-device verdicts.

Runs the ensemble (volume + beaconing) over a conn.log and renders a color-coded
device table with human-readable reasons. Fully local; nothing leaves the host.

Usage:
    VIGILO_LOG=data/iot23/malware-49-1.labeled python -m vigilo.dashboard
    # then open http://127.0.0.1:8088
"""
from __future__ import annotations

import os

from flask import Flask, render_template_string

from vigilo.ensemble import analyze

app = Flask(__name__)

TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8"><title>Vigilo</title>
<style>
 body{background:#0d1117;color:#c9d1d9;font-family:ui-monospace,Menlo,monospace;margin:0;padding:24px}
 h1{margin:0 0 2px;font-size:22px}.sub{color:#8b949e;font-size:13px;margin-bottom:18px}
 .cards{display:flex;gap:14px;margin-bottom:20px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px}
 .card .n{font-size:26px;font-weight:700}.card.alert .n{color:#f85149}.card.ok .n{color:#3fb950}
 .card .l{color:#8b949e;font-size:12px}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #21262d}
 th{color:#8b949e;font-weight:600}
 tr.alert{background:rgba(248,81,73,.07)}
 .pill{padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700}
 .pill.ALERT{background:#f85149;color:#0d1117}.pill.ok{background:#238636;color:#fff}
 .dev{font-weight:600}.reason{color:#d29922}.score{font-variant-numeric:tabular-nums}
 .foot{margin-top:18px;color:#8b949e;font-size:12px}
</style></head><body>
 <h1>🛡 Vigilo</h1>
 <div class="sub">Local network anomaly detection · {{log}} · 100% on-device</div>
 <div class="cards">
   <div class="card"><div class="n">{{n}}</div><div class="l">devices</div></div>
   <div class="card alert"><div class="n">{{n_alert}}</div><div class="l">alerts</div></div>
   <div class="card ok"><div class="n">{{n - n_alert}}</div><div class="l">healthy</div></div>
 </div>
 <table><tr><th>Device</th><th>Status</th><th>Volume</th><th>Beacon</th><th>Flows</th><th>Why</th></tr>
 {% for r in results %}
 <tr class="{{r.verdict}}">
   <td class="dev">{{r.device}}</td>
   <td><span class="pill {{r.verdict}}">{{r.verdict}}</span></td>
   <td class="score">{{r.volume_score}}</td>
   <td class="score">{{r.beacon_score}}</td>
   <td class="score">{{r.flows}}</td>
   <td class="reason">{{ r.reasons | join('; ') }}</td>
 </tr>
 {% endfor %}
 </table>
 <div class="foot">volume = behavioral-forecast anomaly · beacon = periodic-C&amp;C regularity · trained on benign traffic only</div>
</body></html>
"""


@app.route("/")
def index():
    log = os.environ.get("VIGILO_LOG", "data/iot23/malware-49-1.labeled")
    ckpt = os.environ.get("VIGILO_CKPT", "checkpoints/vigilo/vigilo.pt")
    results = analyze(log, ckpt)
    n_alert = sum(r["verdict"] == "ALERT" for r in results)
    return render_template_string(TEMPLATE, results=results, n=len(results),
                                  n_alert=n_alert, log=os.path.basename(log))


def main():
    app.run(host=os.environ.get("VIGILO_HOST", "127.0.0.1"),
            port=int(os.environ.get("VIGILO_PORT", "8088")))


if __name__ == "__main__":
    main()
