"""
Vigilo — unified CLI entry point.

Dispatches to subcommands so users interact with a single ``vigilo`` binary
instead of remembering ``python -m vigilo.<module>`` incantations.

    vigilo train   --logs benign.conn.log --output-dir checkpoints/vigilo
    vigilo detect  --log suspect.conn.log --ckpt checkpoints/vigilo/vigilo.pt
    vigilo serve   --log data/home/home.conn.log
    vigilo report  --log data/home/home.conn.log --out reports/demo.html
    vigilo beacon  --log suspect.conn.log
    vigilo ingest  capture.pcap data/home/home.conn.log
    vigilo version
"""
from __future__ import annotations

import sys


def main():
    commands = {
        "train":   "Train the forecaster on benign traffic",
        "detect":  "Score devices in a conn.log for anomalies",
        "serve":   "Launch the live web dashboard",
        "report":  "Render a static HTML report",
        "beacon":  "Run standalone beaconing detector",
        "ingest":  "Convert a pcap to conn.log via tshark",
        "version": "Print version and exit",
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: vigilo <command> [options]\n")
        print("commands:")
        for cmd, desc in commands.items():
            print(f"  {cmd:<10} {desc}")
        print("\nRun 'vigilo <command> --help' for command-specific options.")
        sys.exit(0)

    cmd = sys.argv.pop(1)

    if cmd == "train":
        from vigilo.train import main as _main
        _main()
    elif cmd == "detect":
        from vigilo.detect import main as _main
        _main()
    elif cmd == "serve":
        from vigilo.dashboard import main as _main
        _main()
    elif cmd == "report":
        from vigilo.report import main as _main
        _main()
    elif cmd == "beacon":
        from vigilo.beaconing import main as _main
        _main()
    elif cmd == "ingest":
        from vigilo.pcap_to_conn import main as _main
        _main()
    elif cmd == "version":
        _print_version()
    else:
        print(f"vigilo: unknown command '{cmd}'")
        print("Run 'vigilo --help' for available commands.")
        sys.exit(1)


def _print_version():
    try:
        from importlib.metadata import version
        v = version("vigilo")
    except Exception:
        v = "0.1.0-dev"
    print(f"vigilo {v}")


if __name__ == "__main__":
    main()
