"""Vigilo — local, self-hosted network anomaly detection.

Learns each device's normal network behavior from Zeek conn.log and flags
deviations (beaconing, scanning, exfiltration, new C&C destinations). Runs
entirely on-device; no traffic data leaves the network.

Engine uses a PdMForecaster (continuous-I/O state-space forecaster):
behavioral feature windows per device -> forecast next window -> anomaly is
forecast-error surprise.
"""

__version__ = "0.1.0"
