#!/usr/bin/env python3
"""Serve only the comparison report on remote loopback for an SSH-forwarded preview."""
import json
from pathlib import Path
import socket
import subprocess
import sys

base = Path('/root/autodl-tmp/drivingeditor_repro')
report = base / 'reports/paper50k_comparison'
report.mkdir(parents=True, exist_ok=True)
with socket.socket() as probe:
    occupied = probe.connect_ex(('127.0.0.1', 8781)) == 0
if occupied:
    print(json.dumps({'status': 'port_in_use', 'port': 8781}))
    sys.exit(1)
with (base / 'logs/paper50k_http.log').open('ab', buffering=0) as log:
    process = subprocess.Popen([sys.executable, str(base / 'scripts/serve_paper_report.py')], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=log, start_new_session=True)
(base / 'logs/paper50k_http.pid').write_text(str(process.pid) + '\n')
print(json.dumps({'pid': process.pid, 'report': str(report), 'bind': '127.0.0.1:8781'}))
