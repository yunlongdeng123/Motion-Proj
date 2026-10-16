import os,signal,json
from pathlib import Path
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
pid=json.loads((r/'contribution_audit_launch.json').read_text())['pid']
cmd=Path('/proc',str(pid),'cmdline')
if (r/'contribution_audit').exists():raise SystemExit('Already started GPU audit; preserve process')
if cmd.exists():
    data=cmd.read_bytes()
    if b'attribute_road.py' not in data or b'--wait' not in data:raise SystemExit('PID identity mismatch')
    os.kill(pid,signal.SIGTERM)
(r/'attribution_waiter_revision.json').write_text(json.dumps({'old_pid':pid,'reason':'restart still-CPU waiting process to load revised global-mass base selection; no GPU or training work interrupted','original_output_created':False},indent=2))
print('Stopped owned CPU waiter',pid)
