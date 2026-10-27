"""Persist an actual tool observation; explicit user override is separate."""
import sys,json,subprocess,base64
from pathlib import Path
used=float(sys.argv[1]);checked=sys.argv[2];reset=int(sys.argv[3]);B=Path(__file__).resolve().parents[2]
d=dict(checked_utc=checked,bucket='codex',windowDurationMins=10080,usedPercent=used,remaining=max(0,100-used),remaining_percent=max(0,100-used),resetsAt=reset,v43_quota_stop_enabled=False,user_override='Ignore quota for now; user will notify when to stop',override_scope='v43_ownership_20261028 only; old queue stays paused')
for folder in [B/'outputs/V43_Ownership_20261027',B/'outputs/V43_Ownership_20261027/gpu_v43']:
    folder.mkdir(exist_ok=True);p=folder/'quota_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
    with (folder/'quota_history.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
script='''import json,base64
from pathlib import Path
d=json.loads(base64.b64decode(PAYLOAD))
for folder in ['persistent_world_20261027','v43_ownership_20261028']:
 R=Path('/root/autodl-tmp/drivingeditor_repro/campaign')/folder;p=R/'quota_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
 with (R/'quota_history.jsonl').open('a') as f:f.write(json.dumps(d)+'\\n')
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028');p=R/'authorization.json';a=json.loads(p.read_text());a['quota_override']='Ignore quota for now; user will notify when to stop';p.write_text(json.dumps(a,indent=2))
print(json.dumps(d))
'''.replace('PAYLOAD',repr(base64.b64encode(json.dumps(d).encode()).decode()))
r=subprocess.run(['ssh','wm-vgpu-1011','python3 -'],input=script.encode(),capture_output=True);assert r.returncode==0,r.stderr.decode();print(r.stdout.decode())
