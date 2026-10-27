from pathlib import Path
import subprocess,json
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';R='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028'
r=subprocess.run(['ssh','wm-vgpu-1011','/root/autodl-tmp/envs/drivingeditor/bin/python '+R+'/code/snapshot_remote.py'],capture_output=True);assert r.returncode==0,r.stderr.decode(errors='replace');d=json.loads(r.stdout);p=O/'remote_state.json';t=p.with_suffix('.tmp');t.write_bytes(r.stdout);t.replace(p)
print(json.dumps(dict(checked=d['checked_utc'],stage=d['campaign']['stage'],context=d['campaign']['context'],formal_complete=[x['context'] for x in d['scenes'] if x['stages'].get('formal10k',{}).get('status',{}).get('phase')=='complete'],control=d['unconstrained_control'].get('phase') if d['unconstrained_control'] else None,free_GiB=d['disk']['free']/2**30)))
