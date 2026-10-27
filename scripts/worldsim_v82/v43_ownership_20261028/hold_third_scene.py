"""Persist latest user scope; preserve already completed work and all checkpoints."""
from pathlib import Path
import subprocess,json
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43'
script='''from pathlib import Path
import json,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028');ctx='473735159277431842_630_095_650_095';p=R/'authorization.json';a=json.loads(p.read_text());a['held_contexts']=[ctx];a['active_development_contexts']=['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000'];a['latest_user_scope']='Case2 stop at10K; do not continue case3; diagnose case1 badcases';a['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());t=p.with_suffix('.tmp');t.write_text(json.dumps(a,indent=2));t.replace(p)
states={}
for family,path in [('training',R/'actor_campaign_v1'/ctx/'formal10k/status.json'),('native_edits',R/'actor_motion_v1'/ctx/'status.json')]:
 d=json.loads(path.read_text());states[family]=dict(status=d,status_mtime_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(path.stat().st_mtime)))
 dpid=Path('/proc/%s/stat'%d['pid']);states[family]['worker_alive']=dpid.exists() and dpid.read_text().split()[2]!='Z' and dpid.read_text().split()[21]==str(d['start_ticks'])
receipt=dict(checked_utc=a['updated_utc'],context=ctx,held=True,completed_before_instruction=True,work_preserved=True,no_new_work_started=True,states=states)
(R/'scene3_user_hold_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
r=subprocess.run(['ssh','wm-vgpu-1011','python3 -'],input=script.encode(),capture_output=True);assert r.returncode==0,r.stderr.decode();(O/'scene3_user_hold_receipt.json').write_bytes(r.stdout);d=json.loads(r.stdout);print(json.dumps(dict(context=d['context'],held=True,states={k:dict(phase=v['status']['phase'],step=v['status'].get('step'),mtime=v['status_mtime_utc'],worker_alive=v['worker_alive']) for k,v in d['states'].items()})))
