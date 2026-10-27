"""Scoped source upload and durable authorization. Never resume old campaign."""
from pathlib import Path
import subprocess,json,base64,zipfile,io,hashlib,datetime
B=Path(__file__).parent
ROOT='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028'
files=[p for p in B.glob('*.py') if p.name!='deploy.py']
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.writestr(p.name,p.read_bytes())
script='''import io,zipfile,base64,json,time,hashlib
from pathlib import Path
R=Path(ROOT);(R/'code').mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile(io.BytesIO(base64.b64decode(DATA)));files={}
for name in z.namelist():
 p=R/'code'/name;assert p.parent==R/'code'
 content=z.read(name)
 if p.exists() and p.read_bytes()!=content:
  b=R/'source_history'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());b.mkdir(parents=True,exist_ok=True);(b/name).write_bytes(p.read_bytes())
 t=p.with_suffix('.tmp');t.write_bytes(content);t.replace(p);files[name]=hashlib.sha256(content).hexdigest()
auth=R/'authorization.json'
if not auth.exists():
 auth.write_text(json.dumps(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),user_request='Continue v43 GPU ownership/structure/semantic experiments and diagnostics, each case 10k-20k; dynamic vehicle support repair, identify vehicle evidence in static branch, finer static semantics',quota_override='先不管额度，要停止了我通知你',quota_stop_enabled=False,user_stop_required=True,old_queue_resumed=False,old_automation_resumed=False,allow_v43_training=True,formal_first_budget=10000,max_per_case_per_arm=20000,quality_accepted=False,human_verdict=None),ensure_ascii=False,indent=2))
receipt=dict(root=str(R),source_sha256=files,authorization=json.loads(auth.read_text()),old_workers_started=False)
(R/'deployment_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False));print(json.dumps(receipt,ensure_ascii=False))
'''.replace('Path(ROOT)','Path('+repr(ROOT)+')').replace('base64.b64decode(DATA)','base64.b64decode('+repr(base64.b64encode(buf.getvalue()).decode())+')')
r=subprocess.run(['ssh','wm-vgpu-1011','python3 -'],input=script.encode(),capture_output=True);assert r.returncode==0,r.stderr.decode(errors='replace')
out=B.parents[1]/'outputs/V43_Ownership_20261027/gpu_v43';out.mkdir(parents=True,exist_ok=True);(out/'deployment_receipt.json').write_bytes(r.stdout);print(r.stdout.decode())
