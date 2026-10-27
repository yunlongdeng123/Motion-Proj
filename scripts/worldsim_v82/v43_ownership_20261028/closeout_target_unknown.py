"""Record completed jobs/resources; no power action and no queue changes."""
from pathlib import Path
import subprocess,json,hashlib,datetime
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43'
script='''import json,time,subprocess,shutil,hashlib
from pathlib import Path
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028');D=R/'target_unknown_v2';records={}
for tag in ['pilot49','formal199','dense199','conditional_prior_pilot','conditional_pair']:
 s=json.loads((D/tag/'controller.json').read_text());assert s['phase']=='complete';records[tag]=s
models={}
for p in list((D/'formal199').glob('*/formal/checkpoint*.pth'))+list((D/'dense199').glob('*/formal/checkpoint*.pth'))+list((D/'conditional_pair').glob('*/formal/checkpoint*.pth')):
 models[str(p)]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'mixed_support_audit_r2').iterdir() if p.is_file()}
motion={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'static_motion_audit').iterdir() if p.is_file()}
result=dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),target_unknown_v2=dict(phase='user_wrapup_experiments_complete_quality_not_accepted',controllers=records,models=models,fullclip_frames=len(json.loads((D/'fullclip_inputs/manifest.json').read_text())['frames']),audit_files=files,static_motion_files=motion,formal_updates=50000,pilot_updates=3000,extra_smoke_updates=980),GPU_compute_query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True),disk=dict(zip(['total','used','free'],shutil.disk_usage(str(R)))),authorization=json.loads((R/'authorization.json').read_text()))
print(json.dumps(result))
'''
r=subprocess.run(['ssh','wm-vgpu-1011','python3 -'],input=script.encode(),capture_output=True);assert r.returncode==0,r.stderr.decode();new=json.loads(r.stdout)
for name,sha in new['target_unknown_v2']['audit_files'].items():assert hashlib.sha256((O/'target_unknown_v2/mixed_support_audit'/name).read_bytes()).hexdigest()==sha,name
for name,sha in new['target_unknown_v2']['static_motion_files'].items():assert hashlib.sha256((O/'target_unknown_v2/static_motion_audit'/name).read_bytes()).hexdigest()==sha,name
state=json.loads((O/'remote_state.json').read_text());state.update(new);state['workers']=[];state['monitor']=dict(scope='target_unknown_v2',phase='complete',NVML_sampled_peak_mib=max(v['NVML_peak_mib'] for v in new['target_unknown_v2']['controllers'].values()),old_rounds_historical=True);state['quota']=json.loads((O/'quota_state.json').read_text());(O/'remote_state.json').write_text(json.dumps(state,indent=2));(O/'target_unknown_v2/remote_closeout.json').write_text(json.dumps(new,indent=2));print(json.dumps(dict(checked_utc=new['checked_utc'],verified_checkpoints=len(new['target_unknown_v2']['models']),audit_files_verified=True,GPU_compute_query=new['GPU_compute_query'])))
