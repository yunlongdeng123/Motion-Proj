"""Refresh current scope/resources; retain the previous reconstruction snapshot."""
from pathlib import Path
import subprocess,json,shutil
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';p=O/'remote_state.json';backup=O/'remote_state_before_first_gpu_review.json'
if not backup.exists():shutil.copy2(p,backup)
script='''import json,time,shutil,subprocess
from pathlib import Path
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028');D=R/'first_gpu_review_v1'
s=json.loads((D/'round_state.json').read_text());s['phase']='experiments_complete_artifacts_verified';(D/'round_state.json').write_text(json.dumps(s,indent=2))
print(json.dumps(dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),authorization=json.loads((R/'authorization.json').read_text()),disk=dict(zip(['total','used','free'],shutil.disk_usage(str(R)))),first_gpu_review=s,GPU_compute_query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True))))
'''
r=subprocess.run(['ssh','wm-vgpu-1011','python3 -'],input=script.encode(),capture_output=True);assert r.returncode==0,r.stderr.decode();new=json.loads(r.stdout);state=json.loads(p.read_text());state.update(new);state['workers']=[];state['monitor']=dict(scope='first_gpu_review_v1',phase='completed',NVML_sampled_peak_mib=3647,CUDA_allocator_peak_separately_recorded=True,old_baseline_monitor_historical=True);state['quota']=json.loads((O/'quota_state.json').read_text());p.write_text(json.dumps(state,indent=2));(O/'first_gpu_review_round_state.json').write_text(json.dumps(new['first_gpu_review'],indent=2));print(json.dumps(dict(checked_utc=new['checked_utc'],active_GPU_apps=new['GPU_compute_query'],disk_free_GiB=new['disk']['free']/1024**3)))
