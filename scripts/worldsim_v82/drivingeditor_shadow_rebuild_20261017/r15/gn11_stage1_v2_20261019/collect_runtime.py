"""Local narrow read-only SSH collector. Never starts or writes remote work."""
from pathlib import Path
import json,os,subprocess

REMOTE=r'''
from pathlib import Path
import json,os,subprocess,datetime,shutil,math
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v2_20261019')
def read(p):return json.loads(p.read_text())
r={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaign':str(C)}
for key,rel in [('status','run30k/status.json'),('launch','training_launch.json'),('gpu_gate','gate_002/gpu_gate_receipt.json')]:
 try:r[key]=read(C/rel)
 except Exception as e:r[key]={'read_error':str(e)}
pid=r.get('launch',{}).get('pid')
if pid:
 p=Path('/proc')/str(pid)
 r['process']={'pid':pid,'exists':p.exists()}
 if p.exists():
  r['process'].update(command=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(),
      wchan=p.joinpath('wchan').read_text().strip(),
      state=next(l for l in p.joinpath('status').read_text().splitlines() if l.startswith('State:')),
      starttime_ticks=int(p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]))
r['disk_free_bytes']=shutil.disk_usage(C).free
r['compute_device_nodes']=[str(p) for p in Path('/dev').glob('nvidia*')]
r['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
try:
 r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True,timeout=8,stderr=subprocess.STDOUT).strip()
 r['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=8,stderr=subprocess.STDOUT).strip()
except (OSError,subprocess.SubprocessError) as e:
 r['gpu_query_error']=str(e);r['gpu']='NO_GPU_DEVICE_NODES' if not r['compute_device_nodes'] else 'GPU_QUERY_UNAVAILABLE'
 r['gpu_processes']=None
if r.get('process',{}).get('exists'):
 r['process']['matches_training_command']='run_candidate.py' in r['process'].get('command','')
r['md0']={}
for key in ['array_state','degraded','sync_action']:
 try:r['md0'][key]=(Path('/sys/block/md0/md')/key).read_text().strip()
 except Exception as e:r['md0'][key]=str(e)
r['published_checkpoint_steps']=sorted(int(p.stem.split('_')[-1]) for p in (C/'run30k/checkpoint_events').glob('iteration_*.json'))
r['audited_steps']=sorted(int(p.stem.split('_')[-1]) for p in (C/'audits').glob('step_*.json'))
p=C/'run30k/v2_objectives.jsonl'
if p.exists():
 try:
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));lines=f.read().decode().splitlines()
  r['v2_last_logged_objectives']=json.loads(lines[-1])
  v=r['v2_last_logged_objectives']
  numbers=[v['ownership_volume_raw'],v['road_total'],*v['road'].values()]
  r['v2_logged_terms_finite']=all(math.isfinite(x) for x in numbers)
  r['v2_log_scope']='pre-backward diagnostics; exact completed updates come from status optimizer counters'
 except Exception as e:r['v2_objective_read_error']=str(e)
for name in ['storage_admission.json','v2_training_returned.json']:
 p=C/'run30k'/name
 if p.exists():r[name]=read(p)
p=C/'training.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));r['log_last_4000_bytes']=f.read().decode(errors='replace')
print(json.dumps(r,allow_nan=False))
'''


if __name__=='__main__':
    W=Path(__file__).resolve().parent;O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011','/usr/bin/python3','-'],
        input=REMOTE,text=True,capture_output=True,timeout=45)
    if p.returncode:raise RuntimeError('read-only collector failed: '+p.stderr[-1000:])
    r=json.loads(p.stdout)
    for dst in [W/'runtime_progress.json',O/'runtime_progress.json']:
        tmp=dst.with_suffix('.json.partial');tmp.write_text(json.dumps(r,indent=2),encoding='utf-8');os.replace(tmp,dst)
    with (W/'runtime_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps({k:r[k] for k in ['collected_utc','status','gpu','published_checkpoint_steps','audited_steps']}))
