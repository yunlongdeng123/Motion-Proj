"""只读采集三个消融与排队控制器；不启动、不写远端文件。"""
from pathlib import Path
import datetime,json,subprocess
W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019/ablations'
REMOTE=r'''
from pathlib import Path
import datetime,json,subprocess,sys
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v2_20261019');g=C/'ablations'
sys.path.insert(0,str(g/'source_bundle'))
from ablation_support import read,process,collective_storage_required
def optional(p):return read(p) if p.exists() else None
def tail(p):
    if not p.exists():return None
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));return f.read().decode(errors='replace')
r=dict(collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),main=optional(C/'run30k/status.json'),
    registry=optional(g/'registry.json'),controller_launch=optional(g/'controller_launch.json'),
    deployment=optional(g/'deployment_receipt.json'),arms={})
for name in ['real_only','prior_only','ownership_only']:
    root=g/name
    r['arms'][name]=dict(status=optional(root/'run30k/status.json'),gate=optional(root/'gate_001/gpu_gate_receipt.json'),
        gate_status=optional(root/'gate_001/status.json'),train_launch=optional(root/'train_launch.json'),
        gate_launch=optional(root/'gate_launch.json'),
        published_steps=sorted(int(p.stem.split('_')[-1]) for p in (root/'run30k/checkpoint_events').glob('*.json') if read(p).get('status')=='complete'),
        audited_steps=sorted(read(p)['step'] for p in (root/'audits').glob('step_*.json')),
        manifest=optional(root/'source_bundle/candidate_manifest.json'))
    status=r['arms'][name]['status'] or r['arms'][name]['gate_status']
    if status and 'failed' in status.get('status',''):r['arms'][name]['log_last4000']=tail(root/('train.log' if r['arms'][name]['status'] else 'gate.log'))
r['controller_process']=process(r['controller_launch']['pid']) if r['controller_launch'] else None
if r['registry']:
    r['processes']={str(pid):process(pid) for pid in [r['registry']['main_pid'],*map(int,r['registry']['children'])]}
    r['disk_required_estimate_bytes']=collective_storage_required(g)
import shutil
r['disk_free_bytes']=shutil.disk_usage(C).free
r['compute_device_nodes']=[str(p) for p in Path('/dev').glob('nvidia*')]
r['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
try:
 r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True,timeout=8,stderr=subprocess.STDOUT).strip()
 r['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=8,stderr=subprocess.STDOUT).strip()
except (OSError,subprocess.SubprocessError) as e:
 r['gpu_query_error']=str(e);r['gpu']='NO_GPU_DEVICE_NODES' if not r['compute_device_nodes'] else 'GPU_QUERY_UNAVAILABLE';r['gpu_processes']=None
r['md0']={n:(Path('/sys/block/md0/md')/n).read_text().strip() for n in ['array_state','degraded','sync_action']}
if r['controller_process'] is None or (r['registry'] and 'FAILED' in r['registry']['status']):r['controller_log_last4000']=tail(g/'controller.log')
print(json.dumps(r))
'''

def main():
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011','/usr/bin/python3','-'],
        input=REMOTE,text=True,encoding='utf-8',capture_output=True,check=True,timeout=40)
    r=json.loads(p.stdout);O.mkdir(parents=True,exist_ok=True)
    (O/'runtime.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    (W/'ablation_runtime.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    with (W/'ablation_runtime_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(dict(time=r['collected_utc'],registry_status=(r['registry'] or {}).get('status'),
        main=(r['main'] or {}).get('iteration'),
        arms={k:dict(status=(v['status'] or v['gate_status'] or {}).get('status','queued'),
            iteration=(v['status'] or {}).get('iteration',0),pid=(v['train_launch'] or v['gate_launch'] or {}).get('pid'),
            gate_passed=bool(v['gate'])) for k,v in r['arms'].items()},
        disk_free_GB=r['disk_free_bytes']/1e9,required_estimate_GB=r.get('disk_required_estimate_bytes',0)/1e9,gpu=r['gpu'])))

if __name__=='__main__':main()
