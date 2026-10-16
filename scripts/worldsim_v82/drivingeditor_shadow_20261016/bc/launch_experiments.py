"""Launch only this authorized, bounded pilot in a new directory."""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path

base=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1')
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--warmup',type=int,default=300);a=p.parse_args()
assert a.name.replace('_','').isalnum() and 1<=a.steps<=1000 and 1<=a.warmup<=300
out=base/'runs'/a.name;out.parent.mkdir(exist_ok=True)
receipt=base/(a.name+'_launch.json')
if out.exists() or receipt.exists():raise SystemExit('Preserve existing run/receipt. Choose a fresh named run after inspecting its state.')
for path in base.glob('*_launch.json'):
    row=json.loads(path.read_text());proc=Path('/proc')/str(row['pid'])/'cmdline'
    if proc.exists() and b'run_experiments.py' in proc.read_bytes():raise SystemExit('An owned pilot is already running; inspect it instead of duplicating.')
cmd=[sys.executable,str(Path(__file__).with_name('run_experiments.py')),'--output',str(out),'--steps',str(a.steps),'--warmup',str(a.warmup)]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
log=base/(a.name+'_console.log')
with log.open('xb') as f:
    worker=subprocess.Popen(cmd,cwd=Path(__file__).parent,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
row={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':worker.pid,'command':cmd,'output':str(out),'console_log':str(log),
     'authorization':'Human explicitly opened GPU and requested experiments, self-validation and periodic progress follow-up on 2026-10-16.',
     'subagents':False,'maximum_C_updates':5*a.steps,'B_multistarts':8,'B_updates_per_start':a.warmup,
     'checkpoint_scope':'new local-road addon only; parent checkpoint read-only'}
receipt.write_text(json.dumps(row,indent=2));print(json.dumps(row))
