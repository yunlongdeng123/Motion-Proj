import datetime,json,os,subprocess,sys
from pathlib import Path
base=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1')
name='plane_1000_v1';out=base/'runs'/name;receipt=base/(name+'_launch.json')
if out.exists() or receipt.exists():raise SystemExit('Preserve existing plane experiment')
for path in base.glob('*_launch.json'):
    old=json.loads(path.read_text());proc=Path('/proc')/str(old['pid'])/'cmdline'
    if proc.exists() and any(x in proc.read_bytes() for x in [b'run_experiments.py',b'run_plane_refinement.py']):raise SystemExit('An owned GPU experiment is already running')
command=[sys.executable,str(Path(__file__).with_name('run_plane_refinement.py')),'--output',str(out),'--steps','1000']
log=base/(name+'_console.log');env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
with log.open('xb') as f:worker=subprocess.Popen(command,cwd=Path(__file__).parent,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
row={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':worker.pid,'command':command,'output':str(out),'console_log':str(log),
  'additional_updates':1000,'reason':'Zero-update supported-plane projection reduced observed shadow error from0.040408 to0.019140; now refit road appearance after changed depth ordering.',
  'source_parameters_and_checkpoint_read_only':True,'new_points_added':0}
receipt.write_text(json.dumps(row,indent=2));print(json.dumps(row))
