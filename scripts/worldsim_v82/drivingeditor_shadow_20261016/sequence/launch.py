import subprocess,sys,json,os,datetime,shutil,hashlib
from pathlib import Path
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
name=sys.argv[1];script=sys.argv[2];extra=sys.argv[3:]
receipt=root/(name+'_launch.json')
if receipt.exists():raise SystemExit('Existing receipt preserved; do not duplicate')
snapshot=root/'source_snapshots'/name
shutil.copytree(root/'implementation',snapshot,ignore=shutil.ignore_patterns('__pycache__'))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')}
(snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
cmd=[sys.executable,str(root/'implementation'/script),*extra]
with (root/(name+'_console.log')).open('xb') as log:
    p=subprocess.Popen(cmd,env=env,cwd=root/'implementation',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
row={'pid':p.pid,'command':cmd,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
receipt.write_text(json.dumps(row,indent=2));print(json.dumps(row))
