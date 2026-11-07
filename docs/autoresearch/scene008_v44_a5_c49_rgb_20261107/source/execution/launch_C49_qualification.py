from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1');R.mkdir(exist_ok=True);(R/'contract').mkdir(exist_ok=True)
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
source=R/'source/qualify_C49_parent_visibility.py'
with (R/'parent_visibility49.log').open('x') as log:p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(source)],stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'unix':time.time(),'planned_native_forwards':98,'planned_dummy_backwards':98,'optimizer_updates':0,'no_automatic_formal_release':True}
(R/'contract/parent_visibility49_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
