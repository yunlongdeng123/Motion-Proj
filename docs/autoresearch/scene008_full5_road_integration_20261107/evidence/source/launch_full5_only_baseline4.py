from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full5_road_integration_20261107_r1');R.mkdir(exist_ok=True);(R/'source').mkdir(exist_ok=True);(R/'contract').mkdir(exist_ok=True)
PY='/root/autodl-tmp/envs/drivingeditor/bin/python';source=R/'source/render_full5_only_baseline4.py';env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
with (R/'baseline4.log').open('x') as log:p=subprocess.Popen([PY,'-B',str(source)],stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'unix':time.time(),'native_renderer_calls_planned':8,'raster_kernel_forwards_planned':24,'optimizer_updates':0,'only_full5_scene_source':True}
(R/'contract/baseline4_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
