from pathlib import Path
import json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1');env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
with (R/'lifecycle_formal_queue.log').open('x') as log:p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(R/'source/C49_lifecycle_then_formal2000.py')],stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'unix':time.time(),'release_condition':'actual>=600s + finite + source/geometry freeze + fullstate restore/next2 PASS','fresh_Cwarm0_formal_budget':2000,'further5k10k_automatic':False};(R/'contract/lifecycle_queue_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
