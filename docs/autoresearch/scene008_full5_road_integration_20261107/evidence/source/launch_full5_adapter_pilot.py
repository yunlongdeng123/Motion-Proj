from pathlib import Path
import json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full5_road_integration_20261107_r1');env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
with (R/'adapter_C4_queue.log').open('x') as log:p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(R/'source/run_full5_adapter_pilot.py')],stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'unix':time.time(),'automatic_training':False,'automatic49':False};(R/'contract/adapter_C4_queue_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
