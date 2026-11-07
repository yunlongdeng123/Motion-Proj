from pathlib import Path
import json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1');env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
with (R/'control49_queue.log').open('x') as log:p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(R/'source/queue_C49_control_final49.py')],stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'unix':time.time(),'condition':'formal actual2000 returned; otherwise no GPU','native_forwards_planned_if_released':196,'optimizer_updates':0,'no_H49':True};(R/'contract/control49_queue_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
