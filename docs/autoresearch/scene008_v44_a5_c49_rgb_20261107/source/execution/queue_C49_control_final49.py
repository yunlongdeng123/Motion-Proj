from pathlib import Path
import json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1');env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
while not (R/'formal2000_r1/returned.json').exists():
 p=Path('/proc')/str(json.loads((R/'contract/formal2000_r1_launch.json').read_text())['pid'])/'stat';assert p.exists() and p.read_text().split()[2]!='Z','Formal exited without a passing actual receipt';time.sleep(1)
result=json.loads((R/'formal2000_r1/returned.json').read_text())
if result['final_step']!=2000:
 (R/'control49_queue_status.json').write_text(json.dumps({'status':'SKIPPED_CONTROL49_FORMAL_STOPPED_BELOW2000','actual_formal_step':result['final_step'],'new_native_forwards':0})+'\n');raise SystemExit(0)
cmd=['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(R/'source/evaluate_C49_control_final49.py')]
with (R/'control_final49_2000.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
(R/'contract/control_final49_2000_launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd,'unix':time.time(),'planned_native_forwards':196,'backwards':0,'optimizer_updates':0,'known_imperfect_control':True},indent=2)+'\n');print(json.dumps({'pid':p.pid,'stage':'control_final49_2000'}),flush=True)
code=p.wait();(R/'control49_queue_status.json').write_text(json.dumps({'status':'COMPLETE_KNOWN_IMPERFECT_CONTROL49' if code==0 else 'FAILED_CONTROL49_PRESERVED','exit_code':code,'no_H49':True,'no_automatic_further_training':True,'shutdown':False},indent=2)+'\n');raise SystemExit(code)
