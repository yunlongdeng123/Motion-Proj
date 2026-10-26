import subprocess,sys,json,time,os
from pathlib import Path
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026');py='/root/autodl-tmp/envs/v42-freeinpaint/bin/python';script='/root/autodl-tmp/freeinpaint_smoke.py'
jobs=[('restore_a37',[py,script,'--restore',str(C/'freeinpaint_pass1/a37/object-removal/inference_state.pth')])]
for actor,passes in [('a6',1),('a37',3),('a6',3)]:
    jobs.append((f'{actor}_passes{passes}',[py,script,'--dataset','gs25','--data_root',str(C/'inputs'),'--scene_name',actor,'--num_input_views','3','--height','336','--width','504','--seed','0','--num_refinement_iters',str(passes),'--save_gs','--output_root',str(C/f'freeinpaint_pass{passes}')]))
status=dict(pid=os.getpid(),jobs=[],training_updates=0,quality_accepted=False,human_verdict=None)
for name,cmd in jobs:
    result=dict(name=name,command=cmd,start=time.time());status['jobs'].append(result);(C/'free_probe_status.json').write_text(json.dumps(status,indent=2))
    with (C/f'free_probe_{name}.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);result['pid']=child.pid;result['start_ticks']=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]);(C/'free_probe_status.json').write_text(json.dumps(status,indent=2));result['returncode']=child.wait()
    result['end']=time.time();(C/'free_probe_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(result),flush=True)
    if result['returncode']!=0:sys.exit(result['returncode'])
status['complete']=True;(C/'free_probe_status.json').write_text(json.dumps(status,indent=2))
