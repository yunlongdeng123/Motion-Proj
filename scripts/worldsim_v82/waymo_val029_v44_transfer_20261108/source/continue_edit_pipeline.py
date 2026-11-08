"""Bounded experiment queue. No shutdown until report, push and local verified retrieval."""
from pathlib import Path
import subprocess,json,time,os
ROOT=Path(__file__).resolve().parent.parent
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def stage(name,command):
    (ROOT/'edit_pipeline_status.json').write_text(json.dumps(dict(stage=name,unix=time.time(),pid=os.getpid(),shutdown=False),indent=2))
    with (ROOT/f'{name}.log').open('x') as log:subprocess.run([PY,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
    try:
        while not (ROOT/'semantic_asset_all_r2/summary.json').exists():time.sleep(2)
        stage('road_gate32_r2',['source/fit_road_gates.py','--output','road_gate32_r2','--steps','32'])
        stage('road_gates',['source/fit_road_gates.py','--output','road_gates','--steps','1000','--resume',str(ROOT/'road_gate32_r2/checkpoint000032.pt')])
        stage('heldout_camera_audit',['source/heldout_camera.py'])
        stage('semantic_all_eval',['source/render_experiment.py','--mode','semantic','--output','semantic_all_eval'])
        stage('final_all_eval',['source/render_experiment.py','--mode','final','--output','final_all_eval'])
        stage('final_heldout_eval',['source/render_experiment.py','--mode','final','--output','final_heldout_eval','--heldout'])
        (ROOT/'edit_pipeline_status.json').write_text(json.dumps(dict(stage='GPU_RENDER_STAGES_COMPLETE_PENDING_CPU_ACCEPTANCE_AND_CLOSEOUT',unix=time.time(),shutdown=False),indent=2))
    except Exception as e:
        (ROOT/'edit_pipeline_status.json').write_text(json.dumps(dict(stage='RECOVERABLE_FAILURE',error=repr(e),unix=time.time(),shutdown=False),indent=2));raise
if __name__=='__main__':main()
