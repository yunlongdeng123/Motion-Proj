from pathlib import Path
import os,json,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=R/'ownership_controls_r12_sam3';receipt=root/'input_pipeline_launch.json'
if receipt.exists():raise FileExistsError('No duplicate input pipeline')
assert (R/'ownership_controls_r11/switch_to_SAM3_stop_complete.json').exists()
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
with (root/'input_pipeline.log').open('xb') as log:
 p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python',str(R/'r12_input_pipeline_controller.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=p.pid,time=time.time(),status='FINITE_INPUT_ONLY_NO_GAUSSIAN_TRAINING',runtime_setup_pid=25277,stop_after='root visual SAM3 input review; no later Gaussian training automatically');receipt.write_text(json.dumps(v,indent=2));print(json.dumps(v))
