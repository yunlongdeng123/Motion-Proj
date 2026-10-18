from pathlib import Path
import subprocess,json,datetime,os
E=Path(__file__).resolve().parent;p=E/'all_vehicle_instance_prompt_launch.json'
assert not p.exists(),'existing finite input task; do not duplicate'
old=E/'all_vehicle_prompt_launch.json';oldpid=json.loads(old.read_text())['pid'];assert not Path('/proc/'+str(oldpid)).exists()
runtime=E.parent/'shadow_rebuild_20261017_v3/ownership_controls_r12_sam3/runtime_sam3_dinov3/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with (E/'all_vehicle_instance_prompt.log').open('x') as f:
 child=subprocess.Popen([str(runtime),'-B','-u',str(E/'infer_vehicle_instance_prompts.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
row=dict(pid=child.pid,kind='finite49_real_instance_segmentation_only',training_updates=0,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),supersedes_semantic_global_top3_identity_selection=True)
with p.open('x') as f:json.dump(row,f,indent=2)
print(json.dumps(row))
