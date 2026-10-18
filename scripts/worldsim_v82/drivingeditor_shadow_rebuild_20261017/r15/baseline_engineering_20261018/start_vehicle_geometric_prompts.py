from pathlib import Path
import sys,subprocess,json,datetime,os
E=Path(__file__).resolve().parent
p=E/'all_vehicle_prompt_launch.json'
assert not p.exists(),'existing finite input task; do not duplicate'
runtime=E.parent/'shadow_rebuild_20261017_v3/ownership_controls_r12_sam3/runtime_sam3_dinov3/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with (E/'all_vehicle_prompt.log').open('x') as f:
 child=subprocess.Popen([str(runtime),'-B','-u',str(E/'infer_vehicle_geometric_prompts.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
row=dict(pid=child.pid,kind='finite49_input_segmentation_only',training_updates=0,at=datetime.datetime.now(datetime.timezone.utc).isoformat())
with p.open('x') as f:json.dump(row,f,indent=2)
print(json.dumps(row))
