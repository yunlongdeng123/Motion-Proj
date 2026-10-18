from pathlib import Path
import subprocess,os,json,datetime
E=Path(__file__).resolve().parent;p=E/'all_vehicle_preparation_launch.json'
assert not p.exists(),'do not duplicate registered preparation'
assert json.loads((E/'all_actor_queue_status.json').read_text())['state']=='COVERAGE_INVENTORY_COMPLETE_PREPARING_FULL_IDENTITY_AND_INITIALIZATION'
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with (E/'all_vehicle_preparation_controller.log').open('x') as f:
 child=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B','-u',str(E/'run_all_vehicle_preparation.py')],cwd=E,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),full49_original_resolution=True,new_training_updates=0)
p.write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
