from pathlib import Path
import os,json,subprocess,datetime
E=Path(__file__).resolve().parent
old=json.loads((E/'B_scope_revision_cancellation_receipt.json').read_text());assert old['training_updates']==0 and not Path('/proc/'+str(old['old_controller'])).exists()
out=E/'intersection_queue_launch.json';assert not out.exists()
with (E/'intersection_controller.log').open('x') as f:
 child=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B','-u',str(E/'run_all_vehicle30k_queue.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
row=dict(pid=child.pid,previous_waiting_controller=old['old_controller'],same_training_budget=30000,scope='LIDAR_AND_CAMERA_INTERSECTION_ONLY',at=datetime.datetime.now(datetime.timezone.utc).isoformat())
with out.open('x') as f:json.dump(row,f,indent=2)
print(json.dumps(row))
