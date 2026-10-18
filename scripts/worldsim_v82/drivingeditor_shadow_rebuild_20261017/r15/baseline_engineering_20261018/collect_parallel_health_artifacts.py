from pathlib import Path
import json,subprocess,shutil
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018';E='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018';py='/root/autodl-tmp/envs/drivingeditor/bin/python'
packet=json.loads(subprocess.check_output(['ssh','wm-vgpu-1011',py,E+'/collect_recovery_progress.py']))
names=['A_recovery_queue_status.json','live_GN11_recovery_status.json','live_direct_instance_status.json','live_B_native_status.json']
for name in names:
 if name in packet:
  (W/name).write_text(json.dumps(packet[name],indent=2));shutil.copy2(W/name,O/name)
for name in ['parallel_5k_durability_receipt.json','remaining_disk_forecast.json']:
 subprocess.run(['scp','wm-vgpu-1011:'+E+'/'+name,str(W/name)],check=True);shutil.copy2(W/name,O/name)
a=packet['live_direct_instance_status.json'];b=packet['live_B_native_status.json'];print(json.dumps(dict(A_step=a['step'],A_state=a['state'],B_step=b['iteration'],B_state=b['status'],guardian=packet['A_recovery_queue_status.json']['state'])))
