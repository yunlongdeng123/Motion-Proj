from pathlib import Path
import json,subprocess,datetime,os
E=Path(__file__).resolve().parent;p=E/'all_vehicle30k_launch.json';assert not p.exists()
with (E/'all_vehicle30k_controller.log').open('x') as f:
 c=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B','-u',str(E/'run_all_vehicle30k_queue.py')],cwd=E,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(pid=c.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),budget=30000,gate='all_vehicle_ready.json requires complete input coverage; no user reapproval needed')
p.write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
