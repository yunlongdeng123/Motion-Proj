from pathlib import Path
import json,subprocess,sys,os,datetime
Q=Path(__file__).resolve().parent
if (Q/'queue_status.json').exists() or (Q/'launch.json').exists():raise FileExistsError('repair queue already exists')
assert 'Ran 26 tests' in (Q/'cpu_tests.log').read_text() and (Q/'cpu_tests.log').read_text().rstrip().endswith('OK')
assert json.loads((Q/'input_preflight.json').read_text())['gaussians']==771821
with (Q/'controller.log').open('a') as log:
    child=subprocess.Popen([sys.executable,str(Q/'controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=Q,start_new_session=True)
r=dict(pid=child.pid,command=[sys.executable,str(Q/'controller.py')],at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    state='LAUNCHED_WAITING_EXISTING_GPU_QUEUE',user_authorized_fresh_4k=True,original_45k_still_paused=True)
(Q/'launch.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
