from pathlib import Path
import sys,subprocess,json,time
ROOT=Path(__file__).resolve().parent.parent
assert not (ROOT/'smoke_lifecycle_status.json').exists()
with (ROOT/'smoke_lifecycle_controller.log').open('x') as log:
    proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'source/run_full_dynamic_smoke_lifecycle.py')],
        stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
(ROOT/'smoke_lifecycle_controller.pid').write_text(str(proc.pid)+'\n')
print(json.dumps(dict(controller_pid=proc.pid,unix=time.time(),root=str(ROOT),formal_auto_release=False)))
