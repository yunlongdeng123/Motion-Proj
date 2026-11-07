from pathlib import Path
import sys,subprocess,json,time
ROOT=Path(__file__).resolve().parent.parent
assert not (ROOT/'formal10k_keep_lifecycle_status.json').exists()
with (ROOT/'formal10k_keep_controller.log').open('x') as log:
    proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'source/observe_full_dynamic_keep10k.py')],
        stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
(ROOT/'formal10k_keep_controller.pid').write_text(str(proc.pid)+'\n')
print(json.dumps(dict(controller_pid=proc.pid,unix=time.time(),no_future_training=True,all49_auto_release=False)))
