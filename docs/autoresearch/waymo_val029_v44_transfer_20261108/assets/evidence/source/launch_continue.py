from pathlib import Path
import subprocess,sys,json,time
root=Path(__file__).resolve().parent.parent
with (root/'native_controller.log').open('x') as log:
    p=subprocess.Popen([sys.executable,'-B',str(root/'source/continue_native.py')],stdout=log,stderr=subprocess.STDOUT,cwd=root,start_new_session=True)
(root/'native_controller_launch.json').write_text(json.dumps(dict(pid=p.pid,time=time.time(),scope='smoke gate, fresh10K, read-only native keep/delete evaluation; no power action')))
print(p.pid)
