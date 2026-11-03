"""10k与重载评价完成后立即做CPU验收、视频和完整状态打包；不训练、不关机。"""
import json,time,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def status(stage,**extra):
    (ROOT/'postprocess_status.json').write_text(json.dumps(dict(stage=stage,**extra),indent=2)+'\n')
try:
    status('WAITING_FOR_10K_AND_RELOAD')
    while True:
        q=json.loads((ROOT/'queue_status.json').read_text())
        if q['stage']=='FAILED':raise RuntimeError(q['error'])
        if q['stage']=='COMPLETE_TRAIN_AND_LAYER_EVALUATION':break
        time.sleep(5)
    status('CPU_ACCEPTANCE_AND_PACKAGING')
    with (ROOT/'package.log').open('x') as f:
        subprocess.run([sys.executable,str(ROOT/'source/package_single.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    status('COMPLETE_PACKAGE_READY',formal_updates=10000,shutdown=False)
except Exception as exc:
    status('FAILED',error=repr(exc));raise
