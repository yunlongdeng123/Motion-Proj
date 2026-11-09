"""等待本任务编辑出口，进行CPU分析与结果归档；始终保持实例开机。"""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent.parent
def main():
    deadline=time.monotonic()+10800
    while True:
        p=ROOT/'edit_pipeline_status.json'
        if p.exists():
            s=json.loads(p.read_text())
            if s['stage']=='RECOVERABLE_FAILURE':raise RuntimeError(s)
            if s['stage']=='GPU_EDITS_COMPLETE_PENDING_ANALYSIS_RETRIEVAL_REPORT_PUSH':break
        if time.monotonic()>deadline:raise RuntimeError('delivery waiting timeout')
        time.sleep(5)
    with (ROOT/'package.log').open('x') as f:subprocess.run([sys.executable,'-B',str(ROOT/'source/analyze_and_package.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    (ROOT/'delivery_status.json').write_text(json.dumps(dict(stage='PACKAGED_PENDING_VERIFIED_LOCAL_RETRIEVAL_REPORT_PUSH',automatic_shutdown=False),indent=2))
if __name__=='__main__':main()
