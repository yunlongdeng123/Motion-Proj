import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text())
s=read(ROOT/'queue_status.json');result=dict(s)
stage=s['stage'];directory=ROOT/({'smoke':'smoke','formal10k':'formal10k','smoke_resume':'smoke_resume'}.get(stage,stage))
if (directory/'progress.json').exists():
    p=read(directory/'progress.json');result.update(iteration=p['iteration'],elapsed_seconds=round(p['elapsed_seconds'],1),
        static_rows=p['static_rows'],dynamic_rows=p['dynamic_rows'],memory_GiB=round(p['reserved_bytes']/2**30,2),loss=p['losses']['loss_hybrid_total'])
if (ROOT/'smoke_summary.json').exists():result['smoke_summary']=read(ROOT/'smoke_summary.json')
if (ROOT/'postprocess_status.json').exists():result['postprocess']=read(ROOT/'postprocess_status.json')
if stage=='FAILED':
    logs=sorted(ROOT.glob('*.log'),key=lambda p:p.stat().st_mtime)
    result['error_log']=logs[-1].read_text()[-4500:]
print(json.dumps(result,ensure_ascii=False))
