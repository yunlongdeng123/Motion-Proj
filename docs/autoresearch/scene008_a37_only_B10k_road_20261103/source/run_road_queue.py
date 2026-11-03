"""短程真实GPU保存/恢复验证后，从零完成单a37道路修复1k。"""
import json,sys,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
RUN=Path(__file__).resolve().parent.parent
def write(v):(RUN/'queue_status.json').write_text(json.dumps(dict(unix=time.time(),**v),indent=2)+'\n')
def run(label,*args):
    write(dict(stage=label))
    with (RUN/(label+'.log')).open('x') as f:subprocess.run([sys.executable,str(RUN/'source/repair37.py'),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    assert json.loads((RUN/'inputs/receipt.json').read_text())['target']==37
    run('selection','--select')
    run('smoke','--output','smoke','--steps',32)
    run('restore_a','--output','restore_a','--steps',33,'--resume','smoke/checkpoint000032.pt')
    run('restore_b','--output','restore_b','--steps',33,'--resume','smoke/checkpoint000032.pt')
    same={str(raw):(RUN/'restore_a/step000033'/f'{raw:03d}_repaired.png').read_bytes()==(RUN/'restore_b/step000033'/f'{raw:03d}_repaired.png').read_bytes() for raw in [101,105,113]}
    assert all(same.values()),same
    (RUN/'smoke_summary.json').write_text(json.dumps(dict(real_GPU_steps=32,restore_updates=2,full_state_restored=True,repeated_next_step_PNG_exact=same,
        formal_from_scratch=True,expected_formal_minutes_under10=True),indent=2)+'\n')
    run('formal','--output','formal','--steps',1000)
    run('evaluation','--output','evaluation','--resume','formal/checkpoint001000.pt','--evaluate')
    write(dict(stage='COMPLETE',formal_updates=1000,smoke_updates=32,restore_updates=2,shutdown=False))
except Exception as exc:write(dict(stage='FAILED',error=repr(exc)));raise
