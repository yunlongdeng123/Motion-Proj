"""P0R1一次性4k门禁：有限等待完整权重，然后顺序渲染/评估，不训练、不重试。"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/external/worldsim_v75/VAD-GS')
RUN=Path('/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000')
OUT=RUN/'engineering_4k'
CONFIG='configs/v76/nuscenes_000_repro_exhaustive.yaml'
PYTHON=sys.executable


def main():
    OUT.mkdir(exist_ok=False)
    state={'task_id':'VADGS-P0R1-000-GATE-4000','pid':os.getpid(),'stage':'waiting_for_checkpoint',
           'completed_stages':[],'timeout_seconds':7200,'checkpoint_iteration':4000,
           'shutdown_when_pipeline_exits':False,'failure_ledger_refs':['V76-F01']}
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary=OUT/'state.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(OUT/'state.json')
    started=time.monotonic()
    previous=None
    stable=0
    checkpoint=RUN/'trained_model/iteration_4000.pth'
    try:
        save()
        while True:
            main_state=json.loads((RUN/'pipeline_state.json').read_text())
            if main_state.get('status')=='failed':
                raise RuntimeError('main pipeline failed; gate will not restart it')
            if time.monotonic()-started>7200:
                raise TimeoutError('4k gate bounded wait expired; no automatic retry')
            current=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns) if checkpoint.exists() else None
            stable=stable+1 if current and current==previous else 0
            previous=current
            # At least 60 seconds of unchanged size/mtime; train saves directly.
            if current and current[0]>0 and stable>=2:
                memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(memory.strip().splitlines()[0])>=12000:
                    break
            time.sleep(30)
        state['checkpoint_bytes']=checkpoint.stat().st_size
        env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        commands=[
            ('sweep', [PYTHON,'script/v76/render_counterfactual.py','--config',CONFIG,'--iteration','4000',
                '--intervention','sweep','--camera','0','--frame','20','--offsets=-3.5,0,0.5,1,2,3.5',
                '--output-dir',str(OUT/'sweep')]),
            ('sweep_summary',[PYTHON,'script/v76/summarize_sweep.py','--input-dir',str(OUT/'sweep')]),
            ('official_test',[PYTHON,'script/v76/evaluate_official_test.py','--config',CONFIG,'--iteration','4000',
                '--output-dir',str(OUT/'official_test')]),
            ('official_test_plot',[PYTHON,'script/v76/plot_official_test.py','--input-dir',str(OUT/'official_test'),
                '--output',str(OUT/'official_test_frame20.png')]),
        ]
        for name,command in commands:
            state.update(stage=name,status='running',command=command)
            save()
            with (OUT/f'{name}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid
                save()
                code=child.wait()
            state['returncode']=code
            if code:
                raise RuntimeError(f'{name} exited {code}')
            state['completed_stages'].append(name)
            save()
        state.update(stage='completed_needs_review',status='completed',child_pid=None)
        save()
    except Exception as exc:
        state.update(status='failed',error=repr(exc),failure_ledger_delta='pending classification')
        save()
        raise


if __name__=='__main__':main()
