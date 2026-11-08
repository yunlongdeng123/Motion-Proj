"""只完成 CPU 数据接线；此控制器没有 GPU 阶段或电源操作。"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent


def status(stage, **fields):
    (ROOT/'cpu_status.json').write_text(json.dumps(dict(stage=stage,unix=time.time(),
        GPU_queries=0,training_updates=0,render_calls=0,**fields),indent=2)+'\n')


def run(name, *args):
    status(name)
    with (ROOT/f'{name}.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'source'/f'{name}.py'),*map(str,args)],
                       stdout=log,stderr=subprocess.STDOUT,env=os.environ,check=True)


def main():
    deadline=time.monotonic()+2700
    while True:
        p=ROOT/'inputs/receipt.json'
        if p.exists() and json.loads(p.read_text())['status']=='PASS_CPU_OBSERVATIONS':
            break
        if time.monotonic()>deadline:
            raise TimeoutError('CPU extraction did not finish within45min; no GPU continuation')
        status('WAITING_FOR_CPU_EXTRACTION')
        time.sleep(5)
    run('verify_cpu_bank','--bank',ROOT/'inputs','--selection',ROOT/'selected_case.json')
    run('build_point_prompts','--bank',ROOT/'inputs')
    run('inspect_background','--bank',ROOT/'inputs','--selection',ROOT/'selected_case.json')
    selection=json.loads((ROOT/'selected_case.json').read_text())
    frames=[]
    for raw in selection['build']:
        prompts=json.loads((ROOT/'inputs/point_prompts'/f'{raw:03d}.json').read_text())
        target=prompts.get(selection['target']['track_id'])
        assert target is not None,raw
        frames.append(dict(raw=raw,positive_returns=target['positive_returns'],
                           positive_points=len(target['positive']),negative_points=len(target['negative'])))
    (ROOT/'inputs/point_prompt_verification.json').write_text(json.dumps(dict(
        status='PASS_CPU_POINT_PROMPT_INPUTS',policy='same-frame OBB unique body returns, camera z-buffer, other-box ray exclusion',
        frames=frames,target_positive_frames=sum(r['positive_points']>0 for r in frames),
        mask_or_owner_ground_truth=False,segmentation_run=False,GPU_queries=0),indent=2))
    status('CPU_PREPARATION_COMPLETE_GPU_NOT_STARTED',build_frames=len(selection['build']),
           holdout_frames=len(selection['holdout']),next_requires='four-frame segmentation QA and real600s lifecycle smoke',
           training_ready=False)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('CPU_PREPARATION_FAILED',exception=repr(exc))
        raise
