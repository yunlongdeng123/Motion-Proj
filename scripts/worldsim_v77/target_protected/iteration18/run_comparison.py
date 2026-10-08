"""Resume one r51 comparison chain; never overlap two GPU models."""
import os, json, time, subprocess, pathlib, fcntl

ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r51')
OUT=ROOT/'r47_comparison'
SCRIPT=pathlib.Path(__file__).with_name('compare_r47.py')
ENVS=pathlib.Path('/root/autodl-tmp/envs')

def read(p): return json.loads(p.read_text()) if p.exists() else {}
def state(stage, **kw):
    p=OUT/'chain_state.json'; tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw),indent=2)+'\n');tmp.replace(p)
def alive(pid):
    p=pathlib.Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'
def run(phase,env):
    state(phase)
    with (OUT/f'{phase}.log').open('a') as log:
        subprocess.run([str(ENVS/env/'bin/python'),'-u',str(SCRIPT),phase],
            stdout=log,stderr=subprocess.STDOUT,check=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))

def main():
    lock=(OUT/'chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state('waiting_extract')
    while read(OUT/'extract_state.json').get('stage')!='complete':
        assert alive(read(OUT/'extract_launcher.json')['pid']),'Extraction stopped before completion'
        time.sleep(10)
    run('prepare','motionproj')
    prepared=read(OUT/'prepare_state.json')
    assert not prepared['pending'] and prepared['completed']==72,prepared
    state('waiting_baseline_GPU')
    while read(ROOT/'drive_state.json').get('stage')!='DELETE_complete_pending_structure_review':
        assert alive(read(ROOT/'baseline_launcher.json')['pid']),'Baseline stopped before completion'
        time.sleep(10)
    while alive(read(ROOT/'baseline_launcher.json')['pid']):time.sleep(2)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not processes,f'Unexpected active GPU process: {processes}'
    run('source-masks','worldsim-v77-sam2')
    run('evaluate','driveeditor')
    result=read(OUT/'evaluation_state.json')
    assert result['stage']=='complete' and len(result['completed'])==72
    state('GPU_complete',GPU_jobs=0,windows=72)

if __name__=='__main__':
    try:main()
    except Exception as error:
        state('error',error=repr(error));raise
