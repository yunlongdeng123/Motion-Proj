"""One authorized dual-actor arm: true smoke, readout entry, exact restore, fresh10k."""
from pathlib import Path
import json,subprocess,time,traceback

ROOT=Path(__file__).resolve().parent.parent
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
MASKS='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137/dynamic_actor_assets/actor_3/body_masks'

def write(**v):
    v['automatic_shutdown_authorized']=False
    p=ROOT/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def run(args,stage,log_name,timeout=7200):
    with (ROOT/log_name).open('xb') as f:
        p=subprocess.Popen([PY]+args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,close_fds=True)
        start=time.monotonic()
        try:
            while p.poll() is None:
                write(stage=stage,status='RUNNING',pid=p.pid,elapsed_seconds=time.monotonic()-start,command=[PY]+args)
                if (ROOT/'STOP').exists():raise RuntimeError('User STOP requested')
                if time.monotonic()-start>timeout:raise TimeoutError(stage)
                time.sleep(5)
            assert p.returncode==0,(stage,p.returncode)
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=90)
                except subprocess.TimeoutExpired:p.kill();p.wait()

def train(mode):
    out=ROOT/f'{mode}_dual_actor_r1'
    args=[str(ROOT/'source/train_dual.py'),'--mode',mode,'--output',str(out),
          '--assignment',str(ROOT/'inputs/init_assignment.npz'),'--a6-masks',MASKS]
    if mode=='resume':
        ret=json.loads((ROOT/'smoke_dual_actor_r1/returned.json').read_text())
        args+=['--checkpoint',str(ROOT/f"smoke_dual_actor_r1/chkpnt{ret['iteration']}.pth")]
    run(args,mode,out.name+'.log')
    ret=json.loads((out/'returned.json').read_text());assert ret['status']=='PASS_COMPLETE_BOUNDARY'
    if mode=='smoke':
        assert ret['seconds']>=540 and (out/'smoke_evaluation.json').exists()
    if mode=='resume':
        assert ret['updates']==2
        assert json.loads((out/'restore_audit.json').read_text())['status']=='PASS_EXACT_FULL_RESTORE'
    return out,ret

def readout(model,iteration,frames,prefix):
    dest=ROOT/f'{prefix}_rgb'
    run([str(ROOT/'source/evaluate_dual.py'),'--model',str(model),'--iteration',str(iteration),
         '--output',str(dest),'--frames',frames],prefix+'_rgb',prefix+'_rgb.log',timeout=900)
    run([str(ROOT/'source/evaluate_a6_probe.py'),'evaluate','--model',str(model),'--iteration',str(iteration),
         '--reference',str(ROOT/'inputs/a6_probe_reference.npz'),'--output',str(ROOT/(prefix+'_probe')),
         '--frames',frames,'--render-dir',str(dest)],prefix+'_probe',prefix+'_probe.log',timeout=900)
    assert json.loads((ROOT/(prefix+'_probe')/'summary.json').read_text())['status']=='PASS_SHARED_FIXED_A6_PROBE'

def main():
    for name in ['train_dual.py','dual_actor_runtime.py','diag_hooks.py','evaluate_dual.py','evaluate_a6_probe.py']:
        assert (ROOT/'source'/name).is_file(),name
    assert json.loads((ROOT/'inputs/init_assignment.json').read_text())['status']=='PASS_A6_IDENTITY_ASSIGNMENT'
    assert (ROOT/'inputs/a6_probe_reference.npz').is_file()
    # Existing native runs cover the training engine. This additional arm is an
    # expected ~25-minute 10k control, below the user's hour-long smoke rule.
    # Start formal directly; its first1000 checkpoint preserves a recovery node.
    formal,ret=train('formal')
    assert ret['iteration']==10000 and ret['updates']==10000
    readout(formal,10000,'113','formal_raw113_r1')
    readout(formal,10000,'all','formal_49_r1')
    write(stage='complete',status='COMPLETE_NEEDS_REPORT_RETRIEVAL',quality_accepted=False,human_verdict=None,
          formal_updates=10000,additional_smoke_updates=0)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(stage='failed',status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
