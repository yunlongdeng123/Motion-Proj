from pathlib import Path
import json,os,subprocess,sys,time,hashlib
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable
ENGINE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def status(stage,**extra):write(ROOT/'queue_status.json',dict(stage=stage,controller_pid=os.getpid(),updated_unix=time.time(),shutdown=False,**extra))
def run(label,*args):
    with (ROOT/(label+'.log')).open('x') as log:
        p=subprocess.Popen([PY,'-B',*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
        status(label,pid=p.pid,log=str(ROOT/(label+'.log')));assert p.wait()==0,(label,p.returncode)
def evaluate(step,model):
    label=f'a5_evaluation_{step//1000}k'
    run(label,ROOT/'source/evaluate_a5.py','--model',model,'--iteration',step,'--output',ROOT/label,
        '--engine',ENGINE,'--dynamic-aliases','37','--delete-name','delete_a37')
    s=json.loads((ROOT/label/'summary.json').read_text())
    assert s['status']=='PASS_FRESH_NATIVE49_A5_EVALUATION' and s['frame_count']==49 and s['fixed_denominator']['pixels']==121128
    assert s['optimization_updates']==0 and s['replay']['complete_all49_keep_delete_replay'] and s['replay']['max_uint8_difference']<=1
    assert s['source_integrity']=='PASS_UNCHANGED_BYTES_MTIME_SHA256'
    return s
def verify_sources():
    engine=json.loads((ROOT.parent/'engine_source_manifest.json').read_text())
    inputs=json.loads((ROOT/'input_source_manifest.json').read_text())
    changes=[]
    for name,receipt in dict(engine,**inputs).items():
        p=Path(name);stat=p.stat()
        if stat.st_size!=receipt['bytes'] or stat.st_mtime_ns!=receipt['mtime_ns'] or hashlib.sha256(p.read_bytes()).hexdigest()!=receipt['sha256']:
            changes.append(name)
    if changes:raise RuntimeError('protected source changed: '+repr(changes))
    return dict(status='PASS_UNCHANGED_BYTES_MTIME_SHA256',files=len(engine)+len(inputs))

def paired_check(step):
    baseline=Path(json.loads((ROOT/'arm_contract.json').read_text())['baseline_root'])
    phases=['formal1k']+(['formal5k'] if step==5000 else [])
    keys=['iteration','frame_id','time_shift','learning_rates']
    def records(root):
        rows=[]
        for phase in phases:
            for line in (root/phase/'paired_steps.jsonl').read_text().splitlines():
                d=json.loads(line);rows.append({k:d[k] for k in keys})
        return rows
    old,new=records(baseline),records(ROOT)
    assert len(old)==len(new)==step
    mismatches=[i+1 for i,(a,b) in enumerate(zip(old,new)) if a!=b]
    result=dict(status='PASS_EXACT_CAMERA_SHIFT_LR_MATCH' if not mismatches else 'FAIL',
        baseline=str(baseline),updates=step,compared_fields=keys,mismatch_iterations=mismatches)
    write(ROOT/f'paired_camera_shift_lr_{step}.json',result)
    assert not mismatches,mismatches[:10]

def main():
    assert ROOT.name in ['control','support'] and ROOT.parent.name=='scene008_a5_static_rgb5_ablation_20261107_r2'
    write(ROOT/'protected_sources_before.json',verify_sources())
    run('formal1k',ROOT/'source/train_loss_arm.py','--mode','formal','--stop-iteration','1000','--output',ROOT/'formal1k')
    first=json.loads((ROOT/'formal1k/returned.json').read_text());assert first['source_iteration']==0 and first['iteration']==1000 and first['actual_additional_iterations']==1000
    paired_check(1000)
    estimate=first['elapsed_seconds']*5+300
    write(ROOT/'real_lifecycle_smoke.json',dict(status='PASS_1K_REAL_SAVE_EVAL_ENTRY',
        observed_1k_seconds=first['elapsed_seconds'],conservative_5k_seconds=estimate,
        long_experiment_threshold_seconds=3600,first_1k_is_formal=True,extra_engineering_updates=0,
        long_smoke_covered=first['elapsed_seconds']>=540,queue_wait_in_estimate=False))
    assert estimate<3600 or first['elapsed_seconds']>=540,'Long-smoke requirement not covered'
    early=evaluate(1000,ROOT/'formal1k')
    write(ROOT/'preview_1k_ready.json',dict(status='READY_FRESH1K_FOUR_ANCHORS',arm=ROOT.name,iteration=1000,
        summary=str(ROOT/'a5_evaluation_1k/summary.json'),all49=early['all49'],
        anchors={str(r):str(ROOT/f'a5_evaluation_1k/anchors/{r:03d}_crop.png') for r in [89,97,105,109]}))
    run('formal5k',ROOT/'source/train_loss_arm.py','--mode','extend','--checkpoint',ROOT/'formal1k/chkpnt1000.pth','--stop-iteration','5000','--output',ROOT/'formal5k')
    last=json.loads((ROOT/'formal5k/returned.json').read_text());audit=json.loads((ROOT/'formal5k/restore_audit.json').read_text())
    assert last['source_iteration']==1000 and last['iteration']==5000 and last['actual_additional_iterations']==4000
    assert audit['status']=='PASS_EXACT_FULL_RESTORE' and not audit['differences']
    paired_check(5000)
    final=evaluate(5000,ROOT/'formal5k')
    write(ROOT/'protected_sources_after.json',verify_sources())
    status('COMPLETE_STATIC_RGB5_FRESH5K_ABLATION_ARM',arm=ROOT.name,formal_updates=5000,learning_rate_horizon=30000,
        all49=final['all49'],quality_accepted=False,human_verdict=None)
if __name__=='__main__':
    try:main()
    except Exception as e:status('FAILED',error=repr(e));raise
