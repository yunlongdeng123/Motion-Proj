"""Own one existing GPU smoke, then pilot and exact-restore two updates only."""
from pathlib import Path
import json,os,sys,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable
def read(p):return json.loads(p.read_text())
def write(p,v):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def status(stage,**kw):write(ROOT/'smoke_lifecycle_status.json',dict(stage=stage,controller_pid=os.getpid(),unix=time.time(),
    shutdown=False,reset_card=False,formal_auto_release=False,**kw))
def run(label,*args):
    with (ROOT/(label+'.log')).open('x') as log:
        proc=subprocess.Popen([PY,'-B',*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
        status(label,child_pid=proc.pid,log=str(ROOT/(label+'.log')))
        assert proc.wait()==0,(label,proc.returncode)
def main():
    launch=read(ROOT/'smoke_launch_receipt.json');pid=launch['pid']
    assert pid==11522 and launch['dynamic_actor_ids']==[5,6,18,26,37]
    status('WAIT_EXISTING_LONG540_SMOKE',existing_train_pid=pid)
    smoke=ROOT/'smoke'
    while not (smoke/'returned.json').exists():
        proc=Path(f'/proc/{pid}')
        if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':raise RuntimeError('Smoke exited before complete bundle')
        time.sleep(2)
    result=read(smoke/'returned.json');step=result['iteration']
    assert result['status']=='PASS_COMPLETE_BOUNDARY' and result['mode']=='smoke' and result['elapsed_seconds']>=540
    assert result['source_iteration']==0 and result['actual_additional_iterations']==step
    event=smoke/f'checkpoint_events/iteration_{step:06d}.json';assert event.is_file()
    for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']:assert (smoke/f'{prefix}{step}.pth').is_file()
    run('smoke_pilot',ROOT/'source/evaluate_scene.py','--model',smoke,'--iteration',step,'--output',ROOT/'smoke_pilot','--scope','pilot')
    pilot=read(ROOT/'smoke_pilot/summary.json')
    assert pilot['status']=='PASS_NATIVE_IDENTITY_EDIT_PILOT_ENGINEERING' and pilot['frame_count']==3 and pilot['raw_ids']==[105,130,137]
    assert pilot['optimization_updates']==0 and not pilot['optimizer_created']
    assert pilot['source_integrity']=='PASS_UNCHANGED_BYTES_MTIME_SHA256' and pilot['parameters_unchanged']
    assert pilot['keep_native_max_abs_difference']<=1e-6 and pilot['keep_again_max_abs_difference']<=1e-6
    assert pilot['cuda_joint_target_contribution_zero'] and pilot['visible_owner37_witness']
    assert pilot['queries']==dict(keep=[],delete_a37=[37],delete_a6=[6],delete_a6_a37=[6,37],keep_again=[])
    run('smoke_resume2step',ROOT/'source/train_full5.py','--mode','resume2step','--checkpoint',smoke/f'chkpnt{step}.pth','--output',ROOT/'smoke_resume2step')
    audit=read(ROOT/'smoke_resume2step/restore_audit.json');entry=read(ROOT/'smoke_resume2step/resume_entry.json')
    assert audit['status']=='PASS_EXACT_FULL_RESTORE' and audit['differences']==[]
    assert entry['status']=='PASS_NEXT_TRAINING_ENTRY' and entry['completed_extra_updates']==2 and entry['entered_iteration']==step+3
    manifest=read(smoke/'runtime_source_manifest.json')
    changed=[p for p,r in manifest['files'].items() if Path(p).stat().st_size!=r['bytes'] or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=r['sha256']]
    assert not changed,changed
    gate=dict(status='PASS_REAL_LONG540_FULL_DYNAMIC_SAVE_PILOT_FULL_RESTORE_NEXT_ENTRY',smoke_iteration=step,
        smoke_updates=step,resume_extra_updates=2,total_new_engineering_updates=step+2,formal_updates=0,
        actual_smoke_train_seconds=result['elapsed_seconds'],dynamic_actor_ids=[5,6,18,26,37],
        tested_queries=pilot['queries'],pilot_frames=[105,130,137],all49_evaluated=False,
        exact_full_restore_differences=[],source_manifest_changed=[],
        quality_accepted=False,visual_review_required=True,formal_auto_release=False,
        shutdown=False,reset_card=False)
    write(ROOT/'smoke_lifecycle_gate.json',gate)
    status('PASS_SMOKE_LIFECYCLE_READY_FOR_OWNER_FRESH5K_GATE',engineering_updates=step+2)
if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED_SMOKE_LIFECYCLE',error=repr(error));raise
