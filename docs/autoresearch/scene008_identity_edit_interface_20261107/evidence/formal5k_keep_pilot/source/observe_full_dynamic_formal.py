"""Observe one authorized native trainer, then pilot only; no training launch."""
from pathlib import Path
import argparse,json,os,sys,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable
ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['formal5k','formal10k'],required=True);opt=ap.parse_args()
phase=opt.phase;end=5000 if phase=='formal5k' else 10000;start=0 if end==5000 else 5000
def read(p):return json.loads(p.read_text())
def write(p,v):
    temporary=p.with_name(p.name+'.tmp');temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(p)
def status(stage,**kw):write(ROOT/(phase+'_lifecycle_status.json'),dict(stage=stage,controller_pid=os.getpid(),unix=time.time(),
    phase=phase,training_launches_allowed=False,shutdown=False,reset_card=False,**kw))
def main():
    launch=read(ROOT/(phase+'_launch_receipt.json'));pid=launch['pid'];model=ROOT/phase
    status('WAIT_EXISTING_FORMAL_COMPLETE_BOUNDARY',existing_trainer_pid=pid)
    while not (model/'returned.json').exists():
        proc=Path(f'/proc/{pid}')
        if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':raise RuntimeError('Formal trainer exited without complete node')
        time.sleep(2)
    result=read(model/'returned.json')
    assert result['status']=='PASS_COMPLETE_BOUNDARY' and result['source_iteration']==start
    assert result['iteration']==end and result['actual_additional_iterations']==5000
    assert (model/f'checkpoint_events/iteration_{end:06d}.json').is_file()
    args=read(model/'effective_config.json')
    assert args['iterations']==args['position_lr_max_steps']==30000
    assert args['dynamic_actor_ids']==[5,6,18,26,37] and args['delete_actor_ids']==[]
    if start:
        audit=read(model/'restore_audit.json');assert audit['status']=='PASS_EXACT_FULL_RESTORE' and audit['differences']==[]
    label=phase+'_pilot'
    with (ROOT/(label+'.log')).open('x') as log:
        proc=subprocess.Popen([PY,'-B',str(ROOT/'source/evaluate_scene.py'),'--model',str(model),
            '--iteration',str(end),'--output',str(ROOT/label),'--scope','pilot'],stdout=log,stderr=subprocess.STDOUT)
        status('PILOT105_130_137',child_pid=proc.pid);assert proc.wait()==0,proc.returncode
    pilot=read(ROOT/label/'summary.json')
    assert pilot['status']=='PASS_NATIVE_IDENTITY_EDIT_PILOT_ENGINEERING' and pilot['frame_count']==3
    assert pilot['source_integrity']=='PASS_UNCHANGED_BYTES_MTIME_SHA256'
    assert pilot['parameters_unchanged'] and pilot['keep_native_max_abs_difference']<=1e-6 and pilot['keep_again_max_abs_difference']<=1e-6
    assert pilot['cuda_joint_target_contribution_zero'] and pilot['visible_owner37_witness']
    steps=[json.loads(line) for line in (model/'paired_steps.jsonl').read_text().splitlines() if line.strip()]
    assert [r['iteration'] for r in steps]==list(range(start+1,end+1))
    if not start:
        smoke=[json.loads(line) for line in (ROOT/'smoke/paired_steps.jsonl').read_text().splitlines() if line.strip()]
        fields=['iteration','frame_id','time_shift','learning_rates']
        assert [{k:r[k] for k in fields} for r in smoke]==[{k:r[k] for k in fields} for r in steps[:len(smoke)]]
    manifest=read(model/'runtime_source_manifest.json')
    changed=[p for p,r in manifest['files'].items() if Path(p).stat().st_size!=r['bytes'] or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=r['sha256']]
    assert not changed,changed
    gate=dict(status='PASS_COMPLETE_FORMAL_NODE_AND_THREE_FRAME_QUERY_ENGINEERING',iteration=end,
        phase_source_iteration=start,phase_new_formal_updates=5000,total_new_formal_updates=end,
        new_engineering_updates_separate=4683,pilot_frames=[105,130,137],all49_evaluated=False,
        source_changed=[],parameters_unchanged=True,quality_accepted=False,visual_review_required=True,
        future_training_auto_release=False,shutdown=False,reset_card=False)
    write(ROOT/(phase+'_pilot_gate.json'),gate)
    status('PASS_FORMAL_NODE_AND_PILOT_AWAIT_OWNER_VISUAL_REVIEW',formal_total=end)
if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED_FORMAL_PILOT',error=repr(error));raise
