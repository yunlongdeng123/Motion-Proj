"""Complete the existing10K trajectory; pilot native KEEP only, no future training."""
from pathlib import Path
import json,sys,os,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parent.parent
API=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit')
def read(p):return json.loads(p.read_text())
def write(p,v):
    temporary=p.with_name(p.name+'.tmp');temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(p)
def status(stage,**kw):write(ROOT/'formal10k_keep_lifecycle_status.json',dict(stage=stage,controller_pid=os.getpid(),unix=time.time(),
    training_launches_allowed=False,all49_auto_release=False,shutdown=False,reset_card=False,**kw))
def main():
    pid=read(ROOT/'formal10k_launch_receipt.json')['pid'];model=ROOT/'formal10k'
    status('WAIT_EXISTING_EXACT5K_TO10K',existing_trainer_pid=pid)
    while not (model/'returned.json').exists():
        proc=Path(f'/proc/{pid}')
        if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':raise RuntimeError('10K trainer exited before complete checkpoint')
        time.sleep(2)
    returned=read(model/'returned.json');audit=read(model/'restore_audit.json')
    assert returned['status']=='PASS_COMPLETE_BOUNDARY' and returned['source_iteration']==5000 and returned['iteration']==10000
    assert returned['actual_additional_iterations']==5000 and audit['status']=='PASS_EXACT_FULL_RESTORE' and audit['differences']==[]
    rows=[]
    for tag in ['formal5k','formal10k']:
        rows.extend(json.loads(line) for line in (ROOT/tag/'paired_steps.jsonl').read_text().splitlines() if line.strip())
    assert [row['iteration'] for row in rows]==list(range(1,10001))
    label='formal10k_keep_pilot';out=ROOT/label
    with (ROOT/(label+'.log')).open('x') as log:
        proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'source/evaluate_reconstruction.py'),'--model',str(model),
            '--iteration','10000','--output',str(out),'--scope','pilot'],stdout=log,stderr=subprocess.STDOUT)
        status('KEEP_RECONSTRUCTION_PILOT97_105_109',child_pid=proc.pid);assert proc.wait()==0,proc.returncode
    pilot=read(out/'summary.json')
    assert pilot['status']=='PASS_NATIVE_KEEP_RECONSTRUCTION_PILOT' and pilot['frame_count']==3 and pilot['raw_ids']==[97,105,109]
    assert pilot['optimization_updates']==0 and pilot['delete_actor_ids']==[] and pilot['parameters_unchanged']
    assert pilot['source_integrity']=='PASS_UNCHANGED_BYTES_MTIME_SHA256' and pilot['identity_partition_max_abs_error']<=1e-5
    assert pilot['full49_fixed_reference']['full_pixels']==121128 and pilot['full49_fixed_reference']['upper_pixels']==25906
    assert set(pilot['current_owner_rows'])=={'5','6','18','26','37'} and all(v>0 for v in pilot['current_owner_rows'].values())
    manifest=read(model/'runtime_source_manifest.json');changed=[];aux=[]
    for name,record in manifest['files'].items():
        p=Path(name)
        same=p.is_file() and p.stat().st_size==record['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
        if not same:
            if p.parent==API and p.name not in ['__init__.py','ownership.py','render.py']:
                aux.append(dict(path=name,scope='Unimported new CLI/scene IO file; explicitly permitted by root; immutable API3core checked separately'))
            else:changed.append(name)
    assert not changed,changed
    gate=dict(status='PASS_COMPLETE_FORMAL10000_AND_NATIVE_KEEP_RECONSTRUCTION_PILOT',formal_total_updates=10000,
        phase_new_updates=5000,new_engineering_updates_separate=4683,old_engineering6599_unchanged=True,
        contiguous_training_iterations=[1,10000],full_restore_differences=[],pilot_frames=[97,105,109],
        all49_evaluated=False,all49_requires_body_image_review=True,body_quality_verdict=None,
        active_native_and_api3core_changed=[],unimported_auxiliary_api_changes=aux,
        identity_current_rows=pilot['current_owner_rows'],source_integrity=pilot['source_integrity'],
        no_future_training_release=True,shutdown=False,reset_card=False)
    write(ROOT/'formal10k_keep_pilot_gate.json',gate)
    status('PASS_10K_KEEP_PILOT_AWAIT_BODY_VISUAL_REVIEW',formal_total=10000)
if __name__=='__main__':
    try:main()
    except BaseException as error:status('FAILED10K_KEEP_CLOSEOUT',error=repr(error));raise
