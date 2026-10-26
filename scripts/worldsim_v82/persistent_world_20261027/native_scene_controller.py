"""Admitted scenes run independently of the remaining candidate quality checks."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from gpu_slot import exclusive
from resource_control import require_start,stop_requested
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().split()[21]
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)

def same_process(pid,start_ticks):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split()
        return fields[2]!='Z' and fields[21]==str(start_ticks)
    except FileNotFoundError:return False

def dependency_result(path):
    """Give the previous scene its complete first diagnostic slot before training.

    A failed first attempt is preserved and explicitly yields the slot for useful
    training while the agent repairs it; it never counts as a completed test.
    """
    if not path.exists():return None
    try:d=json.loads(path.read_text())
    except json.JSONDecodeError:return None
    phase=d.get('phase','')
    if phase in ('single_request_diagnostics_complete_continuous_pending','diagnostics_complete'):
        assert d.get('completed_native_cases',0)>=d.get('expected_native_cases',11) and d.get('completed_backend_cases',0)>=d.get('expected_backend_cases',6)
        return dict(status='diagnostic_complete',snapshot=d)
    if phase=='engineering_repair_needed_preserved' and not same_process(d['pid'],d['start_ticks']):
        child=d.get('child')
        if not child or not same_process(child['pid'],child['start_ticks']):
            return dict(status='failed_attempt_preserved_agent_repair_pending',snapshot=d,not_a_completed_case=True)
    return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--after-motion');p.add_argument('--mask-first-count',type=int,default=0);p.add_argument('--resume-waiting',action='store_true');a=p.parse_args()
    scene=ROOT/'motion_scene_inputs_v1'/a.context;run=ROOT/'native_scene_training_v1'/a.context
    previous=None
    if a.resume_waiting:
        previous=json.loads((run/'status.json').read_text())
        assert previous['context']==a.context and previous['phase'] in ('waiting_gpu','waiting_previous_scene_diagnostics','waiting_new_scene_mask_batch')
        assert not previous.get('child') and not same_process(previous['pid'],previous['start_ticks'])
        assert not any((run/x).exists() for x in ['smoke_fresh','smoke_restore','formal30k'])
        assert not list(run.glob('*.log')),'Resume is only for a zero-work waiting controller'
        archive=run/f'waiting_controller_{previous["pid"]}_{previous["start_ticks"]}.json'
        assert not archive.exists();write(archive,previous)
    else:run.mkdir(parents=True,exist_ok=False)
    state=dict(context=a.context,pid=os.getpid(),start_ticks=ticks(os.getpid()),formal_steps=30000,save_every=5000,per_edit_retraining=False,phase='waiting_gpu',child=None,after_motion_context=a.after_motion,mask_first_count=a.mask_first_count,previous_waiting_controller=previous)
    def status(**kw):state.update(kw);write(run/'status.json',state)
    def launch(phase,cmd,env=None):
        require_start();status(phase=phase,command=cmd)
        with (run/f'{phase}.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);status(child=dict(pid=child.pid,start_ticks=ticks(child.pid)))
            sent=False
            while child.poll() is None:
                if stop_requested() and not sent:
                    child.terminate();sent=True;status(resource_stop_sent=True)
                time.sleep(3)
        status(last_returncode=child.returncode)
        assert child.returncode==0,f'{phase} failed; preserve scene/run/log for engineering repair'
        require_start()
    status()
    try:
        if a.after_motion:
            assert a.after_motion!=a.context
            dependency=ROOT/'motion_suite_execution_v1'/a.after_motion/'status.json'
            status(phase='waiting_previous_scene_diagnostics',dependency=str(dependency))
            while True:
                require_start();result=dependency_result(dependency)
                if result is not None:
                    write(run/'diagnostic_priority_release.json',dict(checked_at=time.time(),dependency=str(dependency),**result));break
                time.sleep(5)
            status(phase='waiting_gpu')
        if a.mask_first_count:
            masks=ROOT/'benchmark_mask_preparation_v1/status.json'
            status(phase='waiting_new_scene_mask_batch',dependency=str(masks))
            while True:
                require_start()
                if masks.exists():
                    d=json.loads(masks.read_text())
                    enough=len(d.get('completed',[]))>=a.mask_first_count
                    finished=d['phase']=='available_masks_complete_target_review_and_training_pending'
                    failed=bool(d.get('errors')) and not same_process(d['pid'],d['start_ticks'])
                    if enough or finished or failed:
                        write(run/'mask_batch_priority_release.json',dict(checked_at=time.time(),requested=a.mask_first_count,completed=len(d.get('completed',[])),state=d,partial_or_failed=not enough));break
                time.sleep(3)
            status(phase='waiting_gpu')
        with exclusive():
            require_start()
            sky=Path('/root/autodl-tmp/drivingeditor_repro/campaign/next_scene30k_20261012')
            settings=json.loads((sky/'cpu_setup_status.json').read_text());weights=settings['weights']
            env=dict(os.environ,PYTHONPATH=settings['sky_pythonpath'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            launch('sky',[settings['sky_python'],str(sky/'extract_sky.py'),'--scene',str(scene),'--config',weights['sky_config']['path'],'--checkpoint',weights['sky_checkpoint']['path']],env)
            write(scene/'native_preparation/sky_returned.json',dict(frames=49,source=str(sky/'extract_sky.py'),settings=weights,real_inference=True))
            base=[PY,str(ROOT/'code/native_scene_worker.py'),'--scene',str(scene)]
            launch('smoke_fresh',base+['--output',str(run/'smoke_fresh'),'--smoke-seconds','300'])
            receipt=json.loads((run/'smoke_fresh/returned.json').read_text());node=json.loads((run/'smoke_fresh/latest_complete_node.json').read_text())['iteration']
            assert receipt['reason']=='smoke_saved' and receipt['training_seconds_since_first_step']>=300
            launch('smoke_restore',base+['--output',str(run/'smoke_restore'),'--smoke-seconds','300','--resume',str(run/'smoke_fresh'/f'chkpnt{node}.pth')])
            restored=json.loads((run/'smoke_restore/restore_verification.json').read_text());receipt2=json.loads((run/'smoke_restore/returned.json').read_text())
            assert restored['all_passed'] and receipt2['training_seconds_since_first_step']>=300 and receipt2['iteration']>node
            write(run/'smoke_gate.json',dict(status='real_gpu_save_eval_strict_restore_and_next_stage_pass',fresh_extra_updates=receipt['observed_finite_step_calls'],restore_extra_updates=receipt2['observed_finite_step_calls'],real_seconds=receipt['training_seconds_since_first_step']+receipt2['training_seconds_since_first_step'],formal_updates=0))
            launch('formal30k',base+['--output',str(run/'formal30k')])
            status(phase='formal30k_complete_evaluation_and_edits_pending',child=None,quality_accepted=False,human_verdict=None)
    except Exception as e:
        status(phase='resource_stop' if stop_requested() else 'engineering_failure_preserved',error=repr(e));raise

if __name__=='__main__':main()
