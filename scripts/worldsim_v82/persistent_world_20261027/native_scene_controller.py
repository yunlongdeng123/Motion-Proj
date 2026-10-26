"""Admitted scenes run independently of the remaining candidate quality checks."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from gpu_slot import exclusive
from resource_control import require_start,stop_requested
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().split()[21]
def write(p,d):p.write_text(json.dumps(d,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);a=p.parse_args()
    scene=ROOT/'motion_scene_inputs_v1'/a.context;run=ROOT/'native_scene_training_v1'/a.context;run.mkdir(parents=True,exist_ok=False)
    state=dict(context=a.context,pid=os.getpid(),start_ticks=ticks(os.getpid()),formal_steps=30000,save_every=5000,per_edit_retraining=False,phase='waiting_gpu')
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
