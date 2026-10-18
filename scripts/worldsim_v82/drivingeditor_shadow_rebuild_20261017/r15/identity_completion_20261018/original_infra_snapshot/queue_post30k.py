"""One-shot serial restore/evaluation/edit export. No training or auto retries."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
POST=BASE/'post30k_r1'
RUN=BASE/'runs/gmc_two_actor_fixed30k_r1'
TRACKS=('NoQttdyd4C16Y_zkGhiFKg','fXVhuN-RuJDvZwnkQ-0YSQ')

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def status(value):
    value.update(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat())
    path=POST/'status.json.partial'
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(path,POST/'status.json')

def stages():
    for step in (10000,20000,30000):
        name='restore_%06d'%step
        yield name,[str(BASE/'infra/check_full_restore.py'),'--checkpoint',str(RUN/('checkpoint_%06d.pt'%step)),
            '--expected-step',str(step),'--output',str(POST/name)]
    yield 'scene_evaluation',[str(BASE/'infra/evaluate_scene.py'),'--spec',str(BASE/'specs/scene0008_023_071.json'),
        '--checkpoint',str(RUN/'checkpoint_030000.pt'),'--expected-step','30000',
        '--fixed-camera-frame','0008033','--output',str(POST/'evaluation'),'--run']
    for actor,track in enumerate(TRACKS):
        for operation in ('delete','translate','rotate'):
            name='actor%d_%s'%(actor,operation)
            command=[str(BASE/'infra/render_edits.py'),'--spec',str(BASE/'specs/scene0008_023_071.json'),
                '--checkpoint',str(RUN/'checkpoint_030000.pt'),'--track',track,'--operation',operation,
                '--output',str(POST/'edits'/name),'--run']
            if operation=='translate':command+=['--translate-local-m','0.5','0','0']
            if operation=='rotate':command+=['--rotate-degrees','30']
            yield name,command

def worker():
    completed=[]
    try:
        for name,arguments in stages():
            command=['nice','-n','10',sys.executable,'-B','-u']+arguments
            with (POST/(name+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
                status({'state':'running_serial_postprocessing','current':name,'child_pid':child.pid,
                    'command':command,'completed':completed,'optimizer_updates':0,'other_tasks_modified':False})
                code=child.wait()
            if code:
                status({'state':'stopped_on_engineering_error_or_priority','current':name,'returncode':code,
                    'completed':completed,'assets_preserved':True,'automatic_retries':False})
                return code
            completed.append(name)
        manifest=read(POST/'evaluation/scene_evaluation_manifest.json')
        if manifest['status']!='SCENE_EVALUATION_EXPORTED':raise ValueError('Missing actual full evaluation')
        if any(len(read(POST/'edits'/name/'edit_receipt.json')['frames'])!=49 for name in completed if name.startswith('actor')):
            raise ValueError('Incomplete actual49 edit receipt')
        status({'state':'POST30K_RESTORE_EVALUATION_AND_SIX_EDITS_EXPORTED_REVIEW_PENDING','completed':completed,
            'actual_restored_steps':[10000,20000,30000],'human_quality_verdict':None,
            'optimizer_updates':0,'other_tasks_modified':False})
        return 0
    except BaseException as error:
        status({'state':'engineering_failed','error':repr(error),'completed':completed,'assets_preserved':True})
        raise

if __name__=='__main__':
    if '--worker' in sys.argv:raise SystemExit(worker())
    if '--launch' not in sys.argv:raise SystemExit('Explicit --launch required')
    training=read(RUN/'status.json')
    if training.get('state')!='complete' or training.get('step')!=30000:raise ValueError('Actual complete 30k required')
    if Path('/proc/37747').exists():raise RuntimeError('Recorded training PID still exists; inspect without killing')
    if POST.exists():raise FileExistsError('Refuse duplicate controller/output')
    POST.mkdir()
    command=['nice','-n','10',sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker']
    with (POST/'controller.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),start_new_session=True)
    receipt={'pid':child.pid,'command':command,'utc':datetime.now(timezone.utc).isoformat(),
        'scope':'three_actual_GPU_state_restores; 49observed+49fixed evaluation; 2actors x3edits x49observed',
        'serial_GPU_only':True,'parent_GPU_use':False,'automatic_retries':False,'optimizer_updates':0,'other_tasks_modified':False}
    (POST/'launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))
