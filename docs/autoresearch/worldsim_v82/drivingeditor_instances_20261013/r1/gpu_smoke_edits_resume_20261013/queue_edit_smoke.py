"""One-shot serial real GMC edit export; stop on error or priority work."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
ROOT = BASE / 'edits_smoke_300_r1'
TRACKS = ('NoQttdyd4C16Y_zkGhiFKg', 'fXVhuN-RuJDvZwnkQ-0YSQ')

def write_status(value):
    value.update(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat())
    temporary = ROOT / 'status.json.partial'
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, ROOT / 'status.json')

def main():
    completed = []
    try:
        for actor, track in enumerate(TRACKS):
            for operation in ('delete', 'translate', 'rotate'):
                name = 'actor%d_%s' % (actor, operation)
                command = ['nice', '-n', '10', sys.executable, '-B', '-u', str(BASE/'infra/render_edits.py'),
                    '--spec', str(BASE/'specs/scene0008_023_071.json'),
                    '--checkpoint', str(BASE/'queue_gmc_r1/gpu_smoke/checkpoint_000300.pt'),
                    '--track', track, '--operation', operation, '--output', str(ROOT/name), '--run']
                if operation == 'translate': command += ['--translate-local-m', '0.5', '0', '0']
                if operation == 'rotate': command += ['--rotate-degrees', '30']
                with (ROOT/(name+'.log')).open('x', encoding='utf-8') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'))
                    write_status({'state':'exporting', 'current':name, 'child_pid':child.pid,
                        'command':command, 'completed':completed, 'other_tasks_modified':False})
                    code = child.wait()
                if code != 0:
                    write_status({'state':'stopped_on_export_error_or_priority', 'current':name,
                        'returncode':code, 'completed':completed, 'assets_preserved':True})
                    return code
                receipt = json.loads((ROOT/name/'edit_receipt.json').read_text(encoding='utf-8'))
                if len(receipt['frames']) != 49:
                    raise ValueError('Expected all 49 actual observed camera/time pairs')
                completed.append(name)
        write_status({'state':'SIX_REAL_EDIT_EXPORTS_COMPLETE_REVIEW_PENDING', 'completed':completed,
            'human_quality_verdict':None, 'other_tasks_modified':False})
        return 0
    except BaseException as error:
        write_status({'state':'engineering_failed', 'error':repr(error), 'completed':completed, 'assets_preserved':True})
        raise

if __name__ == '__main__':
    if '--launch' in sys.argv:
        if ROOT.exists(): raise FileExistsError('Refuse duplicate controller/output')
        ROOT.mkdir(parents=True)
        command=['nice','-n','10',sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker']
        with (ROOT/'controller.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),start_new_session=True)
        launch={'pid':child.pid,'command':command,'utc':datetime.now(timezone.utc).isoformat(),
            'scope':'GMC two actors x delete/0.5m local translation/30deg local yaw x 49 actual frames',
            'GPU_use_in_parent':False,'serial_GPU_exports':True,'other_tasks_modified':False}
        (ROOT/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
        print(json.dumps(launch))
    elif '--worker' in sys.argv:
        raise SystemExit(main())
    else:
        raise SystemExit('Explicit --launch required')
