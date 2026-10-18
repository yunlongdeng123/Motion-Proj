"""One finite queue after the active GN11 worker; no formal instance start.

One corrected 300-step control, two restore/loss checks (six disposable audit
updates total), five full49 branch/deletion exports. Failure preserves assets.
"""
from pathlib import Path
import datetime, json, os, subprocess, sys, time, traceback

OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def state(value, **kw):
    target = OUT / 'review_queue_status.json'
    temporary = target.with_suffix('.json.partial')
    temporary.write_text(json.dumps(dict(state=value, at=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(), **kw), indent=2))
    temporary.replace(target)


def stage(name, cmd, completed):
    state('ENGINEERING_REVIEW_RUNNING', stage=name, completed=completed,
        formal_instance_updates=0, command=cmd)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TORCH_HOME='/root/autodl-tmp/drivingeditor_repro/cache/torch')
    with (OUT / ('review_' + name + '.log')).open('x') as log:
        subprocess.run([sys.executable, '-B', '-u', *cmd], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    completed.append(name)


def main():
    marker = OUT / 'review_queue_started.json'
    if marker.exists():
        raise FileExistsError('no duplicate queue, restart, or unregistered extra budget')
    assert read(OUT / 'stage_binding_receipt.json')['status'] == 'PASSED_ACTUAL_GN11_RIGID_STAGE_BINDING'
    assert read(OUT / 'isolated_control_contract.json')['maximum_new_pilot_updates'] == 300
    marker.write_text(json.dumps(dict(pid=os.getpid(), maximum_new_control_updates=300,
        disposable_audit_updates=6, formal_instance_updates=0)))
    start = time.monotonic()
    complete_state = 'MATCHED_FIVE_GN11_30K_COMPLETE_INSTANCE_PHASE_GATED_FOR_REVIEW'
    while True:
        current = read(OUT / 'full_queue_status.json')
        if current['state'] == complete_state:
            break
        if current['state'].startswith('FAILED'):
            raise RuntimeError('GN11 failure; do not consume further GPU budget')
        if time.monotonic() - start > 6 * 3600:
            raise TimeoutError('finite six-hour GN11 wait exceeded; no automatic restart')
        state('WAITING_FOR_ACTIVE_MATCHED_GN11_NO_GPU_ACQUIRED', GN11_state=current['state'],
            new_control_updates=0, formal_instance_updates=0)
        time.sleep(25)
    completed = []
    arm = 'five_raw_body_isolated'
    pilot = OUT / 'pilots' / arm
    if pilot.exists():
        raise FileExistsError('existing corrected pilot requires explicit checkpoint/step audit')
    spec = OUT.parent / 'identity_completion_20261018/spec5.json'
    stage('correct_isolated_control', [str(OUT / 'isolated_body_implementation/train.py'),
        '--spec', str(spec), '--output', str(pilot), '--iterations', '300', '--seed', '0', '--joint-background',
        '--refine-poses-after', '0', '--pose-translation-limit-m', '.5', '--box-boundary-weight', '1.0',
        '--pose-prior-weight', '.01', '--max-features', '10', '--checkpoint-steps', '100', '300',
        '--probe-steps', '100', '300', '--raw-body-opacity-weight', '100', '--identity-opacity-weight', '100', '--run'], completed)
    assert read(pilot / 'status.json')['state'] == 'complete'
    assert read(pilot / 'status.json')['step'] == 300
    for arm in ['five_replay', 'five_raw_body_isolated']:
        stage('restore_' + arm, [str(OUT / 'pilot_gpu_review.py'), '--mode', 'restore', '--arm', arm], completed)
    for arm in ['three_replay', 'five_replay', 'five_raw_body', 'five_ID_weight', 'five_raw_body_isolated']:
        stage('renders_' + arm, [str(OUT / 'pilot_gpu_review.py'), '--mode', 'renders', '--arm', arm], completed)
    state('ENGINEERING_REVIEW_COMPLETE_AWAITING_ACTUAL_LOSS_RESTORE_AND_VISUAL_DECISION', completed=completed,
        maximum_pilot_updates_total=1500, disposable_audit_updates=6, formal_instance_updates=0,
        next='Review actual receipts and full49, bind the new GN11 point count/source before authorized fresh instance30K',
        human_verdict=None, quality_accepted=False, shutdown_performed=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        state('FAILED_ENGINEERING_REVIEW_PRESERVED_NO_RESTART', error=traceback.format_exc(), formal_instance_updates=0)
        raise
