"""Reliable detached launcher for the exact reviewed observed-static-RGB hypothesis."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
PLAN = BASE / 'campaign/vgpu_bugfix_20261012/loss_audit/next_diagnostics_cpu/observed_static_rgb_S/materialized_plan'
ENV = '/root/autodl-tmp/envs/drivingeditor/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def main():
    binding_path = BASE / 'campaign/vgpu_bugfix_20261012/loss_audit/next_diagnostics_cpu/observed_static_rgb_S/materialized_plan/experiment_binding.json'
    approval_path = PLAN / 'static_anchor_root_coordination_receipt.json'
    status = {'status': 'checking_exclusive_gpu_idle', 'launcher_pid': os.getpid(), 'started_utc': now(), 'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'output': str(BASE / 'campaign/vgpu_bugfix_20261012/continuation_diagnostic/observed_static_rgb30k_to32000')}
    write(PLAN / 'static_anchor_launcher_status.json', status)
    child = None
    result = 1
    try:
        approval = json.loads(approval_path.read_text())
        binding = json.loads(binding_path.read_text())
        if approval.get('decision') != 'APPROVED_SINGLE_VARIABLE_OBSERVED_STATIC_RGB_30K_TO32000' or approval.get('launcher_sha256') != status['launcher_sha256'] or approval.get('experiment_binding_sha256') != hashlib.sha256(binding_path.read_bytes()).hexdigest() or binding['output'] != status['output']:
            raise RuntimeError('Root receipt does not bind actual launcher/binding/output')
        with (BASE / 'campaign/gpu_slot.lock').open('r+') as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
            if apps:
                raise RuntimeError('Unexpected active GPU compute processes: ' + apps)
            status['GPU_idle_and_lock_available_before_launch'] = True
            fcntl.flock(fd, fcntl.LOCK_UN)
        # The root-approved harness acquires and retains the same lock before its GPU stage.
        def interrupted(signum, _frame):
            if child is not None and child.poll() is None:
                child.send_signal(signum)
            raise InterruptedError('Launcher interrupted by signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        command = [ENV, '-u', str(PLAN / 'continuation_static_anchor_harness.py'), '--experiment-binding', str(binding_path), '--root-approved-plan', str(approval_path)]
        with (PLAN / 'static_anchor_training.log').open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            status.update(status='running_root_approved_static_anchor30k_to32000', worker_pid=child.pid, command=command, environment_policy={'PYTHONDONTWRITEBYTECODE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
            write(PLAN / 'static_anchor_launcher_status.json', status)
            (PLAN / 'static_anchor_worker.pid').write_text(str(child.pid) + '\n')
            result = child.wait()
        status.update(status='complete' if result == 0 else 'failed', actual_exit_code=result)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        result = 130 if isinstance(error, InterruptedError) else 1
        status.update(status='failed', actual_exit_code=result, error=repr(error))
    finally:
        status['finished_utc'] = now()
        write(PLAN / 'static_anchor_launcher_status.json', status)
        (PLAN / 'static_anchor_launcher.exit').write_text(str(result) + '\n')
    return result


if __name__ == '__main__':
    sys.exit(main())
