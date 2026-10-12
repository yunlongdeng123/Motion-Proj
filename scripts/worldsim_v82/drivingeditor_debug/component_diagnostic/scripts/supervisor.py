"""One exclusive read-only component diagnostic. No training subprocesses."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
OUT = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic'
PYTHON = '/root/autodl-tmp/envs/drivingeditor/bin/python'


def write(path, value):
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    lock = (BASE / 'campaign/gpu_slot.lock').open('r+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        write(OUT / 'supervisor_status.json', {'status': 'lock_busy', 'actual_exit_code': 73})
        return 73
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    if apps:
        write(OUT / 'supervisor_status.json', {'status': 'unexpected_gpu_apps', 'apps': apps, 'actual_exit_code': 74})
        return 74
    nonce = uuid.uuid4().hex
    owner = {'owner_pid': os.getpid(), 'nonce': nonce, 'purpose': 'Read-only 20k/30k component diagnostic; no training', 'output': str(OUT), 'started_utc': utc()}
    lock.seek(0)
    lock.truncate()
    lock.write(json.dumps(owner))
    lock.flush()
    os.fsync(lock.fileno())
    status = dict(owner, status='waiting_for_reviewed_worker', gpu_lock_held=True)
    write(OUT / 'supervisor_status.json', status)
    child = None

    def signal_handler(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise InterruptedError('Supervisor interrupted: ' + str(signum))

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    result = 1
    try:
        deadline = time.monotonic() + 2400
        while not (OUT / 'worker_ready.json').is_file():
            if time.monotonic() > deadline:
                raise TimeoutError('Worker staging timed out')
            time.sleep(1)
        ready = json.loads((OUT / 'worker_ready.json').read_text())
        import hashlib
        worker = OUT / 'scripts/component_diagnostic.py'
        if hashlib.sha256(worker.read_bytes()).hexdigest() != ready['worker_sha256']:
            raise RuntimeError('Worker SHA differs from ready manifest')
        status.update(status='running_cpu_then_gpu_diagnostic', worker_pid=None, worker_sha256=ready['worker_sha256'])
        write(OUT / 'supervisor_status.json', status)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        with (OUT / 'worker.log').open('w') as log:
            child = subprocess.Popen([PYTHON, '-u', str(worker), '--gpu-lock-nonce', nonce], stdout=log, stderr=subprocess.STDOUT, env=env)
            status['worker_pid'] = child.pid
            write(OUT / 'supervisor_status.json', status)
            result = child.wait()
        status.update(status='complete' if result == 0 else 'failed', actual_exit_code=result, finished_utc=utc())
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        result = 130 if isinstance(exc, InterruptedError) else 1
        status.update(status='failed', actual_exit_code=result, error=repr(exc), finished_utc=utc())
    finally:
        status.update(gpu_lock_held=False, lock_release_utc=utc())
        write(OUT / 'supervisor_status.json', status)
        (OUT / 'actual_exit_code.txt').write_text(str(result) + '\n')
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return result


if __name__ == '__main__':
    sys.exit(main())
