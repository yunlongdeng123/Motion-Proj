"""One detached, guarded fixed-pose read-only GPU inference worker."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
HERE = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic/fixed_pose_temporal32k'
ENV = '/root/autodl-tmp/envs/drivingeditor/bin/python'


def main():
    approval = HERE / 'root_coordination_receipt.json'
    receipt = json.loads(approval.read_text())
    assert receipt['decision'] == 'APPROVED_READONLY_FIXED_POSE_TEMPORAL_32K'
    assert receipt['launcher_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert receipt['plan_sha256'] == hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest()
    assert receipt['script_sha256'] == hashlib.sha256((HERE / 'fixed_pose_temporal_diagnostic.py').read_bytes()).hexdigest()
    assert not (HERE / 'results').exists()
    with (BASE / 'campaign/gpu_slot.lock').open('r+') as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    status = {'status': 'launching_approved_readonly_inference', 'launcher_pid': os.getpid(),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (HERE / 'launcher_status.json').write_text(json.dumps(status, indent=2) + '\n')
    code = 1
    try:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        with (HERE / 'worker.log').open('w') as log:
            worker = subprocess.Popen([ENV, '-u', str(HERE / 'fixed_pose_temporal_diagnostic.py'),
                                       '--root-approved-plan', str(approval)], stdout=log, stderr=subprocess.STDOUT, env=environment)
            status.update(status='running_approved_readonly_inference', worker_pid=worker.pid)
            (HERE / 'worker.pid').write_text(str(worker.pid) + '\n')
            (HERE / 'launcher_status.json').write_text(json.dumps(status, indent=2) + '\n')
            code = worker.wait()
        return code
    finally:
        status.update(status='complete' if code == 0 else 'failed', actual_exit_code=code,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (HERE / 'launcher_status.json').write_text(json.dumps(status, indent=2) + '\n')
        (HERE / 'launcher.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    sys.exit(main())
