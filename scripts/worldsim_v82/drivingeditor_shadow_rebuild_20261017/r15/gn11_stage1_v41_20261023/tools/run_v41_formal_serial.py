"""唯一正式串行控制器：V4.1-R30K → 原V1-B30K → 同口径49帧评价。"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from v41_formal_contract import read, validate_authorization


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(pid):
    return dict(pid=pid, start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])


def save(path, data):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def check_nodes(c, arm):
    run = c/f'arms/{arm}/run30k'
    for step in (10000, 20000, 30000):
        prefixes = ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')
        if arm == 'R':
            node = read(run/f'v41_checkpoint{step}.json')
            assert node['V1_updates'] == node['road_updates'] == step and node['road_points'] == 14034
            assert len(node['files']) == 5
            prefixes += ('road_chkpnt',)
        for prefix in prefixes:
            assert (run/f'{prefix}{step}.pth').stat().st_size > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    opts = p.parse_args()
    c = opts.campaign.resolve()
    validate_authorization(c)
    jobs = [
        ('R_TRAIN', [sys.executable, '-u', str(c/'tools/run_v41_arm.py'), '--campaign', str(c), '--arm', 'R', '--execute']),
        ('B_TRAIN', [sys.executable, '-u', str(c/'tools/run_v41_baseline.py'), '--campaign', str(c), '--execute'])]
    for arm in ('R', 'B'):
        cmd = [sys.executable, '-u', str(c/'tools/evaluate_v41.py'), '--campaign', str(c), '--arm', arm,
               '--iteration', '30000', '--output', str(c/f'formal49/{arm}_30000')]
        if arm == 'R':
            cmd += ['--audit-deletion-gradients']
        jobs.append((arm+'_EVAL49', cmd))
    jobs.append(('NATIVE_CPU_COMPARE', [sys.executable, str(c/'tools/compare_v41_states.py'),
        '--reference', str(c/'arms/B/run30k'), '--candidate', str(c/'arms/R/run30k'),
        '--iteration', '30000', '--native-only', '--output', str(c/'formal_native_comparison30000.json')]))
    print(json.dumps(dict(commands=jobs, execute=opts.execute), indent=2), flush=True)
    if not opts.execute:
        return
    lock = (c/'formal_serial.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (c/'formal_serial_status.json').exists():
        raise FileExistsError('已有正式串行现场，先检查PID和phase；不重复启动')
    assert shutil.disk_usage(c).free >= 30_000_000_000
    state = dict(status='RUNNING', controller=identity(os.getpid()), started_utc=utc(),
        completed=[], phase=None, child=None, quality_accepted=False, human_verdict=None)
    status = c/'formal_serial_status.json'
    save(status, state)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    (c/'formal49').mkdir(exist_ok=False)
    try:
        for name, cmd in jobs:
            assert shutil.disk_usage(c).free >= (12_000_000_000 if name.endswith('_TRAIN') else 4_000_000_000)
            directory = c/('arms/'+name[0] if name.endswith('_TRAIN') else 'formal49')
            directory.mkdir(parents=True, exist_ok=True)
            log = directory/(name.lower()+'.log')
            with log.open('x') as stream:
                child = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=env)
                state.update(phase=name, child=identity(child.pid), phase_started_utc=utc(), command=cmd)
                save(status, state)
                code = child.wait()
            accepted_difference = name == 'NATIVE_CPU_COMPARE' and code == 1 and (c/'formal_native_comparison30000.json').is_file()
            if code and not accepted_difference:
                raise RuntimeError(f'{name} exited {code}; inspect {log}')
            if name.endswith('_TRAIN'):
                check_nodes(c, name[0])
            if name.endswith('_EVAL49'):
                assert read(c/f'formal49/{name[0]}_30000/summary.json')['aggregate49']['frame_count'] == 49
            state['completed'].append(dict(phase=name, returncode=code, finished_utc=utc(), log=str(log)))
            state['child'] = None
            save(status, state)
        state.update(status='TRAINING_EVALUATION_COMPLETE_REPORT_PENDING', finished_utc=utc(), phase='REPORT_PENDING')
    except BaseException as exc:
        state.update(status='FAILED_PRESERVED', error=str(exc), failed_utc=utc())
        raise
    finally:
        save(status, state)


if __name__ == '__main__':
    main()
