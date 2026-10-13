"""GPU 准入只读检查；StreetCrafter 的准备进程也优先，不终止任何他人任务。"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess


def inspect_priority(proc_root='/proc', run=subprocess.run, own_pid=None):
    own_pid = os.getpid() if own_pid is None else own_pid
    competitors = []
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit() or int(directory.name) == own_pid:
            continue
        try:
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', errors='replace').strip()
            cwd = str((directory / 'cwd').resolve())
        except (OSError, PermissionError):
            continue
        # ssh probe、编辑器命令不能仅因参数中出现检查文本而算竞争者。
        executable = command.split(' ', 1)[0].rsplit('/', 1)[-1]
        if executable in ('sshd', 'ssh', 'grep', 'rg'):
            continue
        if executable in ('sh', 'bash') and (' -c ' in command or ' -lc ' in command) and 'inspect_priority' in command:
            continue
        if 'streetcrafter' in (command + ' ' + cwd).lower() or 'street_crafter' in (command + ' ' + cwd).lower():
            competitors.append({'pid': int(directory.name), 'cmdline': command, 'cwd': cwd})
    try:
        result = run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True, timeout=8, check=True)
        gpu = []
        for fields in csv.reader(io.StringIO(result.stdout)):
            if not fields:
                continue
            pid = int(fields[0].strip())
            if pid != own_pid:
                gpu.append({'pid': pid, 'process_name': fields[1].strip(), 'used_memory_mib': fields[2].strip()})
        query_error = None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        gpu, query_error = [], str(error)
    allowed = not competitors and not gpu and query_error is None
    return {'status': 'GPU_ADMISSION_AVAILABLE' if allowed else 'DEFER_GPU_TO_STREETCRAFTER_OR_OTHER_WORK',
            'allowed': allowed, 'streetcrafter_processes': competitors, 'other_compute_processes': gpu,
            'query_error': query_error, 'policy': 'StreetCrafter preparation/training/inference has priority; no kill, no shared environment changes'}


def require_gpu_slot(lock_path, proc_root='/proc'):
    """持有共享锁后再查 GPU；busy 时即时返回，不排一个未来会偷占 GPU 的 worker。"""
    import fcntl
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    stream = lock.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = inspect_priority(proc_root)
        if not receipt['allowed']:
            raise RuntimeError(json.dumps(receipt, ensure_ascii=False))
        return stream, receipt
    except BaseException:
        stream.close()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    receipt = inspect_priority()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    raise SystemExit(0 if receipt['allowed'] else 3)
