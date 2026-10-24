"""独立NVML每秒采样，记录全程/分阶段峰值；不创建CUDA上下文。"""
import argparse
import ctypes
import datetime
import fcntl
import json
import os
from pathlib import Path
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def live(identity):
    try:
        row = Path(f'/proc/{identity["pid"]}/stat').read_text().split()
        return row[2] != 'Z' and row[21] == identity['start_ticks']
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='监控器故障后续记已有峰值，不重置历史')
    opts = p.parse_args()
    c = opts.campaign.resolve()
    lock = (c/'gpu_memory_watch.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    # 使用驱动自带NVML，避免为监控改动训练Python环境。
    nv = ctypes.CDLL('libnvidia-ml.so.1')
    def call(name, *args):
        code = getattr(nv, name)(*args)
        if code:
            raise RuntimeError(f'{name} returned {code}')
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
    class Utilization(ctypes.Structure):
        _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]
    call('nvmlInit_v2')
    handle = ctypes.c_void_p()
    call('nvmlDeviceGetHandleByIndex_v2', ctypes.c_uint(0), ctypes.byref(handle))
    log = c/'gpu_memory_samples.jsonl'
    peak_path = c/'gpu_memory_peak.json'
    if peak_path.exists() and not opts.resume:
        raise FileExistsError('已有显存监控现场，不重置历史峰值')
    peak = dict(status='WATCHING', started_utc=utc(),
        monitor=dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        sampling_seconds=1, sample_count=0, device_peak_used_bytes=0,
        phases={}, limitation='NVML sampled residency may miss sub-second peaks; CUDA allocator peaks separately retained')
    if opts.resume:
        previous = json.loads(peak_path.read_text())
        assert not live(previous['monitor']), '已有监控器仍存活'
        monitor = peak['monitor']
        peak = previous
        peak.setdefault('monitor_restarts', []).append(dict(
            previous_monitor=previous['monitor'], last_sample_utc=previous['last_sample']['utc'], resumed_utc=utc()))
        peak.update(monitor=monitor, status='WATCHING')
    try:
        while True:
            state = json.loads((c/'formal_serial_status.json').read_text())
            memory, utilization = Memory(), Utilization()
            call('nvmlDeviceGetMemoryInfo', handle, ctypes.byref(memory))
            call('nvmlDeviceGetUtilizationRates', handle, ctypes.byref(utilization))
            child = state.get('child') or {}
            row = dict(utc=utc(), phase=state.get('phase'), child_pid=child.get('pid'),
                device_used_bytes=memory.used, device_total_bytes=memory.total,
                utilization=utilization.gpu)
            with log.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            peak['sample_count'] += 1
            peak['last_sample'] = row
            if memory.used >= peak['device_peak_used_bytes']:
                peak.update(device_peak_used_bytes=memory.used, device_peak_sample=row)
            phase = peak['phases'].setdefault(row['phase'], dict(device_peak_used_bytes=0, samples=0))
            phase['samples'] += 1
            phase['device_peak_used_bytes'] = max(phase['device_peak_used_bytes'], memory.used)
            save(peak_path, peak)
            if state['status'] != 'RUNNING' or not live(state['controller']):
                peak.update(status='FINISHED_CONTROLLER_STOPPED', finished_utc=utc(), controller_status=state['status'])
                break
            time.sleep(1)
    finally:
        save(peak_path, peak)
        call('nvmlShutdown')


if __name__ == '__main__':
    main()
