"""单次只读检查正式R/B队列；不启动/重启任务，不读取大模型。"""
import argparse
import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def last_json(path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size-131072))
        for line in reversed(f.read().splitlines()):
            try:
                return json.loads(line)
            except (ValueError, UnicodeDecodeError):
                pass
    return None


def live(record):
    if not record:
        return False
    try:
        fields = Path(f'/proc/{record["pid"]}/stat').read_text().split()
        return fields[2] != 'Z' and fields[21] == record['start_ticks']
    except (FileNotFoundError, ProcessLookupError):
        return False


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    opts = p.parse_args()
    c = opts.campaign.resolve()
    state = read(c/'formal_serial_status.json')
    arms = {}
    for arm in ('R', 'B'):
        run = c/f'arms/{arm}/run30k'
        native, road = last_json(run/'paired_steps.jsonl'), last_json(run/'v41_road_steps.jsonl')
        nodes = {}
        for step in (10000, 20000, 30000):
            prefixes = ['chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state']
            if arm == 'R':
                prefixes.append('road_chkpnt')
            files = {f'{prefix}{step}.pth': (run/f'{prefix}{step}.pth').stat().st_size
                     if (run/f'{prefix}{step}.pth').exists() else 0 for prefix in prefixes}
            manifest = read(run/f'v41_checkpoint{step}.json') if arm == 'R' else read(run/f'v1_checkpoint{step}.json')
            complete = all(files.values()) and (arm == 'B' or manifest is not None)
            if arm == 'R' and complete:
                complete = manifest['V1_updates'] == manifest['road_updates'] == step and manifest['road_points'] == 14034
            nodes[str(step)] = dict(complete=bool(complete), files=files, manifest=manifest)
        arms[arm] = dict(native=native, road=road, finite_latest=finite(native) and finite(road),
            road_points_fixed=road is None or road['road_points'] == 14034,
            nodes=nodes, returned=read(run/'phase_0_30000_returned.json'))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv'],
        capture_output=True, text=True)
    compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv'],
        capture_output=True, text=True)
    result = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        campaign=str(c), serial=state, controller_live=live(state.get('controller')) if state else False,
        child_live=live(state.get('child')) if state else False, arms=arms,
        gpu_memory_peak=read(c/'gpu_memory_peak.json'),
        disk=shutil.disk_usage(c)._asdict(), gpu=gpu.stdout.strip(), compute=compute.stdout.strip())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
