"""一次性读取 V4 串行状态、末尾训练记录和节点；不启动任何训练。"""

from pathlib import Path
import argparse
import json
import math
import shutil
import subprocess


def last_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeError):
            continue
    return None


def process(pid):
    root = Path('/proc')/str(pid)
    if not root.exists():
        return dict(alive=False)
    try:
        fields = (root/'stat').read_text().split()
        return dict(alive=fields[2] != 'Z', start_ticks=fields[21],
                    command=(root/'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace'))
    except FileNotFoundError:
        return dict(alive=False)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    options = parser.parse_args()
    campaign = options.campaign.resolve()
    status_path = campaign/'serial_status.json'
    state = json.loads(status_path.read_text()) if status_path.exists() else None
    arms = {}
    for arm in ('R', 'RP'):
        run = campaign/'arms'/arm/'run30k'
        native = last_json(run/'paired_steps.jsonl')
        road = last_json(run/'v4_road_steps.jsonl')
        nodes = {}
        for step in (10000, 20000, 30000):
            files = {prefix: (run/f'{prefix}{step}.pth').stat().st_size
                     for prefix in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state', 'road_chkpnt')
                     if (run/f'{prefix}{step}.pth').is_file()}
            node = run/f'v4_checkpoint{step}.json'
            nodes[str(step)] = dict(complete=node.exists() and len(files) == 5 and all(files.values()),
                                   files=files)
        arms[arm] = dict(run_exists=run.exists(),
            native_iteration=None if native is None else native.get('iteration'),
            native_loss=None if native is None else native.get('losses', {}).get('loss_hybrid_total'),
            road_iteration=None if road is None else road.get('iteration'),
            road_points=None if road is None else road.get('road_points'),
            road_loss=None if road is None else road.get('loss'),
            latest_records_finite=finite(native) and finite(road), checkpoints=nodes,
            training_returned=(run/'v4_training_returned.json').exists())
    result = dict(serial=state, arms=arms, free_disk_bytes=shutil.disk_usage(campaign).free,
        GPU=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                      '--format=csv,noheader'], text=True).strip())
    if state:
        result['controller_process'] = process(state['pid'])
        if state.get('training_pid'):
            result['training_process'] = process(state['training_pid'])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
