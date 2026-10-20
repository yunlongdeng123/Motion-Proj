"""心跳用纯CPU只读快照；不启动/停止训练，不加载checkpoint，不改配置。"""

import argparse
import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo
from v3_pair_runtime import read, process, matches_worker, PAIR_POLICY


def last_jsonl(path):
    if not path.exists():
        return None
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size-65536))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def snapshot(campaign):
    plan = read(campaign/'plan.json')
    serial = plan.get('max_concurrent_training') == 1
    registry = read(campaign/('four_arm_registry.json' if serial else 'pair_registry.json'))
    batch = registry.get('current_batch') or []
    if serial:
        current = registry.get('current_arm')
        batch = next((items for items in registry.get('logical_batches', []) if current in items), [])
    active_arms = [registry['current_arm']] if serial and registry.get('current_arm') else batch
    now = datetime.datetime.now(datetime.timezone.utc)
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    used, total, utilization = map(int, gpu.split(','))
    arms = {}; healthy = registry['status'] == ('RUNNING_SEQUENTIALLY' if serial else 'RUNNING_TWO_PAIRS')
    for arm in plan['arm_order']:
        root = campaign/'arms'/arm; run = root/'run30k'
        state = read(run/'status.json') if (run/'status.json').exists() else None
        events = []
        for file in sorted((run/'checkpoint_events').glob('iteration_*.json')):
            event = read(file)
            events.append(dict(step=int(file.stem.split('_')[-1]), status=event.get('status'),
                complete=event.get('status') == 'complete' and len(event.get('checkpoints',[])) == 4 and len(event.get('frames',[])) == 49,
                checkpoint_count=len(event.get('checkpoints',[])), frame_count=len(event.get('frames',[]))))
        capacity = last_jsonl(run/'v3_capacity.jsonl')
        alive = bool(state and matches_worker(process(state['pid']), campaign, arm, 'train'))
        age = (now.timestamp()-(run/'status.json').stat().st_mtime) if state else None
        finite = bool(state and all(math.isfinite(state.get(k, 0.)) for k in ['loss','psnr']))
        counts_ok = bool(state and all(v == state['iteration'] for v in state.get('optimizer_step_counts', {}).values())
                         and set(state.get('optimizer_step_counts', {})) == {'static','dynamic','environment'})
        arms[arm] = dict(state=state, process_alive=alive, status_age_seconds=age,
            finite_reported_scalars=finite, update_counts_match=counts_ok, checkpoint_events=events,
            capacity=capacity, latest_objective=last_jsonl(run/'v3_objectives.jsonl'))
        if arm in active_arms:
            healthy &= (bool(state) and (alive or state['status'] == 'complete') and finite and counts_ok
                        and state['status'] in {'training','evaluating','complete'}
                        and (age < 600 or state['status'] == 'complete')
                        and bool(capacity and capacity.get('fixed_road_ids_preserved')))
    free_gpu = (total-used)*1024**2
    healthy &= free_gpu >= PAIR_POLICY['runtime_headroom_warning_bytes']
    disk_free = shutil.disk_usage(campaign).free
    healthy &= disk_free >= 10*1024**3
    fivek_ready = bool(batch) and all(any(e['complete'] for e in arms[arm]['checkpoint_events']) for arm in batch)
    return dict(timestamp_utc=now.isoformat(), timestamp_SGT=now.astimezone(ZoneInfo('Asia/Singapore')).isoformat(),
        method_version=plan['method_version'], registry=registry, current_logical_batch=batch,
        active_training_arms=active_arms, controller_kind='single_slot' if serial else 'two_slot_pairs',
        arms=arms, GPU_used_MiB=used,
        GPU_total_MiB=total, GPU_utilization_percent=utilization, free_disk_bytes=disk_free,
        healthy_now=bool(healthy), both_current_arms_have_5k_node=bool(fivek_ready),
        interpretation='resource/implementation health only; not visual quality acceptance; cumulative CPU snapshots do not prove peak GPU capacity',
        training_mutation=False, human_verdict=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = snapshot(args.campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key:result[key] for key in ['timestamp_SGT','GPU_used_MiB','GPU_total_MiB','healthy_now','both_current_arms_have_5k_node']}))
    print(json.dumps({arm: {'iteration':item['state'].get('iteration') if item['state'] else None,
        'alive':item['process_alive'], 'complete_nodes':sum(event['complete'] for event in item['checkpoint_events'])}
        for arm,item in result['arms'].items()}))
