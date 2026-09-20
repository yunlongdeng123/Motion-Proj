"""小误差独立源：reference G1 通过后只生成 DVGT/class-prior 的减速臂。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONTROL-QUALIFY-01/20260921-r1'
G1 = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONTROL-G1-01/20260921-r1'
BASE = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01/20260921-r1'
OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONTROL-STATE-01/20260921-r1'
TASKS = [('dvgt_metric', 'edited'), ('class_prior', 'edited')]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    assert not OUT.exists()
    q = json.loads((QUAL/'result.json').read_text()); g1 = json.loads((G1/'evaluation.json').read_text())
    assert q['status'] == 'qualified' and g1['reference_gate_passed']
    OUT.mkdir(parents=True)
    protocol = {
        'task_id': 'WS-V75-ACTOR-SLOWDOWN-CONTROL-STATE-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': '7a5f22d2',
        'qualification': str(QUAL), 'reference_g1': str(G1/'evaluation.json'),
        'base_unedited_generation': str(BASE), 'new_tasks': [list(x) for x in TASKS],
        'log': q['selected']['log'], 'actor': q['selected']['actor'],
        'seed': 42, 'frames': 117, 'blocks': 15, 'fps': 30, 'event_frame': 5,
        'fixed': ['same source, event frame5, slowdown factor0.5 and seed42 as large-error source',
                  'reuse frozen unedited sequences; only edited sequence added per state arm'],
        'measurement': 'same five frames, response-vector error, gain and 10px/2-frame material rule as large-error source',
        'stop': 'exactly two new sequences; no retry, source/factor/event/seed/threshold search',
        'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'protocol.json', protocol)
    result = {'status': 'running', 'runs': [], 'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'queue_result.json', result); began = time.monotonic(); script = Path(__file__).with_name('run_actor_removal_generation.py')
    for arm, variant in TASKS:
        cmd = [sys.executable, str(script), '--arm', arm, '--variant', variant, '--output', str(OUT), '--qualification', str(QUAL)]
        with (OUT/f'{arm}-{variant}.log').open('w') as stream:
            child = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        path = OUT/f'{arm}-{variant}/result.json'; terminal = json.loads(path.read_text()) if path.exists() else {'status': 'missing_result'}
        result['runs'].append({'arm': arm, 'variant': variant, 'exit_code': child.returncode, 'result': terminal}); save(OUT/'queue_result.json', result)
        print(json.dumps({'arm': arm, 'variant': variant, 'exit_code': child.returncode, 'status': terminal['status']}), flush=True)
        if child.returncode or terminal['status'] != 'complete':
            result.update(status='oom_stopped' if terminal['status'] == 'oom_stopped' else 'failed_stopped', wall_s=time.monotonic()-began)
            save(OUT/'queue_result.json', result); return
    result.update(status='complete', wall_s=time.monotonic()-began, completed_runs=2,
                  world_model_sequences=2, generation_forwards=30, generated_frames=234,
                  reused_unedited_sequences=3, reused_reference_edited_sequences=1)
    save(OUT/'queue_result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
