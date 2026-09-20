"""运行减速反事实的两个重建状态配对；reference 配对复用已冻结结果。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01/20260921-r2'
REFERENCE = ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-GENERATION-01/20260921-r1'
REFERENCE_UNEDITED = ROOT/'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01/20260921-r1/reference-unedited'
OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-GENERATION-01/20260921-r1'
TASKS = [(arm, variant) for arm in ['dvgt_metric', 'class_prior'] for variant in ['unedited', 'edited']]
LATE = [45, 61, 85, 109, 116]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    assert not OUT.exists()
    q = json.loads((QUAL/'result.json').read_text()); raster = json.loads((QUAL/'raster_result.json').read_text())
    assert q['status'] == 'qualified' and raster['status'] == 'passed'
    assert json.loads((REFERENCE/'evaluation.json').read_text())['reference_gate_passed']
    OUT.mkdir(parents=True)
    protocol = {
        'task_id': 'WS-V75-ACTOR-SLOWDOWN-STATE-GENERATION-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': '7a5f22d2',
        'qualification': str(QUAL), 'reference_unedited': str(REFERENCE_UNEDITED),
        'reference_edited': str(REFERENCE/'reference-edited'),
        'log': q['selected']['log'], 'actor': q['selected']['actor'],
        'arms': ['reference', 'dvgt_metric', 'class_prior'], 'new_tasks': [list(x) for x in TASKS],
        'variants': ['unedited', 'edited'], 'frames': 117, 'blocks': 15, 'fps': 30, 'seed': 42,
        'event_frame': 5, 'late_frames': LATE,
        'fixed': ['same initial RGB/text embeddings', 'same map/non-target actors/camera/generator/seed42',
                  'within-arm conditions exact through frame4', 'same slowdown factor0.5 and event frame5'],
        'evaluation_frozen': json.loads((QUAL/'protocol.json').read_text())['measurement'],
        'material_effect_rule': json.loads((QUAL/'protocol.json').read_text())['material_effect_rule'],
        'stop': 'exactly four new sequences; OOM/error stops; no retry, downsizing, seed/source/factor/event/threshold search',
        'human_verdict': None, 'failure_ledger_delta': 'none',
    }
    save(OUT/'protocol.json', protocol)
    save(OUT/'queue_manifest.json', {'tasks': [{'arm': a, 'variant': v} for a, v in TASKS], 'human_verdict': None})
    result = {'status': 'running', 'runs': [], 'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'queue_result.json', result); began = time.monotonic()
    script = Path(__file__).with_name('run_actor_removal_generation.py')
    for arm, variant in TASKS:
        cmd = [sys.executable, str(script), '--arm', arm, '--variant', variant,
               '--output', str(OUT), '--qualification', str(QUAL)]
        with (OUT/f'{arm}-{variant}.log').open('w') as stream:
            child = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        path = OUT/f'{arm}-{variant}/result.json'
        terminal = json.loads(path.read_text()) if path.exists() else {'status': 'missing_result'}
        result['runs'].append({'arm': arm, 'variant': variant, 'exit_code': child.returncode, 'result': terminal})
        save(OUT/'queue_result.json', result)
        print(json.dumps({'arm': arm, 'variant': variant, 'exit_code': child.returncode, 'status': terminal['status']}), flush=True)
        if child.returncode or terminal['status'] != 'complete':
            result.update(status='oom_stopped' if terminal['status'] == 'oom_stopped' else 'failed_stopped', wall_s=time.monotonic()-began)
            save(OUT/'queue_result.json', result); return
    result.update(status='complete', wall_s=time.monotonic()-began, completed_runs=4,
                  world_model_sequences=4, generation_forwards=60, generated_frames=468,
                  reused_reference_sequences=2, reused_reference_frames=234)
    save(OUT/'queue_result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
