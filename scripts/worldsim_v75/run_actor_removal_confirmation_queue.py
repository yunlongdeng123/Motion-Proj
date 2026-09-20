"""冻结并顺序运行独立源的六段对象移除确认。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01/20260921-r3'
OUT = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01/20260921-r1'
TASKS = [(arm, variant) for arm in ['reference', 'dvgt_metric', 'class_prior'] for variant in ['unedited', 'removed']]
EVAL_FRAMES = [4, 5, 13, 21, 29, 37, 45, 61, 85, 109, 116]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    assert not OUT.exists()
    q = json.loads((QUAL/'result.json').read_text())
    raster = json.loads((QUAL/'raster_result.json').read_text())
    conditioning = json.loads((QUAL/'conditioning/result.json').read_text())
    assert q['status'] == 'qualified' and q['generation_admitted']
    assert raster['status'] == 'passed' and raster['generation_admitted']
    assert conditioning['status'] == 'complete'
    OUT.mkdir(parents=True)
    protocol = {
        'task_id': 'WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': 'ff62e1b2',
        'qualification': str(QUAL), 'source_screen': str(ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-SOURCES-01/20260921-r1/result.json'),
        'log': q['selected']['log'], 'actor': q['selected']['actor'], 'behind': q['selected']['behind'],
        'arms': ['reference', 'dvgt_metric', 'class_prior'], 'variants': ['unedited', 'removed'],
        'tasks': [list(x) for x in TASKS], 'frames': 117, 'blocks': 15, 'fps': 30, 'seed': 42, 'event_frame': 5,
        'fixed': ['logged camera trajectory', 'initial image embeddings and text', 'map and non-target actors', 'generator weights/configuration/random seed'],
        'change': 'target A is present through frame4 and absent from conditions at frame5 onward; unedited pair retains it',
        'role': 'independent fixed-camera state/edit confirmation; no feedback and no pixel-ground-truth counterfactual',
        'information_roles': 'reference uses GT target cuboid; DVGT uses official T=1 readout plus GT size/yaw; class prior uses ordinary size/ego-heading; later RGB and B cuboid evaluation-only',
        'evaluation_frozen': {
            'frames': EVAL_FRAMES,
            'detector': 'same COCO FasterRCNN, score>=0.5, vehicle classes3/6/8, IoU>=0.3',
            'reference_gate': 'unedited A >=6 post-event; removed A <=2; removed B >=4 and strictly greater than unedited B',
            'confirmation_gate': 'class-prior removed differs from reference removed on >=3 A/B decisions and contains A at least2 more post-event samples than reference removed',
            'dvgt_role': 'report mismatch to reference without admission threshold or preferred direction',
            'invariant_readout': 'report matched other annotated vehicles and outside-edit pixel change; no pass from pixel similarity alone',
        },
        'stop': 'exactly six sequences; any OOM/error stops without retry, lower resolution, new seed/source/event time/threshold; no closed-loop feedback',
        'human_verdict': None, 'failure_ledger_refs': ['V75-F01', 'V74-H2-F20', 'V74-H2-F22'], 'failure_ledger_delta': 'none',
    }
    save(OUT/'protocol.json', protocol)
    save(OUT/'queue_manifest.json', {'tasks': [{'arm': a, 'variant': v} for a, v in TASKS], 'human_verdict': None})
    result = {'status': 'running', 'runs': [], 'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'queue_result.json', result); began = time.monotonic()
    script = Path(__file__).with_name('run_actor_removal_generation.py')
    for arm, variant in TASKS:
        command = [sys.executable, str(script), '--arm', arm, '--variant', variant,
                   '--output', str(OUT), '--qualification', str(QUAL)]
        with (OUT/f'{arm}-{variant}.log').open('w') as stream:
            child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        path = OUT/f'{arm}-{variant}/result.json'
        terminal = json.loads(path.read_text()) if path.exists() else {'status': 'missing_result'}
        result['runs'].append({'arm': arm, 'variant': variant, 'exit_code': child.returncode, 'result': terminal})
        save(OUT/'queue_result.json', result)
        print(json.dumps({'arm': arm, 'variant': variant, 'exit_code': child.returncode, 'status': terminal['status']}), flush=True)
        if child.returncode or terminal['status'] != 'complete':
            result.update(status='oom_stopped' if terminal['status'] == 'oom_stopped' else 'failed_stopped', wall_s=time.monotonic()-began)
            save(OUT/'queue_result.json', result); return
    result.update(status='complete', wall_s=time.monotonic()-began, completed_runs=6,
                  world_model_sequences=6, generation_forwards=90, generated_frames=702)
    save(OUT/'queue_result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
