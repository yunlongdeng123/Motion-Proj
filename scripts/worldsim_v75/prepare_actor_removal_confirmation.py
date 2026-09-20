"""冻结独立对象移除确认源，并准备单次 DVGT 输入。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
SCREEN = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-SOURCES-01/20260921-r1'
OUT = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01/20260921-r1'
SCRIPTS = Path('/root/autodl-tmp/motion_proj/scripts/worldsim_v75')
EVAL_FRAMES = [4, 5, 13, 21, 29, 37, 45, 61, 85, 109, 116]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    assert not OUT.exists(), '拒绝覆盖独立确认目录'
    screen = json.loads((SCREEN/'result.json').read_text())
    selected = screen['selected']
    assert screen['status'] == 'qualified' and screen['screened_logs'] == 5
    assert selected['log'] == '47286726-5dd4-4e26-bd2d-5324f429e445'
    assert selected['actor'] == '3ebfe7b2-49d6-45d3-b978-c1df0f564a90'
    assert selected['behind'] == 'be9de385-f16b-43e4-9e44-bd4423cd6c26'
    OUT.mkdir(parents=True)
    protocol = {
        'task_id': 'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01',
        'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': 'ff62e1b2',
        'source_screen': str(SCREEN/'result.json'),
        'role': 'independent source confirmation fixed before reconstruction or generation output',
        'log': selected['log'], 'base': selected['base'],
        'actor': selected['actor'], 'behind': selected['behind'],
        'event_frame': 5,
        'state_arms': ['reference', 'dvgt_metric', 'class_prior'],
        'variants': ['unedited', 'removed'],
        'state_definition': {
            'reference': 'original GT cuboid and trajectory',
            'dvgt_metric': 'single official DVGT-1 seven-camera cutoff readout changes initial translation only; GT dimensions/yaw and relative motion retained',
            'class_prior': 'same DVGT visible-face range with fixed 3.9x1.6x1.56m dimensions and ego-heading orientation; relative motion retained',
        },
        'information_roles': 'initial RGB and text shared; GT map/non-target actors shared; later real RGB and behind-track cuboids evaluation-only; reference target state is an explicitly extra-information diagnostic',
        'reconstruction_gate': 'same cutoff seven-camera T=1 DVGT readout; reference LiDAR core >=6, error <=0.5m, face retention >=0.8; DVGT core >=20 and face retention >=0.8',
        'raster_gate': 'all arms exactly equal before edit; each arm changes at least 50 condition pixels at frames5,30,90',
        'generation': {'seed': 42, 'frames': 117, 'blocks': 15, 'fixed_recorded_camera': True, 'evaluation_frames': EVAL_FRAMES},
        'reference_gate': 'unedited A >=6 post-event samples; removed A <=2; removed B >=4 and greater than unedited B',
        'confirmation_gate': 'reference gate passes; class-prior removed arm differs from reference on >=3 A/B decisions and has A present at least 2 more samples than reference removed',
        'dvgt_role': 'report distance to reference without requiring a direction; it is not used to admit the class-prior confirmation',
        'stop': 'one frozen source, one DVGT forward, six fixed-camera sequences; any OOM/error stops without retry, lower resolution, new source/seed/event time/threshold; no closed-loop feedback in this confirmation',
        'human_verdict': None, 'failure_ledger_delta': 'none',
    }
    save(OUT/'protocol.json', protocol)
    result = {'status': 'started', 'human_verdict': None, 'failure_ledger_delta': 'none'}
    began = time.monotonic()
    try:
        recon = OUT/'reconstruction'
        command = [sys.executable, str(SCRIPTS/'prepare_natural.py'), '--output', str(recon),
                   '--base-dir', selected['base'], '--target', selected['actor'],
                   '--task-id', protocol['task_id']]
        with (OUT/'prepare_reconstruction.log').open('w') as stream:
            child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if child.returncode:
            raise RuntimeError(f'prepare_natural failed rc={child.returncode}')
        native = json.loads((recon/'protocol.json').read_text())
        native.update(
            run_id=OUT.name,
            role='independent_actor_removal_confirmation_readout_fixed_before_model_output',
            source_screen=str(SCREEN/'result.json'),
            target_selection='first real-supported pair in frozen eight-log screen; no model output or reconstruction-error ranking',
            admission_policy='reference_and_raw_support_without_error_ranking',
            followup_if_admitted='construct exactly reference, DVGT-metric and fixed-class-prior state arms; raster probe; then six frozen seed42 fixed-camera sequences',
            generation_frames=EVAL_FRAMES,
            stop_rules=['OOM stops immediately without retry or downsizing',
                        'causal generation requires reliable target LiDAR and raw model core support',
                        'no source, seed, threshold, event time, horizon or state-adapter replacement'],
        )
        save(recon/'protocol.json', native)
        result.update(status='prepared', reconstruction=str(recon),
                      target_detection=native['target_detection'],
                      bbox_network=native['bbox_network'],
                      world_model_generation_calls=0, new_reconstruction_calls=0)
    except BaseException as exc:
        result.update(status='engineering_failed_before_scientific_result',
                      error_type=type(exc).__name__, error=str(exc),
                      world_model_generation_calls=0, new_reconstruction_calls=0)
        raise
    finally:
        result['wall_s'] = time.monotonic()-began
        save(OUT/'prepare_result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
