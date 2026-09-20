"""冻结未来减速的重建状态比较，并准备唯一一次官方 DVGT-1 读出。"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
SOURCE = ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-QUALIFY-01/20260921-r1'
G1 = ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-GENERATION-01/20260921-r1'
OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01/20260921-r1'
SCRIPTS = Path('/root/autodl-tmp/motion_proj/scripts/worldsim_v75')
LATE = [45, 61, 85, 109, 116]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    global OUT
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', default='20260921-r1')
    args = parser.parse_args(); OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01'/args.run_id
    assert not OUT.exists(), '拒绝覆盖冻结目录'
    source = json.loads((SOURCE/'result.json').read_text())
    g1 = json.loads((G1/'evaluation.json').read_text())
    assert source['status'] == 'qualified' and g1['reference_gate_passed']
    selected = source['selected']
    OUT.mkdir(parents=True)
    protocol = {
        'task_id': 'WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01',
        'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': '7a5f22d2',
        'engineering_recovery_of': ('20260921-r1' if args.run_id == '20260921-r2' else None),
        'source_g1': str(G1/'evaluation.json'),
        'log': selected['log'], 'base': selected['base'],
        'actor': selected['actor'], 'behind': selected['behind'],
        'event_frame': 5, 'slowdown_factor': 0.5,
        'question': 'Which reconstruction-state errors change a world model response to the same future slowdown?',
        'state_arms': ['reference', 'dvgt_metric', 'class_prior'],
        'state_definition': {
            'reference': 'original GT cuboid and trajectory',
            'dvgt_metric': 'one official DVGT-1 seven-camera T=1 depth readout changes initial translation only; GT dimensions/yaw and relative motion retained',
            'class_prior': 'same DVGT range with fixed 3.9x1.6x1.56m dimensions and ego-heading orientation; relative motion retained',
        },
        'intervention': 'all arms exact through frame4; from frame5 target pose follows its own unedited trajectory at tau=4+0.5*(frame-4), with SO(3) interpolation',
        'information_roles': 'initial RGB/text/map/non-target actors/camera/seed shared; reference target state and readout size/yaw are explicit extra information; LiDAR is reference/metric-control only',
        'reconstruction_gate': 'reference LiDAR core >=6, center error <=0.5m, face retention >=0.8; DVGT core >=20 and face retention >=0.8',
        'generation_plan': 'reuse frozen seed42 reference unedited/edited; add exactly DVGT and class-prior unedited/edited pairs',
        'late_frames': LATE,
        'measurement': {
            'own_state_unedited': 'detection overlaps arm-specific unedited projection at IoU>=0.3',
            'own_state_edited': 'detection overlaps arm-specific slowed projection at IoU>=0.3 and is >=5px closer to it than to arm-specific unedited projection',
            'pair_response_error': 'norm((edited detection-unedited detection)-(edited projection-unedited projection)) per late frame',
            'source_state_error': 'detection-center distance to corresponding reference-state projection',
        },
        'material_effect_rule': 'relative to reference, an arm is materially worse only if median pair-response error rises by >=10px or own-state edited success loses >=2 of 5 late frames',
        'interpretation': 'pass supports state-interface sensitivity on this source; failure means these reconstruction substitutions are absorbed by the world model under this edit',
        'stop': 'one DVGT forward and four new sequences; no source/factor/event/seed/threshold search; OOM/error stops without downsizing or retry',
        'human_verdict': None, 'failure_ledger_delta': 'none',
    }
    save(OUT/'protocol.json', protocol)
    result = {'status': 'started', 'human_verdict': None, 'failure_ledger_delta': 'none'}
    began = time.monotonic()
    try:
        recon = OUT/'reconstruction'
        cmd = [sys.executable, str(SCRIPTS/'prepare_natural.py'), '--output', str(recon),
               '--base-dir', selected['base'], '--target', selected['actor'], '--task-id', protocol['task_id']]
        with (OUT/'prepare_reconstruction.log').open('w') as stream:
            child = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        if child.returncode:
            raise RuntimeError(f'prepare_natural failed rc={child.returncode}')
        native = json.loads((recon/'protocol.json').read_text())
        native.update(
            run_id=OUT.name,
            role='future_slowdown_state_sensitivity_fixed_after_reference_G1_before_reconstruction_output',
            source_g1=str(G1/'evaluation.json'),
            target_selection='inherited frozen mid-distance source; reference edit gate passed before reconstruction',
            admission_policy='reference_and_raw_support_without_error_ranking',
            followup_if_admitted='construct three frozen state arms; raster probe; reuse reference pair and generate exactly four reconstruction-state sequences',
            generation_frames=LATE,
            stop_rules=['OOM stops immediately without retry or downsizing',
                        'causal generation requires reliable target LiDAR and raw model core support',
                        'no source, state adapter, seed, threshold, event time or slowdown-factor replacement'],
        )
        save(recon/'protocol.json', native)
        result.update(status='prepared', reconstruction=str(recon), target_detection=native['target_detection'],
                      bbox_network=native['bbox_network'], world_model_generation_calls=0, new_reconstruction_calls=0)
    except BaseException as exc:
        result.update(status='engineering_failed_before_scientific_result', error_type=type(exc).__name__, error=str(exc),
                      world_model_generation_calls=0, new_reconstruction_calls=0)
        raise
    finally:
        result['wall_s'] = time.monotonic()-began
        save(OUT/'prepare_result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
