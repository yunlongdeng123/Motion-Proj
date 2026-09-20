"""一次普通世界速度零均值初始化控制；原检测、关联规则和全部噪声不变。"""
import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from evaluate_localization import iou
from following_geometry import scene
from replay_iou_control import new_policy, step
from rgb_idm_policy import MotionTracks
from run_approach_state_control import state_action

ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
OUT = ROOT/'WS-V75-VELOCITY-PRIOR-CONTROL-01/20260920-r1'
CASES = [
    {'log': '02678d04-cc9f-3148-9f95-1ba66347dff9',
     'real': 'WS-V75-APPROACH-BASELINE-01/20260920-association-r2',
     'generated': 'WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2'},
    {'log': '24642607-2a51-384a-90a7-228067956d05',
     'real': 'WS-V75-APPROACH-BASELINE-02/20260920-r1',
     'generated': 'WS-V75-APPROACH-CLOSEDLOOP-02/20260920-r1'}]


class ZeroWorldVelocityTracks(MotionTracks):
    """只替换新目标世界速度均值；保留36(m/s)^2初始速度方差。"""
    def update(self, detections, time_s, ego_velocity):
        return super().update(detections, time_s, np.zeros(2))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def metrics(errors):
    d = np.array(errors)
    return {'n': len(d), 'mean_abs_error_mps2': float(np.mean(abs(d))),
            'median_abs_error_mps2': float(np.median(abs(d))),
            'max_underbraking_mps2': float(np.maximum(d, 0).max()),
            'max_overbraking_mps2': float(np.maximum(-d, 0).max())}


def gate(m):
    return m['median_abs_error_mps2'] <= .5 and m['max_underbraking_mps2'] <= 2 and m['max_overbraking_mps2'] <= 2


def lead_info(lead):
    if lead is None: return None
    return {k: lead[k] for k in ['track_id', 'gap_m', 'lead_speed_mps', 'box', 'world_center']}


def streams(case):
    g = ROOT/case['generated']
    result = [{'arm': a, 'seed': 42, 'path': str(g/a)} for a in ['gt_clean', 'dvgt_metric', 'dvgt_lidar_scaled', 'reference_lidar']]
    for task, seed, arms in [('WS-V75-SHAPE-FEEDBACK-01/20260920-conditioning-r2', 42, ['dvgt_class_prior', 'dvgt_visible_extent']),
                             ('WS-V75-SHAPE-CONFIRM-01/20260920-r1', 43, ['gt_clean', 'dvgt_class_prior', 'dvgt_visible_extent'])]:
        result.extend({'arm': a, 'seed': seed, 'path': str(ROOT/task/case['log']/a)} for a in arms)
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists(); OUT.mkdir(parents=True)
    protocol = {'task_id': 'WS-V75-VELOCITY-PRIOR-CONTROL-01', 'run_id': OUT.name,
                'frozen_utc': datetime.now(timezone.utc).isoformat(),
                'source_revision': 'c92f8a40', 'cases': [{**c, 'streams': streams(c)} for c in CASES],
                'motivation': 'new tracks store world-coordinate position/velocity but initialize velocity to ego velocity; test ordinary zero-mean world-velocity prior once',
                'variant': 'new-track world velocity mean [0,0]; retain covariance diag[2.25,2.25,36,36], all dynamics/noise, distance-gated Hungarian, expiry, detector candidates, lead selection, IDM and timestamps',
                'source_transfer': 'SORT + FilterPy initialize unobserved velocity components with zero mean; analogy only, neither full SORT nor image-to-world velocity equivalence claimed',
                'sources': ['https://github.com/abewley/sort/blob/master/sort.py', 'https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py'],
                'role': 'post-hoc ordinary perception-policy control on two exposed development tasks; saved-ego action replay is not actual changed-policy feedback',
                'fixed_scope': '2 real streams + all18 completed generated streams, 15 decisions each; warm up each from same20 causal real captures; no new detector or model',
                'verification': 'original MotionTracks must exactly reproduce every saved lead, detection state and acceleration before testing variant',
                'primary': {'log': CASES[0]['log'], 'arm': 'reference_lidar', 'seed': 42,
                            'rule': 'after shared startup, mean absolute RGB-minus-condition action error and max underbraking decrease; max overbraking must not increase'},
                'baseline_gate': 'both real streams retain original leader>=.8/gap<=3m/action median<=.5/maxunder,maxover<=2; all4 generated GT streams pass same absolute action limits',
                'if_all_pass': 'at most6 seed42 actual feedback runs: two GT first, then DVGT/referenceLiDAR each task; original117-frame inputs, new policy only; any baseline failure or OOM stops immediately',
                'stop': 'one zero-world-velocity control only; no prior/noise/association/threshold grid; if any gate fails no new generation or method claim',
                'input_roles': 'variant receives only saved detections, known ego/route/ground and legal past RGB; current condition/physical actor states only for evaluation',
                'human_verdict': None, 'failure_ledger_refs': ['V74-H2-F22'], 'failure_ledger_delta': 'none'}
    save(OUT/'protocol.json', protocol)
    began = time.monotonic(); results = []; checks = []; accesses = 0
    for case in protocol['cases']:
        real_path = ROOT/case['real']; real = json.loads((real_path/'result.json').read_text())
        rp = json.loads((real_path/'protocol.json').read_text()); base = Path(rp['base'])
        assert real['status'] == 'complete' and real['real_gate_passed']
        tr = np.load(base/'trajectory.npz'); reference = scene(base)
        jobs = [{'arm': 'real', 'seed': None, 'path': str(real_path)}]+case['streams']
        folder = OUT/case['log']; folder.mkdir()
        # 在任何变体评估之前逐一验证当前case的全部原实现回放。
        for job in jobs:
            tail = real['rows'] if job['arm'] == 'real' else json.loads((Path(job['path'])/'decisions.json').read_text())
            p = new_policy(base, MotionTracks)
            for prior in [r['policy'] for r in real['warmup']+tail]:
                q = step(p, prior)
                assert q['lead'] == prior['lead'] and q['detections'] == prior['detections']
                assert q['acceleration_mps2'] == prior['acceleration_mps2']
                accesses += 1
        checks.append({'log': case['log'], 'streams': len(jobs), 'exact_observation_visits': len(jobs)*35})
        for job in jobs:
            p = new_policy(base, ZeroWorldVelocityTracks)
            for warm in real['warmup']: step(p, warm['policy'])
            tail = real['rows'] if job['arm'] == 'real' else json.loads((Path(job['path'])/'decisions.json').read_text())
            condition = None if job['arm'] == 'real' else json.loads((Path(job['path'])/'condition_scene.json').read_text())
            rows = []
            for prior in tail:
                old = prior['policy']; serial_before = p.tracker.serial; q = step(p, old)
                frame = old['frame']; state = old['ego_state']
                if condition is None:
                    ref = prior['oracle_acceleration_mps2']; cond = ref
                    selected = q['lead']; lead = prior['reference_lead']; proj = prior['projection']
                    correct = bool(selected is None and lead is None or selected and lead and proj and iou(selected['box'],proj['bounds']) >= .3)
                    gap = abs(selected['gap_m']-lead['gap_m']) if correct and lead else None
                else:
                    _, cond = state_action(condition, tr, p, frame, state)
                    _, ref = state_action(reference, tr, p, frame, state)
                    correct = None; gap = None
                    assert abs(old['acceleration_mps2']-ref-prior['acceleration_difference_mps2']) < 1e-10
                rows.append({'frame': frame, 'ego_speed_mps': state['speed_mps'],
                             'reference_action_mps2': ref, 'condition_action_mps2': cond,
                             'old_action_mps2': old['acceleration_mps2'], 'zero_action_mps2': q['acceleration_mps2'],
                             'old_lead': lead_info(old['lead']), 'zero_lead': lead_info(q['lead']),
                             'new_track_ids_zero': list(range(serial_before,p.tracker.serial)),
                             'old_minus_reference_mps2': old['acceleration_mps2']-ref,
                             'zero_minus_reference_mps2': q['acceleration_mps2']-ref,
                             'old_minus_condition_mps2': old['acceleration_mps2']-cond,
                             'zero_minus_condition_mps2': q['acceleration_mps2']-cond,
                             'correct_leader_zero': correct, 'gap_error_m_zero': gap})
            assert len(rows) == 15
            m = {mode: metrics([r[f'{mode}_minus_reference_mps2'] for r in rows]) for mode in ['old', 'zero']}
            intended = {mode: metrics([r[f'{mode}_minus_condition_mps2'] for r in rows[1:]]) for mode in ['old', 'zero']}
            name = f'{job["arm"]}-seed{job["seed"]}'
            save(folder/f'{name}.json', {'status': 'complete', 'source': job['path'], 'rows': rows, 'human_verdict': None})
            summary = {'log': case['log'], **job, 'physical': m, 'condition_after_startup': intended,
                       'absolute_action_gate': gate(m['zero']),
                       'changed_lead_choices': sum((r['old_lead'] is None) != (r['zero_lead'] is None) or
                           (r['old_lead'] is not None and r['zero_lead'] is not None and r['old_lead']['box'] != r['zero_lead']['box']) for r in rows),
                       'detail': str(folder/f'{name}.json')}
            if job['arm'] == 'real':
                correct = float(np.mean([r['correct_leader_zero'] for r in rows]))
                gaps = [r['gap_error_m_zero'] for r in rows if r['gap_error_m_zero'] is not None]
                gap = float(np.median(gaps)) if gaps else None
                summary.update(correct_leader_fraction=correct, median_gap_error_m=gap,
                               real_gate=correct >= .8 and gap is not None and gap <= 3 and gate(m['zero']))
            results.append(summary)
            print(json.dumps({k:v for k,v in summary.items() if k not in ['detail','path']},ensure_ascii=False),flush=True)
    primary = next(x for x in results if x['log']==protocol['primary']['log'] and x['arm']=='reference_lidar')
    old, new = [primary['condition_after_startup'][key] for key in ['old','zero']]
    terms = {'mean_error_decreases': new['mean_abs_error_mps2'] < old['mean_abs_error_mps2'],
             'underbraking_decreases': new['max_underbraking_mps2'] < old['max_underbraking_mps2'],
             'overbraking_not_increased': new['max_overbraking_mps2'] <= old['max_overbraking_mps2']}
    real_pass = all(r['real_gate'] for r in results if r['arm']=='real')
    gt_pass = all(r['absolute_action_gate'] for r in results if r['arm']=='gt_clean')
    assert accesses == 700 and len(results) == 20 and not torch.cuda.is_initialized()
    result = {'status': 'complete', 'cases': results, 'original_replay_checks': checks,
              'exact_observation_visits': accesses, 'unique_observations': 340,
              'primary_terms': terms, 'both_real_pass': real_pass, 'all_four_gt_pass': gt_pass,
              'generation_admitted': real_pass and gt_pass and all(terms.values()),
              'new_model_calls': 0, 'cuda_initialized': False, 'wall_s': time.monotonic()-began,
              'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','original_replay_checks']}),flush=True)


if __name__ == '__main__': main()
