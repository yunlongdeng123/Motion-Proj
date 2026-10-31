"""只读独立审核评分产物的有限域、权重、约束与结论边界。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def close(a, b, tolerance=1e-10):
    assert abs(a - b) <= tolerance, (a, b)


def audit(score_path, output):
    scores = read(score_path)
    contract_path = score_path.parent / 'evaluation_contract.json'
    contract = read(contract_path)
    original_contract = Path(scores.get('contract_source_path', score_path.parents[2] / 'evaluation_contract_pre_score.json'))
    # Windows原件可能CRLF，评分副本按Linux文本重序列化；验证原件bytes与副本内容，不能把换行差异当合同变更。
    assert sha256(original_contract.read_bytes()).hexdigest() == scores['contract_sha256']
    assert read(original_contract) == contract
    assert scores['student_policy_evaluated'] is False
    assert scores['learned_generalization_result'] is None
    assert scores['capability_bin'] == 'C?' and scores['task_completion_threshold'] is None
    assert scores['preedit_eligibility'] == 'unknown_not_assigned_from_evaluation_GT'
    actions = scores['actions']
    expected_count = scores['expected_route_count']
    assert expected_count in (1, 2, 4, 8)
    assert len(actions) == expected_count == scores['evaluated_route_count']
    assert len({a['route_assignment_hash'] for a in actions}) == expected_count
    expected_pairs = {(f, q) for f in scores['evaluated_anchor_frames'] for q in contract['operations']}
    safe = []
    for action in actions:
        observations = action['observations']
        assert len(observations) == 12
        assert {(r['frame'], r['edit']) for r in observations} == expected_pairs
        assert all(r['route_assignment_hash'] == action['route_assignment_hash'] for r in observations)
        if action['reference_available']:
            op_losses = []
            should_pass = True
            for row in observations:
                if row['edit'] == 'none':
                    should_pass &= row['none_raw_rgb_change_mae'] <= contract['development_constraints']['none_raw_rgb_change_mae_max']
                else:
                    regions = row['regions']
                    values = [regions[k]['clamped_mae'] for k in ('old_vacated', 'new_body') if regions[k]['pixels'] > 0]
                    assert values
                    close(sum(values) / len(values), row['normalized_target_loss'])
                    assert row['keep_valid_pixel_denominator'] == regions['keep']['pixels']
                    close(row['keep_worsened_pixels'] / row['keep_valid_pixel_denominator'], row['keep_worsened_fraction'])
                    should_pass &= row['keep_mae_increase'] <= contract['development_constraints']['keep_mae_increase_max']
                    should_pass &= row['keep_worsened_fraction'] <= contract['development_constraints']['keep_fraction_above_increase_threshold_max']
                for region in row['regions'].values():
                    assert region['channel_values'] == 3 * region['pixels']
                    if not region['pixels']:
                        assert region['clamped_mae'] is None and region['raw_mae'] is None
            for q in contract['scored_edit_operations']:
                values = [r['normalized_target_loss'] for r in observations if r['edit'] == q]
                assert len(values) == 3
                op_loss = sum(values) / 3
                close(op_loss, action['per_operation_target_losses'][q])
                op_losses.append(op_loss)
            close(sum(op_losses) / 3, action['joint_target_loss'])
            assert should_pass == action['development_constraint_pass']
            if should_pass:
                safe.append(action)
            else:
                assert action['finite_decision_gap'] is None
        else:
            assert action['joint_target_loss'] is None and action['finite_decision_gap'] is None
            assert action['development_constraint_pass'] is None
            assert type(action['none_preservation_constraint_pass']) is bool
    if safe:
        best = min(safe, key=lambda a: (a['joint_target_loss'], a['action_id']))
        assert best['action_id'] == scores['best_safe_shared_action_id']
        close(best['joint_target_loss'], scores['best_verified_shared_residual'])
        for action in safe:
            close(action['joint_target_loss'] - best['joint_target_loss'], action['finite_decision_gap'])
        assert scores['development_safe_action_count'] == len(safe)
    else:
        assert scores['oracle_status'] == 'safe_oracle_unavailable'
        assert scores['best_safe_shared_action_id'] is None
    report = dict(schema='v43_r4_independent_score_arithmetic_audit_v1', status='PASS',
                  score_path=str(score_path), scene_id=scores['scene_id'],
                  actions_checked=len(actions), native_observation_rows_checked=sum(len(a['observations']) for a in actions),
                  safe_actions_checked=len(safe) if scores['reference_role'] != 'none' else None,
                  checks=['exact12 frame-operation pairs per route', 'one shared route hash across q',
                          'perROI normalization and equal frame/operation weighting',
                          'per-anchor hard constraints and unsafe-gap exclusion',
                          'same shared-domain argmin and nonnegative finite gap arithmetic',
                          'no GT-derived eligibility or task-completion/generalization label'],
                  limits=['arithmetic and contract audit, not re-rendering',
                          'native renderer identity and reference pixels require separate source/output inspection'])
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.scores, args.output)
