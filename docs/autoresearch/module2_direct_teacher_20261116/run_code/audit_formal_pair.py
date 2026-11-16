"""Read-only CPU checks of the two locked 800-step direct-teacher runs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import torch

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
NEW = CAMPAIGN / 'road_direct_teacher_20261116_r1'
OLD = CAMPAIGN / 'road_asset_20261116_r1'


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, default=NEW)
    p.add_argument('--repo', type=Path, default=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42'))
    p.add_argument('--baseline', type=Path, default=OLD / 'phase2/formal_r1/asset')
    p.add_argument('--r8', type=Path, default=OLD / 'asset_r8_guarded')
    p.add_argument('--output', type=Path, default=NEW / 'qa/formal_pair_independent/acceptance.json')
    args = p.parse_args()
    sys.path.insert(0, str(args.repo))
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_coverage_training import tangent_scale_scope

    baseline = load(args.baseline / 'road.npz')
    provenance = load(args.baseline / 'provenance.npz')
    r8 = load(args.r8 / 'road.npz')
    active = np.flatnonzero(baseline['active'])
    n = len(baseline['active'])
    checks = {}
    rows = []
    states = {}
    scopes = {}
    contracts = {}
    pending = []

    def check(name, value):
        checks[name] = bool(value)

    for role in ('direct_C', 'matched_F_control'):
        path = args.campaign / role / 'formal_r1'
        if not (path / 'completed.json').exists():
            pending.append(role)
            continue
        complete = read(path / 'completed.json')
        contract = read(path / 'frozen_contract.json')
        contracts[role] = contract
        scope = load(path / 'row_scopes.npz')
        scopes[role] = scope
        checkpoints = {step: torch.load(path / f'checkpoint{step:06d}.pt', map_location='cpu', weights_only=False)
                       for step in (0,200,500,800)}
        states[role] = checkpoints
        trace = read(path / 'training.json')
        final = checkpoints[800]
        init_model = checkpoints[0]['model']
        model = final['model']
        asset = path / 'stage_joint/asset'
        road = load(asset / 'road.npz')
        prior = load(asset / 'provenance.npz')
        prefix = role + ':'
        check(prefix + 'formal_800_fresh', complete['completed'] and complete['actual_updates'] == 800
              and complete['step'] == 800 and complete['formal'] and complete['schedule'] == [200,300,300]
              and checkpoints[0]['step'] == 0 and not checkpoints[0]['optimizer']['state'])
        check(prefix + 'source_target_code_bound', all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
              for name, expected in contract['input_binding'].items()))
        check(prefix + 'formal_contract_correct', contract['formal'] and contract['role'] == role and contract['seed'] == 101
              and contract['schedule'] == [200,300,300] and contract['train_frames'] == [86,110]
              and contract['diagnostic_only'] == [134,179,198])
        check(prefix + 'all_checkpoints_bound', all(state['contract'] == contract and state['formal_run']
              and state['step'] == step for step, state in checkpoints.items()))
        check(prefix + 'trace_budget_and_domains', len(trace) == 800
              and [row['step'] for row in trace] == list(range(1,801))
              and [row['raw'] for row in trace] == final['trace']
              and final['trace'].count(86) == 400 and final['trace'].count(110) == 400
              and all(row['stage'] == ('color' if row['step'] <= 200 else 'coverage' if row['step'] <= 500 else 'joint') for row in trace))
        check(prefix + 'losses_finite', all(np.isfinite(row['loss']) for row in trace))
        check(prefix + 'inactive_gradient_routes_zero', all(
            all(row['gradient_max'][key] == 0 for key in ('opacity_logits','scale_factor')) if row['stage'] == 'color'
            else row['gradient_max']['colors'] == 0 if row['stage'] == 'coverage' else True for row in trace))
        check(prefix + 'Adam_group_steps_500_600_600', [int(final['optimizer']['state'][i]['step']) for i in range(3)] == [500,600,600])
        check(prefix + 'color_frozen_during_coverage', torch.equal(checkpoints[200]['model']['colors'], checkpoints[500]['model']['colors'])
              and all(torch.equal(checkpoints[200]['optimizer']['state'][0][key], checkpoints[500]['optimizer']['state'][0][key])
                      for key in ('step','exp_avg','exp_avg_sq')))
        check(prefix + 'coverage_frozen_during_color', torch.equal(init_model['opacity_logits'], checkpoints[200]['model']['opacity_logits'])
              and torch.equal(init_model['scale_factor'], checkpoints[200]['model']['scale_factor']))
        check(prefix + 'all_model_buffers_frozen', all(torch.equal(value, model[key]) for key,value in init_model.items()
              if key not in ('colors','opacity_logits','scale_factor')))
        check(prefix + 'active_order_fixed', np.array_equal(active, scope['active_world_rows']))
        guard = provenance['guarded_world_rows'][active] | scope['conservative_K_guard']
        physical0 = (provenance['surface_supported'] & provenance['legal_write']
                     & ~provenance['guarded_world_rows'] & (baseline['minimum_model_depth'] <= 0))[active]
        physical = physical0 & ~scope['conservative_K_guard']
        check(prefix + 'physical_and_color_scopes', np.array_equal(physical, scope['physical_legal'])
              and np.array_equal(scope['color'], physical & (scope['A_seen'] | (scope['C_seen'] & ~provenance['known'][active]))))
        check(prefix + 'opacity_actual_mass_scope', np.array_equal(scope['opacity'], physical & (scope['actual_coverage_mass'] > 0)))
        check(prefix + 'all_guard_rows_frozen', all(not np.any(scope[key] & guard) for key in ('color','opacity','scale')))
        check(prefix + 'known_C_only_frozen', not np.any(scope['color'] & provenance['known'][active] & ~scope['A_seen']))
        for key in ('color','opacity','scale'):
            ids = key + '_ids'
            check(prefix + ids + '_exact', np.array_equal(model[ids].numpy(), np.flatnonzero(scope[key])))
        physics = np.zeros(n, bool)
        physics[active] = physical0
        _, r8_upper, distance, _ = tangent_scale_scope(r8, provenance, physics, max_factor=1.15)
        absolute = r8['gaussian_sigma_m'][:, :2] * r8_upper[:, None]
        check(prefix + 'absolute_R8_upper_exact', np.array_equal(absolute, scope['absolute_sigma_upper_m']))
        world_scale = active[scope['scale']]
        margin = distance[world_scale] - 3*np.max(road['gaussian_sigma_m'][world_scale,:2], axis=1)
        check(prefix + 'bounded_sigma_no_compounding', np.all(road['gaussian_sigma_m'][world_scale,:2] <= absolute[world_scale] + 1e-8)
              and np.all(road['gaussian_sigma_m'][world_scale,:2] >= baseline['gaussian_sigma_m'][world_scale,:2] - 1e-8)
              and np.all(margin > 0))
        check(prefix + 'trusted_sigma_exact', np.array_equal(road['gaussian_sigma_m'][provenance['coverage_trusted']], baseline['gaussian_sigma_m'][provenance['coverage_trusted']]))
        check(prefix + 'direct_color_unit_cube', torch.isfinite(model['colors']).all() and bool(((model['colors'] >= 0) & (model['colors'] <= 1)).all()))
        check(prefix + 'opacity_offsets_monotonic_bound', bool((model['opacity_logits'] >= 0).all())
              and bool((model['opacity_logits'] <= model['opacity_upper_logit']).all()))
        check(prefix + 'tangent_factors_monotonic_bound', bool((model['scale_factor'] >= 1).all())
              and bool((model['scale_factor'] <= model['scale_upper']).all()))
        for key in baseline:
            if key not in ('reference_srgb','road_opacity','gaussian_sigma_m'):
                check(prefix + key + '_exact', np.array_equal(road[key], baseline[key]))
        legal = {key: np.zeros(n,bool) for key in ('color','opacity','scale')}
        for key in legal:
            legal[key][active] = scope[key]
        check(prefix + 'saved_RGB_scope', not np.any(np.any(road['reference_srgb'] != baseline['reference_srgb'],axis=1) & ~legal['color']))
        check(prefix + 'saved_opacity_scope', not np.any((road['road_opacity'] != baseline['road_opacity']) & ~legal['opacity']))
        check(prefix + 'saved_tangent_scope', not np.any(np.any(road['gaussian_sigma_m'][:,:2] != baseline['gaussian_sigma_m'][:,:2],axis=1) & ~legal['scale']))
        check(prefix + 'saved_normal_sigma_exact', np.array_equal(road['gaussian_sigma_m'][:,2], baseline['gaussian_sigma_m'][:,2]))
        check(prefix + 'source_prior_ledger_exact', all(np.array_equal(provenance[key], prior[key], equal_nan=True) for key in
              ('source','confidence','known','unknown','source_world_row','source_raw','source_distance_m','observed_reference_srgb')))
        check(prefix + 'prior_and_effective_RGB_separate', np.array_equal(prior['prior_reference_srgb'], baseline['reference_srgb'])
              and np.array_equal(prior['effective_learned_rgb'], road['reference_srgb']))
        for stage in ('color','coverage','joint'):
            value = read(path / ('stage_' + stage) / 'joint_K.json')
            check(prefix + stage + '_joint_K_77266_exact', sum(row['K_pixels'] for row in value['rows']) == 77266
                  and all(row['joint_K_max'] == 0 and row['changed_pixels'] == 0 for row in value['rows']))
            reload_path = path / ('stage_' + stage) / 'reload.json'
            if reload_path.exists():
                value = read(reload_path)
                check(prefix + stage + '_public_reload', len(value['rows']) == 5
                      and all(row['RGB_float_max'] <= 1e-6 and row['alpha_float_max'] <= 1e-6 for row in value['rows']))
            else:
                pending.append(role + ':' + stage + ' public reload')
        before = read(path / 'evaluation_before.json')
        after = read(path / 'evaluation_after.json')
        check(prefix + 'evaluation_fixed_38416', before['fixed_road_pixels'] == after['fixed_road_pixels'] == 38416)
        rows.append(dict(role=role, formal_updates=800, frame_counts={str(raw): final['trace'].count(raw) for raw in (86,110)},
                         coverage_before=before, coverage_after=after,
                         changed_RGB_rows=int(np.any(road['reference_srgb'] != baseline['reference_srgb'],axis=1).sum()),
                         changed_opacity_rows=int((road['road_opacity'] != baseline['road_opacity']).sum()),
                         changed_tangent_rows=int(np.any(road['gaussian_sigma_m'][:,:2] != baseline['gaussian_sigma_m'][:,:2],axis=1).sum()),
                         minimum_three_sigma_boundary_margin_m=float(margin.min()) if len(margin) else None))

    if len(states) == 2:
        left, right = states['direct_C'], states['matched_F_control']
        check('paired_initial_model_identical', all(torch.equal(value,right[0]['model'][key]) for key,value in left[0]['model'].items()))
        check('paired_camera_trace_identical', left[800]['trace'] == right[800]['trace'])
        check('paired_scopes_identical', all(np.array_equal(scopes['direct_C'][key],scopes['matched_F_control'][key]) for key in
              ('active_world_rows','color','opacity','scale','A_seen','C_seen','conservative_K_guard','absolute_sigma_upper_m','scale_upper_factor')))
        a, b = dict(contracts['direct_C']), dict(contracts['matched_F_control'])
        a.pop('role'); b.pop('role')
        check('paired_contract_only_role_differs', a == b)
    result = dict(schema='independent_direct_teacher_formal_pair.v1',
                  mechanical_status='failed' if not all(checks.values()) else ('incomplete' if pending else 'pass'),
                  checks=checks, pending=pending, runs=rows, formal_updates=sum(row['formal_updates'] for row in rows),
                  human_verdict=None, road_quality_status='requires separate common-mask float comparison and visual review',
                  no_new_training_or_GPU_forward_in_audit=True,
                  interpretation='A is visible training fidelity; C is pseudo-teacher agreement; three other cameras are diagnostics.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['mechanical_status'], checks=len(checks), failed=[key for key,value in checks.items() if not value],
                          pending=pending, output=str(args.output))))


if __name__ == '__main__':
    main()
