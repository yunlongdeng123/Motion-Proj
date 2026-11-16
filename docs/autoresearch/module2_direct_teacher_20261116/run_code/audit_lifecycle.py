"""Independent read-only acceptance of the native 2/2/2 lifecycle smoke.

Checks saved tensors, actual code/data hashes, routed gradients, three stage
entries, full joint keep masks and public saved-loader receipts. No GPU work.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import torch

NEW = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_direct_teacher_20261116_r1')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lifecycle', type=Path, default=NEW / 'phase0/lifecycle_r4')
    p.add_argument('--supervision', type=Path, default=NEW / 'supervision_r3')
    p.add_argument('--repo', type=Path, default=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42'))
    p.add_argument('--output', type=Path, default=NEW / 'qa/lifecycle_r4_independent/acceptance.json')
    args = p.parse_args()
    sys.path.insert(0, str(args.repo))
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_direct_teacher_training import prepare_targets

    root = args.lifecycle
    def checkpoint(path):
        return torch.load(root / path, map_location='cpu', weights_only=False)

    s2 = checkpoint('save_color/checkpoint000002.pt')
    s3 = checkpoint('restore_next/checkpoint000003.pt')
    replay3 = checkpoint('replay_next/checkpoint000003.pt')
    s4 = checkpoint('next_stage/checkpoint000004.pt')
    s6 = checkpoint('next_stage/checkpoint000006.pt')
    tensor_differences = {}
    metadata_differences = []
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    def compare(a, b, path=''):
        if torch.is_tensor(a):
            if a.dtype != b.dtype or a.shape != b.shape:
                metadata_differences.append(path + ':shape/dtype')
            elif not torch.equal(a, b):
                if a.dtype.is_floating_point:
                    tensor_differences[path] = float((a-b).abs().max())
                else:
                    metadata_differences.append(path + ':nonfloat')
        elif isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                metadata_differences.append(path + ':numpy')
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                metadata_differences.append(path + ':keys')
            else:
                for key in a:
                    compare(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (tuple, list)):
            if len(a) != len(b):
                metadata_differences.append(path + ':length')
            else:
                for index, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, path + '/' + str(index))
        elif a != b:
            metadata_differences.append(path + ':value')

    compare(s3, replay3)
    replay_receipt = read(root / 'replay_comparison.json')
    tolerance = replay_receipt['tolerance']
    check('replay_nonparameter_metadata_exact', not metadata_differences)
    check('replay_under_declared_tolerance', all(np.isfinite(v) and v <= tolerance for v in tensor_differences.values()))
    step_records = []
    for label, state, expected in [('color2', s2, [2,0,0]), ('coverage3', s3, [2,1,1]),
                                   ('coverage4', s4, [2,2,2]), ('joint6', s6, [4,4,4])]:
        counts = [int(state['optimizer']['state'][i]['step']) if i in state['optimizer']['state'] else 0 for i in range(3)]
        check(label + '_Adam_steps', counts == expected)
        step_records.append(dict(state=label, actual=counts, expected=expected))
    check('color_values_frozen_in_coverage', torch.equal(s2['model']['colors'], s3['model']['colors'])
          and torch.equal(s2['model']['colors'], s4['model']['colors']))
    check('color_Adam_moments_frozen_in_coverage', all(torch.equal(s2['optimizer']['state'][0][key],
          s4['optimizer']['state'][0][key]) for key in ('step', 'exp_avg', 'exp_avg_sq')))
    check('model_buffers_frozen', all(torch.equal(s2['model'][key], s6['model'][key]) for key in s2['model']
          if key not in ('colors','opacity_logits','scale_factor')))
    check('only_training_cameras_balanced_trace', len(s6['trace']) == 6 and s6['trace'].count(86) == 3 and s6['trace'].count(110) == 3)

    pair_rows = []
    for raw in (86,110):
        with np.load(args.supervision / f'{raw:03d}.npz') as values:
            data = {key: values[key] for key in values.files}
        direct = prepare_targets(data, 'cpu', 'direct_C')
        control = prepare_targets(data, 'cpu', 'matched_F_control')
        differing = [key for key in direct if not torch.equal(direct[key], control[key])]
        check(f'{raw}_paired_target_only_C_differs', differing == ['selected_C_srgb'])
        check(f'{raw}_direct_target_is_original_teacher', torch.equal(direct['selected_C_srgb'], direct['target_C_srgb']))
        check(f'{raw}_control_target_is_immutable_F', torch.equal(control['selected_C_srgb'], torch.as_tensor(data['F_control_C_srgb'])))
        pair_rows.append(dict(raw=raw, differing_fields=differing))

    contracts = []
    scopes = []
    gradient_rows = []
    for label in ('save_color','restore_next','replay_next','next_stage'):
        directory = root / label
        contract = read(directory / 'frozen_contract.json')
        contracts.append(contract)
        check(label + '_source_code_data_binding', all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
              for path, expected in contract['input_binding'].items()))
        with np.load(directory / 'row_scopes.npz') as values:
            scope = {key: values[key] for key in values.files}
        scopes.append(scope)
        check(label + '_all_coverage_masses_saved', all(key in scope for key in (
            'actual_A_mass','actual_C_mass','actual_coverage_mass','actual_A_coverage_mass','actual_C_coverage_mass')))
        check(label + '_opacity_scope_from_actual_coverage', np.array_equal(scope['opacity'],
              scope['physical_legal'] & (scope['actual_coverage_mass'] > 0)))
        check(label + '_all_scopes_exclude_tile_guard', all(not np.any(scope[key] & scope['conservative_K_guard'])
              for key in ('color','opacity','scale')))
        guard = read(directory / 'guard_receipt.json')
        check(label + '_actual_tile_guard_receipt', 'tile-expanded' in guard['method']
              and 'projection-rounding margin' in guard['method'] and guard['five_frame_union_rows'] == int(scope['conservative_K_guard'].sum()))
        for row in read(directory / 'gradient_probe.json')['rows']:
            check(label + f"_{row['raw']}_real_routes", row['A_color_gradient_max'] > 0 and row['C_color_gradient_max'] > 0
                  and row['coverage_opacity_gradient_max'] > 0 and row['coverage_scale_gradient_max'] > 0
                  and all(row[key] == 0 for key in ('photo_opacity_gradient_max','photo_scale_gradient_max','coverage_RGB_gradient_max')))
            check(label + f"_{row['raw']}_nonphoto_C_coverage_gradient", row['C_coverage_outside_photo_weight'] > 0
                  and row['C_coverage_outside_photo_opacity_gradient_max'] > 0)
            gradient_rows.append(dict(run=label, **row))
        if (directory / 'restore.json').exists():
            restored = read(directory / 'restore.json')
            check(label + '_full_restore_exact', restored['full_state_restored'] and restored['differences'] == []
                  and restored['model_optimizer_paramgroups_sampler_queue_trace_all_RNG_exact'])
    check('all_smoke_contracts_identical', all(value == contracts[0] for value in contracts))
    check('no_formal_updates_and_fixed_smoke_schedule', all(not value['formal'] and value['schedule'] == [2,2,2] for value in contracts))
    check('scope_ids_booleans_identical_after_restarts', all(all(np.array_equal(scopes[0][key], scope[key]) for key in (
        'active_world_rows','color','opacity','scale','A_seen','C_seen','conservative_K_guard')) for scope in scopes[1:]))

    for path in list(root.glob('*/joint_K_after.json')) + list(root.glob('*/maximum_joint_K.json')) + list(root.glob('**/stage_*/joint_K.json')):
        value = read(path)
        check(str(path.relative_to(root)), sum(row['K_pixels'] for row in value['rows']) == 77266
              and all(row['joint_K_max'] == 0 for row in value['rows']))
    for path in root.glob('**/stage_*/reload.json'):
        value = read(path)
        check(str(path.relative_to(root)), len(value['rows']) == 5
              and all(row['RGB_float_max'] <= 1e-6 and row['alpha_float_max'] <= 1e-6 for row in value['rows']))
    for path in root.glob('**/evaluation*.json'):
        check(str(path.relative_to(root)), read(path)['fixed_road_pixels'] == 38416)
    expected_entries = {'save_color':[(1,'color',[.01,0.,0.])], 'restore_next':[(3,'coverage',[0.,.05,.005])],
                        'replay_next':[(3,'coverage',[0.,.05,.005])],
                        'next_stage':[(4,'coverage',[0.,.05,.005]),(5,'joint',[.003,.05,.005])]}
    for label, expected in expected_entries.items():
        actual = [(row['step'],row['stage'],row['learning_rates']) for row in read(root / label / 'stage_entries.json')]
        check(label + '_stage_entries', actual == expected)

    result = dict(schema='independent_direct_teacher_lifecycle_acceptance.v1',
                  status='pass' if all(checks.values()) else 'failed',
                  preformal_gate='ready' if all(checks.values()) else 'no-go', checks=checks,
                  replay_tensor_differences=tensor_differences, replay_metadata_differences=metadata_differences,
                  bitwise_next_update=not tensor_differences and not metadata_differences,
                  replay_declared_tolerance=tolerance, optimizer_steps=step_records, paired_targets=pair_rows,
                  actual_gradient_rows=gradient_rows, engineering_accounting=read(root / 'engineering_accounting.json'),
                  formal_updates=0, human_verdict=None,
                  interpretation='Ready for the predeclared paired experiment; no visual-quality or hidden-ground-truth acceptance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], gate=result['preformal_gate'], checks=len(checks),
                          failed=[key for key,value in checks.items() if not value], output=str(args.output))))


if __name__ == '__main__':
    main()
