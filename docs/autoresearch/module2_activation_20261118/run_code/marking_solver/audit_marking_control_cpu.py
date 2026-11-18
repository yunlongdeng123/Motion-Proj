"""只读检查正式预算、完整状态/代码来源与资产冻结；不执行native。"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
M = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_marking_solver_20261118_r1')
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    import torch
    torch.set_num_threads(1)
    result = read(M / 'results.json')
    smoke = read(M / 'smoke_r1/completed.json')
    saved = torch.load(M / 'formal_r1/checkpoint_final.pt', map_location='cpu', weights_only=False)
    trace = read(M / 'formal_r1/trace.json')
    checks = dict(actual_callback_evaluations_500=result['actual_objective_gradient_evaluations'] == 500,
                  trace_exact_1_to_500=[r['evaluation'] for r in trace] == list(range(1, 501)),
                  checkpoint_fortran_evaluations_500=saved['solver']['evaluations'] == 500,
                  checkpoint_native_evaluations_500=saved['native']['native_objective_gradient_calls'] == 500,
                  cap_stop=result['stop_reason'] == 'actual_objective_gradient_evaluation_cap',
                  solver_time_below300=result['solver_elapsed_seconds'] < 300,
                  selected_best_callback_500=result['best_feasible_evaluation'] == 500,
                  native_smoke3=smoke['engineering_actual_objective_gradient_calls'] == 3,
                  full_restore_exact=smoke['complete_restore_exact'],
                  formal_fresh_c0=smoke['formal_entry_fresh_c0'])
    binding = saved['binding']
    for name, expected in binding['code'].items():
        path = M / 'formal_r1_preflight/source_snapshots' / name
        checks['source_snapshot_' + name] = hashlib.sha256(path.read_bytes()).hexdigest() == expected
    fields = {}
    with np.load(BASE / 'road.npz') as a, np.load(M / 'formal_r1/asset/road.npz') as b:
        active = a['active']
        for name in a.files:
            if name != 'reference_srgb':
                checks['road_' + name + '_exact'] = np.array_equal(a[name], b[name])
            else:
                changed = np.any(a[name] != b[name], axis=1)
                checks['inactive_RGB_exact'] = not (changed & ~active).any()
                fields['changed_active_RGB_rows'] = int((changed & active).sum())
                fields['total_active_rows'] = int(active.sum())
                expected = saved['native']['best']['x'].numpy()
                checks['export_equals_selected_native_ci'] = np.array_equal(b[name][active], expected)
                checks['export_ci_box'] = bool(((expected >= 0) & (expected <= 1)).all())
    ledger = ('source', 'known', 'unknown', 'confidence', 'observed_reference_srgb', 'row_rgb',
              'source_world_row', 'source_distance_m', 'source_raw', 'source_uv_native',
              'old_requested', 'requested', 'W_world', 'world_xyz', 'world_keys', 'surface_supported')
    with np.load(BASE / 'provenance.npz') as a, np.load(M / 'formal_r1/asset/provenance.npz') as b:
        for name in ledger:
            checks['prior_ledger_' + name + '_bytes_exact'] = a[name].dtype == b[name].dtype and a[name].shape == b[name].shape and a[name].tobytes() == b[name].tobytes()
    sf = float(saved['solver']['best']['f'])
    nf = float(saved['native']['best']['f'])
    checks['scaled_SSE_and_normalized_f_agree'] = abs(sf / 111618 - nf) < 1e-15
    sg = saved['solver']['best']['g'].reshape(-1, 3) / 111618
    ng = saved['native']['best']['g'].numpy().astype(np.float64)
    checks['solver_gradient_scaling_exact_math'] = float(np.abs(sg-ng).max()) < 1e-18
    counts = dict(formal_actual_FG=500, engineering_actual_FG=3,
                  formal_forward=result['native_forward_calls_in_context'] + result['additional_public_clone_forward_calls'],
                  formal_backward=result['native_backward_calls'],
                  engineering_forward=smoke['native_forward_calls_in_context'] + smoke['additional_public_clone_forward_calls'],
                  engineering_backward=smoke['native_backward_calls'])
    counts.update(total_forward=counts['formal_forward'] + counts['engineering_forward'],
                  total_backward=counts['formal_backward'] + counts['engineering_backward'])
    f0 = result['original_spectral_solver']['normalized_f']
    summary = dict(schema='marking_solver_independent_CPU_runtime_audit.v1',
                   status='pass_mechanical' if all(checks.values()) else 'failed', checks=checks,
                   asset=fields, accounting=counts, normalized_f=nf,
                   normalized_f_relative_reduction_vs_old=(f0-nf)/f0,
                   FW_numeric_proxy=result['certificate']['FW_gap_numeric_proxy'],
                   FW_numeric_proxy_over_f=result['certificate']['FW_gap_numeric_proxy']/nf,
                   smoke_nextstate_bitwise=smoke['next_state_bitwise'],
                   source_scope='same oldS2,151096 active ci, fixed target/D37206/111618, no target rewriting',
                   strict_negative_certificate=False,
                   interpretation='global-L2 improved; raw86 detail still not met; neither budget stop nor unconstrained detail feasibility is a negative capacity certificate',
                   additional_native_calls=0, GPU_executed_by_auditor=False,
                   human_verdict=None, production_promoted=False)
    (M / 'qa').mkdir(exist_ok=True)
    (M / 'qa/runtime_acceptance.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(path=str(M/'qa/runtime_acceptance.json'), status=summary['status'],
                          checks=len(checks), failures=[k for k,v in checks.items() if not v], accounting=counts)))


if __name__ == '__main__':
    main()
