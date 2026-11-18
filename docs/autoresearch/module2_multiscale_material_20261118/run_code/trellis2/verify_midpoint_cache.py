"""只读核验 frozen midpoint cache；不加载 Torch、不运行 native。"""
from pathlib import Path
import json, os
import numpy as np
from precompute_quad_prior import sha
from audit_quadrature import load, stats

P = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
T = P / 'trellis2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU only')
    out = T / 'midpoint_full_r1/acceptance.json'
    if out.exists():
        raise FileExistsError(out)
    full = load(T / 'midpoint_full_r1/row_queries.npz')
    selected = load(T / 'midpoint_audit_r1/row_queries.npz')
    plan = load(P / 'chart_plan_candidate_r1/geometry_plan.npz')
    old = load(T / 'quad_prior_candidate_r1/row_queries.npz')
    receipt = json.loads((T / 'midpoint_full_r1/receipt.json').read_text())
    audit = json.loads((T / 'midpoint_audit_r1/receipt.json').read_text())
    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        if not checks[name]:
            raise AssertionError(name)

    n = len(plan['active_world_rows'])
    check('full_after_fixed541_pass', audit['status'] == 'PASS_MIDPOINT_CPU_QUALIFICATION' and receipt['status'] == 'PASS_FULL_FROZEN_CACHE_AFTER_FIXED541_GATE')
    check('exact_plan_rows', np.array_equal(full['active_world_rows'], plan['active_world_rows']) and np.array_equal(full['local_rows'], np.arange(n)))
    check('unique_world_ids', len(np.unique(full['active_world_rows'])) == n)
    for key in selected:
        if key in full:
            check('fixed541_exact_' + key, np.array_equal(full[key][selected['local_rows']], selected[key], equal_nan=True))
    for key in ['filtered_rgb_T', 'filtered_rgb_A0', 'common_valid_mass', 'material_T_weight_fraction', 'physical_T_quad_mass', 'discrete_gaussian_density_mass']:
        check('finite_' + key, np.isfinite(full[key]).all())
    for key in ['filtered_rgb_T', 'filtered_rgb_A0', 'common_valid_mass', 'material_T_weight_fraction']:
        check('bounded_' + key, np.all(full[key] >= -1e-12) and np.all(full[key] <= 1 + 1e-8))
    check('physical_mass_not_above_density', np.all(full['physical_T_quad_mass'] <= full['discrete_gaussian_density_mass'] + 1e-12))
    copied = full['copied_same_A0']
    check('copied_T_and_A0_exact_old_A0', np.array_equal(full['filtered_rgb_T'][copied], old['filtered_rgb_A0'][copied]) and np.array_equal(full['filtered_rgb_A0'][copied], old['filtered_rgb_A0'][copied]))
    check('copied_has_zero_T_material_mass', np.all(full['material_T_weight_fraction'][copied] == 0))
    computed = full['midpoint_N'] > 0
    check('even_N', np.all(full['midpoint_N'][computed] % 2 == 0))
    upper = np.where(old['all225_quad_in_patch'], .0025, .00125)
    check('fixed_step_bounds', np.all(full['actual_step_m'][computed] <= upper[computed] + 1e-15))
    radius = 5 * np.sqrt(np.linalg.eigvalsh(plan['world_covariance_xy'])[:, -1])
    check('fixed_5sigma_radius_exact', np.array_equal(radius, full['bbox_radius_m']))
    check('same_common_A0_qualification', np.array_equal(load(T / 'world_prior_r1/A0_with_trellis_patch_nodes.npz')['valid'], load(P / 'cpu_initial_r1/A0.npz')['valid']))
    noncopy = ~copied
    result = dict(
        schema='T2_midpoint_full_cache_readonly_acceptance.v1', status='PASS', checks=checks,
        active_rows=n, copied_same_A0_rows=int(copied.sum()), computed_rows=int(computed.sum()),
        rows_with_T_material=int(noncopy.sum()), rows_with_nonzero_T_minus_A0=int(np.any(full['filtered_rgb_T'] != full['filtered_rgb_A0'], axis=1).sum()),
        common_valid_mass_quantiles=np.quantile(full['common_valid_mass'], [0, .01, .5, .99, 1]).tolist(),
        same_midpoint_T_minus_A0=stats(full['filtered_rgb_T'][noncopy].astype(np.float64) - full['filtered_rgb_A0'][noncopy]),
        midpoint_A0_minus_old_GH15=stats(full['filtered_rgb_A0'][noncopy].astype(np.float64) - old['filtered_rgb_A0'][noncopy]),
        source_sha256={str(path): sha(path) for path in [P / 'cpu_initial_r1/A0.npz', P / 'chart_plan_candidate_r1/geometry_plan.npz', T / 'world_prior_r1/mesh_uv_source.npz', T / 'world_prior_r1/basecolor_rgba.png', T / 'midpoint_full_r1/row_queries.npz']},
        GPU_operations=0, optimizer_updates=0, new_inference_calls=0,
        runtime_write_and_fresh_K_pending=True, final_native_render_pending=True, human_verdict=None)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
