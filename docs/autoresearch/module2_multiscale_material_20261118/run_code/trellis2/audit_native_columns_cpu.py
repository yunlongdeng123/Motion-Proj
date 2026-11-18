"""已保存T2两列的CPU源buffer/固定域审计；不重渲染、不优化。"""
from pathlib import Path
import json, os
import numpy as np
from run_native_columns import P, T, FRAMES, VARIANTS, load, sha, bytes_equal, write


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only audit requires CUDA_VISIBLE_DEVICES empty')
    output = T / 'native_evaluation_r1/source_buffer_audit.json'
    if output.exists():
        raise FileExistsError(output)
    prepared = json.loads((T / 'native_preparation_r1/manifest.json').read_text())
    inputs = load(prepared['input_npz'])
    parent = Path(prepared['coverage_parent'])
    road0, p0 = load(parent / 'road.npz'), load(parent / 'provenance.npz')
    proof0 = json.loads((parent / 'persistent/proof.json').read_text())
    rows, eligible = inputs['active_world_rows'], inputs['eligible']
    world_write = np.zeros(len(road0['active']), bool)
    world_write[rows] = eligible
    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        if not checks[name]:
            raise AssertionError(name)

    check('all_prepared_source_hashes_unchanged', all(sha(path) == expected for path, expected in prepared['source_sha256'].items()))
    check('prepared_input_hash_unchanged', sha(prepared['input_npz']) == prepared['input_sha256'])
    check('executed_source_exact_prepared', sha(T / 'native_columns_r1/source.py') == prepared['source_sha256'][str(T / 'code/run_native_columns.py')])
    import torch
    gates0 = torch.load(proof0['gates']['checkpoint']['path'], map_location='cpu', weights_only=False)
    results = {}
    for variant in VARIANTS:
        asset = T / f'native_columns_r1/{variant}/asset'
        road, provenance = load(asset / 'road.npz'), load(asset / 'provenance.npz')
        prefix = variant + '/'
        for key, value in road0.items():
            if key != 'reference_srgb':
                check(prefix + 'road_exact/' + key, bytes_equal(value, road[key]))
        check(prefix + 'effective_RGB_from_exact_prepared_all_active', bytes_equal(road['reference_srgb'][rows], inputs[variant]))
        check(prefix + 'all_nonqualified_inactive_RGB_exact', bytes_equal(road['reference_srgb'][~world_write], road0['reference_srgb'][~world_write]))
        for key, value in p0.items():
            if key != 'effective_row_rgb':
                check(prefix + 'original_provenance_exact/' + key, bytes_equal(value, provenance[key]))
        check(prefix + 'derived_effective_row_RGB_exact', bytes_equal(provenance['effective_row_rgb'], road['reference_srgb']))
        expected_T2 = np.zeros(len(world_write), bool)
        if variant == 'T2_mid':
            expected_T2[rows] = eligible & (inputs['material_T_weight_fraction'] > 0)
        check(prefix + 'separate_effective_T2_identity_exact', bytes_equal(provenance['effective_T2_prior'], expected_T2))
        check(prefix + 'common84266_write_exact', bytes_equal(provenance['effective_material_common_midpoint_write'], world_write))
        check(prefix + 'effective_T2_mass_zero_outside_permission', np.all(provenance['effective_T2_prior_weight_fraction'][~world_write] == 0))
        for filename, original in [('mesh_uv_source.npz', T / 'world_prior_r1/mesh_uv_source.npz'), ('basecolor_rgba.png', T / 'world_prior_r1/basecolor_rgba.png'), ('A0.npz', P / 'cpu_initial_r1/A0.npz')]:
            check(prefix + 'material_source_bytes/' + filename, sha(asset / 'material_source' / filename) == sha(original))
        bundled = load(asset / 'material_source/derived_row_ci.npz')
        check(prefix + 'derived_cache_scope_exact', bytes_equal(bundled['active_world_rows'], rows) and bytes_equal(bundled['eligible'], eligible))
        check(prefix + 'derived_cache_RGB_exact', bytes_equal(bundled['color'], inputs[variant]))
        proof = json.loads((asset / 'persistent/proof.json').read_text())
        for key in ['native_sources', 'ownership', 'topology', 'raster_units']:
            check(prefix + 'proof_prior_components_exact/' + key, proof[key] == proof0[key])
        gates = torch.load(proof['gates']['checkpoint']['path'], map_location='cpu', weights_only=False)
        for key in ['gate', 'contributors']:
            check(prefix + 'original_static_gate_tensor_exact/' + key, torch.equal(gates[key], gates0[key]))
        check(prefix + 'gate_indices_bytes_exact', sha(proof['gates']['indices']['path']) == sha(proof0['gates']['indices']['path']))
        Ktotal, audit_rows = 0, []
        for raw in FRAMES:
            native = load(T / f'native_columns_r1/{variant}/rendered/{raw:03d}_native.npz')
            baseline = load(prepared['parent_float_files'][str(raw)])
            check(prefix + f'{raw}/fixed_road_alpha_exact', bytes_equal(native['road_alpha'], baseline['road_alpha']))
            check(prefix + f'{raw}/fullK_mask_exact', bytes_equal(native['K'], baseline['K']))
            K = native['K']
            Kmax = float(np.abs(native['joint_rgb'][:, K].astype(np.float64) - baseline['joint_rgb'][:, K]).max())
            check(prefix + f'{raw}/jointK_RGB_exact', Kmax == 0)
            check(prefix + f'{raw}/native_RGB_finite', np.isfinite(native['road_rgb']).all() and np.isfinite(native['joint_rgb']).all())
            Ktotal += int(K.sum())
            audit_rows.append(dict(raw=raw, K_pixels=int(K.sum()), joint_K_RGB_max=Kmax, road_alpha_exact=True))
        check(prefix + 'K77266_complete', Ktotal == 77266)
        public = json.loads((asset.parent / 'public_reload.json').read_text())
        check(prefix + 'logged_actual_public_road_exact', public['all_exact'] and public['native_calls'] == 5 and all(row[k] == 0 for row in public['rows'] for k in ['RGB_max', 'alpha_max', 'RGB8_max']))
        results[variant] = dict(common_write_rows=int(world_write.sum()), T2_effective_rows=int(expected_T2.sum()),
            effective_RGB_changed_world_rows=int(np.any(road['reference_srgb'] != road0['reference_srgb'], axis=1).sum()),
            original_known_rows_receiving_T2=int((expected_T2 & p0['known']).sum()),
            original_B32_rows_receiving_T2=int((expected_T2 & p0['B32_new']).sum()),
            native_rows=audit_rows, public_reload='actual native scalar proof; no CPU re-render claimed')
    done = json.loads((T / 'native_columns_r1/completed.json').read_text())
    check('exact_30_forward_0_backward_0_update_0_new_inference', done['native_forward_calls'] == 30 and done['native_backward_calls'] == done['optimizer_updates'] == done['new_inference_calls'] == 0)
    write(output, dict(schema='TRELLIS2_native_source_buffer_CPU_audit.v1', status='PASS', checks=checks, variants=results,
        prior_observation_ledger_unchanged=True, geometry_coverage_topology_unchanged=True,
        CPU_Torch_map_location='cpu', GPU_operations=0, new_inference_calls=0, optimizer_updates=0,
        source_role='C/S-conditioned generated material prior; not independent truth', human_verdict=None))
    print(json.dumps(dict(status='PASS', checks=len(checks), output=str(output), variants=results)), flush=True)


if __name__ == '__main__':
    main()
