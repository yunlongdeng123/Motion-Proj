"""Independently read completed temporal receipts; never edit canonical evidence."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'component_diagnostic/fixed_pose_temporal32k'
RESULTS = ROOT / 'results'
OWN = HERE / 'sgcm_data_audit/fixed_pose_time_review'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = RESULTS / 'diagnostic_manifest.json'
    summary_path = RESULTS / 'temporal_summary.json'
    assert sha(manifest_path) == '69ebc14e8c6e87f48f3917e9a474800457df001f7cdb89ae6ff050ac322422a2'
    assert sha(summary_path) == 'f9a4d71fd73c9af117ee3f2f8d7d05cc30fcaf82b0820d0667cabfc46c6e136d'
    manifest = json.loads(manifest_path.read_bytes())
    summary = json.loads(summary_path.read_bytes())
    plan = json.loads((ROOT / 'plan.json').read_bytes())
    contacts = json.loads((RESULTS / 'contact_sheets/receipt.json').read_bytes())
    assert summary['actual_exit_code'] == 0 and len(manifest['records']) == 110
    assert manifest['actual_render_count'] == 98 and manifest['counterfactual_render_count'] == 12
    assert not manifest['optimizer_restored'] and not manifest['training_or_backward_started']
    assert manifest['original_file_SHA_unchanged'] and not manifest['original_parameters_source_config_modified']
    assert sha(ROOT / 'plan.json') == manifest['plan_sha256'] == '891907aa9630288765225579c7f374b8c175bc0adda90a46abf844558488deb4'
    records = {(r['arm'], r['mode'], r['time']['raw_index']): r for r in manifest['records']}
    assert len(records) == 110
    region = 'non_sky_outside_actor'
    maxima, endpoints = {}, []
    for arm in ('B1', 'V'):
        actual = [records[(arm, 'actual', i)] for i in range(49)]
        assert all(r['fixed_camera_frame_id'] == 'image_0/0017010' and r['RNG_unchanged'] for r in actual)
        anchor = actual[10]
        for r in (r for r in manifest['records'] if r['arm'] == arm):
            assert r['fixed_pose_hashes'] == anchor['fixed_pose_hashes']
            assert r['directions_sha256'] == anchor['directions_sha256']
            assert r['environment_texture_sha256'] == anchor['environment_texture_sha256']
            assert r['regions'][region]['pixel_count'] == 142486
        assert all(x['exact_equal'] for x in summary['arms'][arm]['native_raw10_parity'].values())
        assert all(summary['arms'][arm]['native_png_parity'].values())
        maxima[arm] = {key: max(r['region_change_from_anchor'][region][key]['mean_abs'] for r in actual)
                        for key in summary['arms'][arm]['all49_max_regional_mean_abs_change_from_anchor']}
        assert maxima[arm] == summary['arms'][arm]['all49_max_regional_mean_abs_change_from_anchor']
        for raw in (0, 48):
            for mode in ('actual', 'freeze_SHM_position', 'freeze_temporal_opacity_and_support', 'freeze_both'):
                r = records[(arm, mode, raw)]
                if mode == 'freeze_both':
                    assert all(v['exact_equal'] for d in r['region_change_from_anchor'].values() for v in d.values())
                endpoints.append({'arm': arm, 'time_source_raw_index': raw, 'timestamp_model': r['time']['timestamp'], 'mode': mode,
                                  'static_RGB_MAE_vs_same_arm_anchor': r['region_change_from_anchor'][region]['static_full']['mean_abs'],
                                  'static_RGB_MAE_vs_same_time_actual': r.get('region_change_from_same_time_actual', {}).get(region, {}).get('static_full', {}).get('mean_abs', 0.0)})
    copied_contacts = []
    for row in contacts['records']:
        path = RESULTS / 'contact_sheets' / Path(row['path']).name
        assert sha(path) == row['sha256']
        copied_contacts.append({'file': path.name, 'sha256': sha(path), 'original_source_count': len(row['sources'])})
    assert len(copied_contacts) == 4
    original = json.loads((HERE / 'loss_audit/next_diagnostics_cpu/observed_static_rgb_S/materialized_plan/experiment_binding.json').read_bytes())['original_source_sha256']
    repository = HERE / 'loss_audit/next_diagnostics_cpu/observed_static_rgb_S/frozen_repository'
    unit_sources = {}
    for name in ('scene/waymo_loader2.py', 'scene/gaussian_model.py'):
        assert sha(repository / name) == original[name]
        unit_sources[name] = sha(repository / name)
    loader = (repository / 'scene/waymo_loader2.py').read_text(encoding='utf8')
    model = (repository / 'scene/gaussian_model.py').read_text(encoding='utf8')
    assert 'static_pointcloud = (np.pad(static_pointcloud' in loader and '@ transform.T)[:, :3]' in loader
    assert 'fused_point_cloud = torch.tensor(np.asarray(pcd.points))' in model
    assert plan['model_units_per_meter'] == .1
    result = {'status': 'PASS_ACTUAL_TEMPORAL_RECEIPT_INTERPRETATION_NOT_QUALITY_ACCEPTANCE',
              'manifest_sha256': sha(manifest_path), 'summary_sha256': sha(summary_path),
              'plan_sha256': sha(ROOT / 'plan.json'), 'contact_receipt_sha256': sha(RESULTS / 'contact_sheets/receipt.json'),
              'record_count': 110, 'actual_timestamps_per_arm': 49, 'region': region, 'region_pixel_count': 142486,
              'all49_max_raw_FP32_regional_MAE_vs_anchor': maxima, 'endpoint_static_RGB_deltas': endpoints,
              'contacts': copied_contacts,
              'metric_spatial_unit_evidence': {'actual_PCA_model_units_per_original_world_meter': .1, 'metric_point_displacement_formula': 'norm(model_xyz(t)-model_xyz(anchor)) /0.1', 'same_recorded_transform': [r for path, r in plan['protected_inputs'].items() if path.endswith('/transform_matrix.txt')], 'frozen_loader_and_model_source_sha256': unit_sources, 'chain': 'Loader transforms metric source static/dynamic pointcloud and camera translations by PCA scale; Gaussian xyz is directly pcd.points. Diagnostic already divides point displacement by actual .1. No canonical unit correction needed.', 'depth_note': 'Saved normalized-foreground depth arrays/statistics remain model units; displayed0..100m depth PNG divides by.1. Timestamp values are model-time, not declared wall-clock seconds.'},
              'interpretation': 'At the two sampled counterfactual endpoints, freezing temporal opacity/prefilter support changes static RGB more than freezing SHM position. At raw48, both the change from anchored output and the control-vs-actual differences support this direction. Neither control effect is an additive fraction of the original change.',
              'cycle_caveat': 'raw0 time-.48 differs from raw10 anchor-.28 by one cycle=.2, so SHM position is almost phase-equivalent, up to FP32 errors. This endpoint cannot establish motion insignificance at other timestamps. Only actual outputs, not separate controls, cover all49.',
              'scientific_limits': ['Fixed raw10 pose with other observed timestamps is a counterfactual view/time combination; no non-anchor GT quality or PSNR claim.', 'This shows time dependence of existing static model; freeze_both exact is a control validation, not repair success or proof of author hard-freeze intent.', 'Pixel deltas use raw FP32 RGB, not clipping/GT quality.', 'Potentially visible point stratum means temporal marginal>.05 and effective opacity>=1/255 at some sampled time, not actual frustum/pixel visibility.', 'Source/input/parameter/env/camera/ray/RNG invariants are recorded by the executed source; this independent reader rechecks saved receipts, not new GPU runs.'],
              'architecture_components': 'B1/V32k -> fixed raw10 pose+rays -> observed49 timestamps / endpoint position-or-opacity control -> unchanged renderer -> RGB/alpha/depth delta receipts -> limited static-time consistency conclusion',
              'normal_algorithm_success_claimed': False, 'GPU_or_training_started_by_this_reader': False}
    OWN.mkdir(parents=True, exist_ok=True)
    path = OWN / 'actual_temporal_interpretation_cpu.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'sha256': sha(path), 'actual_records': 110, 'contacts': 4}))


if __name__ == '__main__':
    main()
