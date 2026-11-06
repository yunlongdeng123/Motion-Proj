"""Pure CPU verification of the frozen RCR-1 + current C2/SurfaceLOD inputs.

This module never imports a renderer, constructs a Runtime, initializes CUDA,
loads the multi-GB native checkpoint payloads, or starts a training job.
Native checkpoint identities are checked against the existing RCR-1 receipt.
The small road checkpoint is deserialized with map_location='cpu'.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


PASS = 'PASS_V44_CPU_INPUT_CONTRACT'
BLOCKED = 'BLOCKED_V44_CPU_INPUT_CONTRACT'
GEOMETRY = (
    'xyz', 'normal', 'sigma', 'opacity', 'thickness', 'box',
    'knn_spacing', 'normal_radius', 'normal_confidence', 'generated',
    'conditioning', 'normal_valid', 'out_of_box', 'conditioning_held',
    'source_index', 'conditioning_observation_index',
    'conditioning_raw_frame', 'conditioning_cloud_row',
)
RESERVED = set(GEOMETRY) | {
    'canonical_sh', 'rgb', 'colour_inferred', 'metadata_json',
    'coefficients', 'sh', 'sh_degree', 'geometry_hash', 'provenance_hash',
}


def load_contract(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_contract(contract_path, receipt_path=None):
    """Return a PASS/BLOCKED receipt; write only when receipt_path is supplied.

    A caller must require status == PASS. A CPU PASS establishes frozen input
    compatibility only; it does not establish visual quality or GPU readiness.
    """
    start = time.monotonic()
    result = dict(
        status=BLOCKED, version='v44', contract_path=str(Path(contract_path).resolve()),
        checked_at_utc=datetime.now(timezone.utc).isoformat(), checks=[], failures=[],
        device='cpu', cuda_operations_requested=0, renderer_imports=0,
        native_checkpoint_payloads_loaded=0, multi_GB_checkpoints_hashed=0,
        training_updates=0, calibration_updates=0, new_calibration=False,
        gpu_render_verified=False, quality_accepted=False,
        scope='Frozen inputs and CPU structural compatibility only',
    )

    def check(name, condition, detail=None):
        entry = dict(name=name, passed=bool(condition))
        if detail is not None:
            entry['detail'] = detail
        result['checks'].append(entry)
        if not condition:
            raise ValueError(name)

    try:
        import numpy as np
        import torch
        from scipy.interpolate import BSpline

        torch.set_num_threads(1)
        contract = load_contract(contract_path)
        result['contract_sha256'] = sha_file(contract_path)
        check('contract_schema', contract['schema'] == 'scene008.v44.rgb_asset.frozen_inputs.v1')
        paths = {key: Path(value) for key, value in contract['paths'].items()}
        expected = contract['expected']
        policy = contract['policy']
        check('fine_policy_is_explicit_frozen_transfer',
              policy['calibration_mode'] == 'frozen_transfer_from_rcr1_m2'
              and policy['recalibrated_for_current_asset'] is False
              and policy['new_training_updates'] == 0)
        for name, path in paths.items():
            check('path_exists:' + name, path.exists(), str(path))

        # All expected hashes here are small source/data inputs, never native
        # tensor checkpoints. Enforce that boundary even if a contract is edited.
        checked_bytes = 0
        for path_string, wanted in expected['small_file_sha256'].items():
            path = Path(path_string)
            check('small_file_exists:' + path.name, path.is_file(), path_string)
            check('small_file_size_boundary:' + path.name,
                  path.stat().st_size <= 64 * 1024 * 1024, path.stat().st_size)
            actual = sha_file(path)
            check('sha256:' + path_string, actual == wanted, actual)
            checked_bytes += path.stat().st_size
        result['small_files_hashed'] = len(expected['small_file_sha256'])
        result['small_file_bytes_hashed'] = checked_bytes

        source = load_contract(paths['source_receipt'])
        binding = source['binding']
        check('source_binding_hash', source['binding_sha256'] == expected['rcr1_source_binding_sha256']
              == stable_hash(binding), source['binding_sha256'])
        check('source_complete_m2_receipt', source['M2_tensor_count'] == 27
              and source['M2_all_27_tensors_bitwise_unchanged_CPU'] is True)
        check('native10k_parent', str(paths['native_parent']) == binding['native_parent']
              and binding['native_iteration'] == 10000)
        check('freshroad_parent', str(paths['road_run']) == binding['road_run']
              and binding['road_iteration'] == 1000)
        identities = {}
        for path_string, wanted in binding['parent_identity'].items():
            path = Path(path_string)
            actual = [path.stat().st_size, path.stat().st_mtime_ns]
            check('native_checkpoint_identity:' + path.name, actual == list(wanted), actual)
            identities[path.name] = actual
        result['native_checkpoint_identity'] = identities
        for path_string, wanted in binding['native_metadata_sha256'].items():
            check('native_metadata:' + Path(path_string).name,
                  sha_file(path_string) == wanted)
        replay = load_contract(paths['road_run'] / 'replay_binding.json')
        check('freshroad_replay_binding',
              replay['native_parent'] == binding['native_parent']
              and replay['native_iteration'] == 10000
              and replay['parent_identity'] == binding['parent_identity']
              and replay['parent_metadata_sha256'] == binding['native_metadata_sha256'])

        # This 12 MB road checkpoint is the only tensor checkpoint read here.
        state = torch.load(str(paths['road_checkpoint']), map_location='cpu', weights_only=False)
        repair = state['repair']
        road_contract = state['contract']
        check('road_checkpoint_step_and_native_identity', state['step'] == 1000
              and stable_hash(state['parent_identity']) == stable_hash(binding['parent_identity']))
        check('road_checkpoint_method',
              road_contract['native_parent'] == binding['native_parent']
              and road_contract['native_iteration'] == 10000
              and road_contract['target'] == [37]
              and road_contract['only_static_opacity_gate_trainable'] is True
              and road_contract['road_colour_geometry_opacity_fixed'] is True)
        check('road_repair_cpu_finite', all(
            value.device.type == 'cpu' and bool(torch.isfinite(value).all())
            for value in repair.values()))
        check('road_repair_state_hash', tensor_hash(repair) == binding['repair_state_sha256'])
        with np.load(paths['road_run'] / 'inputs/contributors.npz', allow_pickle=False) as saved:
            selected = saved['ids'].copy()
        ids = repair['ids'].numpy()
        selection = load_contract(paths['road_run'] / 'selection.json')
        check('freshroad_contributor_ids', np.array_equal(selected, ids)
              and ids.dtype == np.int64 and ids.ndim == 1
              and len(ids) == expected['gate_rows'] == selection['candidate_rows']
              and len(np.unique(ids)) == len(ids)
              and ids.min() >= 0 and ids.max() < selection['static_rows'])
        check('road_tensor_schema', list(repair['road.dc'].shape) == [expected['road_rows'], 1, 3]
              and list(repair['gate_raw'].shape) == [len(ids)]
              and list(repair['road.xyz'].shape) == [expected['road_rows'], 3])
        baseline = .5 + .28209479177387814 * repair['road.dc'][:, 0]
        check('road_baseline_hash', tensor_hash({'colours': baseline})
              == binding['road_colour_baseline_sha256'])

        fine_meta = load_contract(paths['fine'] / 'scene_binding.json')
        coarse_meta = load_contract(paths['coarse'] / 'scene_binding.json')
        check('fine_keeps_original_m2_source_binding',
              fine_meta['binding_sha256'] == coarse_meta['binding_sha256'] == source['binding_sha256']
              and fine_meta['binding'] == binding
              and fine_meta['stage'] == 'fine11x9'
              and fine_meta['preflight_only'] is False
              and coarse_meta['stage'] == 'coarse6x5')
        check('fine_baseline_and_coarse_provenance',
              fine_meta['baseline_road_sha256'] == binding['road_colour_baseline_sha256']
              and fine_meta['coarse_source']['path'] == str(paths['coarse'])
              and sha_file(paths['coarse'] / 'coefficients.npz')
              == fine_meta['coarse_source']['coefficients_sha256'])
        with np.load(paths['fine'] / 'coefficients.npz', allow_pickle=False) as saved:
            world = saved['world'].copy()
            colour_baseline = saved['colour_baseline'].copy()
        check('fine_coefficients_finite_bounded_99x3', world.shape == (99, 3)
              and np.isfinite(world).all() and np.abs(world).max() <= .09000001)
        check('fine_exact_same_road_baseline', np.array_equal(colour_baseline, baseline.numpy()))
        check('fine_source_coefficients_hash', sha_file(paths['fine'] / 'coefficients.npz')
              == source['fine_binding']['coefficients_sha256'])
        knots = load_contract(paths['fine'] / 'design.json')['knots']
        check('fine_world_knots', knots == source['fine_binding']['knots']
              and knots['grid'] == [11, 9])
        result['fine_transfer'] = dict(
            mode=policy['calibration_mode'], source_binding_sha256=source['binding_sha256'],
            source_M2_sha256=binding['M2_sha256'],
            source_coefficients_sha256=source['fine_binding']['coefficients_sha256'],
            source_gpu_colour_table_sha256=source['fine_binding']['colour_table_sha256'],
            recalibrated_for_current_asset=False, source_metadata_rewritten=False,
            current_asset_sha256=expected['asset_sha256'],
        )
        check('current_asset_is_c2_not_old_m2',
              expected['asset_sha256'] != binding['M2_sha256']
              and sha_file(paths['asset']) == expected['asset_sha256'])
        check('current_posterior_hash', sha_file(paths['posterior']) == expected['posterior_sha256'])

        with np.load(paths['asset'], allow_pickle=False) as saved:
            asset = {key: saved[key].copy() for key in saved.files}
        count = len(asset['xyz'])
        chart = asset['representation_kind'] == 1
        retained = ~chart
        check('c2_schema_rows', count == expected['asset_rows']
              and int(chart.sum()) == expected['chart_rows']
              and int(retained.sum()) == expected['retained_rows']
              and asset['canonical_sh'].shape == (count, 16, 3)
              and asset['xyz'].shape == asset['normal'].shape == (count, 3)
              and asset['resample_reference_pixel_xy'].shape == (count, 2))
        check('canonical_sh3_actual_nonDC_exact_zero',
              int(asset['sh_degree']) == 3 and np.count_nonzero(asset['canonical_sh'][:, 1:]) == 0)
        check('canonical_rgb_exact_dc', np.array_equal(asset['rgb'],
              asset['canonical_sh'][:, 0] * np.float32(.28209479177387814) + np.float32(.5)))
        check('c2_required_geometry_finite', all(np.isfinite(asset[key]).all() for key in GEOMETRY))
        check('c2_positive_scales_and_peak_opacity',
              (asset['sigma'] > 0).all() and (asset['thickness'] > 0).all()
              and (asset['opacity'] > 0).all() and (asset['opacity'] < 1).all()
              and (asset['opacity'][chart] <= np.float32(.9900001)).all())
        digest = hashlib.sha256()
        for key in sorted(GEOMETRY):
            value = np.ascontiguousarray(asset[key])
            digest.update(key.encode()); digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        check('c2_stored_geometry_hash', digest.hexdigest() == str(asset['geometry_hash']))
        digest = hashlib.sha256()
        for key in sorted(set(asset) - RESERVED):
            value = np.ascontiguousarray(asset[key])
            digest.update(key.encode()); digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        check('c2_stored_provenance_hash', digest.hexdigest() == str(asset['provenance_hash']))
        uv = asset['resample_reference_pixel_xy'][chart].astype(np.float64)
        check('c2_chart_unique_integer_uv', np.array_equal(uv, np.rint(uv))
              and len(np.unique(uv[:, 1] * 1920 + uv[:, 0])) == len(uv)
              and (uv[:, 0] >= 0).all() and (uv[:, 0] < 1920).all()
              and (uv[:, 1] >= 0).all() and (uv[:, 1] < 1280).all())
        with np.load(paths['posterior'], allow_pickle=False) as saved:
            posterior = {key: saved[key].copy() for key in (
                'u_knots_pixels', 'v_knots_pixels', 'control_delta_z_m', 'z_reference_m',
                'reference_K', 'reference_camera_from_local_m')}
        check('posterior_finite_used_arrays', all(np.isfinite(value).all() for value in posterior.values()))
        tu, tv = posterior['u_knots_pixels'], posterior['v_knots_pixels']
        coefficients = posterior['control_delta_z_m'].reshape(len(tu) - 4, len(tv) - 4)
        bu = BSpline(tu, np.eye(len(tu) - 4), 3, extrapolate=False)(uv[:, 0])
        bv = BSpline(tv, np.eye(len(tv) - 4), 3, extrapolate=False)(uv[:, 1])
        depth = float(posterior['z_reference_m']) + np.sum((bu @ coefficients) * bv, axis=1)
        K = posterior['reference_K']
        camera = posterior['reference_camera_from_local_m']
        rays = np.column_stack((uv + .5, np.ones(len(uv)))) @ np.linalg.inv(K).T
        local = (rays * depth[:, None] - camera[:3, 3]) @ np.linalg.inv(camera[:3, :3]).T
        xyz_error = float(np.abs(local.astype(np.float32) - asset['xyz'][chart]).max())
        sigma_error = float(np.abs((.6 * depth / K[0, 0]).astype(np.float32) - asset['sigma'][chart]).max())
        check('posterior_reproduces_canonical_c2_surface', np.isfinite(depth).all()
              and (depth > 0).all() and xyz_error < 2e-6 and sigma_error < 2e-7,
              dict(xyz_maxabs_m=xyz_error, sigma_maxabs_m=sigma_error))

        actual_binding = load_contract(paths['native_parent'] / 'actual_scene_binding.json')
        check('native_same_49_raws', actual_binding['frame_ids']
              == [f'image_0/0008{raw:03d}' for raw in range(89, 138)])
        records = load_contract(paths['native_parent'] / 'cameras.json')
        records = {row['img_name']: row for row in records}
        with np.load(paths['fixed_reference'], allow_pickle=False) as saved:
            fixed_raws = saved['raw_ids'].copy()
            fixed_cameras = saved['model_to_camera'].copy()
            fixed_times = saved['model_time'].copy()
            fixed_layers = saved['layer_names'].tolist()
            fixed_ground = np.squeeze(saved['depth_m'][:, 3])
        check('fixed_reference_49raw_ground_proxy',
              np.array_equal(fixed_raws, np.arange(89, 138))
              and fixed_layers[3] == 'ground_beneath_a37_proxy'
              and fixed_ground.shape == (49, 320, 480))
        for raw in range(89, 138):
            with np.load(paths['observations'] / f'{raw:03d}.npz', allow_pickle=False) as saved:
                obs_camera = saved['model_to_camera'].copy()
                obs_time = float(saved['model_time'])
                obs_K = saved['K'].copy()
                obs_pose = saved['model_from_local'][0].copy()
                rgb = saved['rgb'].copy()
                road_rgb = saved['road_edit_rgb'].copy()
                visible = saved['visible'].copy().astype(bool)
                depth_valid = saved['depth_valid'].copy().astype(bool)
                depth_m = saved['depth_m'].copy()
            check(f'observation_rgb_depth_schema_raw{raw:03d}',
                  rgb.shape == road_rgb.shape == (320, 480, 3)
                  and visible.shape == depth_valid.shape == depth_m.shape == (2, 320, 480)
                  and np.isfinite(rgb).all() and np.isfinite(road_rgb).all()
                  and not (depth_valid & ~visible).any()
                  and np.isfinite(depth_m[depth_valid]).all() and (depth_m[depth_valid] > 0).all())
            row = records[f'0008{raw:03d}']
            record_K = np.array([[row['fx'] / 4, 0, row['cx'] / 4],
                                 [0, row['fy'] / 4, row['cy'] / 4], [0, 0, 1]])
            check(f'camera_time_pose_raw{raw:03d}',
                  np.allclose(obs_camera, fixed_cameras[raw - 89], atol=1e-6, rtol=0)
                  and abs(obs_time - float(fixed_times[raw - 89])) < 1e-6
                  and np.allclose(obs_K, record_K, atol=1e-6, rtol=0)
                  and np.isfinite(obs_pose).all()
                  and (paths['road_run'] / 'inputs/views' / f'{raw:03d}.npz').is_file())
            if raw == 106:
                intrinsic = obs_K.astype(float).copy(); intrinsic[:2] *= 4
                reference_camera = obs_camera.astype(float) @ obs_pose.astype(float)
                reference_camera[:3, :4] /= np.linalg.norm(obs_pose[:3, :3], axis=0).mean()
                check('posterior_uses_current_raw106_camera_and_local_pose',
                      np.allclose(K, intrinsic, atol=1e-6, rtol=0)
                      and np.allclose(camera, reference_camera, atol=2e-6, rtol=0))
        result['asset'] = dict(rows=count, chart_rows=int(chart.sum()), retained_rows=int(retained.sum()),
                              nonDC_nonzero=0, geometry_hash=str(asset['geometry_hash']),
                              provenance_hash=str(asset['provenance_hash']))
        result['native_raws'] = list(range(89, 138))
        result['runtime_checks_deferred'] = [
            'GPU native static/dynamic identity and full actor labels',
            'Original shared-depth CUDA rendering and finite RGB/alpha/depth',
            'Keep/delete asset rows equality and complete a37 removal',
            '49 keep/delete visual acceptance including raw130/raw137 silver residual',
        ]
        result['status'] = PASS
    except Exception as error:
        result['failures'].append(dict(type=type(error).__name__, message=str(error)))
    result['seconds'] = time.monotonic() - start
    result['checks_passed'] = sum(row['passed'] for row in result['checks'])
    if receipt_path is not None:
        destination = Path(receipt_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                               encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('contract', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.dont_write_bytecode = True
    receipt = validate_contract(args.contract, args.receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'checks'},
                     indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if receipt['status'] == PASS else 2


if __name__ == '__main__':
    raise SystemExit(main())
