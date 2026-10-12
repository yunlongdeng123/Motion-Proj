"""Approved-only read-only fixed raw10 pose / static-time diagnostic; no training."""
import argparse
import datetime
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
C = BASE / 'campaign/vgpu_bugfix_20261012'
D = C / 'component_diagnostic'
HERE = D / 'fixed_pose_temporal32k'
PLAN = HERE / 'plan.json'
REPO = BASE / 'campaign/resume_vgpu_baseline_20261011/frozen_repository'
PARENT = BASE / 'runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline'
ARMS = {'B1': C / 'continuation_diagnostic/original30k_to32000',
        'V': C / 'continuation_diagnostic/opacity_outside30k_to32000'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def pin(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns, 'sha256': sha(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def helper():
    spec = importlib.util.spec_from_file_location('proven_FP32_inference_helpers', D / 'scripts/component_diagnostic.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation initialized CUDA')
    audit = read(D / 'cpu_audit.json')
    camera = next(row for row in audit['records']['30000']['frames'] if row['frame_id'] == 'image_0/0017010')
    event_path = PARENT / 'checkpoint_events/iteration_030000.json'
    times = sorted(read(event_path)['camera_manifest'], key=lambda row: row['frame_id'])
    assert [row['frame_id'] for row in times] == [f'image_0/0017{i:03}' for i in range(49)]
    assert times[10]['timestamp'] == camera['target_timestamp_model'] == -.28
    source = {row['path']: row for name, row in audit['frozen_source_modules'].items()
              if name in ('gaussian_renderer/__init__.py', 'scene/__init__.py', 'scene/gaussian_model.py',
                          'scene/cameras.py', 'scene/envlight.py', 'utils/graphics_utils.py',
                          'utils/general_utils.py', 'utils/sh_utils.py')}
    source.update({row['path']: row for row in audit['kernel_pins']})
    for path, row in source.items():
        assert sha(path) == row['sha256'], path
    source[str(D / 'scripts/component_diagnostic.py')] = pin(D / 'scripts/component_diagnostic.py')
    assert source[str(D / 'scripts/component_diagnostic.py')]['sha256'] == '0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa'
    checkpoints, configs, anchors, accepted_region_counts = {}, {}, {}, {}
    for arm, out in ARMS.items():
        manifest_path = out / 'terminal_32000/diagnostic_snapshot_manifest.json'
        snapshot = read(manifest_path)
        assert snapshot['iteration'] == 32000 and snapshot['status'] == 'complete'
        assert read(out / 'status.json')['actual_exit_code'] == 0
        checkpoints[arm] = {}
        for branch in ('static', 'dynamic', 'environment'):
            row = snapshot['checkpoints'][branch]
            actual = pin(row['path'])
            assert actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], (arm, branch)
            checkpoints[arm][branch] = actual
        configs[arm] = snapshot['training_config']
        evaluation_path = out / 'extra_eval/32000/diagnostic_eval.json'
        frame = next(row for row in read(evaluation_path)['frames'] if row['frame_id'] == camera['frame_id'])
        accepted_region_counts[arm] = {key: value['pixel_count'] for key, value in frame['regions'].items()}
        anchors[arm] = {'evaluation': pin(evaluation_path), 'float32_arrays': pin(frame['float32_arrays']),
                        'static_png': pin(frame['outputs']['static']), 'alpha_png': pin(frame['outputs']['alpha_s'])}
    transform = Path(configs['B1']['source_path']) / 'transform_matrix.txt'
    import numpy as np
    matrix = np.loadtxt(transform)
    scale = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    assert np.allclose(scale, scale[0], rtol=1e-9, atol=1e-12) and scale[0] > 0
    used = dict(source)
    for out in ARMS.values():
        used[str(out / 'terminal_32000/diagnostic_snapshot_manifest.json')] = pin(out / 'terminal_32000/diagnostic_snapshot_manifest.json')
    source_data = Path(configs['B1']['source_path'])
    actor_path = source_data / 'dynamic_masks/dynamic_mask_0/0017010.png'
    sky_path = source_data / 'sky/sky_0/0017010.png'
    from PIL import Image
    import cv2
    actor = np.asarray(Image.open(actor_path).convert('L').resize((480, 320), Image.Resampling.NEAREST)) > 0
    sky = cv2.resize((np.asarray(Image.open(sky_path)) > 0).astype(np.float32), (480, 320)) > 0
    masks = {'full': np.ones((320, 480), bool), 'sky': sky, 'non_sky': ~sky, 'actor': actor,
             'outside_actor': ~actor, 'non_sky_outside_actor': ~sky & ~actor,
             'lower_image_non_sky_outside_actor_rows200_319': (~sky & ~actor) & (np.arange(320)[:, None] >= 200)}
    for arm, counts in accepted_region_counts.items():
        assert {key: int(masks[key].sum()) for key in counts} == counts, arm
    for path in (event_path, PARENT / 'cameras.json', transform, Path(camera['source_image']), actor_path, sky_path):
        used[str(path)] = pin(path)
    for rows in checkpoints.values():
        used.update({row['path']: row for row in rows.values()})
    for rows in anchors.values():
        used.update({row['path']: row for row in rows.values()})
    result = {'schema_version': 1, 'status': 'CPU_prepared_GPU_not_started', 'repository': str(REPO),
              'output': str(HERE / 'results'), 'script_sha256': sha(__file__),
              'launcher_sha256': sha(HERE / 'launch_fixed_pose_temporal.py'),
              'camera': camera, 'observed_times': [{'raw_index': i, 'original_frame_id': row['frame_id'], 'timestamp': row['timestamp']} for i, row in enumerate(times)],
              'anchor_raw_index': 10, 'anchor_timestamp': -.28,
              'counterfactual_raw_indices': [0, 48],
              'counterfactuals': {'freeze_SHM_position': 'get_xyz_SHM(t) returns original get_xyz_SHM(anchor); same SH direction uses frozen positions.',
                                  'freeze_temporal_opacity_and_support': 'get_marginal_t(t) returns original marginal(anchor), so effective opacity and original marginal>.05 prefilter both stay anchored.',
                                  'freeze_both': 'Both above independent controls; expected fixed static output across time, diagnostic only.'},
              'actual_render_count': 98, 'counterfactual_render_count': 12,
              'model_units_per_meter': float(scale[0]), 'checkpoints': checkpoints, 'configs': configs, 'anchor_references': anchors,
              'raw10_region_counts': {key: int(value.sum()) for key, value in masks.items()},
              'raw10_saved_eval_region_counts': accepted_region_counts,
              'source_and_kernel_pins': source, 'protected_inputs': used,
              'render_policy': 'Reuse unchanged official combined renderer and proven CPU-strip/GPU inference loader; take static branch. Dynamic is loaded only because official original config requests combined renderer. Never restore Adam or training_state; no training_setup, backward or parameter/source/config writes.',
              'pose_policy': 'Single original raw10 camera; only timestamp scalar changes. Pose/projection/intrinsics/image/size/world directions/environment texture remain exact. is_training=False, time_shift=None, no jitter/AMP.',
              'output_channels': ['static_full', 'static_Gaussian_RGB_derived_as_static_minus_environment', 'environment_transmission', 'static_alpha', 'static_depth_weighted_raw', 'static_depth_normalized_foreground'],
              'depth_preview': 'Normalized foreground depth in meters, fixed0..100m gray; alpha<=1e-5 has no foreground depth. Float32 raw/model-unit arrays preserved.',
              'metrics': 'Per-frame static/foreground/env/alpha/depth change vs same-arm anchored pose/time; full and frozen raw10 mask regions; point XYZ displacement and effective-opacity range across all49, with all-point and potentially visible (marginal>.05 and opacity>=1/255 at any sampledtime) strata.',
              'scientific_limits': ['The49 timestamps are observed but their combination with fixed raw10 pose is a novel camera/time diagnostic. No non-anchor GT quality/PSNR claim.', 'Read-only freezes are counterfactual causal controls, not repairs or author-hard-freeze requirements.', 'Derived Gaussian RGB is official static minus official env contribution, preserving FP32 subtraction semantics.'],
              'allow_GPU': False, 'original_files_modified': False, 'training_started': False}
    write(PLAN, result)
    left, right = torch.tensor([1., 3.]), torch.tensor([2., 2.])
    torch.maximum(left, right, out=left)
    assert torch.equal(left, torch.tensor([2., 3.]))
    torch.minimum(left, right, out=left)
    assert torch.equal(left, torch.tensor([2., 2.]))
    write(HERE / 'cpu_preflight.json', {'status': 'PASS', 'CUDA_initialized': torch.cuda.is_initialized(),
                                     'plan_sha256': sha(PLAN), 'script_sha256': sha(__file__),
                                     'actual_times': 49, 'used_checkpoint_components_verified': 6,
                                     'Torch_maximum_minimum_out_CPU_test': 'PASS',
                                     'Tensor_has_maximum_inplace_method': hasattr(torch.tensor(0.), 'maximum_'),
                                     'raw10_region_counts': result['raw10_region_counts'],
                                     'raw10_saved_eval_six_region_counts_exact_B1_V': True,
                                     'checkpoint_copies_created': 0, 'training_state_or_optimizer_restore_tests_run': False,
                                     'training_or_GPU_started': False})
    print(json.dumps({'status': 'PASS_CPU_only_ready', 'plan_sha256': sha(PLAN), 'script_sha256': sha(__file__), 'launcher_sha256': result['launcher_sha256']}))


class StaticTimeControl:
    def __init__(self, model, anchor, mode):
        self.model, self.anchor, self.mode = model, anchor, mode

    def __getattr__(self, name):
        return getattr(self.model, name)

    def get_xyz_SHM(self, timestamp, times=1):
        return self.model.get_xyz_SHM(self.anchor if self.mode in ('freeze_SHM_position', 'freeze_both') else timestamp, times)

    def get_marginal_t(self, timestamp):
        return self.model.get_marginal_t(self.anchor if self.mode in ('freeze_temporal_opacity_and_support', 'freeze_both') else timestamp)


def run(approval_path):
    plan = read(PLAN)
    approval = read(approval_path)
    assert approval['decision'] == 'APPROVED_READONLY_FIXED_POSE_TEMPORAL_32K'
    assert approval['plan_sha256'] == sha(PLAN) and approval['script_sha256'] == sha(__file__)
    assert approval['launcher_sha256'] == sha(HERE / 'launch_fixed_pose_temporal.py')
    out = Path(plan['output'])
    assert not out.exists(), 'Do not duplicate or overwrite a diagnostic attempt'
    for path, row in plan['protected_inputs'].items():
        assert Path(path).stat().st_size == row['bytes'] and Path(path).stat().st_mtime_ns == row['mtime_ns'], path
    out.mkdir()
    status = {'status': 'initializing', 'owner_pid': os.getpid(), 'started_utc': now(), 'gpu_lock_held': False}
    write(out / 'status.json', status)
    code = 1
    fd = (BASE / 'campaign/gpu_slot.lock').open('r+')
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
        status['gpu_lock_held'] = True
        write(out / 'status.json', status)
        gpu_run(plan, out, status)
        status.update(status='complete', actual_exit_code=0)
        code = 0
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc(), actual_exit_code=1)
        raise
    finally:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()
        status.update(gpu_lock_held=False, finished_utc=now())
        write(out / 'status.json', status)
        (out / 'actual_exit_code.txt').write_text(str(code) + '\n')


def gpu_run(plan, out, status):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.utils import save_image
    from omegaconf import OmegaConf
    sys.path.insert(0, str(REPO))
    import gaussian_renderer
    import scene.gaussian_model as gm
    import scene.cameras as cameras
    import scene.envlight as envlight
    import diff_gaussian_rasterization as raster
    h = helper()
    for module in (gaussian_renderer, gm, cameras, envlight, raster, raster._C):
        path = str(Path(module.__file__).resolve())
        assert path in plan['source_and_kernel_pins'] and sha(path) == plan['source_and_kernel_pins'][path]['sha256'], path
    started = time.monotonic()
    records, point_summaries = [], {}
    torch.cuda.reset_peak_memory_stats()
    row = plan['camera']
    source = Path(plan['configs']['B1']['source_path'])
    import cv2
    actor = np.asarray(Image.open(source / 'dynamic_masks/dynamic_mask_0/0017010.png').convert('L').resize((480, 320), Image.Resampling.NEAREST)) > 0
    sky_raw = np.asarray(Image.open(source / 'sky/sky_0/0017010.png'))
    sky = cv2.resize((sky_raw > 0).astype(np.float32), (480, 320)) > 0
    masks = {'full': np.ones((320, 480), bool), 'sky': sky, 'non_sky': ~sky, 'actor': actor,
             'outside_actor': ~actor, 'non_sky_outside_actor': ~sky & ~actor,
             'lower_image_non_sky_outside_actor_rows200_319': (~sky & ~actor) & (np.arange(320)[:, None] >= 200)}
    assert {key: int(value.sum()) for key, value in masks.items()} == plan['raw10_region_counts']
    with torch.no_grad():
        for arm in ('B1', 'V'):
            cfg = OmegaConf.create(plan['configs'][arm])
            models = []
            for branch, cls in (('static', gm.GaussianModel), ('dynamic', gm.Dynamic_GaussianModel)):
                capture, iteration = torch.load(plan['checkpoints'][arm][branch]['path'], map_location='cpu', weights_only=False)
                assert iteration == 32000 and len(capture) == 18
                capture = list(capture)
                capture[14] = {}  # Discard Adam on CPU before any tensor moves to CUDA.
                model = cls(cfg)
                model.restore(h.cuda_capture(tuple(capture)))
                assert model.optimizer is None
                models.append(model)
                del capture
                gc.collect()
            static, dynamic = models
            capture, iteration = torch.load(plan['checkpoints'][arm]['environment']['path'], map_location='cpu', weights_only=False)
            assert iteration == 32000 and len(capture) == 2
            env = envlight.EnvLight(resolution=cfg.env_map_res).cuda()
            env.restore((torch.nn.Parameter(capture[0].detach().cuda(), requires_grad=False), {}))
            env.eval()
            del capture
            with np.load(plan['anchor_references'][arm]['float32_arrays']['path'], allow_pickle=False) as data:
                anchor_saved = {name: data[name].copy() for name in ('gt', 'static', 'alpha_s', 'static_depth_weighted_raw', 'static_env_contribution')}
            camera = cameras.Camera(colmap_id=row['uid'], uid=row['uid'], R=np.asarray(row['R']), T=np.asarray(row['T']),
                                    image=torch.from_numpy(anchor_saved['gt']), image_name=row['image_name'],
                                    image_path=row['source_image'], resolution=(480, 320), timestamp=-.28,
                                    **{key: row[key] for key in ('fx', 'fy', 'cx', 'cy')})
            directions = camera.get_world_directions(False)
            direction_sha = h.tensor_hash(directions)
            env_sample = env(directions.permute(1, 2, 0)).permute(2, 0, 1)
            geometry = {key: h.tensor_hash(getattr(camera, key)) for key in ('world_view_transform', 'full_proj_transform', 'camera_center')}
            model_hashes = {name: h.tensor_hash(getattr(static, name)) for name in
                            ('_xyz', '_features_dc', '_features_rest', '_opacity', '_t', '_scaling_t', '_velocity', '_scaling', '_rotation')}
            env_sha = h.tensor_hash(env.base)
            reference_xyz = static.get_xyz_SHM(-.28)
            displacement_max = torch.zeros(len(reference_xyz), device='cuda')
            opacity_min = torch.full((len(reference_xyz),), float('inf'), device='cuda')
            opacity_max = torch.zeros_like(opacity_min)
            candidate_any = torch.zeros(len(reference_xyz), dtype=torch.bool, device='cuda')
            background = torch.zeros(3, dtype=torch.float32, device='cuda')
            anchor_values = None

            def render_one(time_row, mode):
                nonlocal anchor_values
                camera.timestamp = time_row['timestamp']
                assert h.tensor_hash(camera.get_world_directions(False)) == direction_sha
                assert all(h.tensor_hash(getattr(camera, key)) == value for key, value in geometry.items())
                pc = static if mode == 'actual' else StaticTimeControl(static, -.28, mode)
                before_rng = h.rng_pins(h.rng_capture())
                package = gaussian_renderer.render(camera, pc, dynamic, cfg, background, env_map=env, is_training=False, time_shift=None)
                assert h.rng_pins(h.rng_capture()) == before_rng
                values = {name: h.tensor_cpu(value) for name, value in
                          (('static_full', package['render_s']), ('static_Gaussian_RGB', package['render_s'] - package['env_map_s']),
                           ('environment_transmission', package['env_map_s']), ('static_alpha', package['static_alpha']),
                           ('static_depth_weighted_raw', package['static_depth']))}
                values['static_depth_normalized_foreground'] = values['static_depth_weighted_raw'] / np.maximum(values['static_alpha'], 1e-5)
                assert all(np.isfinite(value).all() for value in values.values())
                assert h.check_delta(values['environment_transmission'], (1 - values['static_alpha']) * h.tensor_cpu(env_sample))['max_abs'] == 0
                if mode == 'actual' and time_row['raw_index'] == 10:
                    parity = {name: h.check_delta(values[key], anchor_saved[name]) for name, key in
                              (('static', 'static_full'), ('alpha_s', 'static_alpha'), ('static_depth_weighted_raw', 'static_depth_weighted_raw'), ('static_env_contribution', 'environment_transmission'))}
                    assert all(item['exact_equal'] for item in parity.values()), parity
                    anchor_values = {key: value.copy() for key, value in values.items()}
                    write(out / arm / 'anchor_native_float32_parity.json', parity)
                folder = out / arm / mode / f'raw{time_row["raw_index"]:02d}'
                folder.mkdir(parents=True)
                paths = {}
                for name in ('static_full', 'static_Gaussian_RGB', 'environment_transmission', 'static_alpha'):
                    path = folder / (name + '.png')
                    save_image(torch.from_numpy(values[name]).clamp(0, 1), path)
                    paths[name] = str(path)
                if mode == 'actual' and time_row['raw_index'] == 10:
                    png_parity = {key: bool(np.array_equal(np.asarray(Image.open(paths[name])),
                                                          np.asarray(Image.open(plan['anchor_references'][arm][key]['path']))))
                                  for key, name in (('static_png', 'static_full'), ('alpha_png', 'static_alpha'))}
                    assert all(png_parity.values()), png_parity
                    write(out / arm / 'anchor_native_png_parity.json', png_parity)
                depth = values['static_depth_normalized_foreground'] / plan['model_units_per_meter']
                depth = np.where(values['static_alpha'] > 1e-5, depth, 0)
                save_image(torch.from_numpy(depth / 100).clamp(0, 1), folder / 'static_depth_0_to100m.png')
                paths['static_depth_0_to100m'] = str(folder / 'static_depth_0_to100m.png')
                np.savez_compressed(folder / 'float32_arrays.npz', **values)
                regional = {region: {'pixel_count': int(mask.sum()), 'values': {key: h.stats(value[..., mask]) for key, value in values.items()}}
                            for region, mask in masks.items()}
                record = {'arm': arm, 'mode': mode, 'counterfactual': mode != 'actual', 'time': time_row,
                          'fixed_camera_frame_id': row['frame_id'], 'fixed_pose_hashes': geometry, 'directions_sha256': direction_sha,
                          'environment_texture_sha256': env_sha, 'RNG_unchanged': True, 'outputs': paths,
                          'float32_arrays': str(folder / 'float32_arrays.npz'), 'regions': regional}
                record['region_change_from_anchor'] = {region: {key: h.check_delta(value[..., mask], anchor_values[key][..., mask])
                                                               for key, value in values.items()} for region, mask in masks.items()}
                records.append(record)
                status.update(status='rendering', arm=arm, mode=mode, raw_index=time_row['raw_index'], completed_render_count=len(records))
                write(out / 'status.json', status)
                del package
                return values

            # Anchor first gates the original saved camera/loader/FP32 renderer before the sweep.
            anchor_values = render_one(plan['observed_times'][10], 'actual')
            actual_values = {10: anchor_values}
            per_time_points = []
            for time_row in plan['observed_times']:
                timestamp = time_row['timestamp']
                xyz = static.get_xyz_SHM(timestamp)
                marginal = static.get_marginal_t(timestamp)[:, 0]
                opacity = static.get_opacity[:, 0] * marginal
                distance = torch.linalg.vector_norm(xyz - reference_xyz, dim=1) / plan['model_units_per_meter']
                torch.maximum(displacement_max, distance, out=displacement_max)
                torch.minimum(opacity_min, opacity, out=opacity_min)
                torch.maximum(opacity_max, opacity, out=opacity_max)
                candidate_any.logical_or_((marginal > .05) & (opacity >= 1 / 255))
                per_time_points.append({'time': time_row, 'XYZ_displacement_from_anchor_m': {'mean': float(distance.double().mean()), 'max': float(distance.max())},
                                        'effective_opacity': {'mean': float(opacity.double().mean()), 'min': float(opacity.min()), 'max': float(opacity.max()),
                                                              'fraction_below1_over255': float((opacity < 1 / 255).double().mean())},
                                        'temporal_prefilter_fraction_marginal_gt0_05': float((marginal > .05).double().mean())})
                if time_row['raw_index'] != 10:
                    values = render_one(time_row, 'actual')
                    records[-1]['change_from_anchor'] = {key: h.check_delta(value, anchor_values[key]) for key, value in values.items()}
                    if time_row['raw_index'] in (0, 48):
                        actual_values[time_row['raw_index']] = values
            for mode in ('freeze_SHM_position', 'freeze_temporal_opacity_and_support', 'freeze_both'):
                for index in (0, 48):
                    values = render_one(plan['observed_times'][index], mode)
                    records[-1]['change_from_anchor'] = {key: h.check_delta(value, anchor_values[key]) for key, value in values.items()}
                    records[-1]['change_from_same_time_actual'] = {key: h.check_delta(value, actual_values[index][key]) for key, value in values.items()}
                    records[-1]['region_change_from_same_time_actual'] = {region: {key: h.check_delta(value[..., mask], actual_values[index][key][..., mask])
                                                                                 for key, value in values.items()} for region, mask in masks.items()}
                    if mode == 'freeze_both':
                        assert all(value['exact_equal'] for value in records[-1]['change_from_anchor'].values())
            arrays = {'max_XYZ_displacement_from_anchor_m': h.tensor_cpu(displacement_max),
                      'effective_opacity_range_over49': h.tensor_cpu(opacity_max - opacity_min),
                      'effective_opacity_max_over49': h.tensor_cpu(opacity_max)}
            visible = candidate_any.cpu().numpy()
            point_summaries[arm] = {'point_count': len(reference_xyz), 'time_count': 49,
                                    'potentially_visible_candidate_count': int(visible.sum()),
                                    'candidate_definition': 'marginal>.05 and effective opacity>=1/255 at any observed time; not proof of frustum/pixel visibility.',
                                    'per_time_actual_points': per_time_points,
                                    'all_points': {key: h.stats(value) for key, value in arrays.items()},
                                    'potentially_visible_candidate_points': {key: h.stats(value[visible]) for key, value in arrays.items()}}
            assert env_sha == h.tensor_hash(env.base)
            assert all(h.tensor_hash(getattr(static, name)) == value for name, value in model_hashes.items())
            del models, static, dynamic, env, camera, reference_xyz, opacity_min, opacity_max, displacement_max, candidate_any
            gc.collect()
            torch.cuda.empty_cache()
    for path, row in plan['protected_inputs'].items():
        assert sha(path) == row['sha256'], path
    write(out / 'diagnostic_manifest.json', {'status': 'complete_readonly_fixed_pose_temporal_diagnostic',
          'plan_sha256': sha(PLAN), 'runtime_FP32': True, 'AMP': False, 'optimizer_restored': False,
          'training_or_backward_started': False, 'original_parameters_source_config_modified': False,
          'original_file_SHA_unchanged': True, 'actual_render_count': 98, 'counterfactual_render_count': 12,
          'records': records, 'static_point_time_variation': point_summaries,
          'elapsed_seconds': time.monotonic() - started, 'CUDA_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'normal_algorithm_success_claimed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-approved-plan')
    args = parser.parse_args()
    if args.root_approved_plan:
        run(Path(args.root_approved_plan))
    else:
        prepare()
