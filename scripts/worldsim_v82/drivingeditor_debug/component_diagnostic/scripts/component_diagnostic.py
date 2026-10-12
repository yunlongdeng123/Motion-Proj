"""Read-only 20k/30k renderer diagnostics under the single campaign GPU lock.

Original checkpoints, source, data, and saved evaluation files are never written.
Mixed iteration cases are diagnostics only. No training/backward/optimizer step.
"""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
RUN = BASE / 'runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline'
REPO = BASE / 'campaign/resume_vgpu_baseline_20261011/frozen_repository'
OUT = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic'
FRAME_IDS = ['image_0/0017' + str(i).zfill(3) for i in (0, 10, 20, 30, 48)]
GAUSSIAN_FIELDS = ['active_sh_degree', 'xyz', 'features_dc', 'features_rest', 'log_scale', 'rotation', 'opacity_logit', 't', 'log_scale_t', 'velocity', 'max_radii2D', 'xyz_gradient_accum', 't_gradient_accum', 'denom', 'optimizer', 'spatial_lr_scale', 'T', 'velocity_decay']
SOURCE_MODULES = ['train.py', 'gaussian_renderer/__init__.py', 'scene/__init__.py', 'scene/gaussian_model.py', 'scene/cameras.py', 'scene/envlight.py', 'scene/waymo_loader2.py', 'utils/camera_utils.py', 'utils/graphics_utils.py', 'utils/general_utils.py', 'utils/sh_utils.py', 'utils/paper_checkpoint.py', 'utils/paper_losses.py', 'utils/paper_masks.py', 'scene/paper_sgcm.py']
CASES = [
    ('baseline20', 20000, 20000, 20000),
    ('baseline30', 30000, 30000, 30000),
    ('s20_d20_e30_ENV_ONLY_DIAGNOSTIC', 20000, 20000, 30000),
    ('s30_d30_e20_ENV_ONLY_DIAGNOSTIC', 30000, 30000, 20000),
    ('s20_d30_e30_BRANCH_MIX_DIAGNOSTIC', 20000, 30000, 30000),
    ('s30_d20_e30_BRANCH_MIX_DIAGNOSTIC', 30000, 20000, 30000),
]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def tensor_hash(value):
    return digest_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns, 'sha256': sha(path)}


def stats(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    finite = np.isfinite(a)
    total = a.size
    f = a[finite]
    result = {'shape': list(a.shape), 'dtype': str(a.dtype), 'numel': total, 'finite_count': int(finite.sum()), 'nonfinite_count': int(total - finite.sum())}
    if not total or not len(f):
        return result
    q = np.quantile(f, [.01, .05, .5, .95, .99])
    result.update(min=float(f.min()), max=float(f.max()), p01=float(q[0]), p05=float(q[1]), p50=float(q[2]), p95=float(q[3]), p99=float(q[4]), mean=float(f.mean(dtype=np.float64)), std=float(f.std(dtype=np.float64)), fraction_lt0=float(np.mean(f < 0)), fraction_gt1=float(np.mean(f > 1)))
    return result


def tree_summary(value):
    if isinstance(value, torch.Tensor):
        return {'type': 'tensor', 'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device), 'sha256': tensor_hash(value), 'finite': bool(torch.isfinite(value).all())}
    if isinstance(value, np.ndarray):
        return {'type': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': digest_bytes(value.tobytes()), 'finite': bool(np.isfinite(value).all())}
    if isinstance(value, dict):
        return {str(key): tree_summary(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [tree_summary(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def optimizer_stats(opt):
    result = {'param_groups': [], 'tensor_devices': [], 'all_finite': True, 'moment_tensor_count': 0, 'moment_bytes': 0}
    devices = set()
    for group in opt.get('param_groups', []):
        result['param_groups'].append({key: tree_summary(value) for key, value in group.items() if key != 'params'})
    for state in opt.get('state', {}).values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                result['all_finite'] &= bool(torch.isfinite(value).all())
                devices.add(str(value.device))
                result['moment_tensor_count'] += 1
                result['moment_bytes'] += value.numel() * value.element_size()
    result['tensor_devices'] = sorted(devices)
    result['step_values'] = [float(state['step']) for state in opt.get('state', {}).values() if 'step' in state]
    return result


@torch.no_grad()
def gaussian_stats(capture, model_to_world, scale):
    if len(capture) != 18:
        raise ValueError('Unsupported Gaussian capture schema')
    count = len(capture[1])
    fields = {}
    for i in range(1, 14):
        value = capture[i]
        if not isinstance(value, torch.Tensor) or len(value) != count or value.device.type != 'cpu':
            raise ValueError('Capture field count/device mismatch: ' + GAUSSIAN_FIELDS[i])
        fields[GAUSSIAN_FIELDS[i]] = stats(value)
        if fields[GAUSSIAN_FIELDS[i]]['nonfinite_count']:
            raise FloatingPointError('Nonfinite capture field')
        if value.dtype != torch.float32:
            raise TypeError('Expected original float32 Gaussian tensors')
    opacity = torch.sigmoid(capture[6])
    opacity_summary = stats(opacity)
    opacity_summary['threshold_fractions_le'] = {str(th): float((opacity <= th).float().mean()) for th in (.005, .01, .05, .1, .5, .9)}
    xyz = capture[1].detach().numpy()
    world = xyz.astype(np.float64) @ model_to_world[:3, :3].T + model_to_world[:3, 3]
    activated_scale = torch.exp(capture[4])
    result = {'point_count': count, 'active_sh_degree': int(capture[0]), 'spatial_lr_scale': float(capture[15]), 'T': float(capture[16]), 'velocity_decay': float(capture[17]), 'field_statistics': fields, 'opacity_activated': opacity_summary, 'scale_activated_model_units': stats(activated_scale), 'scale_activated_source_world_meters': stats(activated_scale / scale), 'temporal_scale_activated_model_time': stats(torch.exp(capture[8])), 'position_model_unit_axes': [stats(xyz[:, i]) for i in range(3)], 'position_source_world_meter_axes': [stats(world[:, i]) for i in range(3)], 'position_model_radius': stats(np.linalg.norm(xyz, axis=1)), 'position_source_world_radius_meters': stats(np.linalg.norm(world, axis=1)), 'dc_only_rgb_before_directional_sh': stats(.28209479177387814 * capture[2] + .5), 'unit_policy': 'Coordinates are saved normalized model-space values; source-world meters recovered only through the frozen world_to_model affine. Gaussian axis scales divided by affine uniform scale.', 'optimizer_cpu_inventory': optimizer_stats(capture[14])}
    stripped = list(capture)
    stripped[14] = {}
    return result, tuple(stripped)


def camera_manifest(event, camera_path):
    cameras = json.loads(Path(camera_path).read_text())
    by_name = {camera['img_name']: camera for camera in cameras}
    if len(by_name) != len(cameras):
        raise ValueError('Ambiguous saved camera names')
    result = []
    for frame in event['camera_manifest']:
        if frame['frame_id'] not in FRAME_IDS:
            continue
        camera = by_name[frame['image_name']]
        candidates = []
        for rs in event['config']['resolution_scales']:
            res = event['config']['resolution']
            if res in (1, 2, 3, 4, 8, 16, 32):
                scale = rs * res
                size = [round(camera['width'] / scale), round(camera['height'] / scale)]
            else:
                scale = rs * (1 if res == -1 else camera['width'] / res)
                size = [int(camera['width'] / scale), int(camera['height'] / scale)]
            if size == [frame['width'], frame['height']]:
                candidates.append(scale)
        if len(set(candidates)) != 1:
            raise ValueError('Camera resolution scale is ambiguous')
        scale = candidates[0]
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = camera['rotation'], camera['position']
        w2c = np.linalg.inv(c2w)
        row = dict(frame, R=w2c[:3, :3].T.tolist(), T=w2c[:3, 3].tolist(), source_camera_json_id=camera['id'], output_width=frame['width'], output_height=frame['height'], target_timestamp_model=frame['timestamp'])
        if 'fx' in camera:
            row.update({key: camera[key] / scale for key in ('fx', 'fy', 'cx', 'cy')})
        else:
            row.update({key: camera[key] for key in ('FoVx', 'FoVy')})
        result.append(row)
    if set(row['frame_id'] for row in result) != set(FRAME_IDS) or len(result) != 5:
        raise ValueError('Requested frozen camera set is incomplete')
    return sorted(result, key=lambda row: row['frame_id'])


@torch.no_grad()
def cpu_audit(manifest):
    started = time.monotonic()
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU audit unexpectedly initialized CUDA')
    provenance = json.loads((RUN / 'run_provenance.json').read_text())
    if Path(provenance['repository']).resolve() != REPO:
        raise ValueError('Formal child repository pin differs')
    modules = {name: pin(REPO / name) for name in SOURCE_MODULES}
    archive = RUN / 'code_snapshot.tar.gz'
    if sha(archive) != provenance['code_snapshot_sha256']:
        raise ValueError('Child source archive differs from provenance')
    archive_checks = []
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = member.name.removeprefix('./')
            payload = tar.extractfile(member).read()
            actual_path = REPO / relative
            expected = digest_bytes(payload)
            actual = sha(actual_path) if actual_path.is_file() else None
            archive_checks.append({'relative': relative, 'archive_sha256': expected, 'frozen_sha256': actual, 'matches': expected == actual})
    if any(not row['matches'] for row in archive_checks):
        raise ValueError('Frozen source differs from original child code snapshot')
    kernels = []
    for value, expected in provenance['original_rasterizer_pins'].items():
        record = pin(value)
        record.update(recorded_sha256=expected, matches_recorded=record['sha256'] == expected)
        kernels.append(record)
    if any(not row['matches_recorded'] for row in kernels):
        raise ValueError('Original renderer/kernel pin mismatch')
    all_pins = {str(path): pin(path) for path in [RUN / 'run_provenance.json', RUN / 'effective_config.yaml', RUN / 'cameras.json', RUN / 'code_snapshot.tar.gz', RUN / 'resume_candidate_manifest.json', RUN / 'input_files_before_training.json']}
    all_pins.update({row['path']: row for row in modules.values()})
    all_pins.update({row['path']: {key: row[key] for key in ('path', 'sha256', 'bytes', 'mtime_ns')} for row in kernels})
    from omegaconf import OmegaConf
    config_effective = OmegaConf.to_container(OmegaConf.load(RUN / 'effective_config.yaml'), resolve=True)
    events, captures, states, records = {}, {}, {}, {}
    frames_by_iteration = {}
    for iteration in (20000, 30000):
        event_path = RUN / 'checkpoint_events' / ('iteration_' + str(iteration).zfill(6) + '.json')
        event = json.loads(event_path.read_text())
        if event['status'] != 'complete' or event['iteration'] != iteration or Path(event['model_path']).resolve() != RUN:
            raise ValueError('Incomplete or wrong iteration event')
        config_hash = digest_bytes(json.dumps(event['config'], sort_keys=True).encode())
        if config_hash != event['config_sha256'] or event['config'] != config_effective:
            raise ValueError('Event/config/effective config mismatch')
        state_path = Path(event['checkpoints']['training_state']['path'])
        state_record = pin(state_path)
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['iteration'] != iteration or state['config'] != event['config']:
            raise ValueError('Trainingstate iteration/config mismatch')
        if state_record['bytes'] != event['checkpoints']['training_state']['bytes']:
            raise ValueError('Trainingstate size differs from complete event')
        affine = np.asarray(state['sgcm']['metadata']['world_to_model'], dtype=np.float64)
        if affine.shape != (4, 4) or not np.allclose(affine, np.loadtxt(Path(event['source_path']) / 'transform_matrix.txt'), rtol=1e-10, atol=1e-10):
            raise ValueError('Saved scene transform differs from source affine')
        singular = np.linalg.svd(affine[:3, :3], compute_uv=False)
        if not np.allclose(singular, singular[0], rtol=1e-5):
            raise ValueError('Nonuniform scene affine scale')
        model_to_world = np.linalg.inv(affine)
        branch_stats, branch_captures, branch_pins = {}, {}, {}
        for branch in ('static', 'dynamic', 'environment'):
            path = Path(event['checkpoints'][branch]['path']).resolve()
            if path.parent != RUN:
                raise ValueError('Checkpoint is outside intended formal child')
            record = pin(path)
            if record['bytes'] != event['checkpoints'][branch]['bytes']:
                raise ValueError('Truncated component file')
            capture, saved_iteration = torch.load(path, map_location='cpu', weights_only=False)
            if saved_iteration != iteration:
                raise ValueError('Component iteration mismatch')
            print('CPU_COMPONENT_LOADED', iteration, branch, record['sha256'], flush=True)
            if branch == 'environment':
                texture = capture[0]
                if len(capture) != 2 or tuple(texture.shape) != (6, event['config']['env_map_res'], event['config']['env_map_res'], 3) or texture.dtype != torch.float32 or texture.device.type != 'cpu':
                    raise ValueError('Environment tensor/config mismatch')
                branch_stats[branch] = {'global': stats(texture), 'faces': [{'face_index': i, 'global': stats(texture[i]), 'rgb_channels': [stats(texture[i, ..., c]) for c in range(3)]} for i in range(6)], 'optimizer_cpu_inventory': optimizer_stats(capture[1]), 'texture_tensor_sha256': tensor_hash(texture), 'interpretation': 'Raw linear stored cubemap values, unbounded trainable radiance; report below0/above1 without modifying texture. PNG previews clamp only for display.'}
                if branch_stats[branch]['global']['nonfinite_count']:
                    raise FloatingPointError('Nonfinite environment texture')
                branch_captures[branch] = (texture, {})
            else:
                branch_stats[branch], branch_captures[branch] = gaussian_stats(capture, model_to_world, singular[0])
            branch_pins[branch] = dict(record, component_iteration=int(saved_iteration), config_sha256=config_hash)
            all_pins[str(path)] = record
            del capture
            gc.collect()
        event_pin = pin(event_path)
        all_pins[str(event_path)], all_pins[str(state_path)] = event_pin, state_record
        orders = state['camera_orders']
        active_ids = next(row['ids'] for row in orders if row['scale'] == event['config']['resolution_scales'][state['scale_index']])
        if set(active_ids) != {row['frame_id'] for row in event['camera_manifest']} or len(active_ids) != 49:
            raise ValueError('Saved continuation cameras differ from event49 cameras')
        frames_by_iteration[iteration] = camera_manifest(event, RUN / 'cameras.json')
        records[str(iteration)] = {'event_pin': event_pin, 'config_sha256': config_hash, 'config_equals_effective_equals_trainingstate': True, 'checkpoints': dict(branch_pins, training_state=state_record), 'scene_uniform_model_units_per_meter': float(singular[0]), 'world_to_model': affine.tolist(), 'model_to_world': model_to_world.tolist(), 'components': branch_stats, 'training_state': tree_summary(state), 'frames': frames_by_iteration[iteration]}
        states[iteration], events[iteration], captures[iteration] = state, event, branch_captures
        print('CPU_ITERATION_COMPLETE', iteration, flush=True)
        write(OUT / ('cpu_checkpoint_audit_' + str(iteration) + '.json'), records[str(iteration)])
    if frames_by_iteration[20000] != frames_by_iteration[30000]:
        raise ValueError('20k and30k frozen camera bindings differ')
    input_pins = []
    source = Path(events[20000]['source_path'])
    for row in frames_by_iteration[20000]:
        input_pins.extend([pin(row['source_image']), pin(source / 'sky/sky_0' / (row['image_name'] + '.png')), pin(source / 'dynamic_masks/dynamic_mask_0' / (row['image_name'] + '.png'))])
    input_pins.append(pin(source / 'transform_matrix.txt'))
    all_pins.update({row['path']: row for row in input_pins})
    audit = {'status': 'cpu_readonly_audit_complete', 'created_utc': utc(), 'cuda_initialized': torch.cuda.is_initialized(), 'repository': str(REPO), 'source_archive': pin(archive), 'archive_file_hash_checks': archive_checks, 'frozen_source_modules': modules, 'kernel_pins': kernels, 'records': records, 'input_pins': input_pins, 'elapsed_seconds': time.monotonic() - started, 'original_files': all_pins}
    if audit['cuda_initialized']:
        raise RuntimeError('CPU checkpoint audit initialized CUDA')
    write(OUT / 'cpu_audit.json', audit)
    manifest.update(status='cpu_audit_complete', cpu_audit_pin=pin(OUT / 'cpu_audit.json'), originals_before=all_pins)
    write(OUT / 'diagnostic_manifest.json', manifest)
    return audit, events, captures, states, frames_by_iteration[20000]


def cuda_capture(capture):
    result = []
    for i, value in enumerate(capture):
        if isinstance(value, torch.Tensor):
            value = value.detach().cuda()
            if 1 <= i <= 9:
                value = torch.nn.Parameter(value, requires_grad=False)
        result.append(value)
    if result[14]:
        raise RuntimeError('Optimizer state leaked into GPU inference')
    return tuple(result)


def tensor_cpu(value):
    return value.detach().float().cpu().numpy().copy()


def region_stats(values, masks, gt):
    result = {}
    for region, mask in masks.items():
        count = int(mask.sum())
        summary = {'pixel_count': count, 'metrics': {}}
        for key, value in values.items():
            pixels = value[..., mask]
            summary['metrics'][key] = stats(pixels)
            if key in ('composite', 'static') and count:
                clamped = np.clip(pixels, 0, 1)
                error = clamped.astype(np.float64) - gt[..., mask]
                mse = np.mean(error * error)
                summary['metrics'][key].update(clamped_rgb_mae_vs_gt=float(np.mean(np.abs(error))), clamped_rgb_mse_vs_gt=float(mse), clamped_rgb_psnr_db=float(-10 * math.log10(mse)) if mse else None)
            if key == 'dynamic' and count:
                preview = np.floor(np.clip(pixels, 0, 1) * 255. + .5)
                summary['metrics'][key]['rgb_support_gt2_uint8_pixel_fraction'] = float(np.any(preview > 2, axis=0).mean())
            if key in ('alpha_s', 'alpha_d', 'alpha') and count:
                summary['metrics'][key]['fraction_gt_0_1'] = float(np.mean(pixels > .1))
                summary['metrics'][key]['fraction_gt_0_5'] = float(np.mean(pixels > .5))
                summary['metrics'][key]['fraction_gt_0_9'] = float(np.mean(pixels > .9))
        result[region] = summary
    return result


def check_delta(left, right):
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {'max_abs': float(np.max(np.abs(difference))), 'mean_abs': float(np.mean(np.abs(difference))), 'exact_equal': bool(np.array_equal(left, right))}


def rgba_package(package, environment_sample, gt):
    keys = {'composite': 'render', 'static': 'render_s', 'dynamic': 'render_d', 'alpha': 'alpha', 'alpha_s': 'static_alpha', 'alpha_d': 'dynamic_alpha', 'composite_depth_weighted_raw': 'depth', 'static_depth_weighted_raw': 'static_depth', 'composite_env_contribution': 'env_map', 'static_env_contribution': 'env_map_s'}
    values = {name: tensor_cpu(package[key]) for name, key in keys.items()}
    values['environment_sample'] = tensor_cpu(environment_sample)
    values['static_foreground_contribution'] = values['static'] - values['static_env_contribution']
    values['composite_foreground_contribution'] = values['composite'] - values['composite_env_contribution']
    values['gt'] = tensor_cpu(gt)
    values['composite_depth_normalized_foreground'] = values['composite_depth_weighted_raw'] / np.maximum(values['alpha'], 1e-5)
    values['static_depth_normalized_foreground'] = values['static_depth_weighted_raw'] / np.maximum(values['alpha_s'], 1e-5)
    for prefix, alpha in [('composite', values['alpha']), ('static', values['alpha_s'])]:
        norm = values[prefix + '_depth_normalized_foreground']
        values[prefix + '_depth_blended_harmonic_900'] = 1 / np.maximum(alpha / np.maximum(norm, 1e-5) + (1 - alpha) / 900, 1e-5)
    if any(not np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError('Nonfinite rendered data')
    return values


def rng_capture():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state().clone(), 'cuda': [value.clone() for value in torch.cuda.get_rng_state_all()]}


def rng_restore(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def rng_pins(state):
    return {'python': digest_bytes(repr(state['python']).encode()), 'numpy': digest_bytes(repr((state['numpy'][0], state['numpy'][2:])).encode() + state['numpy'][1].tobytes()), 'torch': tensor_hash(state['torch']), 'cuda': [tensor_hash(value) for value in state['cuda']]}


@torch.no_grad()
def gpu_diagnostic(manifest, audit, events, captures, states, frames):
    started = time.monotonic()
    sys.path.insert(0, str(REPO))
    from PIL import Image, ImageDraw
    import cv2
    from omegaconf import OmegaConf
    from torchvision.utils import save_image
    from scene.gaussian_model import GaussianModel, Dynamic_GaussianModel
    from scene.envlight import EnvLight
    from scene.cameras import Camera
    from gaussian_renderer import render
    import diff_gaussian_rasterization
    torch.cuda.init()
    if torch.is_grad_enabled() or torch.is_autocast_enabled():
        raise RuntimeError('Expected no-grad FP32 no-AMP inference')
    models = {}
    torch.cuda.reset_peak_memory_stats()
    for iteration in (20000, 30000):
        args = OmegaConf.create(events[iteration]['config'])
        static = GaussianModel(args)
        dynamic = Dynamic_GaussianModel(args)
        static.restore(cuda_capture(captures[iteration]['static']))
        dynamic.restore(cuda_capture(captures[iteration]['dynamic']))
        environment = EnvLight(resolution=args.env_map_res).cuda()
        environment.restore((torch.nn.Parameter(captures[iteration]['environment'][0].detach().cuda(), requires_grad=False), {}))
        environment.eval()
        if static.optimizer is not None or dynamic.optimizer is not None or hasattr(environment, 'optimizer'):
            raise RuntimeError('Inference created an optimizer')
        for model in (static, dynamic):
            if any(value.dtype != torch.float32 or value.requires_grad for value in (model._xyz, model._features_dc, model._features_rest, model._scaling, model._rotation, model._opacity, model._t, model._scaling_t, model._velocity)):
                raise RuntimeError('Inference model precision or requires_grad differs')
        models[iteration] = (static, dynamic, environment)
    manifest['gpu_runtime'] = {'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'device_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory, 'FP32': True, 'autocast': False, 'autograd': False, 'kernel_python': pin(diff_gaussian_rasterization.__file__), 'kernel_binary': pin(diff_gaussian_rasterization._C.__file__), 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'optimizer_restored': False, 'source_modified': False}
    # The rasterizer uses FP32 tensors. No change to saved radiance/opacity/scale or renderer source.
    source = Path(events[20000]['source_path'])
    masks_by_id = {}
    gts = {}
    for row in frames:
        width, height = row['output_width'], row['output_height']
        with Image.open(row['source_image']) as img:
            rgb = np.asarray(img.convert('RGB')) / 255.
        rgb = cv2.resize(rgb, (width, height)) if rgb.shape[:2] != (height, width) else rgb
        gts[row['frame_id']] = torch.from_numpy(rgb).float().permute(2, 0, 1)
        sky_raw = np.asarray(Image.open(source / 'sky/sky_0' / (row['image_name'] + '.png')))
        # Exact loader sky-mask semantics: positive mask -> float32 linear resize -> positive.
        sky = cv2.resize((sky_raw > 0).astype(np.float32), (width, height)) > 0
        actor = np.asarray(Image.open(source / 'dynamic_masks/dynamic_mask_0' / (row['image_name'] + '.png')).convert('L').resize((width, height), Image.Resampling.NEAREST)) > 0
        shadow = source / 'dynamic_masks/shadow_mask_0' / (row['image_name'] + '.png')
        if shadow.exists():
            raise RuntimeError('Unexpected unpinned shadow mask in this silver actor scene')
        masks_by_id[row['frame_id']] = {'full': np.ones((height, width), dtype=bool), 'sky': sky, 'non_sky': ~sky, 'actor': actor, 'outside_actor': ~actor, 'non_sky_outside_actor': ~sky & ~actor}
        mask_dir = OUT / 'masks'
        mask_dir.mkdir(exist_ok=True)
        Image.fromarray((sky * 255).astype(np.uint8)).save(mask_dir / (row['image_name'] + '_sky.png'))
        Image.fromarray((actor * 255).astype(np.uint8)).save(mask_dir / (row['image_name'] + '_actor.png'))
    baseline_cache = {}
    case_records = []

    def camera_for(row):
        intrinsics = {key: row[key] for key in (('fx', 'fy', 'cx', 'cy') if 'fx' in row else ('FoVx', 'FoVy'))}
        return Camera(colmap_id=row['uid'], uid=row['uid'], R=np.asarray(row['R']), T=np.asarray(row['T']), image=gts[row['frame_id']], image_name=row['image_name'], image_path=row['source_image'], resolution=(row['output_width'], row['output_height']), timestamp=row['target_timestamp_model'], **intrinsics)

    for name, si, di, ei in CASES:
        start = time.monotonic()
        output = OUT / 'cases' / name
        output.mkdir(parents=True, exist_ok=False)
        torch.cuda.reset_peak_memory_stats()
        args = OmegaConf.create(events[si]['config'])
        background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda', dtype=torch.float32)
        record = {'case_id': name, 'status': 'rendering', 'diagnostic_mixed_components': len({si, di, ei}) != 1, 'normal_run_or_success_claimed': False, 'components': {branch: audit['records'][str(it)]['checkpoints'][branch] for branch, it in [('static', si), ('dynamic', di), ('environment', ei)]}, 'frozen_source_modules': audit['frozen_source_modules'], 'kernel_pins': manifest['gpu_runtime'], 'camera_json_pin': audit['original_files'][str(RUN / 'cameras.json')], 'config_sha256': events[si]['config_sha256'], 'is_training': False, 'FP32': True, 'AMP': False, 'frames': []}
        if record['diagnostic_mixed_components']:
            record['purpose'] = 'Cross-iteration component diagnosis only; cannot be presented as a normal checkpoint or algorithm result'
        for row in frames:
            camera = camera_for(row)
            package = render(camera, models[si][0], models[di][1], args, background, env_map=models[ei][2], is_training=False)
            environment_sample = models[ei][2](camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
            values = rgba_package(package, environment_sample, gts[row['frame_id']])
            paths = {}
            for channel, pixels in [('gt', gts[row['frame_id']]), ('composite', package['render']), ('static', package['render_s']), ('dynamic', package['render_d']), ('environment_sample', environment_sample), ('static_foreground_contribution', package['render_s'] - package['env_map_s']), ('static_env_contribution', package['env_map_s']), ('alpha_s', package['static_alpha']), ('alpha_d', package['dynamic_alpha']), ('alpha', package['alpha'])]:
                destination = output / channel / (row['image_name'] + '.png')
                destination.parent.mkdir(exist_ok=True)
                save_image(pixels.clamp(0, 1), destination)
                paths[channel] = str(destination)
            float_path = output / 'float32' / (row['image_name'] + '.npz')
            float_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(float_path, **values)
            parity = {}
            if si == di == ei:
                eval_row = next(value for value in events[si]['frames'] if value['frame_id'] == row['frame_id'])
                for channel in ('gt', 'composite', 'static', 'dynamic'):
                    actual = np.asarray(Image.open(paths[channel]).convert('RGB'))
                    reference_path = eval_row['channels'][channel]
                    expected = np.asarray(Image.open(reference_path).convert('RGB'))
                    parity[channel] = dict(check_delta(actual, expected), reference_pin=pin(reference_path), actual_pin=pin(paths[channel]))
                if not all(value['exact_equal'] for value in parity.values()):
                    raise ValueError('Baseline PNG parity failure: ' + name + '/' + row['frame_id'])
                baseline_cache[(si, row['frame_id'])] = values
            formulas = {'static_env_is_transmittance_times_environment': check_delta(values['static_env_contribution'], (1 - values['alpha_s']) * values['environment_sample']), 'composite_env_is_transmittance_times_environment': check_delta(values['composite_env_contribution'], (1 - values['alpha']) * values['environment_sample']), 'static_is_foreground_plus_env': check_delta(values['static'], values['static_foreground_contribution'] + values['static_env_contribution']), 'composite_is_foreground_plus_env': check_delta(values['composite'], values['composite_foreground_contribution'] + values['composite_env_contribution'])}
            frame_record = {'frame': row, 'outputs': paths, 'float32_arrays': pin(float_path), 'regions': region_stats(values, masks_by_id[row['frame_id']], values['gt']), 'baseline_png_parity': parity, 'formula_checks': formulas}
            record['frames'].append(frame_record)
            print('GPU_FRAME_COMPLETE', name, row['frame_id'], flush=True)
            del package, camera, environment_sample, values
        torch.cuda.synchronize()
        record.update(status='complete', actual_exit_code=0, elapsed_seconds=time.monotonic() - start, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), cuda_current_allocated_bytes=torch.cuda.memory_allocated())
        write(output / 'case_manifest.json', record)
        case_records.append(record)
        manifest['cases'] = [{'case_id': value['case_id'], 'manifest_pin': pin(OUT / 'cases' / value['case_id'] / 'case_manifest.json'), 'status': value['status']} for value in case_records]
        write(OUT / 'diagnostic_manifest.json', manifest)
    semantics = []
    for iteration in (20000, 30000):
        args = OmegaConf.create(events[iteration]['config'])
        background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda', dtype=torch.float32)
        observed_rng = rng_capture()
        state = states[iteration]
        saved_rng = {'python': state['python_rng'], 'numpy': state['numpy_rng'], 'torch': state['torch_rng'], 'cuda': state['cuda_rng']}
        rng_restore(saved_rng)
        try:
            for row in frames:
                camera = camera_for(row)
                baseline = baseline_cache[(iteration, row['frame_id'])]
                rng_before = rng_capture()
                train_package = render(camera, *models[iteration][:2], args, background, env_map=models[iteration][2], is_training=True)
                rng_after = rng_capture()
                # Replay ONLY the direction jitter from the exact pre-render RNG; renderer has no other random draws.
                rng_restore(rng_before)
                directions_jitter = camera.get_world_directions(True)
                replay_rng_after = rng_capture()
                if rng_pins(replay_rng_after) != rng_pins(rng_after):
                    raise RuntimeError('Renderer uses unaccounted random draws beyond environment jitter')
                sampled_jitter = models[iteration][2](directions_jitter.permute(1, 2, 0)).permute(2, 0, 1)
                train = rgba_package(train_package, sampled_jitter, gts[row['frame_id']])
                dirs_eval = camera.get_world_directions(False)
                direction_difference = check_delta(tensor_cpu(directions_jitter), tensor_cpu(dirs_eval))
                rng_restore(rng_after)
                # This diagnostic only replaces this new Camera instance's directions with its own evaluation directions.
                # It does not change the formal Camera class, renderer, checkpoint, source, or tensor values.
                original_directions = camera.get_world_directions
                camera.get_world_directions = lambda train=False: original_directions(False)
                fixed_package = render(camera, *models[iteration][:2], args, background, env_map=models[iteration][2], is_training=True)
                fixed_rng_after = rng_capture()
                sampled_fixed = models[iteration][2](original_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
                fixed = rgba_package(fixed_package, sampled_fixed, gts[row['frame_id']])
                camera.get_world_directions = original_directions
                train_output = OUT / 'train_eval_semantics' / str(iteration)
                output_paths = {}
                for mode, result in [('true_official_jitter', train), ('true_diagnostic_fixed_eval_directions', fixed)]:
                    for channel in ('composite', 'static', 'dynamic', 'environment_sample'):
                        destination = train_output / mode / channel / (row['image_name'] + '.png')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        save_image(torch.from_numpy(result[channel]).clamp(0, 1), destination)
                        output_paths[mode + '/' + channel] = str(destination)
                    data_path = train_output / mode / 'float32' / (row['image_name'] + '.npz')
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(data_path, **result)
                semantic = {'iteration': iteration, 'frame_id': row['frame_id'], 'components': audit['records'][str(iteration)]['checkpoints'], 'camera': row, 'outputs': output_paths, 'official_true_rng_before': rng_pins(rng_before), 'official_true_rng_after': rng_pins(rng_after), 'jitter_replay_rng_matches_official_render': rng_pins(replay_rng_after) == rng_pins(rng_after), 'fixed_eval_directions_consumes_no_rng': rng_pins(fixed_rng_after) == rng_pins(rng_after), 'direction_jitter_delta': direction_difference, 'official_train_eval_delta': {key: check_delta(train[key], baseline[key]) for key in ('composite', 'static', 'dynamic', 'alpha', 'alpha_s', 'alpha_d', 'composite_depth_weighted_raw', 'static_depth_weighted_raw', 'composite_depth_normalized_foreground', 'static_depth_normalized_foreground', 'composite_depth_blended_harmonic_900', 'static_depth_blended_harmonic_900')}, 'fixed_directions_train_eval_delta': {key: check_delta(fixed[key], baseline[key]) for key in ('composite', 'static', 'dynamic', 'alpha', 'alpha_s', 'alpha_d', 'composite_depth_weighted_raw', 'static_depth_weighted_raw')}, 'formula_checks': {'official_jitter_composite_delta_equals_env_delta': check_delta(train['composite'] - baseline['composite'], (1 - baseline['alpha']) * (train['environment_sample'] - baseline['environment_sample'])), 'official_jitter_static_delta_equals_env_delta': check_delta(train['static'] - baseline['static'], (1 - baseline['alpha_s']) * (train['environment_sample'] - baseline['environment_sample'])), 'official_dynamic_train_equals_eval_foreground_plus_shared_jitter_env': check_delta(train['dynamic'], baseline['dynamic'] + (1 - baseline['alpha_d']) * train['environment_sample']), 'fixed_directions_dynamic_train_equals_eval_foreground_plus_shared_eval_env': check_delta(fixed['dynamic'], baseline['dynamic'] + (1 - baseline['alpha_d']) * baseline['environment_sample'])}, 'regions_true_official_jitter': region_stats(train, masks_by_id[row['frame_id']], train['gt']), 'regions_true_fixed_eval_directions': region_stats(fixed, masks_by_id[row['frame_id']], fixed['gt'])}
                semantics.append(semantic)
                print('TRAIN_EVAL_SEMANTICS_COMPLETE', iteration, row['frame_id'], flush=True)
                del train_package, fixed_package, sampled_jitter, sampled_fixed, train, fixed, camera, directions_jitter, dirs_eval
        finally:
            rng_restore(observed_rng)
        if rng_pins(rng_capture()) != rng_pins(observed_rng):
            raise RuntimeError('Semantics audit failed to restore all observed RNG')
    semantic_report = {'status': 'complete', 'records': semantics, 'classification': 'Official training dynamic RGB includes shared environment; eval dynamic RGB intentionally represents foreground without environment. Composite/static use the same alpha compositing expression. Camera training jitter affects environment directions only.', 'diagnostic_camera_override': 'Separate true_diagnostic_fixed_eval_directions calls override only the newly constructed Camera instance to remove direction jitter. Official renderer and original Camera source remain byte-identical.', 'depth_policy': 'Renderer returns alpha-weighted raw composite/static depth; foreground normalization uses corresponding composite alpha or static_alpha. Train and eval use depth/alpha.clamp_min(1e-5) for composite, with the same harmonic900 environment blend. Static depth normalization here is diagnostic only, paired with static_alpha.', 'no_formal_bug_fix_applied': True, 'actual_exit_code': 0}
    write(OUT / 'train_eval_semantics.json', semantic_report)
    # Contact sheets contain the actual unmodified clamped renderer PNGs.
    width, height = frames[0]['output_width'], frames[0]['output_height']
    contact_dir = OUT / 'contact_sheets'
    contact_dir.mkdir(exist_ok=True)
    for row in frames:
        sheet = Image.new('RGB', (width * 4, (height + 32) * len(case_records)), '#15171b')
        draw = ImageDraw.Draw(sheet)
        for i, case in enumerate(case_records):
            f = next(r for r in case['frames'] if r['frame']['frame_id'] == row['frame_id'])
            for j, channel in enumerate(('gt', 'composite', 'static', 'dynamic')):
                draw.text((j * width + 8, i * (height + 32) + 10), case['case_id'] + ' / ' + channel, fill='white')
                img = Image.open(f['outputs'][channel]).convert('RGB')
                if img.size != (width, height):
                    raise ValueError('Contact source image size differs')
                sheet.paste(img, (j * width, i * (height + 32) + 32))
        sheet.save(contact_dir / (row['image_name'] + '.png'))
    torch.cuda.synchronize()
    manifest.update(status='gpu_diagnostic_complete', gpu_elapsed_seconds=time.monotonic() - started, gpu_case_count=len(case_records), selected_camera_count=len(frames), train_eval_semantic_records=len(semantics), baseline_all_four_channel_pngs_exact_equal=True, cuda_current_allocated_bytes=torch.cuda.memory_allocated(), cuda_current_reserved_bytes=torch.cuda.memory_reserved(), contact_sheets=[pin(path) for path in sorted(contact_dir.glob('*.png'))], semantics_pin=pin(OUT / 'train_eval_semantics.json'))
    write(OUT / 'diagnostic_manifest.json', manifest)
    return case_records, semantics


def finish_no_modification(manifest):
    records = []
    for path, before in manifest['originals_before'].items():
        after = pin(path)
        records.append({'path': path, 'before_sha256': before['sha256'], 'after_sha256': after['sha256'], 'sha256_matches': before['sha256'] == after['sha256'], 'mtime_unchanged': before['mtime_ns'] == after['mtime_ns'], 'bytes_unchanged': before['bytes'] == after['bytes']})
    if not all(row['sha256_matches'] and row['mtime_unchanged'] and row['bytes_unchanged'] for row in records):
        raise RuntimeError('Original file changed during read-only diagnostic')
    report = {'status': 'all_original_hashes_and_mtimes_unchanged', 'records': records, 'count': len(records), 'training_executed': False, 'optimizer_step_executed': False, 'source_or_checkpoint_modified': False}
    write(OUT / 'no_modification_sha.json', report)
    manifest['no_modification_pin'] = pin(OUT / 'no_modification_sha.json')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--gpu-lock-nonce', required=True)
    args = cli.parse_args()
    lock = json.loads((BASE / 'campaign/gpu_slot.lock').read_text())
    supervisor = json.loads((OUT / 'supervisor_status.json').read_text())
    if lock['nonce'] != args.gpu_lock_nonce or supervisor['owner_pid'] != os.getppid() or not supervisor['gpu_lock_held']:
        raise RuntimeError('Worker is not owned by the authorized exclusive supervisor')
    manifest = {'schema_version': 1, 'status': 'cpu_audit_running', 'started_utc': utc(), 'host': 'wm-vgpu-1011', 'formal_run_readonly': str(RUN), 'frozen_repository': str(REPO), 'output': str(OUT), 'GPU_lock_nonce': args.gpu_lock_nonce, 'supervisor_pid': os.getppid(), 'worker_pid': os.getpid(), 'worker_pin': pin(__file__), 'FP32_no_AMP': True, 'no_training_no_backward_no_optimizer_step': True, 'numeric_modifications': [], 'author_algorithm_success_claimed': False, 'mixed_components_are_diagnostic_only': True, 'cases_requested': CASES, 'camera_ids_requested': FRAME_IDS}
    write(OUT / 'diagnostic_manifest.json', manifest)
    start = time.monotonic()
    try:
        audit, events, captures, states, frames = cpu_audit(manifest)
        gpu_diagnostic(manifest, audit, events, captures, states, frames)
        finish_no_modification(manifest)
        manifest.update(status='complete', actual_exit_code=0, finished_utc=utc(), elapsed_seconds=time.monotonic() - start)
    except BaseException as exc:
        import traceback
        manifest.update(status='failed', actual_exit_code=1, error=repr(exc), traceback=traceback.format_exc(), finished_utc=utc(), elapsed_seconds=time.monotonic() - start)
        write(OUT / 'diagnostic_manifest.json', manifest)
        raise
    write(OUT / 'diagnostic_manifest.json', manifest)
    print('DIAGNOSTIC_COMPLETE', str(OUT / 'diagnostic_manifest.json'), flush=True)


if __name__ == '__main__':
    main()
