"""Fresh, zero-update native49 a5 evaluation for the frozen owner5 engine.

Run in the existing drivingeditor environment. The declared dynamic aliases
describe ALL PLYs pooled into the native Dynamic_GaussianModel; render_s removes
that entire pooled branch. There is no post-hoc identity partition or custom
Nov4/Nov5 ownership class. Current native10K is hashed as a reference, not loaded.

Example:
  python -B evaluate_a5.py --model RUN/formal30k --iteration 30000 \
    --output RUN/a5_eval30k --dynamic-aliases 37 6 --delete-name delete_a37_a6

Only fixed actor_labels==1 samples are stored as float RGB/alpha. Full-frame
PNG previews are clamped display copies. No optimizer is constructed; the
engine, checkpoint, inputs and native event images are read-only.
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
import gc
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ENGINE = CAMPAIGN / 'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
OBSERVATIONS = CAMPAIGN / 'scene008_instances_20261105_r1/observations/frames'
FULL5 = CAMPAIGN / 'r15_gn11_stage1_v41_20261023/arms/B/run30k'
CURRENT = CAMPAIGN / 'scene008_native_ownership_20261105/run_r3/formal'
RAWS = tuple(range(89, 138))
ANCHORS = (89, 97, 105, 109)
THRESHOLDS = (.1, .5, .9, .99)
MATRIX_ATOL = 1e-5
TIME_ATOL = 2e-7  # Observation-bank model_time is stored as float32.


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source changed while hashing: {path}')
    return dict(path=str(path), bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                sha256=digest)


def inventory(paths):
    return {str(path): file_identity(path) for path in sorted(set(paths), key=str)}


def raw_id(name):
    match = re.search(r'0008(\d{3})$', str(name))
    if match is None:
        raise ValueError(f'Not a Scene008 FRONT frame ID: {name}')
    return int(match.group(1))


def checkpoint_paths(model, iteration):
    return [model / f'{prefix}{iteration}.pth' for prefix in
            ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')]


def source_model_files(model, iteration):
    required = checkpoint_paths(model, iteration) + [
        model / 'effective_config.json', model / 'cameras.json',
        model / 'actual_scene_binding.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    event = model / f'checkpoint_events/iteration_{iteration:06d}.json'
    if event.is_file():
        required.append(event)
    config = read_json(model / 'effective_config.json')
    for value in config.get('followup_scene_spec', {}).get('actor_ply_paths', []):
        path = Path(value)
        if not path.is_absolute():
            path = model / path
        if not path.is_file():
            raise FileNotFoundError(path)
        required.append(path)
    return required


def parse_source_models(options):
    models = dict(evaluated=(options.model, options.iteration),
                  full5_b30k=(FULL5, 30000), current_native10k=(CURRENT, 10000))
    for value in options.source_model:
        try:
            name, spec = value.split('=', 1)
            directory, step = spec.rsplit('@', 1)
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
                raise ValueError('invalid source name')
            models[name] = (Path(directory).resolve(), int(step))
        except ValueError as error:
            raise ValueError('--source-model expects NAME=MODEL_DIRECTORY@ITERATION') from error
    if models['evaluated'] != (options.model, options.iteration):
        raise ValueError('The evaluated source entry cannot be overridden')
    return models


def event_references(model, iteration):
    event_path = model / f'checkpoint_events/iteration_{iteration:06d}.json'
    references = {}
    status = 'NO_CHECKPOINT_EVENT'
    if event_path.is_file():
        event = read_json(event_path)
        status = 'EVENT_PRESENT'
        for row in event.get('frames', []):
            raw = raw_id(row['frame_id'])
            if raw not in RAWS:
                continue
            channels = row.get('channels', {})
            found = {}
            for name, key in (('keep', 'composite'), ('delete', 'static')):
                if key not in channels:
                    continue
                path = Path(channels[key])
                if not path.is_absolute():
                    path = model / path
                if not path.is_file():
                    raise FileNotFoundError(f'Event references missing native PNG: {path}')
                found[name] = path.resolve()
            if raw in references:
                raise ValueError(f'Duplicate checkpoint event frame: {raw}')
            references[raw] = found
    return event_path, references, status


def observation_headers(options, records, binding):
    by_raw = {raw_id(row['img_name']): row for row in records}
    frame_ids = binding['frame_ids']
    if len(frame_ids) != len(binding['timestamps']):
        raise ValueError('actual_scene_binding frame_ids/timestamps lengths differ')
    times = {raw_id(name): float(stamp) for name, stamp in
             zip(frame_ids, binding['timestamps'])}
    if len(by_raw) != 49 or len(records) != 49 or set(by_raw) != set(RAWS):
        raise ValueError('cameras.json must contain exactly FRONT raw089-137')
    if len(times) != 49 or len(frame_ids) != 49 or set(times) != set(RAWS):
        raise ValueError('actual_scene_binding timestamps must cover raw089-137 exactly')
    if not np.isfinite(list(times.values())).all():
        raise ValueError('actual_scene_binding contains a non-finite timestamp')
    frames = []
    union = np.zeros((320, 480), dtype=bool)
    denominator_hash = hashlib.sha256()
    for raw in RAWS:
        path = options.observations / f'{raw:03d}.npz'
        with np.load(path, allow_pickle=False) as data:
            required = ('raw', 'rgb', 'actor_labels', 'K', 'model_to_camera', 'model_time')
            if any(key not in data for key in required):
                raise ValueError(f'{path}: missing fixed observation/camera fields')
            if int(np.asarray(data['raw']).item()) != raw:
                raise ValueError(f'{path}: raw ID mismatch')
            labels = data['actor_labels']
            rgb = data['rgb']
            if labels.shape != (320, 480) or rgb.shape != (320, 480, 3):
                raise ValueError(f'{path}: expected 480x320 labels and HWC RGB')
            if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                raise ValueError(f'{path}: invalid fixed observation RGB')
            mask = np.ascontiguousarray(labels == 1)
            K = np.asarray(data['K'], dtype=np.float64)
            W = np.asarray(data['model_to_camera'], dtype=np.float64)
            timestamp = float(np.asarray(data['model_time']).item())
        row = by_raw[raw]
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = row['rotation'], row['position']
        expected_W = np.linalg.inv(c2w)
        expected_K = np.array([[row['fx'] / 4, 0, row['cx'] / 4],
                               [0, row['fy'] / 4, row['cy'] / 4], [0, 0, 1.]])
        if K.shape != (3, 3) or W.shape != (4, 4):
            raise ValueError(f'raw{raw}: camera matrix dimensions differ')
        if not np.allclose(K, expected_K, atol=MATRIX_ATOL, rtol=0):
            raise ValueError(f'raw{raw}: observation/native intrinsics mismatch')
        if not np.allclose(W, expected_W, atol=MATRIX_ATOL, rtol=0):
            raise ValueError(f'raw{raw}: observation/native model_to_camera mismatch')
        if not np.isfinite(timestamp) or abs(timestamp - times[raw]) > TIME_ATOL:
            raise ValueError(f'raw{raw}: observation/native timestamp mismatch')
        digest = hashlib.sha256(mask.tobytes()).hexdigest()
        denominator_hash.update(np.asarray(raw, dtype='<i2').tobytes())
        denominator_hash.update(mask.tobytes())
        union |= mask
        frames.append(dict(raw=raw, path=path, mask=mask, K=K, W=expected_W,
                           timestamp=times[raw], pixels=int(mask.sum()), mask_sha256=digest))
    total = sum(frame['pixels'] for frame in frames)
    if total != options.expected_mask_pixels:
        raise ValueError(f'Fixed a5 denominator mismatch: {total} != {options.expected_mask_pixels}')
    y, x = np.nonzero(union)
    crop = list(options.crop) if options.crop is not None else [
        max(0, int(x.min()) - 12), max(0, int(y.min()) - 12),
        min(480, int(x.max()) + 13), min(320, int(y.max()) + 13)]
    return frames, denominator_hash.hexdigest(), crop


def rgb8(array):
    return np.uint8(np.floor(np.clip(array, 0, 1) * 255 + .5))


def tensor_hwc(tensor):
    array = tensor.detach().cpu().numpy()
    if array.shape != (3, 320, 480) or not np.isfinite(array).all():
        raise ValueError('Invalid native RGB output')
    return np.ascontiguousarray(array.transpose(1, 2, 0), dtype=np.float32)


def tensor_alpha(tensor):
    array = tensor.detach().cpu().numpy()
    if array.shape == (1, 320, 480):
        array = array[0]
    if array.shape != (320, 480) or not np.isfinite(array).all():
        raise ValueError('Invalid native alpha output')
    return np.ascontiguousarray(array, dtype=np.float32)


def border_inside(mask):
    padded = np.pad(mask, 1, constant_values=False)
    eroded = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + 320, dx:dx + 480]
    return mask & ~eroded


def metrics(keep, deleted, gt, keep_alpha, deleted_alpha):
    count = len(gt)
    if not count:
        return dict(pixels=0, keep_rgb_mae=None, delete_rgb_mae=None,
                    keep_delete_rgb_change=None, sums={})
    a, b = np.clip(keep.astype(np.float64), 0, 1), np.clip(deleted.astype(np.float64), 0, 1)
    g = gt.astype(np.float64)
    sums = dict(keep_rgb_error=float(np.abs(a - g).mean(1).sum()),
                delete_rgb_error=float(np.abs(b - g).mean(1).sum()),
                keep_delete_rgb_change=float(np.abs(a - b).mean(1).sum()),
                keep_raw_rgb_error=float(np.abs(keep.astype(np.float64) - g).mean(1).sum()),
                delete_raw_rgb_error=float(np.abs(deleted.astype(np.float64) - g).mean(1).sum()),
                keep_alpha=float(keep_alpha.astype(np.float64).sum()),
                delete_alpha=float(deleted_alpha.astype(np.float64).sum()))
    result = dict(pixels=count, keep_rgb_mae=sums['keep_rgb_error'] / count,
                  delete_rgb_mae=sums['delete_rgb_error'] / count,
                  keep_delete_rgb_change=sums['keep_delete_rgb_change'] / count,
                  keep_raw_rgb_mae=sums['keep_raw_rgb_error'] / count,
                  delete_raw_rgb_mae=sums['delete_raw_rgb_error'] / count,
                  keep_mean_alpha=sums['keep_alpha'] / count,
                  delete_mean_alpha=sums['delete_alpha'] / count, sums=sums)
    for name, alpha in (('keep', keep_alpha), ('delete', deleted_alpha)):
        result[name + '_coverage_counts'] = {str(value): int((alpha >= value).sum()) for value in THRESHOLDS}
        result[name + '_coverage'] = {key: value / count for key, value in result[name + '_coverage_counts'].items()}
    return result


def aggregate(rows, key):
    entries = [row[key] for row in rows]
    count = sum(item['pixels'] for item in entries)
    sums = {name: sum(item['sums'].get(name, 0) for item in entries)
            for name in ('keep_rgb_error', 'delete_rgb_error', 'keep_delete_rgb_change',
                         'keep_raw_rgb_error', 'delete_raw_rgb_error', 'keep_alpha', 'delete_alpha')}
    result = dict(pixels=count, weighting='fixed pixel-frame count, not unweighted frame mean')
    mapping = dict(keep_rgb_mae='keep_rgb_error', delete_rgb_mae='delete_rgb_error',
                   keep_delete_rgb_change='keep_delete_rgb_change', keep_raw_rgb_mae='keep_raw_rgb_error',
                   delete_raw_rgb_mae='delete_raw_rgb_error', keep_mean_alpha='keep_alpha', delete_mean_alpha='delete_alpha')
    result.update({name: sums[value] / count if count else None for name, value in mapping.items()})
    for name in ('keep', 'delete'):
        result[name + '_coverage'] = {
            str(value): sum(item.get(name + '_coverage_counts', {}).get(str(value), 0) for item in entries) / count
            if count else None for value in THRESHOLDS}
    return result


def contact_sheet(path, arrays, labels, crop=None):
    images = [Image.fromarray(array) for array in arrays]
    if crop is not None:
        images = [image.crop(tuple(crop)) for image in images]
    width, height = images[0].size
    result = Image.new('RGB', (width * len(images), height + 26), '#eef3f6')
    draw = ImageDraw.Draw(result)
    if len(images) != len(labels):
        raise ValueError('contact sheet images/labels lengths differ')
    for index, (image, label) in enumerate(zip(images, labels)):
        result.paste(image, (index * width, 26))
        draw.text((index * width + 7, 6), label, fill='#193b4a')
    result.save(path)


def restore_models(options, config):
    import torch
    from omegaconf import OmegaConf
    from scene import GaussianModel, Dynamic_GaussianModel, EnvLight

    if config.get('dual_actor_schema_version') is not None:
        raise ValueError('Custom ownership checkpoints are outside this native pooled-PLY evaluator')
    ply_paths = config.get('followup_scene_spec', {}).get('actor_ply_paths', [])
    if len(ply_paths) != len(options.dynamic_aliases):
        raise ValueError('Declared dynamic aliases must describe all pooled input PLYs')
    args = OmegaConf.create(config)
    static, dynamic = GaussianModel(args), Dynamic_GaussianModel(args)
    env = EnvLight(resolution=args.env_map_res).cuda()
    model_paths = checkpoint_paths(options.model, options.iteration)[:3]
    models_to_restore = (static, dynamic, env)
    if len(model_paths) != len(models_to_restore):
        raise ValueError('checkpoint/model lengths differ')
    for path, model in zip(model_paths, models_to_restore):
        payload, step = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
        if int(step) != options.iteration:
            raise ValueError(f'Wrong checkpoint iteration: {path}: {step}')
        payload = list(payload)
        if model is env:
            payload[0] = torch.nn.Parameter(payload[0].detach().cuda(), requires_grad=False)
        else:
            for index in range(1, 11):
                payload[index] = payload[index].detach().cuda()
            model.time_duration = tuple(config['time_duration'])
        model.restore(tuple(payload), training_args=None)
        if getattr(model, 'optimizer', None) is not None:
            raise RuntimeError(f'Unexpected optimizer in evaluation: {type(model).__name__}')
        for value in vars(model).values():
            if torch.is_tensor(value):
                value.requires_grad_(False)
        del payload
        gc.collect()
    return args, static, dynamic, env


def evaluate(options, frames, denominator_sha, crop, references, replay_status, models):
    import torch
    sys.path.insert(0, str(options.engine))
    from scene.cameras import Camera
    from gaussian_renderer import render

    if not torch.cuda.is_available():
        raise RuntimeError('A real CUDA environment is required for native evaluation')
    torch.set_num_threads(1)
    config = read_json(options.model / 'effective_config.json')
    args, static, dynamic, env = restore_models(options, config)
    bg = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    rows = []
    replay_max = 0
    replay_checks = 0
    float_dir, anchor_dir = options.output / 'a5_float', options.output / 'anchors'
    float_dir.mkdir()
    anchor_dir.mkdir()
    for frame in frames:
        raw, mask, K, W = (frame[key] for key in ('raw', 'mask', 'K', 'W'))
        camera = Camera(raw, W[:3, :3].T.copy(), W[:3, 3].copy(),
                        cx=K[0, 2], cy=K[1, 2], fx=K[0, 0], fy=K[1, 1],
                        image=torch.zeros((3, 320, 480)), image_name=f'0008{raw:03d}',
                        resolution=(480, 320), timestamp=frame['timestamp'])
        actual_W = camera.world_view_transform.detach().cpu().numpy().T
        if not np.allclose(actual_W, W, atol=MATRIX_ATOL, rtol=0):
            raise ValueError(f'raw{raw}: constructed native Camera matrix mismatch')
        if abs(float(camera.timestamp) - frame['timestamp']) > TIME_ATOL:
            raise ValueError(f'raw{raw}: constructed Camera timestamp mismatch')
        with torch.no_grad():
            package = render(camera, static, dynamic, args, bg, env_map=env, is_training=False)
        # These are the original owner5 renderer keys; no replacement rasterizer.
        keep = tensor_hwc(package['render'])
        deleted = tensor_hwc(package['render_s'])
        keep_alpha = tensor_alpha(package['alpha'])
        deleted_alpha = tensor_alpha(package['static_alpha'])
        with np.load(frame['path'], allow_pickle=False) as data:
            gt = np.ascontiguousarray(data['rgb'], dtype=np.float32)
            if not np.array_equal(data['actor_labels'] == 1, mask):
                raise RuntimeError(f'raw{raw}: observation mask changed after preflight')
        visible = metrics(keep[mask], deleted[mask], gt[mask], keep_alpha[mask], deleted_alpha[mask])
        boundary = border_inside(mask)
        edge = metrics(keep[boundary], deleted[boundary], gt[boundary], keep_alpha[boundary], deleted_alpha[boundary])
        indices = np.column_stack(np.nonzero(mask)).astype(np.int16)
        np.savez_compressed(float_dir / f'{raw:03d}.npz', raw=np.int16(raw), yx=indices,
                            keep_rgb=keep[mask], delete_rgb=deleted[mask], gt_rgb=gt[mask],
                            keep_alpha=keep_alpha[mask], delete_alpha=deleted_alpha[mask],
                            mask_sha256=np.asarray(frame['mask_sha256']),
                            model_time=np.float64(frame['timestamp']))
        images = dict(keep=rgb8(keep), delete=rgb8(deleted), gt=rgb8(gt))
        directory = options.output / 'frames' / f'{raw:03d}'
        directory.mkdir(parents=True)
        for name, image in images.items():
            file_name = options.delete_name if name == 'delete' else name
            Image.fromarray(image).save(directory / f'{file_name}.png')
        replay = {}
        for name, path in references.get(raw, {}).items():
            with Image.open(path) as image:
                expected = np.asarray(image.convert('RGB'))
            if expected.shape != images[name].shape:
                raise ValueError(f'raw{raw}: native replay image shape mismatch: {path}')
            delta = np.abs(images[name].astype(np.int16) - expected.astype(np.int16))
            maximum = int(delta.max())
            replay[name] = dict(source=str(path), max_uint8_difference=maximum,
                                mean_uint8_difference=float(delta.mean()))
            replay_max = max(replay_max, maximum)
            replay_checks += 1
            if maximum > options.replay_tolerance:
                raise RuntimeError(f'raw{raw}: {name} native PNG replay differs by {maximum}')
        if raw in ANCHORS:
            panel = [images['gt'], images['keep'], images['delete']]
            labels = [f'raw{raw} GT', 'keep', options.delete_name]
            contact_sheet(anchor_dir / f'{raw:03d}_full.png', panel, labels)
            contact_sheet(anchor_dir / f'{raw:03d}_crop.png', panel, labels, crop)
            Image.fromarray(mask.astype(np.uint8) * 255).save(anchor_dir / f'{raw:03d}_fixed_mask.png')
        rows.append(dict(raw=raw, fixed_mask_sha256=frame['mask_sha256'],
                         model_time=frame['timestamp'], visible=visible, boundary=edge,
                         native_png_replay=replay))
        print(json.dumps(dict(raw=raw, pixels=visible['pixels'],
                              keep_rgb_mae=visible['keep_rgb_mae'], delete_rgb_mae=visible['delete_rgb_mae'])), flush=True)
        del package, camera
    result = dict(status='PASS_FRESH_NATIVE49_A5_EVALUATION', frames=rows,
                  frame_count=len(rows), raw_ids=list(RAWS), iteration=options.iteration,
                  model=str(options.model), optimizer_created=False, optimization_updates=0,
                  dynamic_aliases=options.dynamic_aliases, delete_name=options.delete_name,
                  deletion='render_s: all declared pooled dynamic PLYs removed',
                  a5_preservation_expected=5 not in options.dynamic_aliases,
                  native_static_rows=len(static.get_xyz), native_dynamic_rows=len(dynamic.get_xyz),
                  fixed_denominator=dict(rule='observations actor_labels == 1 (a5)',
                                         pixels=sum(frame['pixels'] for frame in frames),
                                         sha256=denominator_sha, new_alpha_used_to_select_pixels=False),
                  all49=aggregate(rows, 'visible'), boundary_all49=aggregate(rows, 'boundary'),
                  metric_units='sRGB [0,1] channel-mean absolute error per fixed pixel; primary RGB is clamped; raw RGB MAE also retained',
                  alpha_thresholds=list(THRESHOLDS),
                  alpha_caveat='Total native alpha can be filled by background or another surface. High alpha coverage alone does not establish a5 body completeness or correct ownership.',
                  observation_scope='Fixed original same-scene observed RGB; no clean deletion or hidden-surface ground truth',
                  camera_checks=dict(K_atol=MATRIX_ATOL, model_to_camera_atol=MATRIX_ATOL,
                                     time_atol=TIME_ATOL, all49_passed=True),
                  anchor_raw_ids=list(ANCHORS), fixed_crop_xyxy=crop,
                  crop_rule='Fixed CLI rectangle for visualization only; mask metrics retain the full original image domain. No render alpha used for crop or denominator.',
                  replay=dict(event_status=replay_status, checked_images=replay_checks,
                              max_uint8_difference=replay_max if replay_checks else None,
                              tolerance=options.replay_tolerance,
                              expected_complete_images=98,
                              complete_all49_keep_delete_replay=replay_checks == 98,
                              unavailable_is_not_a_pass=replay_checks == 0),
                  model_source_entries={name: dict(model=str(path), iteration=step) for name, (path, step) in models.items()},
                  engine=str(options.engine), torch_version=torch.__version__,
                  cuda_runtime=torch.version.cuda, quality_accepted=False, human_verdict=None)
    write_json(options.output / 'summary.json', result)
    columns = ('raw', 'pixels', 'keep_rgb_mae', 'delete_rgb_mae', 'keep_delete_rgb_change',
               'keep_mean_alpha', 'delete_mean_alpha')
    with (options.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(raw=row['raw'], **{key: row['visible'].get(key) for key in columns[1:]}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dynamic-aliases', type=int, nargs='+', required=True)
    parser.add_argument('--delete-name', required=True)
    parser.add_argument('--observations', type=Path, default=OBSERVATIONS)
    parser.add_argument('--engine', type=Path, default=ENGINE)
    parser.add_argument('--source-model', action='append', default=[], metavar='NAME=PATH@ITERATION',
                        help='Add/override source manifests; defaults: evaluated, full5_b30k, current_native10k')
    parser.add_argument('--expected-mask-pixels', type=int, default=121128)
    parser.add_argument('--replay-tolerance', type=int, default=1)
    parser.add_argument('--crop', type=int, nargs=4, default=[0, 160, 175, 320],
                        metavar=('X0', 'Y0', 'X1', 'Y1'), help='Fixed display crop only; does not alter the all49 mask domain')
    options = parser.parse_args()
    for key in ('model', 'output', 'observations', 'engine'):
        setattr(options, key, getattr(options, key).resolve())
    if options.iteration <= 0 or options.expected_mask_pixels <= 0 or options.replay_tolerance < 0:
        parser.error('Iteration/mask count must be positive and replay tolerance nonnegative')
    if len(set(options.dynamic_aliases)) != len(options.dynamic_aliases) or any(value <= 0 for value in options.dynamic_aliases):
        parser.error('Dynamic aliases must be unique positive integers')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', options.delete_name) or options.delete_name in ('.', '..', 'gt', 'keep'):
        parser.error('Use a distinct simple --delete-name without path separators')
    x0, y0, x1, y1 = options.crop
    if not (0 <= x0 < x1 <= 480 and 0 <= y0 < y1 <= 320):
        parser.error('Crop must lie inside the 480x320 image')
    models = parse_source_models(options)
    protected_roots = [options.engine, options.observations] + [path for path, _ in models.values()]
    if any(options.output == root or options.output.is_relative_to(root) or root.is_relative_to(options.output)
           for root in protected_roots):
        raise ValueError('Output must be separate from every protected model/engine/observation directory')
    source_groups = {name: source_model_files(path, step) for name, (path, step) in models.items()}
    event, references, replay_status = event_references(options.model, options.iteration)
    observations = [options.observations / f'{raw:03d}.npz' for raw in RAWS]
    engine_files = sorted(options.engine.rglob('*.py'))
    if not engine_files or not (options.engine / 'gaussian_renderer/__init__.py').is_file():
        raise FileNotFoundError('Frozen native engine source is unavailable')
    paths = [Path(__file__).resolve(), *observations, *engine_files]
    for files in source_groups.values():
        paths.extend(files)
    paths.extend(path for channels in references.values() for path in channels.values())
    options.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before = inventory(paths)
    write_json(options.output / 'source_manifest_before.json',
               dict(files=before, script=str(Path(__file__).resolve()),
                    script_sha256=sha256(__file__), groups={name: list(map(str, files)) for name, files in source_groups.items()},
                    engine_source_files=list(map(str, engine_files)),
                    observations=list(map(str, observations))))
    error = None
    result = None
    try:
        frames, denominator_sha, crop = observation_headers(
            options, read_json(options.model / 'cameras.json'),
            read_json(options.model / 'actual_scene_binding.json'))
        write_json(options.output / 'fixed_domain.json',
                   dict(rule='actor_labels==1', total_pixels=sum(frame['pixels'] for frame in frames),
                        sha256=denominator_sha, crop_xyxy=crop,
                        frames=[dict(raw=frame['raw'], pixels=frame['pixels'], mask_sha256=frame['mask_sha256']) for frame in frames]))
        result = evaluate(options, frames, denominator_sha, crop, references, replay_status, models)
    except BaseException:
        error = traceback.format_exc()
    finally:
        after = inventory(paths)
        changed = [path for path in before if before[path] != after.get(path)]
        write_json(options.output / 'source_integrity.json',
                   dict(status='PASS_UNCHANGED' if not changed else 'FAIL_SOURCE_CHANGED',
                        scope='Only the explicitly inventoried consumed source/checkpoint/observation/engine/event-image files; not a claim about the entire remote dataset or filesystem',
                        checked_files=len(before), changed=changed,
                        checked_fields=['bytes', 'mtime_ns', 'sha256'], before=before, after=after,
                        script_sha256=sha256(__file__), seconds=time.monotonic() - started))
    if changed:
        error = (error or '') + '\nProtected source files changed: ' + repr(changed)
    if error:
        write_json(options.output / 'failure.json', dict(status='FAILED', traceback=error))
        if (options.output / 'summary.json').is_file():
            if result is None:
                result = read_json(options.output / 'summary.json')
            result['status'] = 'FAIL_SOURCE_INTEGRITY' if changed else 'FAIL_EVALUATION'
            write_json(options.output / 'summary.json', result)
        raise RuntimeError(error)
    result['source_integrity'] = 'PASS_UNCHANGED_BYTES_MTIME_SHA256'
    result['script_sha256'] = sha256(__file__)
    result['seconds'] = time.monotonic() - started
    write_json(options.output / 'summary.json', result)
    print(json.dumps(dict(status=result['status'], all49=result['all49'],
                          source_integrity=result['source_integrity']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
