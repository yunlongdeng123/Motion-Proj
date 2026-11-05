"""Native BUILD readout with a37-only deletion from a packed dynamic checkpoint."""
import os

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


ENGINE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
RAWS = tuple(range(89, 138))
WIDTH, HEIGHT = 480, 320


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checkpoint_identity(paths):
    records = {}
    for path in paths:
        digest = hashlib.sha256()
        before = path.stat()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Checkpoint changed while hashing: {path}')
        records[path.name] = dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                                  sha256=digest.hexdigest())
    return records


def load_component(torch, path, model, iteration, time_duration=None):
    payload, step = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
    if step != iteration:
        raise ValueError(f'{path.name} holds iteration {step}, expected {iteration}')
    payload = list(payload)
    if time_duration is None:
        payload[0] = torch.nn.Parameter(payload[0].detach().cuda(), requires_grad=False)
    else:
        for index in range(1, 11):
            payload[index] = payload[index].detach().cuda()
        model.time_duration = tuple(time_duration)
    model.restore(tuple(payload), training_args=None)
    del payload
    gc.collect()


def cpu(value):
    return value.detach().cpu().numpy()


def rgb8(value):
    rgb = value.transpose(1, 2, 0)
    return np.uint8(np.floor(np.clip(rgb, 0, 1) * 255 + .5))


def native_camera(Camera, torch, raw, record, timestamp):
    if (record['width'], record['height']) != (WIDTH * 4, HEIGHT * 4):
        raise ValueError(f'Unexpected source resolution for raw{raw}')
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = record['rotation'], record['position']
    world_to_camera = np.linalg.inv(c2w)
    fx, fy, cx, cy = (record[key] / 4 for key in ('fx', 'fy', 'cx', 'cy'))
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera = Camera(raw, world_to_camera[:3, :3].T.copy(), world_to_camera[:3, 3].copy(),
                    cx=cx, cy=cy, fx=fx, fy=fy, image=torch.zeros((3, HEIGHT, WIDTH)),
                    image_name=f'0008{raw:03d}', resolution=(WIDTH, HEIGHT), timestamp=timestamp)
    return camera, intrinsic, world_to_camera


def stat(values):
    pixels = int(values.size)
    total = float(values.sum(dtype=np.float64))
    return dict(pixels=pixels, sum=total, mean=total / pixels if pixels else None)


def image_metrics(composite, deleted, observed, masks):
    maps = dict(composite_vs_observed=np.abs(composite - observed).mean(axis=2),
                delete_a37_vs_observed=np.abs(deleted - observed).mean(axis=2),
                edit_delta=np.abs(deleted - composite).mean(axis=2))
    return {name: {metric: stat(values[mask]) for metric, values in maps.items()}
            for name, mask in masks.items()}


def pool_metrics(rows):
    pooled = {}
    for name, metrics in rows[0]['observed_RGB'].items():
        for metric in metrics:
            records = [row['observed_RGB'][name][metric] for row in rows]
            pixels = sum(value['pixels'] for value in records)
            total = sum(value['sum'] for value in records)
            pooled[f'observed_RGB/{name}/{metric}'] = dict(
                pixels=pixels, sum=total, mean=total / pixels if pixels else None)
    return pooled


def load_reference(path):
    if path is None:
        return None
    names = ('schema', 'raw_ids', 'K', 'model_to_camera', 'model_time', 'depth_m',
             'observed_rgb', 'actor_labels', 'observed_road_guard')
    with np.load(path, allow_pickle=False) as archive:
        ref = {name: archive[name] for name in names}
    if ref['schema'].item() != 'scene008.fixed_a6_probe.v1' or tuple(ref['raw_ids']) != RAWS:
        raise ValueError('Unexpected fixed a6 and road reference package')
    if (ref['observed_rgb'].shape != (len(RAWS), HEIGHT, WIDTH, 3)
            or ref['observed_rgb'].dtype != np.uint8
            or ref['actor_labels'].shape != (len(RAWS), HEIGHT, WIDTH)
            or ref['observed_road_guard'].shape != (len(RAWS), HEIGHT, WIDTH)):
        raise ValueError('Fixed reference has incompatible pixel layout')
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--iteration', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', choices=('113', 'all'), default='113')
    parser.add_argument('--float-npz', '--npz', action='store_true')
    parser.add_argument('--reference', type=Path,
                        help='Optional fixed-a6-probe NPZ for shared visible and observed-road RGB metrics')
    parser.add_argument('--actor-layers', action='store_true',
                        help='Write isolated dynamic layers for every requested view; raw113 always includes them')
    opt = parser.parse_args()
    start = time.monotonic()

    sys.path.insert(0, str(ENGINE))
    import cv2
    import torch
    from omegaconf import OmegaConf
    from scene import Dynamic_GaussianModel, EnvLight, GaussianModel
    from scene.cameras import Camera
    from gaussian_renderer import render
    from utils.paper_masks import _read_mask
    from dual_actor_runtime import actor_ids, hide_actors, install_identity_serialization, render_without_a37

    install_identity_serialization()
    torch.set_num_threads(1)
    config = read_json(opt.model / 'effective_config.json')
    args = OmegaConf.create(config)
    source = Path(config['source_path'])
    a6_masks = Path(config['dual_actor_a6_masks'])
    prefixes = ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt')
    files = [opt.model / f'{prefix}{opt.iteration}.pth' for prefix in prefixes]
    training_state = opt.model / f'training_state{opt.iteration}.pth'
    checked_files = files + ([training_state] if training_state.is_file() else [])
    identity = checkpoint_identity(checked_files)
    static, dynamic = GaussianModel(args), Dynamic_GaussianModel(args)
    env = EnvLight(resolution=args.env_map_res).cuda()
    load_component(torch, files[0], static, opt.iteration, config['time_duration'])
    load_component(torch, files[1], dynamic, opt.iteration, config['time_duration'])
    load_component(torch, files[2], env, opt.iteration)
    ids = actor_ids(dynamic)
    counts = {str(actor): int((ids == actor).sum()) for actor in (6, 37)}
    if min(counts.values()) <= 0 or sum(counts.values()) != len(dynamic.get_xyz):
        raise ValueError(f'Invalid dynamic actor partition: {counts}')
    original_opacity = dynamic._opacity.detach().clone()

    records = {row['img_name']: row for row in read_json(opt.model / 'cameras.json')}
    stamps = read_json(opt.model / 'actual_scene_binding.json')['timestamps']
    if len(stamps) != len(RAWS) or not np.isfinite(stamps).all():
        raise ValueError('Expected the original 49 finite FRONT timestamps')
    ref = load_reference(opt.reference)
    bg = torch.full((3,), float(args.white_background), device='cuda')
    out = opt.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raws = (113,) if opt.frames == '113' else RAWS
    rows = []
    for raw in raws:
        index, fid = raw - RAWS[0], f'0008{raw:03d}'
        camera, intrinsic, world_to_camera = native_camera(Camera, torch, raw, records[fid], stamps[index])
        with torch.no_grad():
            full = render(camera, static, dynamic, args, bg, env_map=env, is_training=False)
            layers = dict(composite=cpu(full['render']), static_only=cpu(full['render_s']))
            if opt.float_npz:
                layers.update(composite_alpha=cpu(full['alpha']), composite_depth=cpu(full['depth']),
                              static_alpha=cpu(full['static_alpha']), static_depth=cpu(full['static_depth']))
            del full
            deleted = render_without_a37(camera, static, dynamic, args, bg, env)
            layers['delete_a37'] = cpu(deleted['render'])
            a6_alpha = cpu(deleted['dynamic_alpha'])[0]
            if opt.float_npz:
                layers.update(delete_a37_alpha=cpu(deleted['alpha']), delete_a37_depth=cpu(deleted['depth']),
                              a6_only_dynamic_alpha=a6_alpha)
            if opt.actor_layers or raw == 113:
                layers['a6_only_dynamic'] = cpu(deleted['render_d'])
            del deleted
            if opt.actor_layers or raw == 113:
                with hide_actors(dynamic, [6]):
                    a37 = render(camera, static, dynamic, args, bg, env_map=env, is_training=False)
                    layers['a37_only_dynamic'] = cpu(a37['render_d'])
                del a37
        if not torch.equal(dynamic._opacity, original_opacity):
            raise RuntimeError(f'Packed dynamic opacity did not restore after raw{raw}')

        with Image.open(source / 'image/image_0' / f'{fid}.png') as image:
            observed = np.asarray(image.convert('RGB'), dtype=np.float64) / 255
        gt = cv2.resize(observed, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        layers['gt'] = gt.transpose(2, 0, 1)
        if not all(np.isfinite(value).all() for value in layers.values()):
            raise ValueError(f'Nonfinite native readout at raw{raw}')
        folder = out / f'{raw:03d}'
        folder.mkdir()
        images = {name: rgb8(value) for name, value in layers.items() if value.shape == (3, HEIGHT, WIDTH)}
        for name, image in images.items():
            Image.fromarray(image).save(folder / f'{name}.png')
        if opt.float_npz:
            np.savez_compressed(folder / 'native_layers.npz', **layers)

        if ref is None:
            a6_body = cpu(_read_mask(str(a6_masks / f'{fid}.png'), WIDTH, HEIGHT))[0] > 0
            a37_body = cpu(_read_mask(str(source / 'dynamic_masks/dynamic_mask_0' / f'{fid}.png'),
                                      WIDTH, HEIGHT))[0] > 0
            masks = dict(a6_visible=a6_body & ~a37_body)
            truth = images['gt']
        else:
            for name, actual in (('K', intrinsic), ('model_to_camera', world_to_camera),
                                 ('model_time', stamps[index])):
                if not np.allclose(ref[name][index], actual, rtol=0, atol=1e-6):
                    raise ValueError(f'Fixed probe reference {name} disagrees at raw{raw}')
            masks = dict(a6_visible=ref['actor_labels'][index] == 4,
                         a6_measured_visible=ref['depth_m'][index, 0] > 0,
                         observed_road_guard=ref['observed_road_guard'][index].astype(bool, copy=False))
            truth = ref['observed_rgb'][index]
        metrics = image_metrics(images['composite'].astype(np.float32) / 255,
                                images['delete_a37'].astype(np.float32) / 255,
                                truth.astype(np.float32) / 255, masks)
        rows.append(dict(raw=raw, frame_id=fid, observed_RGB=metrics,
                         a6_only_dynamic_alpha_on_visible=stat(a6_alpha[masks['a6_visible']]),
                         dynamic_opacity_restored_exactly=True))

    if checkpoint_identity(checked_files) != identity:
        raise RuntimeError('Checkpoint size, timestamp or SHA256 changed during readout')
    pooled = pool_metrics(rows)
    visible = {name: pooled[f'observed_RGB/a6_visible/{name}'] for name in
               ('composite_vs_observed', 'delete_a37_vs_observed', 'edit_delta')}
    summary = dict(status='PASS_DUAL_NATIVE_READOUT', iteration=opt.iteration,
                   frame_count=len(rows), raw_ids=list(raws), frames=rows, pooled=pooled,
                   retained_weighted={'6': dict(pixels=visible['edit_delta']['pixels'],
                       composite_vs_observed_RGB_MAE=visible['composite_vs_observed']['mean'],
                       delete_vs_observed_RGB_MAE=visible['delete_a37_vs_observed']['mean'],
                       deletion_MAE=visible['edit_delta']['mean'])},
                   actor_ids=[6, 37], actor_counts=counts, static_rows=len(static.get_xyz),
                   dynamic_rows=len(dynamic.get_xyz), dynamic_opacity_restored_exactly=True,
                   deletion_render='native joint-depth render with dynamic actor ID 37 opacity suppressed',
                   static_only_render='native static readout excluding both dynamic actors',
                   checkpoint_identity=identity, checkpoint_files_modified=False, updates=0,
                   reference=str(opt.reference) if ref is not None else None,
                   rgb_contract='Observed RGB guards use output PNG sRGB and same-view observed RGB',
                   probe_render_dir=str(out), evaluation_domain='observed FRONT training views BUILD',
                   native_engine=str(ENGINE), seconds=time.monotonic() - start)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                    encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'frames'},
                     ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
