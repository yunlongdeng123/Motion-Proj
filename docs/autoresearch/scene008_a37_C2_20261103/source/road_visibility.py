"""OBB camera visibility and low-dimensional direct-light visibility on road.

Only the stable road raster rows are changed. The native gate and time state
remain active. Observed road RGB is a fixed radiance proxy, not known albedo.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import gc
import json
import sys
import time

RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')
R3 = RUN.parent / 'scene008_a37_only_B10k_road_20261103_r3'
META = RUN.parent / 'r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json'
SETTINGS = RUN / 'evidence/road_visibility_probe/sun_only/selected_settings.json'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def ray_obb(origins, directions, center, axes, half, maximum=None, epsilon=1e-5):
    """Vectorized slab test in model coordinates; axes are OBB column vectors."""
    import torch
    q = (origins - center) @ axes
    d = directions @ axes
    parallel = d.abs() < 1e-9
    outside_parallel = (parallel & (q.abs() > half)).any(1)
    safe = torch.where(parallel, torch.ones_like(d), d)
    t0 = (-half - q) / safe
    t1 = (half - q) / safe
    lower = torch.where(parallel, torch.full_like(d, -float('inf')), torch.minimum(t0, t1))
    upper = torch.where(parallel, torch.full_like(d, float('inf')), torch.maximum(t0, t1))
    near = lower.max(1).values
    far = upper.min(1).values
    hit = ~outside_parallel & (far >= torch.maximum(near, torch.full_like(near, epsilon)))
    if maximum is not None:
        hit &= near < maximum - epsilon
    return hit


class RoadVisibility:
    """No GT access: pose/raw, camera, road rows and frozen setting only."""
    def __init__(self, e, settings=None):
        import numpy as np
        self.e = e
        self.meta = json.loads(META.read_text())
        self.settings = dict(settings if settings is not None else json.loads(SETTINGS.read_text()))
        self.retained_always = tuple(sorted(set(int(alias) for alias in self.settings.get('retained_aliases_always', []))))
        if set(self.retained_always) - {6}:
            raise ValueError('This interface currently supports only a6 in retained_aliases_always')
        if self.retained_always and self.settings.get('camera_visibility', True):
            raise ValueError('Formal retained-a6 visibility permits sun visibility only, not coarse-OBB camera clipping')
        self.a6_geometry = dict(np.load(RUN / 'inputs/a6_geometry_minimal.npz', allow_pickle=False)) if 6 in self.retained_always else None
        self.scale = float(self.meta['meter_to_model'])
        self.cache = {}
        self.last_stats = {}
        self.extra_poses = None
        path = RUN / 'inputs/a6_raw_diagnostic_metadata.json'
        if path.exists():
            self.extra_poses = {int(row['raw']): row for row in json.loads(path.read_text())['actors']['a37']['trajectory']}
        anchor = np.asarray(self.meta['coordinate_anchor'])[:3, :3]
        model = np.asarray(self.meta['world_to_model'])[:3, :3]
        self.sun_model = model @ anchor @ np.asarray(self.settings['sun_direction_anchor'], dtype=float)
        self.sun_model /= np.linalg.norm(self.sun_model)

    def box(self, raw, alias=37):
        import numpy as np
        if alias == 6:
            if self.a6_geometry is None:
                raise ValueError('a6 geometry was not enabled by retained_aliases_always')
            matrix = np.asarray(self.a6_geometry['model_from_canonical'][raw], dtype=float)
            scale = np.linalg.norm(matrix[:3, :3], axis=0).mean()
            dimensions = self.a6_geometry['box_lwh_per_raw'][raw] if 'box_lwh_per_raw' in self.a6_geometry else self.a6_geometry['box_lwh_m']
            return dict(center=matrix[:3, 3], axes=matrix[:3, :3] / scale,
                        half=np.asarray(dimensions, dtype=float) * scale / 2)
        if alias != 37:
            raise ValueError(f'Unsupported retained alias {alias}')
        fid = f'0008{raw:03d}'
        if fid in self.meta['frames']:
            box = self.meta['frames'][fid]['selected_boxes_model'][self.meta['actors'].index(37)]
            return {k: np.asarray(box[k], dtype=float) for k in ('center', 'axes', 'half')}
        if self.extra_poses is None or raw not in self.extra_poses:
            raise ValueError(f'No a37 physical pose for raw{raw}')
        row = self.extra_poses[raw]
        transform = np.asarray(self.meta['world_to_model'])
        return dict(center=transform[:3, :3] @ np.asarray(row['center_world']) + transform[:3, 3],
            axes=transform[:3, :3] @ np.asarray(row['orientation_world']).reshape(3, 3) / self.scale,
            half=np.asarray(row['dimensions']) * self.scale / 2)

    def apply(self, raw, include_target, rr, camera=None):
        import numpy as np
        import torch
        result = dict(rr)
        retained = tuple(sorted(set(self.retained_always) | ({37} if include_target else set())))
        if not retained:
            self.last_stats = dict(raw=raw, retained_aliases=[], road_rows=len(rr['means3D']), camera_blocked=0, sun_blocked=0)
            return result
        camera = camera or self.e.camera(raw)
        device = rr['means3D'].device
        center_cam = camera.camera_center.detach()
        key = (int(raw), retained, tuple(center_cam.cpu().tolist()), len(rr['means3D']))
        if key not in self.cache:
            tensor = lambda x: torch.as_tensor(x, device=device, dtype=rr['means3D'].dtype)
            xyz = rr['means3D'].detach()
            vector = center_cam[None] - xyz
            length = vector.norm(dim=1)
            direction = vector / length[:, None].clamp_min(1e-8)
            sun = tensor(self.sun_model)[None].expand_as(xyz)
            blocked_camera = torch.zeros(len(xyz), device=device, dtype=torch.bool)
            blocked_sun = torch.zeros_like(blocked_camera)
            for alias in retained:
                box = self.box(raw, alias)
                center, axes, half = (tensor(box[k]) for k in ('center', 'axes', 'half'))
                assert float((axes.T @ axes - torch.eye(3, device=device)).abs().max()) < 1e-4
                blocked_camera |= ray_obb(xyz, direction, center, axes, half, maximum=length)
                blocked_sun |= ray_obb(xyz, sun, center, axes, half)
            self.cache[key] = (blocked_camera, blocked_sun)
        blocked_camera, blocked_sun = self.cache[key]
        if self.settings.get('camera_visibility', True):
            result['mask'] = rr['mask'] & ~blocked_camera
        if self.settings.get('sun_visibility', True):
            ambient = float(self.settings['ambient'])
            # Union the retained occluders first, then apply the ambient floor
            # exactly once. Overlapping shadows never multiply ambient twice.
            irradiance = torch.where(blocked_sun, torch.full_like(rr['means3D'][:, 0], ambient), torch.ones_like(rr['means3D'][:, 0]))
            if rr['colors_precomp'] is not None:
                colour = rr['colors_precomp']
            else:
                colour = .5 + .28209479177387814 * rr['shs'][:, 0]
            linear = torch.where(colour <= .04045, colour / 12.92, ((colour + .055).clamp_min(0) / 1.055) ** 2.4)
            shaded = (linear * irradiance[:, None]).clamp_min(0)
            srgb = torch.where(shaded <= .0031308, 12.92 * shaded, 1.055 * shaded.clamp_min(1e-10) ** (1 / 2.4) - .055)
            if rr['colors_precomp'] is not None:
                result['colors_precomp'] = srgb
            else:
                result['shs'] = torch.cat((((srgb - .5) / .28209479177387814)[:, None], rr['shs'][:, 1:]), 1)
        self.last_stats = dict(raw=int(raw), retained_aliases=list(retained), road_rows=len(rr['means3D']),
            camera_blocked=int(blocked_camera.sum()), sun_blocked=int(blocked_sun.sum()),
            camera_visible_sun_blocked=int((blocked_sun & ~blocked_camera).sum()))
        return result


def apply_road_visibility(e, raw, include_target, rr, camera=None):
    """Root interface; append feature tags after this call."""
    if not hasattr(e, '_road_visibility'):
        e._road_visibility = RoadVisibility(e)
    return e._road_visibility.apply(raw, include_target, rr, camera=camera)


def render_probe(e, raw, include_target, visibility=None):
    import torch
    from editor_common import prepare_rasterization, road_kwargs, merge_rows, raster
    camera = e.camera(raw)
    n = e.n
    kw = prepare_rasterization(camera, n.static, n.args)
    kw['means2D'] = torch.zeros_like(kw['means3D'])
    ids = e.repair.ids
    kw['opacities'] = kw['opacities'].index_copy(0, ids, kw['opacities'][ids] * e.repair.gate_raw.sigmoid()[:, None])
    if include_target:
        dy = prepare_rasterization(camera, n.dynamic, n.args, dynamic=True)
        dy['means2D'] = torch.zeros_like(dy['means3D'])
        kw = merge_rows(kw, dy)
    channels = kw['shs'].shape[1] if kw['shs'] is not None else 1
    rr = road_kwargs(camera, e.repair.road, n.args, channels, precomputed_color=kw['colors_precomp'] is not None)
    if visibility is not None:
        rr = visibility.apply(raw, include_target, rr, camera)
    kw['features'] = torch.cat((kw['features'], torch.zeros_like(kw['features'][:, :1])), 1)
    rr['features'] = torch.cat((rr['features'], torch.ones_like(rr['features'][:, :1])), 1)
    package = raster(camera, n.args, n.static.active_sh_degree, merge_rows(kw, rr), n.bg)
    env = n.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
    package['rgb'] = package['rgb'] + (1 - package['alpha']) * env
    return package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUN / 'evidence/road_visibility_probe')
    parser.add_argument('--sun-only', action='store_true', help='Disable coarse-OBB camera clipping; change stable road RGB only')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(R3 / 'source'))
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_erosion
    import torch
    import repair37
    from utils.loss_utils import psnr, ssim
    started = time.monotonic()
    output = args.output / 'sun_only' if args.sun_only and args.output == RUN / 'evidence/road_visibility_probe' else args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'calibration').mkdir()
    (output / 'all49').mkdir()
    torch.set_num_threads(1)
    e = repair37.Experiment()
    ck = torch.load(str(R3 / 'formal/checkpoint001000.pt'), map_location='cpu', weights_only=False)
    assert ck['parent_identity'] == e.n.before and ck['step'] == 1000
    e.repair.load_state_dict(ck['repair'], strict=True)
    e.repair.requires_grad_(False)
    del ck
    gc.collect()
    def rgb8(t):
        return np.uint8(np.floor(np.clip(t.detach().cpu().numpy().transpose(1, 2, 0), 0, 1) * 255 + .5))
    def ground_truth(raw):
        path = Path(e.n.config['source_path']) / 'image/image_0' / f'0008{raw:03d}.png'
        image = cv2.resize(np.asarray(Image.open(path).convert('RGB')) / 255., (480, 320)).astype(np.float32)
        return torch.as_tensor(image.transpose(2, 0, 1).copy(), device='cuda')
    def measure(raw, package, truth):
        pred = package['rgb'].clamp(0, 1)
        d = e.view(raw)
        target = d['actor_labels'] == 5
        masks = dict(target=target, retained_a6=d['actor_labels'] == 4,
            near_target_road=binary_dilation(target, iterations=12) & ~target & (d['actor_labels'] == 0),
            repair_without_target=d['repair_mask'] & ~target,
            road_boundary=binary_dilation(d['repair_mask'], iterations=2) & ~binary_erosion(d['repair_mask'], iterations=2) & ~d['other_vehicle_guard'])
        metrics = dict(psnr=float(psnr(pred, truth)), ssim=float(ssim(pred[None], truth[None])), masks={})
        for key, mask in masks.items():
            t = torch.as_tensor(mask, device='cuda')
            metrics['masks'][key] = dict(samples=int(mask.sum()), gt_srgb_MAE=float((pred - truth).abs()[:, t].mean()) if mask.any() else None,
                road_alpha_T_mean=float(package['extra'][0, t].mean()) if mask.any() else None)
        return metrics
    candidates = {}
    for light, direction in dict(top=[0, 0, 1], directional=[.6, -.3, 1]).items():
        for ambient in (.15, .30, .50):
            name = f'{light}_ambient{ambient:.2f}'
            candidates[name] = dict(schema='c2.road_visibility.v1', retained_geometry=[37],
                camera_visibility=not args.sun_only, sun_visibility=True, ambient=ambient,
                sun_direction_anchor=direction, calibration_domain='BUILD raw105,113,118',
                colour_interpretation='fixed observed/calibrated radiance proxy, not physical albedo',
                method='road-row camera segment OBB visibility plus ambient+(1-ambient)*direct-sun OBB visibility in linear RGB',
                native_time_state_preserved=True, native_static_gate_still_enabled=True, optimizer_updates=0)
    diagnostic = {} if args.sun_only else dict(camera_only=dict(next(iter(candidates.values())), sun_visibility=False),
                      shadow_only=dict(candidates['directional_ambient0.30'], camera_visibility=False))
    helpers = {name: RoadVisibility(e, setting) for name, setting in {**candidates, **diagnostic}.items()}
    calibration = []
    with torch.no_grad():
        for raw in (105, 113, 118):
            gt = ground_truth(raw)
            record = dict(raw=raw, methods={})
            packages = dict(legacy=render_probe(e, raw, True))
            for name, helper in helpers.items():
                packages[name] = render_probe(e, raw, True, helper)
            for name, package in packages.items():
                record['methods'][name] = measure(raw, package, gt)
                if name in helpers:
                    record['methods'][name]['geometry'] = helpers[name].last_stats
                Image.fromarray(rgb8(package['rgb'])).save(output / 'calibration' / f'{raw:03d}_{name}.png')
            calibration.append(record)
            print(json.dumps(dict(stage='visibility_calibration', raw=raw, psnr={k:v['psnr'] for k,v in record['methods'].items()})), flush=True)
        scores = {name: float(np.mean([row['methods'][name]['psnr'] for row in calibration])) for name in candidates}
        selected_name = max(scores, key=scores.get)
        selected = dict(candidates[selected_name], candidate=selected_name, calibration_score= scores[selected_name],
            selection='max mean per-frame PSNR over three explicit BUILD calibration frames; no independent validation',
            frozen_before_all49_evaluation=True)
        write(output / 'selected_settings.json', selected)
        write(output / 'calibration.json', dict(candidates=candidates, diagnostics=diagnostic, scores=scores, selected=selected_name, rows=calibration))
        helper = RoadVisibility(e, selected)
        all49 = []
        replay_max = 0
        for raw in range(89, 138):
            gt = ground_truth(raw)
            package = render_probe(e, raw, True, helper)
            full_metrics = measure(raw, package, gt)
            stats = dict(helper.last_stats)
            deleted = render_probe(e, raw, False, helper)
            old = np.asarray(Image.open(R3 / 'evaluation/all49' / f'{raw:03d}_repaired.png').convert('RGB'))
            diff = int(np.abs(rgb8(deleted['rgb']).astype(np.int16) - old.astype(np.int16)).max())
            replay_max = max(replay_max, diff)
            assert diff <= 1, (raw, diff)
            row = dict(raw=raw, full=full_metrics, geometry=stats, deletion_replay_max_uint8=diff)
            all49.append(row)
            Image.fromarray(rgb8(package['rgb'])).save(output / 'all49' / f'{raw:03d}_full.png')
            Image.fromarray(rgb8(deleted['rgb'])).save(output / 'all49' / f'{raw:03d}_delete37.png')
            if raw in (105, 113, 118):
                Image.fromarray(rgb8(package['extra'].expand(3, -1, -1))).save(output / f'{raw:03d}_road_alpha_T.png')
                legacy = np.asarray(Image.open(RUN / 'evidence/baseline_replay/frames' / f'{raw:03d}/repaired_full.png'))
                native = np.asarray(Image.open(RUN / 'evidence/baseline_replay/frames' / f'{raw:03d}/native_full.png'))
                Image.fromarray(np.concatenate((native, legacy, rgb8(package['rgb'])), axis=1)).save(output / f'{raw:03d}_native_legacy_visibility.png')
            if raw == 89 or (raw - 88) % 10 == 0 or raw == 137:
                print(json.dumps(dict(stage='visibility_all49', raw=raw, seconds=time.monotonic() - started, full=full_metrics['psnr'])), flush=True)
        native_summary = json.loads((RUN / 'evidence/baseline_replay/summary.json').read_text())['methods']
        summary = dict(optimizer_updates=0, selected=selected, full49=dict(psnr=float(np.mean([row['full']['psnr'] for row in all49])),
            ssim=float(np.mean([row['full']['ssim'] for row in all49]))), old_baseline=native_summary,
            deletion_replay_max_uint8=replay_max, parent_unchanged=e.unchanged(),
            noedit_repair_always_enabled=True, target_dynamic_retained=True, a6_geometry_not_duplicated=True,
            camera_visibility_enabled=not args.sun_only, sun_only_probe=args.sun_only,
            frames=49, independent_test=False, fixed_observed_road_RGB_is_not_albedo=True,
            renderer_reads_GT=False, calibration_does_read_BUILD_GT=True,
            source_box=META.as_posix(), seconds=time.monotonic() - started)
        summary['meets_original_native_full'] = summary['full49']['psnr'] >= native_summary['native_full']['psnr'] and summary['full49']['ssim'] >= native_summary['native_full']['ssim']
        assert e.unchanged()
        write(output / 'metrics.json', dict(summary=summary, rows=all49))
        write(output / 'summary.json', summary)
        print(json.dumps(dict(stage='VISIBILITY_COMPLETE', summary=summary)), flush=True)
    del helper, helpers, e
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
