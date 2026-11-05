"""Freeze shared a6 reference rays on CPU and read native static opacity on CUDA."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


WIDTH, HEIGHT = 480, 320
RAWS = tuple(range(89, 138))
LAYERS = (
    'a6_measured_visible',
    'a6_observed_reprojected_unknown',
    'a6_pvd_generated_unknown',
    'ground_beneath_a37_proxy',
)
SCHEMA = 'scene008.fixed_a6_probe.v1'
MARKER_ALPHA = .008


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_camera_geometry(cameras, binding):
    records = {row['img_name']: row for row in read_json(cameras)}
    stamps = np.asarray(read_json(binding)['timestamps'], dtype=np.float64)
    if stamps.shape != (len(RAWS),):
        raise ValueError('Expected the original 49 BUILD timestamps')
    intrinsics, model_to_camera = [], []
    for raw in RAWS:
        row = records[f'0008{raw:03d}']
        if (row['width'], row['height']) != (4 * WIDTH, 4 * HEIGHT):
            raise ValueError(f'Unexpected native camera resolution raw{raw}')
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = row['rotation'], row['position']
        intrinsics.append([[row['fx'] / 4, 0, row['cx'] / 4],
                           [0, row['fy'] / 4, row['cy'] / 4], [0, 0, 1]])
        model_to_camera.append(np.linalg.inv(c2w))
    return np.asarray(intrinsics), np.asarray(model_to_camera), stamps


def center_depth_splats(canonical, canonical_to_camera, intrinsic, scale, sigma=None):
    """Fixed center-depth splats at native CUDA pixel centers, with no learned weights."""
    camera_xyz = canonical @ canonical_to_camera[:3, :3].T + canonical_to_camera[:3, 3]
    u, v, z_model = camera_xyz.T
    valid = np.isfinite(camera_xyz).all(1) & (z_model > .2)
    u, v, z_model = u[valid], v[valid], z_model[valid]
    u = intrinsic[0, 0] * u / z_model + intrinsic[0, 2] - .5
    v = intrinsic[1, 1] * v / z_model + intrinsic[1, 2] - .5
    z_m = z_model / scale
    # Measured points use a fixed 1.25-pixel disk. PVD patch scales only set
    # footprint support, capped at 2 pixels. Both store the point's center z.
    radius = np.full(len(z_m), 1.25) if sigma is None else np.clip(
        2 * sigma[valid] * np.sqrt(intrinsic[0, 0] * intrinsic[1, 1]) / z_m, .75, 2.)
    x0, y0 = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    depth = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x, y = x0 + dx, y0 + dy
            inside = ((x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
                      & ((x - u) ** 2 + (y - v) ** 2 <= radius ** 2))
            np.minimum.at(depth, (y[inside], x[inside]), z_m[inside])
    return np.where(np.isfinite(depth), depth, 0).astype(np.float32)


def prepare(options):
    k, cameras, stamps = load_camera_geometry(options.cameras, options.binding)
    view_paths = [options.views / f'{raw:03d}.npz' for raw in RAWS]
    road_paths = [options.road_input / 'views' / f'{raw:03d}.npz' for raw in RAWS]
    road_reference = options.road_input / 'road_reference.npz'
    with np.load(options.geometry, allow_pickle=False) as source:
        if not np.array_equal(source['raw_ids'], np.arange(199)):
            raise ValueError('a6 geometry must contain the raw-indexed 199-frame trajectory')
        poses = source['model_from_canonical'][list(RAWS)]
        scales = np.linalg.svd(source['model_from_world'][:3, :3], compute_uv=False)
    scale = float(scales.mean())
    if not np.allclose(scales, scale, rtol=1e-8) or scale <= 0:
        raise ValueError('Model-to-metre conversion requires a similarity transform')
    with np.load(options.pvd, allow_pickle=False) as surface:
        generated = surface['prior_origin'] & ~surface['observed_anchor']
        pvd_xyz = surface['canonical_xyz'][generated]
        pvd_sigma = surface['sigma_tangent_m'][generated]
    if not len(pvd_xyz) or not np.isfinite(pvd_xyz).all() or not (pvd_sigma > 0).all():
        raise ValueError('Missing or invalid fixed PVD generated surface rows')
    with np.load(road_reference, allow_pickle=False) as source:
        road_strong = source['original_strong'] & source['measured']

    measured_points = []
    for path in view_paths:
        with np.load(path, allow_pickle=False) as view:
            measured_points.append(view['point_canonical_xyz'][view['point_rgb_valid']])

    depths, rgbs, labels, visible_road, counts = [], [], [], [], []
    for i, (raw, view_path, road_path) in enumerate(zip(RAWS, view_paths, road_paths)):
        with np.load(view_path, allow_pickle=False) as view, np.load(road_path, allow_pickle=False) as road:
            if (int(view['raw_frame']) != raw or abs(float(view['model_time']) - stamps[i]) > 1e-8
                    or not np.allclose(view['K'], k[i], atol=1e-6, rtol=0)
                    or not np.allclose(view['model_to_camera'], cameras[i], atol=1e-6, rtol=0)
                    or not np.allclose(view['model_from_canonical'], poses[i], atol=1e-6, rtol=0)):
                raise ValueError(f'Frozen a6 and native camera/time/pose disagree at raw{raw}')
            identity = view['actor_labels']
            visible = view['visible_evidence_mask'] & (identity == 4)
            a37 = view['a37_visible_mask']
            a6_box = view['a6_box_candidate_mask']
            unknown = (view['a37_occlusion_unknown_mask'] & ~view['raw_reviewed_silhouette_mask']
                       & ((identity == 0) | (identity == 5)))
            measured = view['measured_a6_depth_m']
            measured = np.where(visible & np.isfinite(measured) & (measured > 0), measured, 0)
            other_views = np.concatenate([points for j, points in enumerate(measured_points) if j != i])
            canonical_to_camera = cameras[i] @ poses[i]
            transferred = center_depth_splats(other_views, canonical_to_camera, k[i], scale)
            transferred = np.where(unknown & (measured == 0), transferred, 0)
            prior = center_depth_splats(pvd_xyz, canonical_to_camera, k[i], scale, pvd_sigma)
            prior = np.where(unknown & (measured == 0) & (transferred == 0), prior, 0)

            unit = road['unit_id']
            road_depth = road['depth_m']
            if unit.min() < -1 or unit.max() >= len(road_strong):
                raise ValueError(f'Invalid original road unit index at raw{raw}')
            ground = ((unit >= 0) & road['valid'] & road_strong[np.maximum(unit, 0)]
                      & np.isfinite(road_depth) & (road_depth > 0))
            road_guard = (ground & (identity == 0) & ~a6_box
                          & ~road['other_vehicle_guard'] & ~road['shadow_proxy'])
            road_under = np.where(ground & a37 & ~a6_box, road_depth, 0)
            frame = np.asarray([measured, transferred, prior, road_under], dtype=np.float32)
            if (frame > 0).sum(axis=0).max() > 1 or not np.isfinite(frame).all():
                raise ValueError(f'Invalid or overlapping reference depth layers raw{raw}')
            depths.append(frame)
            rgbs.append(np.rint(np.clip(view['image_srgb'], 0, 1) * 255).astype(np.uint8))
            labels.append(identity)
            visible_road.append(road_guard)
            counts.append(dict(raw=raw, **dict(zip(LAYERS, map(int, (frame > 0).sum(axis=(1, 2))))),
                               observed_road_guard=int(road_guard.sum())))

    receipt = dict(
        schema=SCHEMA, raw_ids=list(RAWS), sources={
            name: dict(path=str(path), sha256=sha(path)) for name, path in (
                ('geometry', options.geometry), ('pvd', options.pvd), ('cameras', options.cameras),
                ('binding', options.binding), ('road_reference', road_reference))},
        a6_views_sha256=[sha(path) for path in view_paths], road_views_sha256=[sha(path) for path in road_paths],
        meter_to_model=scale, rows=counts, pvd_generated_rows=len(pvd_xyz),
        provenance={
            LAYERS[0]: 'Same-frame actual a6 LiDAR support in strict reviewed visible silhouette.',
            LAYERS[1]: 'Other BUILD frames measured visible a6 points reprojected by the original trajectory into the possible a37 occlusion domain. Unvalidated at this ray.',
            LAYERS[2]: 'Existing fixed generated PVD patch centers in the possible a37 occlusion domain, excluding measured/reprojected support. Hypothesis only.',
            LAYERS[3]: 'Fixed original-strong measured road geometry under the observed a37 silhouette and outside the a6 box. Ground-depth proxy only.',
            'observed_road_guard': 'Original-strong measured road pixels outside actors, a6 box, other-vehicle guard and the shadow proxy. RGB comes from the same original a6 view.',
        }, hidden_geometry_GT=False, edited_RGB_GT=False, training_updates=0,
        quality_accepted=False, human_verdict=None)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open('xb') as stream:
        np.savez_compressed(stream, schema=SCHEMA, raw_ids=RAWS, layer_names=LAYERS, K=k,
                            model_to_camera=cameras, model_time=stamps, meter_to_model=scale,
                            model_from_canonical=poses, depth_m=np.asarray(depths),
                            observed_rgb=np.asarray(rgbs), actor_labels=np.asarray(labels),
                            observed_road_guard=np.asarray(visible_road))
    receipt.update(reference=str(options.output), reference_sha256=sha(options.output))
    write_json(options.output.with_suffix('.json'), receipt)
    print(json.dumps(dict(status='PASS_FIXED_REFERENCE_PREPARED', reference_sha256=receipt['reference_sha256'],
                          raw113=counts[113 - RAWS[0]], total_pixels={key: sum(r[key] for r in counts)
                          for key in (*LAYERS, 'observed_road_guard')})), flush=True)


def marker_positions(view, projection, pixels, model_z):
    """Invert the actual CUDA projection at the rasterizer's pixel-center NDC."""
    xy = np.asarray(pixels, dtype=np.float64)
    z = np.asarray(model_z, dtype=np.float64)
    view, projection = np.asarray(view, dtype=np.float64), np.asarray(projection, dtype=np.float64)
    ndc = (2 * xy + 1) / np.array([WIDTH, HEIGHT]) - 1
    w = z * projection[2, 3] + projection[3, 3]
    if np.abs(projection[:2, 3]).max() > 1e-8:
        raise ValueError('Unexpected native projection for pixel marker inversion')
    rhs = ndc * (w[:, None] + 1e-7) - z[:, None] * projection[2, :2] - projection[3, :2]
    xy_camera = rhs @ np.linalg.inv(projection[:2, :2])
    homogeneous = np.column_stack((xy_camera, z, np.ones_like(z)))
    return np.ascontiguousarray((homogeneous @ np.linalg.inv(view))[:, :3], dtype=np.float32)


def marker_kwargs(template, xyz, model_z, torch):
    marker = {key: None if value is None else value.new_zeros((len(xyz), *value.shape[1:]))
              for key, value in template.items()}
    marker['means3D'] = xyz
    marker['features'][:, 0] = model_z
    marker['features'][:, 1:] = 1
    marker['opacities'].fill_(MARKER_ALPHA)
    marker['mask'].fill_(True)
    sigma = 1e-5
    if marker['cov3D_precomp'] is not None:
        marker['cov3D_precomp'][:, [0, 3, 5]] = sigma ** 2
    else:
        marker['scales'].fill_(sigma)
        marker['rotations'][:, 0] = 1
    return marker


def front_opacity(camera, args, static, raster, prepare_rasterization, depths, scale, margin, verify, torch):
    """Sample native ray transmittance with independently calibrated one-pixel markers."""
    native = prepare_rasterization(camera, static, args)
    xyz = native['means3D']
    native['means2D'] = torch.zeros_like(xyz)
    if native['features'].shape != (len(xyz), 2):
        raise ValueError('Expected native static depth/alpha channels')
    native['features'] = torch.cat((native['features'], torch.zeros_like(xyz[:, :1])), dim=1).contiguous()
    background = xyz.new_zeros(3)

    def draw(values):
        return raster(camera, args, static.active_sh_degree, values, background)

    y, x = np.nonzero(depths > 0)
    cutoff = (depths[y, x] - margin) * scale
    if not len(x) or (cutoff <= .2).any():
        raise ValueError('Reference ray cutoff is outside the native positive depth range')
    positions = marker_positions(camera.world_view_transform.cpu().numpy(), camera.projection_matrix.cpu().numpy(),
                                 np.column_stack((x, y)), cutoff)
    y, x = (torch.as_tensor(v, device=xyz.device) for v in (y, x))
    z = xyz.new_tensor(cutoff)
    marker = marker_kwargs(native, xyz.new_tensor(positions), z, torch)
    alone = draw(marker)['extra'][0]
    normalization = alone[y, x]
    outside = torch.ones((HEIGHT, WIDTH), device=xyz.device, dtype=torch.bool)
    outside[y, x] = False
    peak_error = float((normalization - MARKER_ALPHA).abs().max())
    leak = float(alone[outside].abs().max())
    if peak_error > 2e-6 or leak > 1e-7 or not bool(torch.isfinite(alone).all()):
        raise RuntimeError(f'Native marker calibration failed: center {peak_error}, off-ray {leak}')
    combined = {key: None if native[key] is None else torch.cat((native[key], marker[key]), dim=0).contiguous()
                for key in native}
    visibility = draw(combined)['extra'][0, y, x] / normalization
    del combined
    if not bool(torch.isfinite(visibility).all()) or float(visibility.min()) < -1e-6 or float(visibility.max()) > 1.00001:
        raise RuntimeError('Native marker transmittance outside [0,1]')
    verify_count, verify_error, verify_saturated = 0, 0., 0
    if verify:
        # Independent native check on raw113 at known z values. A zero marker
        # can also mean raster early termination on an already opaque prefix.
        row_depth = native['features'][:, 0]
        ordered = torch.argsort(z)
        sample = ordered[torch.linspace(0, len(z) - 1, 13, device=z.device).long().unique()]
        for i in sample.tolist():
            values = dict(native, mask=(native['mask'] & (row_depth < z[i])).contiguous())
            native_alpha = float(draw(values)['alpha'][0, y[i], x[i]])
            t = float(visibility[i])
            error = abs(native_alpha - (1 - t))
            if t <= 1e-6 and native_alpha >= .989:
                verify_saturated += 1
            elif error > 5e-4:
                raise RuntimeError(f'Raw113 prefix/marker mismatch at ray {i}: {native_alpha}, {t}')
            verify_count += 1
            verify_error = max(verify_error, error)
    opacity = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    opacity[y.cpu().numpy(), x.cpu().numpy()] = (1 - visibility.clamp(0, 1)).cpu().numpy()
    return opacity, dict(marker_center_max_abs=peak_error, marker_off_ray_max_abs=leak,
                         native_prefix_check_rays=verify_count, native_prefix_check_max_abs=verify_error,
                         native_prefix_saturation_exceptions=verify_saturated)


def image_metrics(render_dir, index, raw, ref, image):
    frame = render_dir / f'{raw:03d}'
    truth = ref['observed_rgb'][index].astype(np.float32) / 255
    masks = dict(a6_visible=ref['actor_labels'][index] == 4,
                 a6_measured_visible=ref['depth_m'][index, 0] > 0,
                 observed_road_guard=ref['observed_road_guard'][index])
    arrays, source_hashes = [], {}
    for name in ('composite', 'delete_a37'):
        path = frame / f'{name}.png'
        with image.open(path) as src:
            array = np.asarray(src.convert('RGB'), dtype=np.float32) / 255
        if array.shape != truth.shape:
            raise ValueError(f'Unaligned render shape at {path}')
        arrays.append(array)
        source_hashes[name] = sha(path)
    composite, deleted = arrays
    maps = dict(composite_vs_observed=np.abs(composite - truth).mean(axis=2),
                delete_a37_vs_observed=np.abs(deleted - truth).mean(axis=2),
                edit_delta=np.abs(deleted - composite).mean(axis=2))
    return {name: {metric: stat(values[mask]) for metric, values in maps.items()}
            for name, mask in masks.items()}, source_hashes


def stat(values):
    return dict(pixels=int(values.size), sum=float(values.sum(dtype=np.float64)),
                mean=float(values.mean(dtype=np.float64)) if values.size else None)


def evaluate(options):
    # Import the same owner5 engine and raster helper used in prior readouts.
    import evaluate_diag as native
    from attribute_diag import load_raster
    from gaussian_renderer import prepare_rasterization

    torch = native.torch
    torch.set_num_threads(1)
    started = time.monotonic()
    options.output.mkdir(parents=True, exist_ok=False)
    with np.load(options.reference, allow_pickle=False) as source:
        ref = {key: source[key] for key in source.files}
    if (ref['schema'].item() != SCHEMA or tuple(ref['raw_ids']) != RAWS
            or tuple(ref['layer_names']) != LAYERS):
        raise ValueError('Unexpected fixed reference package')
    scale = float(ref['meter_to_model'])
    for key, actual in zip(('K', 'model_to_camera', 'model_time'), load_camera_geometry(
            options.model / 'cameras.json', options.model / 'actual_scene_binding.json')):
        if not np.allclose(actual, ref[key], rtol=0, atol=1e-6):
            raise ValueError(f'Checkpoint does not use the shared camera contract: {key}')
    config = read_json(options.model / 'effective_config.json')
    args = native.OmegaConf.create(config)
    static = native.GaussianModel(args)
    checkpoint = options.model / f'chkpnt{options.iteration}.pth'
    payload, step = torch.load(str(checkpoint), map_location='cpu', weights_only=False, mmap=True)
    if step != options.iteration:
        raise ValueError('Static checkpoint iteration mismatch')
    payload = list(payload)
    for i in range(1, 11):
        payload[i] = payload[i].detach().cuda()
    static.time_duration = tuple(config['time_duration'])
    static.restore(tuple(payload), training_args=None)
    del payload
    raster, raster_source = load_raster()

    rows, pooled = [], {}
    raws = (113,) if options.frames == '113' else RAWS
    with torch.no_grad():
        for raw in raws:
            i = raw - RAWS[0]
            w = ref['model_to_camera'][i]
            k = ref['K'][i]
            camera = native.Camera(raw, w[:3, :3].T.copy(), w[:3, 3].copy(), cx=k[0, 2], cy=k[1, 2],
                                   fx=k[0, 0], fy=k[1, 1], image=torch.zeros((3, HEIGHT, WIDTH)),
                                   image_name=f'0008{raw:03d}', resolution=(WIDTH, HEIGHT),
                                   timestamp=float(ref['model_time'][i]))
            depth_layers = ref['depth_m'][i]
            opacity, calibration = front_opacity(camera, args, static, raster, prepare_rasterization,
                                                 depth_layers.max(axis=0), scale, options.margin_m,
                                                 raw == 113, torch)
            row = dict(raw=raw, calibration=calibration, static_front={})
            domain = depth_layers > 0
            for name, mask in zip(LAYERS, domain):
                values = opacity[mask]
                row['static_front'][name] = dict(stat(values),
                    opacity_over_half_pixels=int((values > .5).sum()),
                    opacity_over_95pct_pixels=int((values > .95).sum()))
            if options.render_dir is not None:
                row['observed_RGB'], row['render_source_sha256'] = image_metrics(
                    options.render_dir, i, raw, ref, native.Image)
            folder = options.output / f'{raw:03d}'
            folder.mkdir()
            np.savez_compressed(folder / 'front_opacity.npz', opacity=opacity, reference_depth_m=depth_layers)
            rgb = np.stack((opacity * (domain[1] | domain[2]), opacity * domain[0],
                            opacity * domain[3]), axis=-1)
            native.Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(folder / 'front_opacity.png')
            rows.append(row)
            print(json.dumps(dict(raw=raw, static_front=row['static_front'])), flush=True)

    def pool(key, records):
        pixels = sum(item['pixels'] for item in records)
        total = sum(item['sum'] for item in records)
        pooled[key] = dict(pixels=pixels, mean=total / pixels if pixels else None)

    for name in LAYERS:
        pool('static_front/' + name, [r['static_front'][name] for r in rows])
    if options.render_dir is not None:
        for name, metrics in rows[0]['observed_RGB'].items():
            for metric in metrics:
                pool(f'observed_RGB/{name}/{metric}', [r['observed_RGB'][name][metric] for r in rows])
    summary = dict(status='PASS_SHARED_FIXED_A6_PROBE', schema=SCHEMA, iteration=options.iteration,
                   reference=str(options.reference), reference_sha256=sha(options.reference),
                   checkpoint=str(checkpoint), checkpoint_size=checkpoint.stat().st_size,
                   raster_source=str(raster_source), raster_source_sha256=sha(raster_source),
                   model_rows=len(static.get_xyz), frames=list(raws), rows=rows, pooled=pooled,
                   metric='1 - native marker transmittance through static Gaussian centers before reference depth minus margin',
                   margin_m=options.margin_m, marker_alpha=MARKER_ALPHA,
                   native_saturation='Unreached marker after native ray termination is reported as fully occluded.',
                   render_dir=str(options.render_dir) if options.render_dir is not None else None,
                   rgb_contract='Input composite and true a37-only deletion retain a6. RGB is a uint8 observed-fidelity guard only.',
                   evaluation_domain='BUILD original 49 training views and fixed reference rays',
                   hidden_geometry_GT=False, edited_RGB_GT=False, training_updates=0,
                   checkpoint_files_modified=False, quality_accepted=False, human_verdict=None,
                   seconds=time.monotonic() - started)
    write_json(options.output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    setup = commands.add_parser('prepare')
    for name in ('geometry', 'views', 'pvd', 'cameras', 'binding', 'road-input', 'output'):
        setup.add_argument('--' + name, type=Path, required=True)
    run = commands.add_parser('evaluate')
    for name in ('model', 'reference', 'output'):
        run.add_argument('--' + name, type=Path, required=True)
    run.add_argument('--iteration', type=int, default=10000)
    run.add_argument('--frames', choices=('113', 'all'), default='113')
    run.add_argument('--margin-m', type=float, default=.20)
    run.add_argument('--render-dir', type=Path, help='Verified per-frame composite and a37-only-delete PNGs')
    options = parser.parse_args()
    if options.command == 'prepare':
        prepare(options)
    else:
        if not np.isfinite(options.margin_m) or options.margin_m < 0:
            parser.error('--margin-m must be finite and nonnegative')
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        evaluate(options)


if __name__ == '__main__':
    main()
