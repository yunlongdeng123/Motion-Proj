"""Default plan only. --execute-cpu reads remote assets with CUDA hidden, never renders on GPU."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(Path(__file__).with_name('experiment_plan.json')))
    parser.add_argument('--execute-cpu', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    if not args.execute_cpu:
        print(json.dumps({'status': 'PLAN_ONLY_WAITING_FOR_USER_GPU_HOST', 'GPU_started': False,
                          'remote_assets_checked': False, 'target': plan['primary_case']}))
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '2'
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from shadow_math import project_gaussians, replay, rotation_matrix, project_points, ray_plane
    torch.set_num_threads(2)
    root = Path(plan['remote_root'])
    case = plan['primary_case']
    sid, frame_id = case['scene_id'], case['frame_id']
    checkpoint = root / case['checkpoint_relative']
    checkpoint_identity = {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                           'mtime_ns': checkpoint.stat().st_mtime_ns}
    evaluation = root / case['evaluation_relative']
    manifest = json.loads((evaluation / 'scene_evaluation_manifest.json').read_text(encoding='utf-8'))
    row = next(f for f in manifest['streams']['observed']['frames'] if f['frame_id'] == frame_id)
    assert manifest['checkpoint']['step'] == 30000 and Path(manifest['checkpoint']['path']) == checkpoint
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert cp['step'] == 30000 and cp['schema'] == 'drivingeditor_rigid_scene_v1'
    assert cp['spec']['scene_id'] == sid
    source = Path(cp['spec']['source_repository'])
    def methods(path):
        wanted = {'get_xyz_SHM', 'get_inst_velocity', 'get_marginal_t', 'capture'}
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
                if isinstance(n, ast.FunctionDef) and n.name in wanted}
    actual_methods = methods(source/'scene/gaussian_model.py')
    expected_methods = methods(Path(__file__).with_name('audited_gaussian_model.py'))
    if actual_methods != expected_methods or len(actual_methods) != 4:
        raise ValueError('Remote Gaussian temporal/capture semantics differ from audited snapshot; review before replay')
    # renderer import is tensor math only; no rasterizer is imported by evaluate_sh_rgb.
    sys.path.insert(0, str(root/'infra'))
    from renderer import evaluate_sh_rgb
    readiness = json.loads((root/'metadata'/sid/'CPU_readiness.json').read_text(encoding='utf-8'))
    plane = np.asarray(readiness['SGCM_geometry']['plane_model'], np.float64)
    transform = np.asarray(manifest['normalization'], np.float64)
    scale = float(np.linalg.norm(transform[:3, :3], axis=0).mean())
    assert np.allclose(np.linalg.norm(transform[:3, :3], axis=0), scale)
    cap = cp['background']
    assert len(cap) == 18 and cp['background_exclusion']['removed_points'] == 0
    assert len(cap[1]) == case['expected_background_points']
    ref = json.loads(Path(cp['spec']['reference_config']).read_text(encoding='utf-8'))
    array = lambda t: t.detach().numpy().astype(np.float32, copy=False)

    def state(capture, time, camera):
        omega = 2*math.pi/float(capture[16])
        xyz = capture[1]+capture[9]*torch.sin((float(time)-capture[7])*omega)/omega
        sigma = capture[8].exp()
        marginal = torch.exp(-.5*((capture[7]-float(time))/sigma)**2).reshape(-1)
        directions = torch.nn.functional.normalize(xyz-torch.tensor(camera['camera_center']), dim=-1, eps=1e-8)
        colors = evaluate_sh_rgb(torch.cat((capture[2], capture[3]), 1), directions, int(capture[0]))
        return {'xyz': array(xyz), 'scales': array(capture[4].exp()), 'rotation': array(capture[5]),
                'opacity': array(capture[6].sigmoid()).ravel()*array(marginal),
                'colors': array(colors), 'mask': array(marginal) > .05}

    x0, y0, x1, y1 = case['ROI_xyxy_exclusive']
    xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    pixels = np.stack((xx.ravel(), yy.ravel()), 1).astype(np.float32)
    current = state(cap, row['model_time'], row['camera'])
    projected = project_gaussians(current, row['camera'], ref['neg_fov'])
    rgb, alpha, mass, _ = replay(projected, pixels)
    with np.load(evaluation/row['npz'], allow_pickle=False) as saved:
        expected = saved['static_raw_rgb'][:, yy.ravel(), xx.ravel()].T.copy()
        observed_gt = saved['gt'].copy()
        observed_masks = saved['observed_masks'].copy()
    residual = np.abs(rgb-expected)
    checks = {'static_raw_RGB_MAE': float(residual.mean()), 'static_raw_RGB_max_abs': float(residual.max()),
              'MAE_limit': .0001, 'max_abs_limit': .002, 'CPU_equation_replay_is_approximate': True}
    passed = checks['static_raw_RGB_MAE'] <= checks['MAE_limit'] and checks['static_raw_RGB_max_abs'] <= checks['max_abs_limit']
    out = Path(args.output or root/'black_shadow_audit_20261016_v1'/'cpu_attribution')
    out.mkdir(parents=True, exist_ok=False)
    (out/'replay_validation.json').write_text(json.dumps(dict(checks, passed=passed), indent=2), encoding='utf-8')
    if not passed:
        raise AssertionError('CPU replay failed against actual saved FP32; do not interpret IDs or relax tolerance silently')
    ranked = np.argsort(mass)[::-1]
    ranked = ranked[mass[ranked] > 0][:case['top_contributor_count']]
    selected = projected['ids'][ranked]
    original, original_step = torch.load(cp['spec']['background_checkpoint'], map_location='cpu', weights_only=False)
    assert original_step == 30000 and len(original[1]) == len(cap[1])
    times = np.asarray([f['model_time'] for f in manifest['streams']['observed']['frames']], np.float64)
    omega = 2*math.pi/float(cap[16])
    n = plane[:3]/np.linalg.norm(plane[:3])
    entries = []
    for rank, (pid, ix) in enumerate(zip(selected, ranked)):
        pid, ix = int(pid), int(ix)
        base, velocity = array(cap[1][pid]), array(cap[9][pid])
        center, sigma = float(cap[7][pid].item()), float(cap[8][pid].exp().item())
        xyz_t = base+np.sin((times-center)*omega)[:, None]*velocity/omega
        marginal = np.exp(-.5*((times-center)/sigma)**2)
        uv_t, z_t = project_points(xyz_t, row['camera'])
        xyz = projected['xyz'][ix]
        signed_height = float((xyz @ plane[:3]+plane[3])/np.linalg.norm(plane[:3])/scale)
        rotation = rotation_matrix(projected['rotation'][ix:ix+1])[0]
        sigma_n = float(np.linalg.norm((rotation.T @ n)*projected['scales'][ix])/scale)
        effective_v = velocity*np.exp(-sigma/float(cap[16])/2*float(cap[17]))
        entries.append({'rank': rank+1, 'static_id': pid, 'ROI_mean_weight': float(mass[ix]/len(pixels)),
            'center_camera_depth_m': float(projected['depth'][ix]/scale),
            'center_signed_height_m': signed_height, 'normal_3sigma_halfextent_m': 3*sigma_n,
            'scale_axes_m': (projected['scales'][ix]/scale).tolist(),
            'center_xyz_model': xyz.tolist(), 'SH_RGB_at_target': projected['colors'][ix].tolist(),
            'raw_velocity_model_per_model_time': velocity.tolist(),
            'regularizer_velocity_model_per_model_time': effective_v.tolist(),
            'temporal_sigma_model_time': sigma, 'temporal_center': center,
            'effective_opacity': projected['opacity'][ix].item(),
            'trajectory_model': xyz_t.tolist(), 'marginal49': marginal.tolist(),
            'fixed_camera_uv49': uv_t.tolist(), 'fixed_camera_depth_m49': (z_t/scale).tolist(),
            'position_span_m': float(np.linalg.norm(xyz_t.max(0)-xyz_t.min(0))/scale),
            'original_GN11_xyz_model': array(original[1][pid]).tolist(),
            'original_GN11_scale_axes_m': (array(original[4][pid].exp())/scale).tolist(),
            'original_GN11_raw_velocity': array(original[9][pid]).tolist(),
            'original_GN11_opacity': float(original[6][pid].sigmoid().item())})
    probes = replay(projected, np.asarray(case['probe_pixels_xy'], np.float32))[3]
    # Ground intersection is a geometric hypothesis, never a ground-truth depth/mask.
    road_points = [ray_plane(p, row['camera'], plane) for p in case['probe_pixels_xy']]
    if any(p is None for p in road_points):
        raise ValueError('ROI road ray has no positive ground intersection; inspect geometry before continuing')
    world_road = np.c_[road_points, np.ones(len(road_points))] @ np.linalg.inv(transform).T
    observation_rows = []
    sheets = [Image.new('RGB', (700, 840), '#102333') for _ in road_points]
    for frame_index, frame in enumerate(manifest['streams']['observed']['frames']):
        uv, depth = project_points(road_points, frame['camera'])
        with np.load(evaluation/frame['npz'], allow_pickle=False) as saved:
            gt = saved['gt']; masks = saved['observed_masks']
            for k, (xy, z) in enumerate(zip(uv, depth)):
                x, y = np.floor(xy+.5).astype(int)
                inside = z > .2 and 0 <= x < 480 and 0 <= y < 320
                observation_rows.append({'probe': k, 'frame_id': frame['frame_id'], 'xy': xy.tolist(),
                    'depth_m': float(z/scale), 'in_view': bool(inside),
                    'RGB': gt[:, y, x].tolist() if inside else None,
                    'selected_body_occludes': bool(masks[:, y, x].any()) if inside else None,
                    'other_vehicle_occlusion': 'not_verified', 'shadow_state': 'requires_actual_RGB_review',
                    'clean_background_observation_confirmed': False})
                tile = Image.new('RGB', (100, 120), '#102333')
                label = ImageDraw.Draw(tile)
                label.text((2, 2), frame['frame_id'], fill='white')
                if inside:
                    actual = Image.fromarray(np.rint(np.moveaxis(gt.clip(0, 1), 0, -1)*255).astype(np.uint8))
                    tile.paste(actual.crop((x-32,y-32,x+32,y+32)).resize((96,96)), (2,20))
                    label.rectangle((48,66,52,70), outline='#ff6c65')
                    label.text((2,106), 'body%d shadow?'%bool(masks[:,y,x].any()), fill='#ffce91')
                else:
                    label.text((2,45), 'OUT OF VIEW', fill='#ffce91')
                sheets[k].paste(tile, ((frame_index%7)*100, (frame_index//7)*120))
    for k, sheet in enumerate(sheets):
        sheet.save(out/('real49_road_probe%d_UNREVIEWED.png'%k))
    # Full 49 views are recorded. No "clear road" verdict from mask absence alone.
    output = {'schema': 'black_shadow_CPU_attribution_v1', 'scene': sid, 'frame_id': frame_id,
        'GPU_used': False, 'training_updates': 0, 'model_parameters_modified': False,
        'checkpoint_identity': checkpoint_identity,
        'diagnostic_ROI_is_not_shadow_GT': True, 'physical_scale': scale,
        'ground_fit_reference': readiness['SGCM_geometry']['ground_fit'], 'ground_plane_model': plane.tolist(),
        'CPU_FP32_replay': checks, 'selected_static_ids': selected.tolist(), 'top_contributors': entries,
        'pixel_traces': probes, 'road_points_world_m': world_road[:, :3].tolist(),
        'ground_ray_hypothesis_only': True, 'real_observations': observation_rows,
        'classification': 'PENDING_GPU_ALPHA_DEPTH_AND_CROSS_VIEW_RGB_REVIEW',
        'source_model_sha256': hashlib.sha256((source/'scene/gaussian_model.py').read_bytes()).hexdigest()}
    (out/'attribution.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(out/'replay_ROI.npz', pixels=pixels, static_raw_RGB=rgb, static_alpha=alpha,
                        selected_static_ids=selected)
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status': 'CPU_ATTRIBUTION_ONLY_GPU_VALIDATION_PENDING', 'GPU_used': False,
                      'contributors': len(entries), 'output': str(out)}))


if __name__ == '__main__':
    main()
