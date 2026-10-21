"""Post-training, zero-update evaluation of the four V3 assets.

No training entry point, V3Context, road installation, optimizer, or shutdown is
called. CPU preflight is deliberately not a GPU evaluation receipt.
"""

import argparse
from contextlib import contextmanager
import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ARMS = ('full', 'real_only', 'prior_only', 'ownership_only')
STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
FRAMES = tuple(f'0008{i:03d}' for i in range(89, 138))
CHANNELS = {'gt', 'composite', 'static', 'dynamic', 'road_raw_pre_environment',
            'road_surface_color', 'road_alpha'}
VERSION = 'V3_FINAL49_ZERO_UPDATE_EVALUATION_R2'
RETENTION_VERSION = 'V3_COMPLETED_CHECKPOINT_RETENTION_R1'
COMPONENT_PREFIXES = dict(static='chkpnt', dynamic='dynamic_chkpnt',
                          environment='env_light_chkpnt', training_state='training_state')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False, ensure_ascii=False)


def retention_removals(campaign):
    """Only the user's exact full/real_only old-step cleanup is an exception.

    Publication events and evaluation assets are never removed or rewritten.
    Missing checkpoints without a completed, event-bound receipt still fail.
    """
    campaign = Path(campaign).resolve()
    receipt = campaign/'checkpoint_retention.json'
    if not receipt.exists():
        return {}
    data = read(receipt)
    if (data.get('version') != RETENTION_VERSION or data.get('status') != 'complete'
            or data.get('campaign') != str(campaign)
            or data.get('authorization') != 'user_requested_full_real_only_keep_latest_30k'
            or data.get('keep_iteration') != 30000
            or data.get('protected_assets_unchanged') is not True
            or data.get('latest30k_sha256_unchanged') is not True):
        raise RuntimeError('invalid checkpoint retention receipt')
    expected = {(a, s, k) for a in ('full', 'real_only')
                for s in STEPS[:-1] for k in COMPONENT_PREFIXES}
    removed = {}
    for item in data.get('removed', []):
        key = (item['arm'], item['iteration'], item['component'])
        if key not in expected or key in removed:
            raise RuntimeError('retention scope expanded or duplicated')
        arm, step, name = key
        run = campaign/'arms'/arm/'run30k'
        path = run/f'{COMPONENT_PREFIXES[name]}{step}.pth'
        event_path = run/'checkpoint_events'/f'iteration_{step:06d}.json'
        if (item['path'] != str(path) or path.exists() or path.is_symlink()
                or item['event_sha256'] != hashlib.sha256(event_path.read_bytes()).hexdigest()
                or read(event_path)['checkpoints'][name] != dict(path=str(path), bytes=item['bytes'])
                or item['bytes'] <= 0):
            raise RuntimeError('retention does not match original publication')
        removed[key] = item
    if set(removed) != expected:
        raise RuntimeError('retention receipt is incomplete')
    return removed


def require_complete(campaign):
    """Reject gates, old failures, partial runs, and mismatched logical budgets."""
    campaign = Path(campaign).resolve()
    registry = read(campaign/'four_arm_registry.json')
    if (registry['status'] != 'FOUR_ARMS_TRAINING_COMPLETE_EVALUATION_PENDING'
            or registry['max_concurrent_training'] != 1
            or tuple(registry['completed_arms']) != ARMS):
        raise RuntimeError('all four fresh30K runs must finish before GPU evaluation')
    result = {}
    removed = retention_removals(campaign)
    for arm in ARMS:
        run = campaign/'arms'/arm/'run30k'
        state = read(run/'status.json')
        if (state['status'] != 'complete' or state['iteration'] != 30000
                or state['arm'] != arm or state['second_stage_updates'] != 0
                or state['optimizer_step_counts'] != dict.fromkeys(
                    ('static', 'dynamic', 'environment'), 30000)):
            raise RuntimeError('not exactly30K optimizer.step calls: '+arm)
        for step in STEPS:
            event = read(run/'checkpoint_events'/f'iteration_{step:06d}.json')
            if (event['status'] != 'complete' or event['iteration'] != step
                    or set(event['checkpoints']) != {
                        'static', 'dynamic', 'environment', 'training_state'}):
                raise RuntimeError('incomplete four-component publication')
            for name, value in event['checkpoints'].items():
                path = Path(value['path']).resolve()
                if not path.is_relative_to(run.resolve()) or value['bytes'] <= 0:
                    raise RuntimeError('missing or escaped checkpoint')
                if (arm, step, name) in removed:
                    continue
                if not path.is_file() or path.stat().st_size != value['bytes']:
                    raise RuntimeError('missing or escaped checkpoint')
            frames = event['frames']
            if (len(frames) != 49 or {f['image_name'] for f in frames} != set(FRAMES)
                    or len({f['frame_id'] for f in frames}) != 49):
                raise RuntimeError('formal49 identity mismatch')
            for frame in frames:
                if (frame['width'], frame['height']) != (480, 320):
                    raise RuntimeError('wrong evaluation resolution')
                paths = list(frame['channels'].values()) + [
                    frame['road_float'], frame['hybrid_metrics']]
                if set(frame['channels']) != CHANNELS or not all(Path(p).is_file() for p in paths):
                    raise RuntimeError('formal49 publication assets missing')
        result[arm] = event
    return result


@contextmanager
def preserve_transform_file(path):
    """Native loader recomputes PCA but must not rewrite the shared input file."""
    path = Path(path).resolve()
    expected = np.loadtxt(path)
    original = np.savetxt

    def check_only(destination, value, *args, **kwargs):
        if (Path(destination).resolve() != path
                or not np.allclose(value, expected, atol=1e-8, rtol=1e-8)):
            raise RuntimeError('loader attempted an unexpected write/transform change')

    np.savetxt = check_only
    try:
        yield
    finally:
        np.savetxt = original


def file_state(paths):
    return {str(Path(p)): (Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
            for p in paths}


def raster_uv(points, camera):
    import torch
    shape = points.shape
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    clip = homogeneous @ camera.full_proj_transform.to(points)
    uv = ((clip[..., :2]/clip[..., 3:].clamp_min(1e-8)+1)*points.new_tensor(
        [camera.image_width, camera.image_height])-1)/2
    valid = ((clip[..., 3] > 0) & torch.isfinite(uv).all(-1)
             & (uv[..., 0] >= 0) & (uv[..., 0] <= camera.image_width-1)
             & (uv[..., 1] >= 0) & (uv[..., 1] <= camera.image_height-1))
    assert uv.shape == (*shape[:-1], 2)
    return uv, valid


def assert_close(actual, expected, name):
    import torch
    expected = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
    if (actual.shape != expected.shape or not torch.isfinite(actual).all()
            or not torch.isfinite(expected).all()
            or not torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)):
        raise RuntimeError('saved30K/native deletion replay mismatch: '+name)
    return float((actual-expected).abs().max())


def frame_contract_diagnostics(package, target, mask, depth, policy, dssim):
    """Same frozen training definitions, evaluated without gradients or updates."""
    from utils.loss_utils import ssim
    from utils.v3_objectives import (body_joint_rgb_loss, static_front_occlusion_loss,
                                      dynamic_color_leak_losses)
    front, front_metrics = static_front_occlusion_loss(package['static_depth'],
        package['static_alpha'], depth, mask, policy['static_front_depth_margin_ratio'],
        policy['static_front_temperature_ratio'])
    leak, ratio, leak_metrics = dynamic_color_leak_losses(
        package['dynamic_premultiplied_rgb'], package['dynamic_alpha'], mask,
        policy['dynamic_boundary_ignore_px'], policy['dynamic_leak_top_fraction'],
        policy['dynamic_leak_top_weight'], policy['dynamic_radiance_ratio_cap'],
        policy['dynamic_ratio_slack'])
    result = dict(front_metrics, **leak_metrics)
    result.update(body_joint_rgb_objective=body_joint_rgb_loss(package['render'], target,
        mask, ssim, dssim, policy['body_l1_normalization']),
        body_masked_structural_loss=1-ssim(package['render']*mask, target*mask),
        front_soft_objective=front, dynamic_color_objective=leak,
        dynamic_color_ratio_objective=ratio)
    return {k: float(v) for k, v in result.items()}


def road_state_diagnostics(static, data, numpy, scale):
    from utils.v3_multiview_road import geometry_objective
    common, terms = geometry_objective(static, data, scale)
    rows = static.road_unit_mask.nonzero().flatten()
    ids = static._road_ids()[rows]
    color = (static._features_dc[rows, 0]*.28209479177387814+.5).clamp_min(0)
    unit_color = color.new_zeros((len(data['provenance']), 3)).index_copy(0, ids, color)
    edges = data['edges'].long()
    valid = data['prior_graph_edge_valid'].bool()
    error = (unit_color[edges[:, 0]]-unit_color[edges[:, 1]]).square().mean(1)
    boundary = valid & (data['provenance'][edges] == 1).any(1)
    prior = (data['provenance'] == 2) & data['is_query']
    return dict(geometry={k: float(v) for k, v in terms.items()},
        common_geometry_objective=float(common), continuity_edges=int(valid.sum()),
        prior_continuity=float((error*valid).sum()/valid.sum().clamp_min(1)),
        prior_boundary_DC_RGB_RMSE=float(error[boundary].mean().sqrt()) if boundary.any() else None,
        prior_query_units=int(prior.sum()), prior_query_without_context_path=int(
            (prior & (data['prior_graph_context_hops'] < 0)).sum()),
        components=int(numpy['prior_graph_component'].max()+1),
        no_UNKNOWN_bridges=True, no_hidden_road_RGB_ground_truth=True)


def evaluate(campaign, arm, output, event):
    started = time.monotonic()
    import torch
    from omegaconf import OmegaConf

    engine = campaign/'engine'
    sys.path.insert(0, str(engine))
    from scene import GaussianModel, Dynamic_GaussianModel, EnvLight
    from scene.waymo_loader2 import readWaymoInfo_add
    from utils.camera_utils import cameraList_from_camInfos
    from utils.general_utils import seed_everything
    from utils.hybrid30k_runtime import camera_id
    from utils.paper_masks import load_dynamic_mask
    from utils.v3_supervision_diagnostics import (supervision_metrics, center_in_boxes,
                                                  sampled_front_contributions)
    from utils.v3_road_diagnostics import road_visibility
    from utils.v3_multiview_road import sample_map
    from utils.v3_road_prior import rendered_coverage
    from utils.v3_road_contract import ROAD_POLICY, ROAD_VERSION
    from gaussian_renderer import render, render_road_only, render_contribution_features
    from PIL import Image

    run = campaign/'arms'/arm/'run30k'
    config = read(run/'effective_config.json')
    if (config['seed'] != 0 or config['iterations'] != 30000
            or config['v3_arm'] != arm or config['start_checkpoint'] is not None
            or config['resolution_scales'] != [4] or config['initialization']):
        raise RuntimeError('fixed scientific configuration mismatch')
    protected = list(event['checkpoint_paths'].values()) + [
        event['checkpoints']['training_state']['path'], run/'effective_config.json',
        run/'status.json', Path(config['source_path'])/'transform_matrix.txt']
    before = file_state(protected)
    args = OmegaConf.create(dict(config, model_path=str(output)))
    seed_everything(0)
    with preserve_transform_file(Path(config['source_path'])/'transform_matrix.txt'):
        info = readWaymoInfo_add(args)
    cameras = cameraList_from_camInfos(info.train_cameras, 4, args)
    cameras = sorted(cameras, key=lambda c: float(c.timestamp))
    spec = config['followup_scene_spec']
    if ([camera_id(c) for c in cameras] != spec['frame_ids_with_camera']
            or any(abs(float(c.timestamp)-t) > 1e-7
                   for c, t in zip(cameras, spec['model_timestamps']))
            or len(cameras) != 49 or info.test_cameras):
        raise RuntimeError('native final49 camera binding mismatch')
    models = [GaussianModel(args), Dynamic_GaussianModel(args),
              EnvLight(resolution=args.env_map_res).cuda()]
    for name, model in zip(('static', 'dynamic', 'environment'), models):
        payload, iteration = torch.load(event['checkpoints'][name]['path'],
                                       map_location='cpu', weights_only=False)
        if iteration != 30000:
            raise RuntimeError('wrong checkpoint iteration')
        payload = list(payload)
        if name == 'environment':
            payload[0] = torch.nn.Parameter(payload[0].detach().cuda(), requires_grad=False)
        else:
            for i in range(1, 11):
                payload[i] = payload[i].detach().cuda()
            model.time_duration = info.time_duration
        model.restore(tuple(payload), training_args=None)
        if getattr(model, 'optimizer', None) is not None:
            raise RuntimeError('evaluation unexpectedly created an optimizer')
        del payload
        gc.collect()
    static, dynamic, environment = models
    bundle_path = Path(config['v3_bundle'])
    meta = read(bundle_path)
    if (meta['version'] != ROAD_VERSION or tuple(meta['frame_order']) != FRAMES
            or set(meta['actors']) != {6, 37, 5, 26, 18}):
        raise RuntimeError('final49 must retain main49/five-ID road evidence')
    with np.load(bundle_path.parent/meta['units'], allow_pickle=False) as z:
        numpy = {k: z[k].copy() for k in z.files}
    data = {k: torch.as_tensor(v, device='cuda') for k, v in numpy.items()
            if k in ('model_xyz', 'surface_supported', 'provenance', 'is_query',
                     'source_rgb', 'source_depth', 'base_weight', 'coverage_model_xyz',
                     'normal_model', 'edges', 'prior_graph_edge_valid', 'prior_graph_context_hops')}
    for key in ('model_xyz', 'normal_model'):
        data[key] = data[key].to(static._xyz)
    expected_ids = torch.nonzero(data['surface_supported']).flatten()
    ids = static._road_ids()[static.road_unit_mask]
    if len(ids) != 14034 or not torch.equal(ids.sort().values, expected_ids.sort().values):
        raise RuntimeError('fixed14034 world-road identity mismatch')
    np.savez_compressed(output/'world_units.npz', **{k: numpy[k] for k in (
        'unit_ids', 'world', 'model_xyz', 'provenance', 'is_query', 'surface_supported',
        'query_frame', 'query_actor', 'query_unit_id', 'query_pixel', 'source_count')})
    road_state = road_state_diagnostics(static, data, numpy, float(meta['meter_to_model']))
    saved_frames = {f['image_name']: f for f in event['frames']}
    background = torch.zeros(3, device='cuda')
    deleted_args = OmegaConf.create(dict(config, including_dynamic=False, adding=False))
    rows = []
    for index, camera in enumerate(cameras):
        frame = Path(camera_id(camera)).stem
        record = saved_frames[frame]
        if (camera.image_width, camera.image_height) != (480, 320):
            raise RuntimeError('native camera size differs from main49')
        actual = camera.world_view_transform.detach().cpu().numpy().T @ np.asarray(meta['world_to_model'])
        actual[:3] /= float(meta['meter_to_model'])
        if not np.allclose(actual, meta['frames'][frame]['world_to_camera'], atol=2e-4, rtol=2e-4):
            raise RuntimeError('world-unit/camera binding differs: '+frame)
        package = render(camera, static, dynamic, args, background, env_map=environment,
                         is_training=False)
        deleted = render(camera, static, None, deleted_args, background,
                         env_map=environment, is_training=False)
        replay = {'actual_target_branch_removal': assert_close(
            deleted['render'], package['render_s'], 'deleted branch '+frame)}
        old = read(record['hybrid_metrics'])
        with np.load(old['arrays'], allow_pickle=False) as z:
            replay['target_RGB'] = assert_close(camera.original_image.cuda(), z['gt'], 'GT '+frame)
            for key, old_key in (('render', 'composite'), ('render_s', 'static'),
                                 ('dynamic_premultiplied_rgb', 'dynamic'),
                                 ('dynamic_alpha', 'alpha_d'), ('static_alpha', 'alpha_s'),
                                 ('depth', 'depth_accum_model')):
                replay[key] = assert_close(package[key], z[old_key], key+' '+frame)
        road = render_road_only(camera, static, args)
        with np.load(record['road_float'], allow_pickle=False) as z:
            for key, old_key in (('rgb', 'rgb_sum'), ('alpha', 'alpha'), ('depth_sum', 'depth_sum')):
                replay['road_'+key] = assert_close(road[key], z[old_key], 'road '+key+' '+frame)
        uv, _ = raster_uv(data['model_xyz'].to(static._xyz), camera)
        contribution = {}
        colors = (.5 + .28209479177387814*static._features_dc[:, 0]).clamp_min(0)

        def contributions(flags, joint):
            features = torch.cat((flags, colors*flags), 1)
            value = render_contribution_features(package, features, joint)
            contribution['joint' if joint else 'background'] = value[:4]
            return value

        unit_arrays, visibility = road_visibility(package, road, static.road_unit_mask,
            torch.arange(len(data['provenance']), device='cuda'), uv,
            data['surface_supported'], data['provenance'], data['is_query'],
            torch.as_tensor(numpy['gate_in_view'][index], device='cuda'), contributions)
        verified = ((data['base_weight'][index] > 0) & (data['provenance'] == 1)
                    & data['is_query'] & unit_arrays['in_view'])
        unit_rgb = {'road': sample_map(road['rgb'], uv)/sample_map(road['alpha'], uv).clamp_min(1e-5),
                    'deleted': sample_map(deleted['render'], uv), 'joint': sample_map(package['render'], uv)}
        errors = {key: float((value-data['source_rgb'][index]).abs()[verified].mean())
                  if verified.any() else None for key, value in unit_rgb.items()}
        outside = ((data['base_weight'][index] > 0) & (data['provenance'] == 1)
                   & ~data['is_query'] & unit_arrays['in_view'])
        outside_errors = {key: float((value-data['source_rgb'][index]).abs()[outside].mean())
                          if outside.any() else None for key, value in unit_rgb.items()}
        road_depth = sample_map(road['depth_sum'], uv)[:, 0]/sample_map(
            road['alpha'], uv)[:, 0].clamp_min(1e-5)
        depth_error = (road_depth-data['source_depth'][index]).abs()/float(meta['meter_to_model'])
        coverage_uv, valid = raster_uv(data['coverage_model_xyz'].to(static._xyz), camera)
        for key in ('gate_in_view', 'gate_angle_ok', 'gate_resolution_ok'):
            valid &= torch.as_tensor(numpy[key][index], device='cuda')[:, None]
        valid &= (data['surface_supported'] & data['is_query'] & (data['provenance'] == 2))[:, None]
        _, subalpha = rendered_coverage(road['alpha'], coverage_uv, valid,
                                       valid.sum(), 1., ROAD_POLICY['coverage_target'])
        gt = camera.original_image.cuda()
        mask = load_dynamic_mask(config['source_path'], camera)
        supervision, maps = supervision_metrics(package, gt, mask, camera.pts_depth,
                                                float(meta['meter_to_model']))
        supervision.update(frame_contract_diagnostics(package, gt, mask, camera.pts_depth,
            config['v3_policy'], config['lambda_dssim']))
        boxes = meta['frames'][frame]['selected_boxes_model']
        tensor = lambda x: torch.as_tensor(x, device=static._xyz.device, dtype=static._xyz.dtype)
        inside = center_in_boxes(package['background_means3D'],
            tensor([b['center'] for b in boxes]).reshape(-1, 3),
            tensor([b['axes'] for b in boxes]).reshape(-1, 3, 3),
            tensor([b['half'] for b in boxes]).reshape(-1, 3))
        samples, sample_metrics = sampled_front_contributions(package, mask, camera.pts_depth,
            inside, static.road_unit_mask,
            lambda flags, joint: render_contribution_features(package, flags, joint),
            config['v3_policy']['static_front_depth_margin_ratio'])
        supervision['static_centers_inside_selected_boxes'] = int(inside.sum())
        supervision['static_total_centers'] = len(inside)
        raw = dict(joint=package['render'], deleted_five=deleted['render'], gt=gt,
            dynamic_raw=package['dynamic_premultiplied_rgb'], dynamic_alpha=package['dynamic_alpha'],
            static_alpha=package['static_alpha'], joint_alpha=package['alpha'],
            joint_depth_sum=package['depth'], static_depth_sum=package['static_depth'],
            dynamic_depth_sum=package['dynamic_depth'],
            road_surface_color=road['rgb']/road['alpha'].clamp_min(1e-5),
            road_raw=road['rgb'], road_alpha=road['alpha'], road_depth_sum=road['depth_sum'],
            road_in_background=contribution['background'], road_in_joint=contribution['joint'],
            **maps, **samples)
        folder = output/frame
        folder.mkdir()
        for name in ('joint', 'deleted_five', 'dynamic_raw', 'road_raw', 'road_surface_color'):
            pixels = (raw[name].permute(1, 2, 0).clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(pixels).save(folder/(name+'.png'))
        unit_arrays.update(verified_exposure=verified, prior_subcell_alpha=subalpha,
                           prior_subcell_valid=valid, source_RGB=data['source_rgb'][index],
                           source_base_weight=data['base_weight'][index],
                           verified_outside_query=outside, road_depth_model=road_depth,
                           source_depth_model=data['source_depth'][index], **unit_rgb)
        for filename, arrays in [('render_float.npz', raw), ('world_unit_float.npz', unit_arrays)]:
            values = {k: v.detach().cpu().numpy() for k, v in arrays.items()}
            if any(v.dtype.kind in 'fc' and not np.isfinite(v).all() for v in values.values()):
                raise FloatingPointError('non-finite final49 output: '+frame)
            np.savez_compressed(folder/filename, **values)
        mse = (package['render'].clamp(0, 1)-gt.clamp(0, 1)).square().mean()
        row = dict(frame=frame, timestamp=float(camera.timestamp), replay_max_errors=replay,
            reconstruction=dict(PSNR=float(-10*torch.log10(mse.clamp_min(1e-12))),
                L1=float((package['render'].clamp(0, 1)-gt.clamp(0, 1)).abs().mean()),
                scope='observed reconstruction, not edited GT'),
            supervision=supervision, sampled_front=sample_metrics, visibility=visibility,
            sampled_front_scope='at most14 measured body pixels; native center-z ordering, not full-frame front mass',
            verified_exposure=dict(units=int(verified.sum()), RGB_L1=errors,
                road_depth_MAE_m=float(depth_error[verified].mean()) if verified.any() else None),
            verified_outside_query=dict(units=int(outside.sum()), RGB_L1=outside_errors,
                scope='accepted observed world-context anchors, not unique pixel area or hidden GT'),
            prior_coverage=dict(subcells=int(valid.sum()),
                alpha_mean=float(subalpha[valid].mean()) if valid.any() else None,
                below_target_fraction=float((subalpha[valid]<ROAD_POLICY['coverage_target']).float().mean())
                    if valid.any() else None))
        write_new(folder/'metrics.json', row)
        rows.append(row)
        print(json.dumps(dict(arm=arm, evaluated_frames=len(rows), frame=frame)), flush=True)
        del package, deleted, road, raw, unit_arrays, contribution, colors, maps, samples, inside
        gc.collect()
    if file_state(protected) != before:
        raise RuntimeError('training assets or shared input modified during evaluation')
    summary = dict(version=VERSION, status='FORMAL49_RENDERED_REPLAYED_AND_FINITE', arm=arm,
        iteration=30000, frames=rows, optimizer_updates=0, optimizer_created=False,
        source_reconstruction_metrics=read(event['metrics']),
        zero_update_evaluation_wall_seconds=time.monotonic()-started,
        evaluation_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        evaluation_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        evaluation_peak_scope='this evaluation process; not training or device-wide peak',
        fixed_road_ids=14034, training_assets_unchanged=True, quality_accepted=False,
        terminal_road_state=road_state,
        human_verdict=None, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        interpretation='No edited-road GT. Verified exposure errors only use same-world accepted sources. '
        'Attenuation may be legal occlusion; source selection/gradient is not causal benefit. '
        'Deletion removes the shared five-target branch, retaining all ordinary background PVG motion.')
    write_new(output/'evaluation.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-final-evaluation-after-user-approval', action='store_true')
    arguments = parser.parse_args()
    campaign, output = arguments.campaign.resolve(), arguments.output.resolve()
    events = require_complete(campaign)
    if not output.is_relative_to(campaign/'final49') or output.exists():
        raise RuntimeError('fresh output under campaign/final49 required; never overwrite')
    if not arguments.execute_final_evaluation_after_user_approval:
        print(json.dumps(dict(status='CPU_PREFLIGHT_ONLY_NOT_RENDERED', optimizer_updates=0)))
        return
    import fcntl
    with (campaign.parent/'gpu_slot.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True)
        if apps.strip():
            raise RuntimeError('GPU compute process exists; evaluation must not overlap training')
        events = require_complete(campaign)
        output.mkdir(parents=True, exist_ok=False)
        try:
            import torch
            from unittest.mock import patch
            with patch.object(torch.optim.Adam, 'step', side_effect=RuntimeError('evaluation forbids Adam')):
                with torch.no_grad():
                    evaluate(campaign, arguments.arm, output, events[arguments.arm])
        except BaseException as error:
            write_new(output/'failure.json', dict(status='FAILED_PRESERVED_NO_RETRY', error=repr(error),
                                                optimizer_updates=0, quality_accepted=False))
            raise


if __name__ == '__main__':
    main()
