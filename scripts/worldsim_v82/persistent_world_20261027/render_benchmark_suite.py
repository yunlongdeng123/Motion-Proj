"""Read-only execution of concrete eight-family requests; not formal acceptance.

The original motion diagnostic suite and its world-SH baseline are unchanged.
Ego camera requests use calibrated world poses, and source assets are loaded
from a different frozen scene. Unknown camera domains are not given fake masks.
"""
import argparse, copy, gc, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation
from resource_control import require_start
from benchmark_geometry import ROOT, read, model_camera, sensor_camera, ActorTrack, asset_transfer
from render_motion_suite import NativeMotion, Camera, prepare_rasterization, actual_native_render, helpers, torch, cpu, save, quat_product, Dynamic_GaussianModel
from sh_rotation_control import rotate


class BenchmarkNative(NativeMotion):
    def camera(self, row, case, query=False):
        if 'camera_c2w_world' not in row:
            return super().camera(row, case, query)
        r = self.records[row['camera_index']]
        c2w = model_camera(self.T, row['camera_c2w_world'])
        w = np.linalg.inv(c2w)
        timestamp = row['dynamic_query_model_time'] if query or case.get('benchmark_family') == 'ego_speed' else row['camera_model_time']
        return Camera(row['camera_index'], w[:3, :3].T.copy(), w[:3, 3].copy(),
                      cx=r['cx']/4, cy=r['cy']/4, fx=r['fx']/4, fy=r['fy']/4,
                      image=torch.zeros(3, 320, 480), image_name=r['img_name'],
                      resolution=(480, 320), timestamp=timestamp)


class ForeignAsset:
    def __init__(self, context):
        self.context = context
        self.run = ROOT/'native_scene_training_v1'/context/'formal30k'
        returned = read(self.run/'returned.json')
        assert returned['reason'] == 'native_returned' and returned['iteration'] == 30000
        from omegaconf import OmegaConf
        self.args = OmegaConf.create(read(self.run/'effective_config.json'))
        assert len(read(Path(self.args.source_path)/'dynamic_multi_actor_preparation.json')['actors']) == 1, 'Foreign asset must have explicit single-actor ownership'
        self.T = np.loadtxt(Path(self.args.source_path)/'transform_matrix.txt')
        self.path = self.run/'dynamic_chkpnt30000.pth'
        self.before = (self.path.stat().st_size, self.path.stat().st_mtime_ns)
        self.dynamic = Dynamic_GaussianModel(self.args)
        payload, step = torch.load(str(self.path), map_location='cpu', mmap=True)
        assert step == 30000
        payload = list(payload)
        for i in range(1, 11):
            payload[i] = payload[i].detach().cuda()
        self.dynamic.time_duration = tuple(self.args.time_duration)
        self.dynamic.restore(tuple(payload), training_args=None)
        for value in vars(self.dynamic).values():
            if torch.is_tensor(value):
                value.requires_grad_(False)
        self.versions = {k:v._version for k,v in vars(self.dynamic).items() if torch.is_tensor(v)}
        self.track = ActorTrack(context)
        del payload
        gc.collect()

    def unchanged(self):
        return self.before == (self.path.stat().st_size, self.path.stat().st_mtime_ns) and self.versions == {k:v._version for k,v in vars(self.dynamic).items() if torch.is_tensor(v)}


def insertion(native, asset, destination_track, row):
    camera = native.camera(row, dict(family='no_edit'))
    static = prepare_rasterization(camera, native.static, native.args)
    static['means2D'] = torch.zeros_like(static['means3D'])
    static['features'] = torch.cat((static['features'], torch.zeros_like(static['features'])), 1)
    A, R, ratio, audit = asset_transfer(asset.track, destination_track, row, asset.T, native.T)
    own = prepare_rasterization(camera, native.dynamic, native.args, dynamic=True)
    query_camera = copy.copy(camera)
    query_camera.timestamp = row['dynamic_query_model_time']
    foreign = prepare_rasterization(query_camera, asset.dynamic, asset.args, dynamic=True)
    check_mean = cpu(foreign['means3D'][:32]).astype(float)
    check_scale = cpu(foreign['scales'][:32]).astype(float)
    check_rotation = Rotation.from_quat(np.roll(cpu(foreign['rotations'][:32]), -1, axis=1)).as_matrix()
    foreign['means3D'] = foreign['means3D'] @ torch.as_tensor(A[:3, :3].T, device='cuda', dtype=torch.float32) + torch.as_tensor(A[:3, 3], device='cuda', dtype=torch.float32)
    foreign['scales'] = foreign['scales']*ratio
    q = torch.as_tensor(np.roll(Rotation.from_matrix(R).as_quat(), 1), device='cuda', dtype=torch.float32)
    foreign['rotations'] = quat_product(q, foreign['rotations'])
    foreign['shs'] = rotate(foreign['shs'], R)
    expected_mean = check_mean @ A[:3, :3].T+A[:3, 3]
    mean_error = float(np.max(np.abs(cpu(foreign['means3D'][:32])-expected_mean)))
    covariance = np.einsum('nij,nj,nkj->nik', check_rotation, check_scale**2, check_rotation)
    expected_covariance = A[:3, :3] @ covariance @ A[:3, :3].T
    new_rotation = Rotation.from_quat(np.roll(cpu(foreign['rotations'][:32]), -1, axis=1)).as_matrix()
    actual_covariance = np.einsum('nij,nj,nkj->nik', new_rotation, cpu(foreign['scales'][:32]).astype(float)**2, new_rotation)
    covariance_relative_error = float(np.max(np.abs(expected_covariance-actual_covariance))/max(1e-12, np.max(np.abs(expected_covariance))))
    assert mean_error < 2e-5 and covariance_relative_error < 3e-5
    audit.update(GPU_mean_transform_max_abs_model_units=mean_error, GPU_covariance_transform_relative_error=covariance_relative_error)
    assert asset.dynamic.active_sh_degree == native.static.active_sh_degree
    kwargs = static
    for dynamic, tag in [(own, [1., 0.]), (foreign, [0., 1.])]:
        dynamic['means2D'] = torch.zeros_like(dynamic['means3D'])
        z = torch.cat((dynamic['means3D'], torch.ones_like(dynamic['means3D'][:, :1])), 1) @ camera.world_view_transform[:, 2:3]
        dynamic['features'] = torch.cat((z, torch.ones_like(z), torch.ones_like(z)*tag[0], torch.ones_like(z)*tag[1]), 1)
        kwargs = helpers.merge_rows(kwargs, dynamic)
    out = helpers.raster(camera, native.args, native.static.active_sh_degree, kwargs, native.bg)
    env = native.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
    out['rgb'] = out['rgb']+(1-out['alpha'])*env
    return out, audit


def adapt_row(row):
    result = copy.deepcopy(row)
    result['camera_model_time'] = row['model_time']
    result['dynamic_query_model_time'] = row['query_model_time']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', nargs='+', required=True)
    parser.add_argument('--single-frame', action='store_true')
    args = parser.parse_args()
    require_start()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    base_run = ROOT/'native_scene_training_v1'/args.context/'formal30k'
    native = BenchmarkNative(base_run, 30000)
    sensor = read(native.source/'sensor_preparation.json')
    specs = [read(ROOT/'benchmark_cases_v1/scenes'/args.context/'requests'/(family+'.json')) for family in args.families]
    selection = read(native.source/'native_preparation/selection_reviewed.json')
    assert all(spec['target_track'] == selection['track'] for spec in specs), 'Do not silently reuse another target'
    indices = [15] if args.single_frame else range(31)
    identity_max = 0.
    for i in indices:
        row = adapt_row(specs[0]['frames'][i])
        row['camera_c2w_world'] = sensor_camera(native.source, sensor['frames'][row['camera_index']], args.context).tolist()
        row['dynamic_query_model_time'] = row['camera_model_time']
        camera = native.camera(row, dict(family='no_edit'))
        baseline = native.render(row, dict(family='no_edit'))
        actual = actual_native_render(camera, native.static, native.dynamic, native.args, native.bg, env_map=native.env)['render']
        error = float((baseline['rgb']-actual).abs().max())
        identity_max = max(identity_max, error)
        assert error <= 2e-6
    records, skipped = [], []
    foreign = None
    target_track = ActorTrack(args.context)
    mapping = dict(ego_speed='no_edit', ego_lane_change='no_edit', ego_direction='no_edit', vehicle_speed='speed', vehicle_lane_change='translate', vehicle_direction='turn', vehicle_removal='delete')
    for spec in specs:
        require_start()
        family = spec['family']
        if not spec['input_screen_eligible']:
            skipped.append(dict(family=family, reason=spec['input_screen_reasons']))
            continue
        folder = args.output/family
        for part in ['native_rgb', 'float', 'mask']:
            (folder/part).mkdir(parents=True)
        if family == 'vehicle_insertion':
            foreign = ForeignAsset(spec['source_asset_context'])
            native.sh_frame = 'world'
        case = dict(family=mapping.get(family, 'no_edit'), benchmark_family=family)
        rows = []
        for i in indices:
            require_start()
            row = adapt_row(spec['frames'][i])
            if family == 'vehicle_insertion':
                out, asset_audit = insertion(native, foreign, target_track, row)
            else:
                out = native.render(row, case)
                asset_audit = None
            assert all(torch.isfinite(out[key]).all() for key in ['rgb', 'alpha', 'depth_sum', 'extra']), 'Non-finite edited native output; preserve attempt'
            name = f'{row["raw"]:03d}'
            camera_edit = family.startswith('ego_')
            if camera_edit or family == 'vehicle_insertion':
                mask = np.zeros((320, 480), dtype=bool)
            else:
                original_row = copy.deepcopy(row)
                original_row['dynamic_query_model_time'] = row['camera_model_time']
                baseline = native.render(original_row, dict(family='no_edit'))
                body = np.array(Image.open(native.source/'dynamic_masks/dynamic_mask_0'/(row['frame_id']+'.png')).resize((480, 320), Image.Resampling.NEAREST)) > 0
                support = binary_dilation(body, iterations=2) | (cpu(baseline['extra'])[0] > .05)
                mask = support & ~(cpu(out['extra'])[0] > .05)
            save(folder/'native_rgb'/(name+'.png'), out['rgb'])
            Image.fromarray(mask.astype('uint8')*255).save(folder/'mask'/(name+'.png'))
            np.savez_compressed(folder/'float'/(name+'.npz'), rgb=cpu(out['rgb']), alpha=cpu(out['alpha']), depth_sum=cpu(out['depth_sum']), dynamic_weight=cpu(out['extra']))
            rows.append(dict(raw=row['raw'], generation_pixels=int(mask.sum()),
                             mask_status='unknown_domain_pending_native_only' if camera_edit else ('insertion_no_old_hole_new_asset_protected' if foreign is not None and family == 'vehicle_insertion' else 'original_support_new_object_protected'),
                             input_collision_tracks=row['request_box_collision_tracks'],
                             input_collision_status='ego_or_retimed_collision_not_checked' if camera_edit or family == 'vehicle_speed' else ('not_required_for_deletion' if family == 'vehicle_removal' else 'annotated_vehicle_boxes_only_not_drivable_space'),
                             asset_alignment=asset_audit))
        result = dict(case_id=spec['case_id'], family=family, frames=len(rows), rows=rows,
                      source_request=str(ROOT/'benchmark_cases_v1/scenes'/args.context/'requests'/(family+'.json')),
                      native_executed=True, fixed_backend_executed=False, formal_ready=False,
                      counterfactual_RGB_GT=False, camera_unknown_mask_complete=False if camera_edit else None,
                      original_SH_world_basis=True if family != 'vehicle_insertion' else None,
                      imported_asset_SH_rigidly_rotated=family == 'vehicle_insertion',
                      imported_asset_relighting=False, quality_accepted=False, human_verdict=None)
        (folder/'summary.json').write_text(json.dumps(result, indent=2))
        records.append({k:v for k,v in result.items() if k != 'rows'})
        if foreign is not None:
            assert foreign.unchanged()
            del foreign
            foreign = None
            gc.collect()
            torch.cuda.empty_cache()
        print('BENCHMARK_NATIVE_COMPLETE', family, flush=True)
    assert native.unchanged()
    summary = dict(context=args.context, status='native_eight_family_diagnostic_batch_complete',
                   completed=records, skipped=skipped, adapter_actual_native_max_difference=identity_max,
                   master_files_and_tensor_versions_unchanged=True, seconds=time.monotonic()-started,
                   cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                   source_time_policy='supported source model time; physical source seconds separately recorded; not learned new dynamics',
                   new_training_updates=0, formal_admitted_cases=0, quality_accepted=False, human_verdict=None)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    from gpu_slot import exclusive
    with exclusive(), torch.no_grad():
        main()
