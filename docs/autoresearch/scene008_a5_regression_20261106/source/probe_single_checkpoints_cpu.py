"""Read-only CPU support audit of saved single-a37 10000 and smoke11977.

No engine import, renderer call, optimizer, or CUDA allocation. Torch checkpoints
are memory mapped on CPU; all arithmetic uses NumPy. Prints strict JSON unless
--output is supplied. Run with the existing drivingeditor Python environment.

Projection footprints below are explicitly a CPU capacity proxy, not a native
raster replay or a causal attribution. The fixed upper25% of the a5 label bbox is
a diagnostic image region, not ground-truth semantic roof annotation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import collections
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
SINGLE = BASE / 'scene008_a37_only_B10k_20261103_r1/formal10k'
NEW = BASE / 'scene008_a5_old_single30k_20261106_r1'
PREP = BASE / 'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137'
ENGINE = BASE / 'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
RAWS = (89, 97, 105, 109)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def identity(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path))


def quantiles(values):
    values = np.asarray(values).reshape(-1)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite checkpoint-derived parameter statistic')
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'), map(float, np.quantile(values, [0, .1, .5, .9, 1]))))


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -700, 700)))


def xyz_at(payload, stamp):
    # Exact native get_xyz_SHM formula, evaluated on CPU.
    phase = (stamp - np.asarray(payload[7]).reshape(-1, 1)) * (2 * np.pi / float(payload[16]))
    return np.asarray(payload[1], dtype=np.float64) + np.asarray(payload[9]) * np.sin(phase) / (2 * np.pi / float(payload[16]))


def region_stats(indices, payload, marginal, scale, rate):
    sigma = np.exp(np.asarray(payload[8])[indices].reshape(-1).astype(np.float64))
    opacity = sigmoid(np.asarray(payload[6])[indices].reshape(-1).astype(np.float64))
    velocity = np.linalg.norm(np.asarray(payload[9])[indices].astype(np.float64), axis=1)
    spatial = np.exp(np.asarray(payload[4])[indices].astype(np.float64)).max(1) / scale
    effective = opacity * marginal[indices]
    active = marginal[indices] > .05
    return dict(rows=len(indices), active_marginal_gt_005=int((marginal[indices] > .05).sum()),
                active_fraction=float(active.mean()) if len(indices) else None,
                effective_opacity_ge_1_over_255=int((effective >= 1 / 255).sum()),
                opacity=quantiles(opacity), marginal_t=quantiles(marginal[indices]),
                effective_opacity=quantiles(effective), effective_opacity_sum=float(effective.sum()),
                active_base_opacity=quantiles(opacity[active]), active_effective_opacity=quantiles(effective[active]),
                t_center=quantiles(np.asarray(payload[7])[indices]), sigma_t=quantiles(sigma),
                max_spatial_sigma_m=quantiles(spatial), velocity_amplitude_m_per_second=quantiles(velocity / scale * rate),
                max_harmonic_displacement_m=quantiles(velocity / (2 * np.pi / float(payload[16])) / scale))


def footprint_proxy(indices, xyz, payload, marginal, K, W, query_yx):
    """First-order projected Gaussian capacity, without other-scene occlusion.

    Uses +0.3px^2 low-pass, no native tile/frustum/clamped-FoV replay. A high
    value is support capacity, not actual a5 alpha or correctness.
    """
    if len(query_yx) == 0:
        return None
    if len(indices) == 0:
        return dict(candidate_gaussians=0, mean=0., coverage_ge_half=0., zero_capacity_pixels=len(query_yx))
    opacity = sigmoid(np.asarray(payload[6])[indices].reshape(-1).astype(np.float64)) * marginal[indices]
    take = opacity >= 1 / 255
    indices, opacity = indices[take], opacity[take]
    x = xyz[indices] @ W[:3, :3].T + W[:3, 3]
    q = np.asarray(payload[5])[indices].astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    w, a, b, c = q.T
    rotation = np.stack((1-2*(b*b+c*c), 2*(a*b-w*c), 2*(a*c+w*b),
                         2*(a*b+w*c), 1-2*(a*a+c*c), 2*(b*c-w*a),
                         2*(a*c-w*b), 2*(b*c+w*a), 1-2*(a*a+b*b)), axis=1).reshape(-1, 3, 3)
    scales = np.exp(np.asarray(payload[4])[indices].astype(np.float64))
    L = rotation * scales[:, None, :]
    covariance = L @ L.transpose(0, 2, 1)
    covariance = W[:3, :3][None] @ covariance @ W[:3, :3].T[None]
    z = np.maximum(x[:, 2], 1e-8)
    J = np.zeros((len(indices), 2, 3))
    J[:, 0, 0], J[:, 0, 2] = K[0, 0] / z, -K[0, 0] * x[:, 0] / z**2
    J[:, 1, 1], J[:, 1, 2] = K[1, 1] / z, -K[1, 1] * x[:, 1] / z**2
    cov2 = J @ covariance @ J.transpose(0, 2, 1) + np.eye(2)[None] * .3
    inverse = np.linalg.inv(cov2)
    uv = (x @ K.T)[:, :2] / z[:, None] - .5
    query = query_yx[:, ::-1].astype(np.float64)
    log_transmission = np.zeros(len(query))
    support_count = np.zeros(len(query), dtype=np.int32)
    for start in range(0, len(indices), 128):
        delta = query[None] - uv[start:start+128, None]
        energy = np.einsum('npi,nij,npj->np', delta, inverse[start:start+128], delta, optimize=True)
        alpha = np.minimum(.99, opacity[start:start+128, None] * np.exp(-.5 * np.maximum(energy, 0)))
        alpha[alpha < 1 / 255] = 0
        support_count += (alpha > 0).sum(0)
        log_transmission += np.log1p(-alpha).sum(0)
    capacity = -np.expm1(log_transmission)
    return dict(candidate_gaussians=len(indices), mean=float(capacity.mean()),
                coverage_ge_half=float((capacity >= .5).mean()),
                zero_capacity_pixels=int((support_count == 0).sum()),
                capacity=quantiles(capacity), gaussian_count_per_pixel=quantiles(support_count))


def native_fixed_metrics(eval_path, frame, part):
    path = eval_path / 'a5_float' / ('%03d.npz' % frame['raw'])
    with np.load(path, allow_pickle=False) as z:
        yx = z['yx']
        if not np.array_equal(yx, np.column_stack(np.nonzero(frame['mask']))):
            raise ValueError('Native float indices differ from fixed original mask')
        selection = part[yx[:, 0], yx[:, 1]]
        keep, deleted, gt = (z[name][selection].astype(np.float64) for name in ('keep_rgb', 'delete_rgb', 'gt_rgb'))
        ka, da = (z[name][selection] for name in ('keep_alpha', 'delete_alpha'))
    return dict(pixels=len(keep), keep_rgb_mae=float(np.abs(np.clip(keep, 0, 1)-gt).mean()),
                delete_rgb_mae=float(np.abs(np.clip(deleted, 0, 1)-gt).mean()),
                keep_delete_rgb_change=float(np.abs(np.clip(keep, 0, 1)-np.clip(deleted, 0, 1)).mean()),
                keep_mean_alpha=float(ka.mean()), delete_mean_alpha=float(da.mean()),
                keep_alpha_ge_half=float((ka >= .5).mean()), keep_alpha_ge_099=float((ka >= .99).mean()),
                keep_alpha=quantiles(ka), mean_keep_rgb=keep.mean(0).tolist())


def frame_data(options, actor_track, raw):
    path = options.observations / ('%03d.npz' % raw)
    with np.load(path, allow_pickle=False) as z:
        mask = z['actor_labels'] == 1
        y, x = np.nonzero(mask)
        if not len(y):
            raise ValueError('Selected anchor has no observed a5 mask')
        upper = mask & (np.indices(mask.shape)[0] <= y.min() + .25 * (y.max()-y.min()))
        values = dict(raw=raw, mask=mask, upper=upper, lower=mask & ~upper,
                      K=z['K'].copy(), W=z['model_to_camera'].copy(), world_W=z['world_to_camera'].copy(), stamp=float(z['model_time']),
                      depth_m=z['scene_depth_m'].copy())
    annotation_path = options.prepared / 'annotations' / ('0008%03d.json' % raw)
    annotation = read(annotation_path)
    actor = next(row for row in annotation['laser_labels'] if row['id'] == actor_track)
    box = actor['box']
    values.update(ego=np.asarray(annotation['frame_pose_world_from_vehicle']).reshape(4, 4),
                  center=np.array([box['center_x'], box['center_y'], box['center_z']]),
                  half=np.array([box['length'], box['width'], box['height']]) / 2,
                  heading=box['heading'], annotation_path=annotation_path,
                  box_source='original laser_labels.box, not camera_synced_box or semantic point truth')
    return values


def checkpoint_stats(options, model, iteration, frames, scale, model_from_world):
    import torch
    result = dict(model=str(model), iteration=iteration, branches={})
    roof_families = {}
    for branch, prefix in (('static', 'chkpnt'), ('dynamic', 'dynamic_chkpnt')):
        payload, actual = torch.load(str(model / ('%s%d.pth' % (prefix, iteration))),
                                    map_location='cpu', mmap=True, weights_only=False)
        if actual != iteration or len(payload) != 18 or any(value.device.type != 'cpu' for value in payload if torch.is_tensor(value)):
            raise ValueError('Unexpected checkpoint schema/device')
        payload = tuple(value.detach() if torch.is_tensor(value) else value for value in payload)
        N = len(payload[1])
        rate = 10 * float(read(model / 'effective_config.json')['frame_interval'])
        lineage = payload[14].get('sgcm_lineage_v1', {}).get('lineage_ids')
        values = dict(rows=N, period_model_time=float(payload[16]),
                      period_physical_seconds=float(payload[16]) / rate,
                      raw_parameter_nonfinite_counts={name:int((~np.isfinite(np.asarray(payload[index]))).sum())
                              for name,index in (('xyz',1),('log_scale',4),('rotation',5),('opacity_logit',6),('t',7),('log_sigma_t',8),('velocity',9))},
                      global_t_center=quantiles(np.asarray(payload[7])),
                      global_sigma_t=quantiles(np.exp(np.asarray(payload[8]).astype(np.float64))),
                      global_base_opacity=quantiles(sigmoid(np.asarray(payload[6]).astype(np.float64))), frames=[])
        for frame in frames:
            raw, stamp, mask = frame['raw'], frame['stamp'], frame['mask']
            xyz = xyz_at(payload, stamp)
            sigma = np.exp(np.asarray(payload[8]).reshape(-1).astype(np.float64))
            marginal = np.exp(-.5 * ((np.asarray(payload[7]).reshape(-1)-stamp) / sigma)**2)
            ego_from_model = np.linalg.inv(frame['ego']) @ np.linalg.inv(model_from_world)
            ego_xyz = xyz @ ego_from_model[:3, :3].T + ego_from_model[:3, 3]
            c, s = np.cos(frame['heading']), np.sin(frame['heading'])
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
            local = (ego_xyz-frame['center']) @ rotation
            near = np.all(np.abs(local) <= frame['half'] + .3, axis=1)
            roof = near & (local[:, 2] >= frame['half'][2]-.4)
            camera_xyz = xyz @ frame['W'][:3, :3].T + frame['W'][:3, 3]
            z = camera_xyz[:, 2]
            uv = (camera_xyz @ frame['K'].T)[:, :2] / np.maximum(z[:, None], 1e-10) - .5
            finite = (z > .001) & np.isfinite(uv).all(1)
            ix = np.clip(np.rint(uv[:, 0]), 0, 479).astype(np.int32)
            iy = np.clip(np.rint(uv[:, 1]), 0, 319).astype(np.int32)
            in_image = finite & (uv[:, 0] >= -.5) & (uv[:, 0] < 479.5) & (uv[:, 1] >= -.5) & (uv[:, 1] < 319.5)
            hit_visible = near & in_image & mask[iy, ix]
            hit_upper = near & in_image & frame['upper'][iy, ix]
            row = dict(raw=raw, observed_pixels=int(mask.sum()), upper25_pixels=int(frame['upper'].sum()),
                       near_a5_OBB=region_stats(np.flatnonzero(near), payload, marginal, scale, rate),
                       OBB_roof_band=region_stats(np.flatnonzero(roof), payload, marginal, scale, rate),
                       centers_in_visible_a5_mask=region_stats(np.flatnonzero(hit_visible), payload, marginal, scale, rate),
                       centers_in_upper25_mask=region_stats(np.flatnonzero(hit_upper), payload, marginal, scale, rate),
                       global_time_gate_dropped_rows=int((marginal <= .05).sum()))
            if branch == 'static':
                active_near = np.flatnonzero(near & (marginal > .05) & (z > .001))
                row['upper25_projected_capacity_proxy'] = footprint_proxy(
                    active_near, xyz, payload, marginal, frame['K'], frame['W'], np.column_stack(np.nonzero(frame['upper'])))
            selected = hit_upper & (marginal > .05)
            depth = frame['depth_m'][iy, ix]
            measured = selected & (depth > 0)
            row['upper_center_lidar_depth_proxy'] = dict(rows=int(measured.sum()),
                    mean_abs_error_m=float(np.abs(z[measured]/scale-depth[measured]).mean()) if measured.any() else None,
                    within_025m_rows=int((np.abs(z[measured]/scale-depth[measured]) <= .25).sum()))
            if branch == 'static' and raw == 105 and lineage is not None:
                counts = collections.Counter(np.asarray(lineage)[roof].tolist())
                roof_families = counts
                row['roof_initial_family_top12'] = [dict(initial_source_id=int(key), descendants=value) for key, value in counts.most_common(12)]
            values['frames'].append(row)
            del xyz, sigma, marginal, ego_xyz, local, camera_xyz
        result['branches'][branch] = values
        del payload
        gc.collect()
    return result, roof_families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-a', type=Path, default=SINGLE)
    parser.add_argument('--iteration-a', type=int, default=10000)
    parser.add_argument('--model-b', type=Path, default=NEW / 'smoke_retry1')
    parser.add_argument('--iteration-b', type=int, default=11977)
    parser.add_argument('--eval-a', type=Path, default=NEW / 'reference_a5_single10k_retry1')
    parser.add_argument('--eval-b', type=Path, default=NEW / 'smoke_a5_evaluation')
    parser.add_argument('--observations', type=Path, default=BASE / 'scene008_instances_20261105_r1/observations/frames')
    parser.add_argument('--prepared', type=Path, default=PREP)
    parser.add_argument('--engine', type=Path, default=ENGINE)
    parser.add_argument('--bundle', type=Path, default=BASE / 'r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
    parser.add_argument('--script-sha', default=None)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    prep_path = options.prepared / 'dynamic_multi_actor_preparation.json'
    actor = read(prep_path)['actors'][0]
    track = actor['raw_track_id']
    if track != '3S_GJHyCGGK4ypFPVKOOqA':
        raise ValueError('Unexpected canonical a5 track')
    bundle = read(options.bundle)
    frames = [frame_data(options, track, raw) for raw in RAWS]
    for frame in frames:
        expected = np.asarray(bundle['world_to_model']) @ np.linalg.inv(frame['world_W'])
        expected[:3, :3] /= float(bundle['meter_to_model'])
        if not np.allclose(np.linalg.inv(frame['W']), expected, atol=1e-5, rtol=0):
            raise ValueError('Bundle world/model map does not match the fixed observation camera')
    paths = [prep_path, options.bundle, options.engine/'scene/gaussian_model.py',
             options.engine/'gaussian_renderer/__init__.py', options.engine/'train.py']
    for model, step, evaluation in ((options.model_a, options.iteration_a, options.eval_a),
                                    (options.model_b, options.iteration_b, options.eval_b)):
        paths += [model/('%s%d.pth'%(prefix,step)) for prefix in ('chkpnt','dynamic_chkpnt')]
        paths += [model/'effective_config.json', model/'actual_scene_binding.json', evaluation/'summary.json']
        paths += [evaluation/'a5_float'/('%03d.npz'%raw) for raw in RAWS]
    paths += [frame['annotation_path'] for frame in frames]
    paths += [options.observations/('%03d.npz'%raw) for raw in RAWS]
    log_path = options.model_b.parent/'smoke_retry1.log'
    paths.append(log_path)
    originals = [identity(path) for path in paths]
    summaries = [read(options.eval_a/'summary.json'), read(options.eval_b/'summary.json')]
    if summaries[0]['fixed_denominator']['sha256'] != summaries[1]['fixed_denominator']['sha256']:
        raise ValueError('Two evaluations have different fixed pixel domains')
    native = []
    for label, evaluation in (('original10000', options.eval_a), ('smoke11977', options.eval_b)):
        native.append(dict(label=label, eval_directory=str(evaluation), frames=[dict(raw=frame['raw'],
                    all_visible=native_fixed_metrics(evaluation, frame, frame['mask']),
                    upper25=native_fixed_metrics(evaluation, frame, frame['upper']),
                    lower75=native_fixed_metrics(evaluation, frame, frame['lower'])) for frame in frames]))
    checkpoint_a, families_a = checkpoint_stats(options, options.model_a, options.iteration_a, frames,
                                               float(bundle['meter_to_model']), np.asarray(bundle['world_to_model']))
    checkpoint_b, families_b = checkpoint_stats(options, options.model_b, options.iteration_b, frames,
                                               float(bundle['meter_to_model']), np.asarray(bundle['world_to_model']))
    log = log_path.read_text(errors='replace')
    sgcm = [json.loads(value) for value in re.findall(r'SGCM_STATS (\{[^\r\n]*\})', log)]
    config = read(options.model_b/'effective_config.json')
    resets = [step for step in range(options.iteration_a+1, options.iteration_b+1)
              if step < config['densify_until_iter'] and step % config['opacity_reset_interval'] == 0]
    unchanged = all(path.stat().st_size == record['bytes'] and path.stat().st_mtime_ns == record['mtime_ns']
                    for path, record in zip(paths, originals))
    if not unchanged or torch.cuda.is_initialized():
        raise RuntimeError('Unexpected source change or CUDA initialization')
    A, B = checkpoint_a['branches']['static']['rows'], checkpoint_b['branches']['static']['rows']
    net_sgcm = sum(row.get('points_after',0)-row.get('points_before',0) for row in sgcm)
    output = dict(status='CPU_CHECKPOINT_SUPPORT_AUDIT_COMPLETE', gpu_used=False,
                  torch_cuda_initialized=torch.cuda.is_initialized(), optimization_updates=0,
                  source_files=originals, source_size_mtime_unchanged_after_probe=unchanged,
                  readonly_scope='Only these inventoried files; SHA captured once, size/mtime rechecked after CPU reads. No claim about all remote inputs.',
                  script_sha256=options.script_sha or digest(__file__),
                  a5_raw_track_id=track, raw_anchors=list(RAWS),
                  regions=dict(upper25='Original actor_labels==1 intersect upper25% of its image bbox; fixed diagnostic proxy, not semantic roof GT',
                               OBB='Original annotated a5 box expanded by0.3m; roof band local z>=half-height-0.4m; centers only, not exact surface ownership'),
                  native_all49_from_existing_evaluations=[dict(iteration=summary['iteration'], all49=summary['all49'],
                         boundary_all49=summary['boundary_all49']) for summary in summaries],
                  native_fixed_anchor_metrics=native, checkpoint_parameters=[checkpoint_a, checkpoint_b],
                  topology=dict(static_rows_a=A,static_rows_b=B,net_static_rows=B-A,sgcm_events=len(sgcm),
                                sgcm_net_rows=net_sgcm,ordinary_and_other_net_remainder=B-A-net_sgcm,
                                sgcm_pruned=sum(row.get('pruned',0) for row in sgcm),
                                sgcm_cloned=sum(row.get('cloned',0) for row in sgcm),
                                sgcm_split_parents=sum(row.get('split_parents',0) for row in sgcm),
                                sgcm_iterations=[row['iteration'] for row in sgcm],
                                opacity_reset_candidates_between_snapshots=resets,
                                opacity_reset_interval=config['opacity_reset_interval'], next_reset_iteration=12000,
                                note='No opacity reset boundary crossed from10000 to11977; SGCM events logged, ordinary net inferred by conservation only'),
                  raw105_roof_family_overlap=dict(a_unique=len(families_a), b_unique=len(families_b),
                         shared=len(set(families_a)&set(families_b)), absent_in_later_selected_roof=len(set(families_a)-set(families_b)),
                         newly_present_in_later_selected_roof=len(set(families_b)-set(families_a)),
                         note='Common initial-source families only; not child UIDs, not exact births/prune histories'),
                  source_formula_locations=dict(time_kernel='scene/gaussian_model.py:144-158,182-183',
                         native_static_time_gate='gaussian_renderer/__init__.py:342-417',
                         prune='scene/gaussian_model.py:698-727', opacity_reset='train.py:781-783; gaussian_model.py:414-417'),
                  mechanism_facts=[
                         'Native opacity is sigmoid(logit)*exp(-0.5*((timestamp-t)/sigma_t)^2); render participation uses marginal_t>0.05.',
                         'Ordinary prune uses base sigmoid opacity<configured threshold, not the time-marginal effective opacity; a spatially nearby but temporally inactive primitive can survive.',
                         'Opacity reset clips base opacity to at most0.01; no3000-step reset boundary occurs between the compared10000 and11977 states.',
                         'Raw105 upper-mask/OBB selection contains many temporally inactive static rows. Those rows can represent other times; inactivity alone is not an implementation bug or proof they should be active now.'
                         ],
                  visual_review_record=dict(source='2026-11-06 parent and reporting agent inspection of saved anchors/105_crop.png',
                         finding='11977 still has a visibly missing roof and blue translucent/misaligned material above/front of a5. Aggregate RGB MAE improvement is not body restoration.'),
                  limitations=['Two snapshots cannot locate a unique harmful training event or prove a single cause.',
                         'OBB membership and Gaussian center projection are diagnostics, not semantic point ground truth.',
                         'CPU footprint is a first-order unoccluded capacity proxy, not native sorted/tiled alpha or a CUDA counterfactual.',
                         'An OBB-center-selected subset may miss Gaussian footprints whose centers are outside the padded box; its capacity shortfall cannot be subtracted from total alpha to attribute the remainder to background.',
                         'Total native alpha can be supplied by background or another surface; high alpha does not prove a5 roof completeness.',
                         'The additional1977 updates are smoke overhead only, not part of original10K or new formal budget.'],
                  seconds=time.monotonic()-started)
    try:
        import resource
        output['peak_CPU_RSS_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except ImportError:
        output['peak_CPU_RSS_bytes'] = None
    text=json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if options.output:
        protected=[options.model_a,options.model_b,options.engine,options.prepared,options.observations]
        target=options.output.absolute()
        if any(target.is_relative_to(path.absolute()) for path in protected):
            raise ValueError('Output may not modify protected source directories')
        options.output.write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()
