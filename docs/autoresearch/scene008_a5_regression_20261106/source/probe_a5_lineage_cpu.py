"""Read-only CPU comparison of actual single10K and the11977 smoke branch."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
import torch

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD = B / 'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137'
SINGLE = B / 'scene008_a37_only_B10k_20261103_r1'
CONTINUATION = B / 'scene008_a5_old_single30k_20261106_r1'
OBS = B / 'scene008_instances_20261105_r1/observations/frames'


def ply(path):
    vertices = PlyData.read(path)['vertex']
    return (np.column_stack([vertices[key] for key in ['x', 'y', 'z']]).astype(np.float64),
            np.asarray(vertices['nx'], dtype=np.float64))


def array(tensor):
    return tensor.detach().numpy()


def stats(values):
    values = np.asarray(values).reshape(-1)
    return dict(count=len(values), quantiles={str(q): float(np.quantile(values, q)) for q in [0, .1, .5, .9, 1]}) if len(values) else dict(count=0)


def main():
    config = json.loads((SINGLE / 'formal10k/effective_config.json').read_text())
    xyz, timestamps = ply(OLD / 'points3d.ply')
    a37, _ = ply(OLD / 'points3d_dynamic_4.ply')
    a5, a5_times = ply(OLD / 'points3d_dynamic_0.ply')
    distance, _ = cKDTree(a37).query(xyz, distance_upper_bound=.1, workers=4)
    keep = distance >= .1
    raw_static = xyz[keep]
    static_times = timestamps[keep]
    distance, _ = cKDTree(a5).query(raw_static, distance_upper_bound=2e-5, workers=4)
    original_a5 = distance < 2e-5
    a5_ids = np.flatnonzero(original_a5)
    transform = np.loadtxt(OLD / 'transform_matrix.txt')
    assert len(raw_static) == 591672 and len(a5_ids) == 1496
    state = torch.load(SINGLE / 'formal10k/training_state10000.pth', map_location='cpu', weights_only=False)
    sgcm = state['sgcm']
    runtime_flags = state['model_runtime']
    plane = np.asarray(sgcm['plane_model'])
    to_anchor = np.asarray(sgcm['model_to_anchor'])
    regions = [np.asarray(region) for region in sgcm['regions_anchor']]
    scale = float(sgcm['model_scale'])
    cycle = float(config['cycle'])
    decay = float(config['velocity_decay'])
    del state
    cameras = json.loads((SINGLE / 'formal10k/cameras.json').read_text())
    camera = next(row for row in cameras if row['img_name'] == '0008105')
    c2w = np.eye(4)
    c2w[:3, :3] = camera['rotation']
    c2w[:3, 3] = camera['position']
    world_to_camera = np.linalg.inv(c2w)
    with np.load(OBS / '105.npz', allow_pickle=False) as data:
        a5_mask = data['actor_labels'] == 1
        camera_time = float(data['model_time'])
    ys, xs = np.nonzero(a5_mask)
    upper = a5_mask & (np.arange(320)[:, None] <= ys.min() + .35 * (ys.max() - ys.min()))

    def classify(points):
        anchors = points @ to_anchor[:3, :3].T + to_anchor[:3, 3]
        inside = np.zeros(len(points), bool)
        for region in regions:
            inside |= ((anchors >= region[0]) & (anchors <= region[1])).all(1)
        dist = np.abs(points @ plane[:3] + plane[3]) / scale
        return dict(inside=inside, prune=inside & (dist > .5), clone=inside & (dist > .25) & (dist <= .5))

    initial_model = raw_static @ transform[:3, :3].T + transform[:3, 3]
    initial_classes = classify(initial_model[a5_ids])
    stages = []
    for label, root, step in [('original_single10k', SINGLE / 'formal10k', 10000),
                              ('extra_smoke11977', CONTINUATION / 'smoke_retry1', 11977)]:
        path = root / f'chkpnt{step}.pth'
        before = path.stat()
        capture, iteration = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
        assert iteration == step
        lineage = array(capture[14]['sgcm_lineage_v1']['lineage_ids'])
        selected = np.isin(lineage, a5_ids)
        current_ids = lineage[selected]
        points = array(capture[1])[selected].astype(np.float64)
        center = array(capture[7])[selected, 0].astype(np.float64)
        sigma = np.exp(array(capture[8])[selected, 0].astype(np.float64))
        opacity = 1 / (1 + np.exp(-array(capture[6])[selected, 0].astype(np.float64)))
        velocity = array(capture[9])[selected].astype(np.float64)
        instantaneous_velocity = velocity * np.exp(-sigma[:, None] / cycle / 2 * decay)
        marginal = np.exp(-.5 * ((camera_time - center) / sigma) ** 2)
        effective = opacity * marginal
        posed = points + velocity * (np.sin((camera_time - center) * 2 * np.pi / cycle) / (2 * np.pi / cycle))[:, None]
        view = posed @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = view[:, 2]
        projected_x = view[:, 0] / np.where(z == 0, np.inf, z) * camera['fx'] / 4 + camera['cx'] / 4
        projected_y = view[:, 1] / np.where(z == 0, np.inf, z) * camera['fy'] / 4 + camera['cy'] / 4
        px, py = np.rint(projected_x).astype(int), np.rint(projected_y).astype(int)
        valid = (z > 0) & (px >= 0) & (px < 480) & (py >= 0) & (py < 320) & (marginal > .05)
        in_body = np.zeros(len(px), bool)
        in_upper = np.zeros(len(px), bool)
        in_body[valid] = a5_mask[py[valid], px[valid]]
        in_upper[valid] = upper[py[valid], px[valid]]
        classes = classify(points)
        origin = initial_model[current_ids]
        row = dict(stage=label, checkpoint=str(path), iteration=step, bytes=before.st_size,
                   total_static_rows=len(lineage), active_SH_degree=int(capture[0]),
                   saved_screen_radius=dict(over20px=int((array(capture[10]) > 20).sum()),
                                            nonzero=int((array(capture[10]) > 0).sum()),
                                            maximum=float(array(capture[10]).max())),
                   a5_initial_lineage_ids=len(a5_ids), surviving_a5_lineages=len(np.unique(current_ids)),
                   extinct_a5_lineages=len(a5_ids) - len(np.unique(current_ids)), a5_descendant_rows=int(selected.sum()),
                   peak_opacity=stats(opacity), temporal_sigma=stats(sigma), temporal_center=stats(center),
                   raw_velocity_model=stats(np.linalg.norm(velocity, axis=1)),
                   damped_velocity_model=stats(np.linalg.norm(instantaneous_velocity, axis=1)),
                   canonical_displacement_world_m=stats(np.linalg.norm(points - origin, axis=1) / scale),
                   sgcm_current_classes={key: int(value.sum()) for key, value in classes.items()},
                   raw105_center_support_proxy=dict(a5_pixels=int(a5_mask.sum()), upper35pct_bbox_mask_pixels=int(upper.sum()),
                       a5_body_projected_descendants=int(in_body.sum()), upper_projected_descendants=int(in_upper.sum()),
                       a5_body_effective_opacity_sum=float(effective[in_body].sum()), upper_effective_opacity_sum=float(effective[in_upper].sum()),
                       upper_effective_opacity=stats(effective[in_upper])),
                   source_mtime_and_size_unchanged=(path.stat().st_mtime_ns == before.st_mtime_ns and path.stat().st_size == before.st_size))
        assert row['source_mtime_and_size_unchanged']
        stages.append(row)
        del capture, lineage
        gc.collect()

    print(json.dumps(dict(status='PASS_READ_ONLY_CPU_A5_LINEAGE_AND_SCHEDULE_PROBE', GPU_used=False,
                         cuda_initialized=torch.cuda.is_initialized(), training_updates=0,
                         a5_source_rows=len(a5), sampled_a5_static_rows=len(a5_ids),
                         raw_a5_time_range=[float(a5_times.min()), float(a5_times.max())],
                         static_a5_time_range=[float(static_times[a5_ids].min()), float(static_times[a5_ids].max())],
                         initial_a5_sgcm_classes={key: int(value.sum()) for key, value in initial_classes.items()},
                         config_t_grad=bool(config['t_grad']), checkpoint10k_runtime_flags=runtime_flags,
                         schedule=dict(opacity_reset_static_only=[3000, 6000, 9000, 12000], actual_latest_step=11977,
                                       reset12000_executed_in_saved_comparison=False,
                                       SH_degree_schedule=[2000, 4000, 6000], maximum_SH_degree=3,
                                       static_time_splits_enable_after_iteration=7500,
                                       static_no_time_split_at10k=False, generic_densification_interval=100,
                                       generic_densification_nodes_between_saved_states=list(range(10100, 11901, 100))),
                         stages=stages,
                         caveats=['Lineage membership records descendants of exact-matched initial a5 source rows; other source lineages can also learn the a5 appearance.',
                                  'Projected-center/effective-opacity statistics ignore occlusion, covariance footprint and pixel transmittance; they are not actual rendered alpha attribution.',
                                  'The upper35percent bbox mask is a fixed observed-pixel upper-body proxy, not a semantic roof annotation.',
                                  'Two endpoints do not isolate the causal contribution of reset recovery versus densification, pruning or ordinary SGD.']), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
