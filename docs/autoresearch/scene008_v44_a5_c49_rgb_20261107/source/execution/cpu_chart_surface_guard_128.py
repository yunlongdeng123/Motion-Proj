"""Read-only all-positive-C2 first-hit classification; no renderer or variant."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import ast
import hashlib
import json
import time

import numpy as np
from scipy.spatial import Delaunay, cKDTree

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B / 'scene008_v44_a5_c49_rgb_20261107_r1'
OLD = B / 'scene008_v44_a5_road_fusion_integration_20261107_r1'
RED = B / 'scene008_road_seam_contact_20261107_r1/red'
R0 = B / 'scene008_road_r0_v44_20261107_r1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def original_nodes(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    assert {node.name for node in selected} == set(names)
    # The original mathematical bodies are compiled unchanged. No import or
    # main() from the old diagnostic can read RGB/GT or choose the query domain.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def counts(label, domain):
    return {name: int((domain & (label == value)).sum()) for value, name in
            ((0, 'unknown'), (1, 'chart_first'), (2, 'ground_first'), (3, 'near_contact'))}


def main():
    out = R / 'review/chart_firsthit128_model_only'
    out.mkdir(exist_ok=False)
    start = time.time()
    kernel = RED / 'audit_raw128_contact.py'
    surface_source = R0 / 'review/localize_lateral_domain.py'
    ns = dict(np=np, cKDTree=cKDTree)
    original_nodes(kernel, ['opaque_hits'], ns)
    original_nodes(surface_source, ['Surface', 'rays'], ns)
    asset_path = B / 'scene008_a6_insert_artifacts_20261105/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz'
    obs_path = B / 'scene008_instances_20261105_r1/observations/frames/128.npz'
    frame_path = B / 'r15_gn11_fullclip_cpu_20261026/frames/128.npz'
    baseline_path = OLD / 'baseline4/frames/128/delete_float.npz'
    # Only fixed geometry/matte and camera arrays are used from model inputs.
    with np.load(asset_path, allow_pickle=False) as z:
        chart = z['representation_kind'] == 1
        uv, vertices = z['resample_reference_pixel_xy'][chart], z['xyz'][chart].astype(float)
        alpha = np.minimum(z['opacity'][chart] / np.float32(.99), 1)
    with np.load(obs_path, allow_pickle=False) as z:
        pose, model_to_camera, K = z['model_from_local'][0], z['model_to_camera'], z['K']
    with np.load(frame_path, allow_pickle=False) as z:
        ground_K, ground_w2l = z['K'], z['w2l']
    with np.load(baseline_path, allow_pickle=False) as z:
        c2_mass = z['group_alpha_T'][0].copy()
    selected = c2_mass > 0
    assert selected.shape == (320, 480) and selected.sum() == 1589
    pixels = np.argwhere(selected)[:, ::-1]
    triangles = Delaunay(uv).simplices
    edges = np.stack([np.linalg.norm(uv[triangles[:, i]] - uv[triangles[:, j]], axis=1)
                      for i, j in ((0, 1), (1, 2), (2, 0))], axis=1)
    triangles = triangles[edges.max(1) <= 1.5]
    bundle_path = B / 'r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json'
    bundle = json.loads(bundle_path.read_text())
    scale = float(bundle['meter_to_model'])
    readiness = Path(bundle['original_readiness'])
    plane = np.asarray(json.loads(readiness.read_text())['SGCM_geometry']['ground_fit']['plane_anchor'])
    pool_path = R0 / 'surface/measured_ground_support_pool.npz'
    with np.load(pool_path, allow_pickle=False) as z:
        pool = z['local_xyz'].copy()
    surface = ns['Surface'](pool, plane)
    ground, ground_valid = ns['rays'](surface, ground_K, ground_w2l)
    ground_depth = (ground.reshape(-1, 3) @ ground_w2l[:3, :3].T + ground_w2l[:3, 3])[:, 2].reshape(320, 480)
    camera_to_actor = np.linalg.inv(pose) @ np.linalg.inv(model_to_camera)
    directions = (np.c_[pixels + .5, np.ones(len(pixels))] @ np.linalg.inv(K).T) @ camera_to_actor[:3, :3].T * scale
    benchmark_ids = np.linspace(0, len(pixels)-1, 16).astype(int)
    t = time.time()
    sample = ns['opaque_hits'](vertices, triangles, camera_to_actor[:3, 3],
                               directions[benchmark_ids], alpha, (.01, .5))
    sample_seconds = time.time() - t
    print(json.dumps(dict(stage='small16_model_only_rays', seconds=sample_seconds,
                         estimated_full_seconds=sample_seconds*len(pixels)/16)), flush=True)
    t = time.time()
    hits = ns['opaque_hits'](vertices, triangles, camera_to_actor[:3, 3],
                            directions, alpha, (.01, .5))
    full_seconds = time.time() - t
    assert np.array_equal(np.isnan(sample), np.isnan(hits[:, benchmark_ids]))
    assert np.array_equal(sample[np.isfinite(sample)], hits[:, benchmark_ids][np.isfinite(sample)])
    chart_depth = np.full((2, 320, 480), np.nan)
    gap = np.full((2, 320, 480), np.nan)
    label = np.zeros((2, 320, 480), np.uint8)
    for mi in range(2):
        chart_depth[mi, selected] = hits[mi]
        gap[mi, selected] = ground_depth[selected] - hits[mi]
        valid = selected & ground_valid & np.isfinite(gap[mi])
        label[mi, valid & (gap[mi] > .05)] = 1
        label[mi, valid & (gap[mi] < -.05)] = 2
        label[mi, valid & (np.abs(gap[mi]) <= .05)] = 3
    # This is a compatibility audit against an old registered diagnostic, not
    # a source of query selection. Its RGB/GT arrays are never read.
    old_map_path = RED / 'raw128_fixed_red_geometry_v2.npz'
    with np.load(old_map_path, allow_pickle=False) as z:
        old_roi, old_labels = z['roi'], z['labels']
        old_ground, old_ground_valid = z['ground_depth_m'], z['ground_valid']
        old_cohort = z['roi'] & (z['baseline_assetW'] > .9) & (z['assetW_loss'] > .01)
    overlap = old_roi & selected
    comparison = []
    for mi, threshold in enumerate((.01, .5)):
        mismatch = int((label[mi, overlap] != old_labels[mi, overlap]).sum())
        assert mismatch == 0
        comparison.append(dict(matte=threshold, overlap_pixels=int(overlap.sum()),
            old_fixed_cohort_pixels=int(old_cohort.sum()),
            old_cohort_in_current_model_guard=int((old_cohort & selected).sum()),
            old_cohort_counts=counts(label[mi], old_cohort & selected),
            label_mismatch_pixels=mismatch))
    assert np.array_equal(ground_valid, old_ground_valid)
    assert np.array_equal(ground_depth, old_ground)
    permitted = selected & (label[0] == 2)
    archive = out / 'model_only_first_hit_maps.npz'
    np.savez_compressed(archive, selected_C2_positive=selected, labels=label,
        ground_valid=ground_valid, ground_camera_z_m=ground_depth,
        chart_camera_z_m=chart_depth, ground_minus_chart_m=gap,
        valid_ground_first_matte001=permitted,
        valid_ground_first_both_old_mattes=(label[0] == 2) & (label[1] == 2) & selected)
    report = dict(status='PASS_READONLY_CPU_MODEL_ONLY_FIRST_HIT_CLASSIFICATION', raw=128,
        state='delete', selection='saved latest-baseline C2 joint alphaT > 0; no RGB/GT',
        selected_pixels=int(selected.sum()), chart_vertices=len(vertices), triangles=len(triangles),
        original_kernel_sha256=sha(kernel), original_surface_source_sha256=sha(surface_source),
        mathematical_AST_bodies_unchanged=True, two_sided=True, UV_edge_max_px=1.5,
        old_fixed_matte_thresholds=[.01, .5], ordering_margin_m=.05,
        classifications={str(v): counts(label[mi], selected) for mi, v in enumerate((.01, .5))},
        matte_disagreement_pixels=int((selected & (label[0] != label[1])).sum()),
        only_allow_proposal='valid ground_first; chart_first/contact/unknown remain protected',
        visibility_policy_or_mask_applied=False, old_diagnostic_comparisons=comparison,
        ground_map_replay_exact=True, map_file=archive.name, map_sha256=sha(archive),
        small16_seconds=sample_seconds, full1589_seconds=full_seconds,
        elapsed_seconds=time.time()-start,
        input_sha256={str(p): sha(p) for p in (asset_path, obs_path, frame_path, baseline_path,
                                              bundle_path, readiness, pool_path, old_map_path)},
        caveats=['Open chart, not closed or independently validated vehicle surface.',
                 'Non-chart rows and absent chart hits remain unknown, never ground-owned.',
                 'Classification does not prove visible thin-edge removal or vehicle preservation.'],
        GPU_used=False, renderer_forwards=0, model_updates=0)
    (out / 'classification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_sha256',)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
