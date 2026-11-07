"""Registered four-frame model-only C2/ground first-hit maps; no GPU variant."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import importlib.util
import json
import time

import numpy as np
from scipy.spatial import Delaunay, cKDTree

SOURCE = Path(__file__).with_name('cpu_chart_surface_guard_128.py')
spec = importlib.util.spec_from_file_location('fixed_first_hit_cpu_helpers', SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
B, R, OLD, RED, R0 = helper.B, helper.R, helper.OLD, helper.RED, helper.R0
RAWS = [94, 105, 128, 137]


def main():
    out = R / 'review/chart_firsthit4_model_only'
    out.mkdir(exist_ok=False)
    start = time.time()
    ns = dict(np=np, cKDTree=cKDTree)
    kernel = RED / 'audit_raw128_contact.py'
    surface_source = R0 / 'review/localize_lateral_domain.py'
    helper.original_nodes(kernel, ['opaque_hits'], ns)
    helper.original_nodes(surface_source, ['Surface', 'rays'], ns)
    asset_path = B / 'scene008_a6_insert_artifacts_20261105/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz'
    with np.load(asset_path, allow_pickle=False) as z:
        chart = z['representation_kind'] == 1
        uv, vertices = z['resample_reference_pixel_xy'][chart], z['xyz'][chart].astype(float)
        alpha = np.minimum(z['opacity'][chart]/np.float32(.99), 1)
    triangles = Delaunay(uv).simplices
    edges = np.stack([np.linalg.norm(uv[triangles[:, i]]-uv[triangles[:, j]], axis=1)
                      for i, j in ((0, 1), (1, 2), (2, 0))], axis=1)
    triangles = triangles[edges.max(1) <= 1.5]
    bundle_path = B / 'r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json'
    bundle = json.loads(bundle_path.read_text())
    readiness = Path(bundle['original_readiness'])
    plane = np.asarray(json.loads(readiness.read_text())['SGCM_geometry']['ground_fit']['plane_anchor'])
    pool_path = R0 / 'surface/measured_ground_support_pool.npz'
    with np.load(pool_path, allow_pickle=False) as z:
        pool = z['local_xyz'].copy()
    surface, scale = ns['Surface'](pool, plane), float(bundle['meter_to_model'])
    sources = {str(p): helper.sha(p) for p in (SOURCE, Path(__file__), kernel,
        surface_source, asset_path, bundle_path, readiness, pool_path)}
    rows = []
    for raw in RAWS:
        t = time.time()
        guards = {}
        for state in ('keep', 'delete'):
            p = OLD / f'baseline4/frames/{raw:03d}/{state}_float.npz'
            with np.load(p, allow_pickle=False) as z:
                component = z['group_alpha_T'][0]
            assert np.isfinite(component).all() and np.all(component >= 0)
            guards[state] = component > 0
            sources[str(p)] = helper.sha(p)
        selected = guards['keep'] | guards['delete']
        pixels = np.argwhere(selected)[:, ::-1]
        obs_path = B / f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz'
        frame_path = B / f'r15_gn11_fullclip_cpu_20261026/frames/{raw:03d}.npz'
        with np.load(obs_path, allow_pickle=False) as z:
            pose, camera, K = z['model_from_local'][0], z['model_to_camera'], z['K']
        with np.load(frame_path, allow_pickle=False) as z:
            ground_K, ground_w2l = z['K'], z['w2l']
        sources[str(obs_path)], sources[str(frame_path)] = helper.sha(obs_path), helper.sha(frame_path)
        ground, valid_ground = ns['rays'](surface, ground_K, ground_w2l)
        ground_z = (ground.reshape(-1, 3) @ ground_w2l[:3, :3].T+ground_w2l[:3, 3])[:, 2].reshape(320, 480)
        camera_to_actor = np.linalg.inv(pose) @ np.linalg.inv(camera)
        direction = (np.c_[pixels+.5, np.ones(len(pixels))] @ np.linalg.inv(K).T) @ camera_to_actor[:3, :3].T * scale
        hits = ns['opaque_hits'](vertices, triangles, camera_to_actor[:3, 3], direction, alpha, (.01, .5))
        labels = np.zeros((2, 320, 480), np.uint8)
        gaps = np.full((2, 320, 480), np.nan)
        chart_z = np.full((2, 320, 480), np.nan)
        for mi in range(2):
            chart_z[mi, selected] = hits[mi]
            gaps[mi, selected] = ground_z[selected]-hits[mi]
            valid = selected & valid_ground & np.isfinite(gaps[mi])
            labels[mi, valid & (gaps[mi] > .05)] = 1
            labels[mi, valid & (gaps[mi] < -.05)] = 2
            labels[mi, valid & (np.abs(gaps[mi]) <= .05)] = 3
        state_info = {}
        for state in ('keep', 'delete'):
            state_info[state] = dict(C2_positive_pixels=int(guards[state].sum()),
                classifications=helper.counts(labels[0], guards[state]),
                C2_guard_pixels_allowed_to_remove=int((guards[state] & (labels[0] == 2)).sum()))
        path = out/f'{raw:03d}_first_hit_maps.npz'
        np.savez_compressed(path, labels=labels, ground_valid=valid_ground,
            ground_camera_z_m=ground_z, chart_camera_z_m=chart_z, ground_minus_chart_m=gaps,
            selected_C2_positive_keep=guards['keep'], selected_C2_positive_delete=guards['delete'],
            permitted_ground_first_keep=guards['keep'] & (labels[0] == 2),
            permitted_ground_first_delete=guards['delete'] & (labels[0] == 2))
        row = dict(raw=raw, union_C2_positive_pixels=int(selected.sum()), states=state_info,
            old_matte001_vs05_disagreement_pixels=int((selected & (labels[0] != labels[1])).sum()),
            map_path=str(path), map_sha256=helper.sha(path), seconds=time.time()-t)
        if raw == 128:
            prior = R/'review/chart_firsthit128_model_only/model_only_first_hit_maps.npz'
            with np.load(prior, allow_pickle=False) as z:
                assert np.array_equal(labels[:, guards['delete']], z['labels'][:, guards['delete']])
            row['prior_128_DELETE_labels_exact'] = True
        rows.append(row)
        print(json.dumps(row), flush=True)
    old128 = json.loads((RED/'raw128_contact_audit_v2.json').read_text())
    old130_path = R0/'review/asset_ground_conflict/canonical_chart_35ray_first_hit_probe.json'
    old130 = json.loads(old130_path.read_text())
    prior = R/'review/chart_firsthit128_model_only/classification.json'
    compatibility128 = json.loads(prior.read_text())['old_diagnostic_comparisons']
    report = dict(status='PASS_CPU_FOUR_FRAME_FIRST_HIT_MAPS_NO_RENDER_VARIANT', raw_ids=RAWS,
        states=['keep', 'delete'], rows=rows, selector='baseline C2 alphaT>0 independently per state',
        source_input_sha256=sources, registered_guard_update='C2_positive AND NOT valid_ground_first(gap<-.05,matte.01)',
        unchanged_guards=['all semantic5 positive model alphaT', 'all KEEP37 positive model alphaT'],
        unknown_contact_chart_first_all_remain_protected=True,
        diagnostic_only_old_matte05=True, no_threshold_tuning=True,
        old_71_ray_comparison=dict(raw128_fixed36=compatibility128,
            raw130_fixed35_summary=old130['summary'], raw130_proof_sha256=helper.sha(old130_path),
            raw128_proof_sha256=helper.sha(RED/'raw128_contact_audit_v2.json')),
        current_method_does_not_read_query_RGB_or_GT_to_choose_guard=True,
        no_rendering_policy_applied=True, open_chart_not_ground_truth=True,
        renderer_forwards=0, model_updates=0, GPU_used=False, seconds=time.time()-start)
    (out/'classification4.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
