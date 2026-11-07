"""Two pre-registered DELETE pixels: exact native full-parent contribution.

Posthoc diagnostic only: observations describe the points and never choose or
modify query masks/parameters. Exactly2 joint forwards +2 dummy backwards.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
NEW = B/'scene008_v44_a5_c49_rgb_20261107_r1'
SEM = B/'scene008_v44_full5_integration_20261107_r1'
REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CONTRACT = R/'contract/white_point_parent_diagnostic_v1.json'
OUT = R/'white_point_parent_diagnostic_v1'
POINTS = [(128, 301, 221), (137, 300, 227)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def asset(manifest_path):
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    p = manifest_path.parent/m['new_road']['file']
    assert sha(p) == m['new_road']['sha256']
    with np.load(p, allow_pickle=False) as z:
        a = {key: z[key].copy() for key in z.files}
    return m, a, p


def main():
    c = json.loads(CONTRACT.read_text())
    assert c['status'] == 'SEALED_BEFORE_TWO_POINT_GPU'
    assert c['source_sha256'] == sha(__file__) and c['planned_forwards'] == c['planned_backwards'] == 2
    OUT.mkdir(exist_ok=False)
    started = time.time()
    sys.path[:0] = [str(R/'source'), str(REPO), str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
    import torch
    from v44_runtime import V44Runtime
    from foreground_road_culling_H4_v2 import ForegroundRoadCulling
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration, C_MANIFEST
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import baseline_appearance_binding
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    manifests = dict(Cwarm=C_MANIFEST, formal2k=NEW/'formal_export2000_r1/scene_manifest.json',
        J=R/'J4_road_parent_operator/J_candidate_material_v1/scene_manifest.json',
        K=R/'J4_road_parent_operator/K_candidate_material_v1/scene_manifest.json')
    materials = {name: asset(path) for name, path in manifests.items()}
    tables = {name: values[1]['parent_render_colours_srgb'] for name, values in materials.items()}
    qualified = NEW/'parent_visibility49/qualification_manifest.json'
    qm = json.loads(qualified.read_text())
    selection = qualified.parent/qm['selection_file']
    assert sha(selection) == qm['selection_sha256']
    with np.load(selection) as z:
        ids, fitmass = z['eligible_parent_ids'].copy(), z['parent_mass'].copy()
    eligible = np.zeros(10098, bool)
    eligible[ids] = True
    assert np.array_equal(ids, np.flatnonzero(fitmass > 0)) and eligible.sum() == 2119
    a = materials['formal2k'][1]
    mapping = a['child_to_parent_ids']
    assert np.array_equal(mapping, np.repeat(np.arange(10098), 4))
    base = V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
    sem = SemanticActor5V44Integration(base,
        semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',
        full5_transform_path=R/'contract/full5_transform_matrix.txt')
    road = RoadFusionV44Integration(sem, manifests['formal2k'])
    h2 = ForegroundRoadCulling(road, max_iterations=64)
    h2root = R/'H4_model_foreground_culling_v2'
    for item in json.loads((h2root/'receipt.json').read_text())['records']:
        raw, state = item['raw'], item['state']
        with np.load(h2root/f'frames/{raw:03d}/{state}_visibility_mask.npz') as z:
            mask, guard = z['selected_road_child_rows'], z['model_foreground_guard']
        record = item['meta']['foreground_road_culling']
        assert record['converged']
        h2.cache[(raw, state == 'delete')] = dict(mask=torch.as_tensor(mask, device='cuda'),
            guard=torch.as_tensor(guard, device='cuda'), record=record, baseline=None)
    binding = baseline_appearance_binding(sem)
    inputs = [CONTRACT, Path(__file__), qualified, selection, h2root/'receipt.json']
    for name, (m, _, p) in materials.items():
        inputs.extend([Path(manifests[name]), p])
    inputs_sha = {str(p): sha(p) for p in inputs}
    write(OUT/'protocol.json', dict(contract_sha256=sha(CONTRACT), source_inputs_sha256=inputs_sha,
        points=POINTS, query_model='fixedH2+formal2k, noI/H3', actual_model_updates=0))
    calls = backs = 0
    records = []
    for raw, x, y in POINTS:
        scene = h2.scene(raw, delete=True)
        rows = {key: value.detach() if torch.is_tensor(value) else value for key, value in scene['rows'].items()}
        offset = int(scene['meta']['latest_semantic_prefix_rows'])
        assert len(rows['mask']) == offset+40392 and not bool((rows['features'][:, 16] != 0).any())
        probe = torch.ones((10098, 1), device='cuda', dtype=torch.float32, requires_grad=True)
        feature = torch.cat((probe.new_zeros((offset, 1)), probe.index_select(0, road.parent_indices)), 0)
        rows['features'] = torch.cat((rows['features'][:, :16], feature, rows['features'][:, 17:]), 1)
        assert all(not v.requires_grad for key, v in rows.items() if torch.is_tensor(v) and key != 'features')
        native = base.raster(scene['cam'], base.pipe, 0, rows, scene['bg'])
        calls += 1
        write(OUT/'kernel_progress.json', dict(raw=raw, native_forwards=calls, dummy_backwards=backs, optimizations=0))
        alpha = native['alpha'].detach().cpu().numpy()[0]
        ds = native['depth_sum'].detach().cpu().numpy()[0]
        values = dict(rgb=(native['rgb']+(1-native['alpha'])*scene['env'].detach()).detach().cpu().numpy(),
            alpha=alpha, depth_sum=ds, depth_m=ds/np.maximum(alpha, 1e-8)/base.scale,
            group_alpha_T=native['extra'][:8].detach().cpu().numpy())
        with np.load(NEW/f'H2_material2000_retry1/frames/{raw}/delete_float.npz') as z:
            for key, value in values.items():
                assert np.array_equal(value, z[key]), (raw, key)
            original_new_rgb = z['newroad_rgb_T'][:, y, x].copy()
        read = native['extra'][14]
        ones_error = float((read-native['extra'][7]).abs().max().detach())
        assert ones_error <= 1e-6
        mass = torch.autograd.grad(read[y, x], probe)[0].detach().cpu().numpy()[:, 0]
        backs += 1
        write(OUT/'kernel_progress.json', dict(raw=raw, native_forwards=calls, dummy_backwards=backs, optimizations=0))
        assert mass.shape == (10098,) and np.isfinite(mass).all() and mass.min() >= 0
        weight = float(values['group_alpha_T'][7, y, x])
        error = abs(float(mass.sum(dtype=np.float64))-weight)
        assert error <= 1e-6 and error/max(weight, 1e-30) <= 1e-5
        positive = np.flatnonzero(mass > 0)
        order = positive[np.argsort(-mass[positive], kind='stable')]
        reconstructions = {}
        version_caches = dict(Cwarm=h2root, formal2k=NEW/'H2_material2000_retry1',
            J=R/'J4_road_parent_operator/lp_candidate_render_v1',
            K=R/'J4_road_parent_operator/K_candidate_render_v1')
        for name, colours in tables.items():
            reconstructed = mass.astype(np.float64) @ colours.astype(np.float64)
            with np.load(version_caches[name]/f'frames/{raw}/delete_float.npz') as z:
                reference = z['newroad_rgb_T'][:, y, x]
            delta = float(np.abs(reconstructed-reference).max())
            assert delta <= 1e-6, (raw, name, delta)
            reconstructions[name] = dict(reconstructed_newroad_RGB_T=reconstructed.tolist(),
                cached_newroad_RGB_T=reference.tolist(), max_abs_error=delta)
        observation = B/f'scene008_instances_20261105_r1/observations/frames/{raw}.npz'
        with np.load(observation) as z:
            roles = dict(actor_label=int(z['actor_labels'][y, x]), visible6=bool(z['visible'][0, y, x]),
                visible37=bool(z['visible'][1, y, x]), road=bool(z['road'][y, x]),
                road_edit=bool(z['road_edit'][y, x]), protect=bool(z['protect'][y, x]))
        target = next(item for item in qm['targets'] if item['raw'] == raw)
        assert sha(target['path']) == target['sha256']
        with np.load(target['path']) as z:
            target_roles = dict(fit_mask=bool(z['fit_mask'][y, x]), protect_mask=bool(z['protect_mask'][y, x]))
        parents = []
        warm = materials['Cwarm'][1]
        for parent in order:
            parent = int(parent)
            info = dict(parent_id=parent, visible_alphaT=float(mass[parent]),
                fraction_of_newroad_alphaT=float(mass[parent]/weight), qualified_C49=bool(eligible[parent]),
                C49_all49_fit_mass=float(fitmass[parent]), parent_anchor_local_xyz=a['parent_local_xyz'][parent].tolist(),
                parent_key=a['parent_keys'][parent].tolist(), exact_child_rows=list(range(4*parent, 4*parent+4)),
                H2_active_children=h2.cache[(raw, True)]['mask'][4*parent:4*parent+4].cpu().tolist(),
                colours={name: colours[parent].tolist() for name, colours in tables.items()},
                RGB_T_contributions={name: (float(mass[parent])*colours[parent].astype(np.float64)).tolist() for name, colours in tables.items()})
            for key in ('parent_reference_srgb', 'parent_source_valid_count', 'parent_source_valid_count49',
                        'parent_source_agree_count', 'parent_appearance_provenance', 'writeback_eligible_parents'):
                if key in warm:
                    info['original_C_'+key] = warm[key][parent].tolist()
            if 'parent_source_valid' in warm and 'parent_source_raw' in warm:
                valid = warm['parent_source_valid'][:, parent]
                info['original_valid_source_raw_ids'] = warm['parent_source_raw'][valid].tolist()
            parents.append(info)
        p = OUT/f'{raw}_parent_mass.npz'
        np.savez_compressed(p, parent_alphaT=mass, eligible_C49=eligible,
            C49_all49_fit_mass=fitmass, positive_parent_ids=order)
        records.append(dict(raw=raw, state='delete', pixel_xy=[x, y], roles=roles,
            target_roles=target_roles, full_newroad_weight=weight, parent_mass_sum_error=error,
            ones_equals_newroad_alphaT_maxabs=ones_error, positive_parents=len(order),
            eligible2119_contribution=float(mass[eligible].sum(dtype=np.float64)),
            frozen7979_contribution=float(mass[~eligible].sum(dtype=np.float64)),
            material_reconstructions=reconstructions, parents_in_mass_order=parents,
            mass_file=p.name, mass_sha256=sha(p), GT_roles_are_diagnostic_only=True))
        print(json.dumps(dict(raw=raw, positive=len(order), eligible_mass=records[-1]['eligible2119_contribution'],
            frozen_mass=records[-1]['frozen7979_contribution'], top_ids=order[:8].tolist())), flush=True)
        del native, probe, feature, rows, scene, read
    road.assert_unchanged(full=True)
    assert binding == baseline_appearance_binding(sem)
    for path, expected in inputs_sha.items():
        assert sha(path) == expected
    assert calls == backs == 2 and h2.actual_dummy_backwards == 0
    write(OUT/'receipt.json', dict(status='PASS_TWO_POINT_FULL_PARENT_VISIBILITY_DIAGNOSTIC_NO_OPTIMIZATION',
        actual_native_forwards=calls, actual_dummy_backwards=backs, original_parameter_updates=0,
        optimizations=0, records=records, all_model_and_source_inputs_unchanged=True,
        source_sha256=sha(__file__), contract_sha256=sha(CONTRACT), seconds=time.time()-started,
        source_parent_RGB_eligibility_is_not_a_query_mask=True,
        new_query_policy=False, automatic_fit=False, automatic49=False, shutdown=False))


if __name__ == '__main__':
    main()
