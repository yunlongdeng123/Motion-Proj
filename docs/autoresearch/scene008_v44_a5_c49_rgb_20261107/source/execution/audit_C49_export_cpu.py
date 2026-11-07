"""Independent CPU audit for a C49 render-asset export and fixed-H2 reuse."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B / 'scene008_v44_a5_c49_rgb_20261107_r1'
H2 = B / 'scene008_v44_a5_road_fusion_integration_20261107_r1/H4_model_foreground_culling_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(scene_manifest, evaluation_table, output):
    scene_manifest, evaluation_table, output = map(Path, (scene_manifest, evaluation_table, output))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(scene_manifest.read_text())
    old_manifest_path = scene_manifest.parent / manifest['lineage']['source_C_manifest_file']
    old_manifest = json.loads(old_manifest_path.read_text())
    old_asset_path = Path(manifest['lineage']['source_C_asset_path'])
    new_asset_path = scene_manifest.parent / manifest['new_road']['file']
    assert sha(old_asset_path) == old_manifest['new_road']['sha256']
    assert sha(new_asset_path) == manifest['new_road']['sha256']
    qualification = scene_manifest.parent / manifest['lineage']['C49_export']['qualification_file']
    qm = json.loads(qualification.read_text())
    qpath = scene_manifest.parent / manifest['lineage']['C49_export']['selection_file']
    assert sha(qualification) == manifest['lineage']['C49_export']['qualification_sha256']
    assert sha(qpath) == qm['selection_sha256']
    old, new, q, evaluation = (np.load(p, allow_pickle=False) for p in
                               (old_asset_path, new_asset_path, qpath, evaluation_table))
    assert set(new.files) == set(old.files)
    changed, digest = [], hashlib.sha256()
    for key in old.files:
        assert new[key].shape == old[key].shape and new[key].dtype == old[key].dtype
        if new[key].tobytes() != old[key].tobytes():
            changed.append(key)
        if key not in ('render_colours_srgb', 'parent_render_colours_srgb'):
            assert new[key].tobytes() == old[key].tobytes(), key
            for part in (key, str(new[key].dtype), str(new[key].shape)):
                digest.update(part.encode()+b'\0')
            digest.update(new[key].tobytes())
    assert set(changed).issubset({'render_colours_srgb', 'parent_render_colours_srgb'})
    assert np.array_equal(q['eligible_parent_ids'], np.flatnonzero(q['parent_mass'] > 0))
    assert np.array_equal(evaluation['eligible_parent_ids'], q['eligible_parent_ids'])
    assert np.array_equal(evaluation['parent_RGB'], new['parent_render_colours_srgb'])
    zero = q['parent_mass'] == 0
    assert np.array_equal(new['parent_render_colours_srgb'][zero], old['parent_render_colours_srgb'][zero])
    assert np.array_equal(new['render_colours_srgb'],
                          new['parent_render_colours_srgb'][new['child_to_parent_ids']])
    for item in manifest['replay_sources'].values():
        assert sha(scene_manifest.parent / item['file']) == item['sha256']
    assert manifest['renderer_policy'] == old_manifest['renderer_policy']
    assert manifest['frozen_v44_binding'] == old_manifest['frozen_v44_binding']
    hr = json.loads((H2/'receipt.json').read_text())
    assert hr['runtime_receipt']['native_road_receipt']['road_asset_sha256'] == sha(old_asset_path)
    masks = []
    for raw in (94, 105, 128, 137):
        for state in ('keep', 'delete'):
            p = H2 / f'frames/{raw:03d}/{state}_visibility_mask.npz'
            z = np.load(p, allow_pickle=False)
            assert z['selected_road_child_rows'].shape == (40392,) and z['selected_road_child_rows'].dtype == bool
            assert z['model_foreground_guard'].shape == (320, 480) and z['model_foreground_guard'].dtype == bool
            masks.append(dict(raw=raw, state=state, path=str(p), sha256=sha(p),
                              active_rows=int(z['selected_road_child_rows'].sum())))
    export_info = manifest['lineage']['C49_export']
    report = dict(status='PASS_INDEPENDENT_CPU_C49_EXPORT_AND_H2_GEOMETRY_REUSE_PRECONDITION',
        step=export_info['checkpoint_step'], stage=export_info['stage'], scene_manifest=str(scene_manifest),
        scene_manifest_sha256=sha(scene_manifest), new_asset_sha256=sha(new_asset_path),
        source_C_asset_sha256=sha(old_asset_path), changed_array_keys=changed,
        all_non_render_RGB_arrays_dtype_shape_bytes_exact=True, non_RGB_array_digest=digest.hexdigest(),
        zero_mass_parent_count=int(zero.sum()), zero_mass_parentRGB_exact=True,
        full_parentRGB_equal_saved_public_node_table=True, child_inheritance_exact=True,
        renderer_policy_and_frozen_V44_binding_exact=True,
        H2_cached_receipt_sha256=sha(H2/'receipt.json'), H2_contract_sha256=hr['contract_sha256'],
        H2_cached_visibility_inputs=masks, new_H_material_contract_required=True,
        reuse_scope='Original eight H2 model-derived masks only, conditioned on unchanged source scene/geometry and renderer.',
        GPU_replay_not_performed_by_this_audit=True, audit_added_native_forwards=0,
        audit_added_model_updates=0, auditor_source_sha256=sha(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'H2_cached_visibility_inputs'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-manifest', required=True, type=Path)
    parser.add_argument('--evaluation-table', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.scene_manifest, args.evaluation_table, args.output)
