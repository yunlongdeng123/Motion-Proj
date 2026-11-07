"""Lossless CPU export of the single certified K LP point; never re-solves."""
from pathlib import Path
import copy
import hashlib
import json
import shutil

import numpy as np

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
NEW = B/'scene008_v44_a5_c49_rgb_20261107_r1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    lp = R/'J4_road_parent_operator/K_single_v1'
    point = lp/'K_candidate_parent_RGB.npz'
    certificate = lp/'result.json'
    assert sha(point) == '41a570ba2f5c7544b8ddeaef730142368f660234a3b53721d88570b0dea1903a'
    cert = json.loads(certificate.read_text())
    assert cert['status'] == 'FEASIBLE_CPU_CERTIFIED_PENDING_REAL8F'
    assert cert['actual_solver_calls'] == 1 and cert['solver_success']
    assert cert['schema'] == 'K_PARENT_RGB_SINGLE_LP_CPU_RESULT_V1'
    assert len(cert['float32_bins']) == 106 and all(x['original_pass'] for x in cert['float32_bins'])
    assert cert['float64_certificate_pass'] and cert['float32_certificate_pass']
    assert cert['fixed7979_bitwise_unchanged']
    scene = NEW/'formal_export2000_r1/scene_manifest.json'
    parent = json.loads(scene.read_text())
    asset_path = scene.parent/parent['new_road']['file']
    assert sha(asset_path) == '1287d260b80196bf7f3f4555b77f4d4879ba6bc7e56c63dbe6ac098e66a02952'
    with np.load(asset_path, allow_pickle=False) as z:
        old = {key: z[key].copy() for key in z.files}
    with np.load(point, allow_pickle=False) as z:
        colours, ids, eligible, delta = (z[key].copy() for key in
            ('full_parent_RGB', 'eligible_parent_ids', 'eligible_parent_RGB', 'delta_from_formal2k'))
        assert np.array_equal(colours, z['parent_RGB'])
    matrix = np.load(R/'J4_road_parent_operator/road_parent_operator.npz', allow_pickle=False)
    assert sha(R/'J4_road_parent_operator/road_parent_operator.npz') == cert['operator_npz_sha256']
    assert np.array_equal(ids, matrix['eligible_parent_ids']) and ids.shape == (2119,)
    assert colours.dtype == np.float32 and colours.shape == (10098, 3)
    assert np.isfinite(colours).all() and colours.min() >= 0 and colours.max() <= 1
    assert np.array_equal(eligible, colours[ids])
    assert np.array_equal(delta, eligible.astype(np.float64)-old['parent_render_colours_srgb'][ids].astype(np.float64))
    fixed = np.ones(10098, bool)
    fixed[ids] = False
    assert np.array_equal(colours[fixed], old['parent_render_colours_srgb'][fixed])
    new = {key: value.copy() for key, value in old.items()}
    new['parent_render_colours_srgb'] = colours
    new['render_colours_srgb'] = colours[old['child_to_parent_ids']]
    out = R/'J4_road_parent_operator/K_candidate_material_v1'
    out.mkdir(exist_ok=False)
    target = out/'new_road_K_single_LP.npz'
    np.savez_compressed(target, **new)
    allowed, changed = {'parent_render_colours_srgb', 'render_colours_srgb'}, []
    with np.load(target, allow_pickle=False) as z:
        assert set(z.files) == set(old)
        for key in old:
            assert z[key].dtype == old[key].dtype and z[key].shape == old[key].shape
            if z[key].tobytes() != old[key].tobytes():
                changed.append(key)
            if key not in allowed:
                assert z[key].tobytes() == old[key].tobytes(), key
            else:
                assert z[key].tobytes() == new[key].tobytes()
    manifest = copy.deepcopy(parent)
    for item in parent['replay_sources'].values():
        source = scene.parent/item['file']
        assert sha(source) == item['sha256']
        shutil.copyfile(source, out/item['file'])
    provenance = out/'provenance'
    provenance.mkdir()
    for source, name in ((scene, 'formal2k_scene_manifest.json'),
            (certificate, 'single_LP_result.json'), (point, 'K_candidate_parent_RGB.npz'),
            (R/'J4_road_parent_operator/operator_manifest.json', 'operator_manifest.json'),
            (Path(__file__), 'export_K_parent_material.py')):
        shutil.copyfile(source, provenance/name)
    manifest['new_road'].update(file=target.name, sha256=sha(target),
        material_variant='K_single_LP_2119_parent_RGB_on_formal2k_geometry')
    details = dict(formal2k_parent_asset_sha256=sha(asset_path),
        formal2k_scene_manifest_sha256=sha(scene),
        candidate_RGB_file='provenance/K_candidate_parent_RGB.npz', candidate_RGB_sha256=sha(point),
        LP_certificate_file='provenance/single_LP_result.json', LP_certificate_sha256=sha(certificate),
        LP_source_sha256=cert['source_sha256'], LP_protocol_sha256=cert['protocol_sha256'],
        operator_manifest_sha256=cert['operator_manifest_sha256'], operator_npz_sha256=cert['operator_npz_sha256'],
        CPU_solver_calls=1, Adam_updates=0, original_parameter_updates=0,
        eligible_parent_rows=2119, frozen_parent_rows=7979,
        fixed7979_parent_RGB_bitwise_unchanged=True,
        actual_changed_array_keys=changed, only_allowed_changed_arrays=sorted(allowed),
        all_other_array_dtype_shape_bytes_exact=True, no_resolve_no_clip_no_reprojection=True,
        actual_colour_point='exact float32 certified NPZ, without numerical modification',
        actual_native_quality_validation='PENDING_ROOT_PREAUTHORIZED8F_ON_FIXED_H2_MASKS',
        adopted=False)
    manifest['lineage'] = dict(formal2k_parent_manifest_file='provenance/formal2k_scene_manifest.json',
        formal2k_parent_manifest_sha256=sha(scene), formal2k_parent_asset_path=str(asset_path),
        K_single_LP=details)
    manifest['material_writeback'].update(K_single_CPU_LP_calls=1, K_Adam_updates=0,
        K_original_parameters_updated=0, current_parent_colour_key='parent_render_colours_srgb',
        current_child_colour_key='render_colours_srgb')
    manifest['known_unresolved'] += [
        'K LP certifies the original six pooled and106 local seam inequalities only on four BUILD views; all photometric protections and images require actual native replay.',
        'Existing H2 cached visibility must be applied; this scene asset alone does not include that per-view child mask.',
        'No I coverage or H3 guard modification is included.']
    manifest.update(exporter_sha256=sha(__file__), GPU_parity_state_at_export='K_REAL8F_PENDING',
                    GPU_parity_receipt='native_validation_pending.json', adoption_claim=False, human_verdict=None)
    write(out/'scene_manifest.json', manifest)
    write(out/'native_validation_pending.json', dict(status='NOT_RUN_BY_CPU_EXPORTER', native_forwards=0,
        note='Use fixed original H2 cache; root separately authorizes exactly8 native forwards.'))
    write(out/'export_receipt.json', dict(status='PASS_CPU_EXACT_SINGLE_LP_MATERIAL_EXPORT_PENDING_NATIVE8F',
        scene_manifest_sha256=sha(out/'scene_manifest.json'), asset_sha256=sha(target),
        source_exporter_sha256=sha(__file__), **details))
    print(json.dumps(dict(scene_manifest=str(out/'scene_manifest.json'),
        scene_manifest_sha256=sha(out/'scene_manifest.json'), asset_sha256=sha(target),
        changed_arrays=changed, frozen7979_exact=True, native_forwards=0), indent=2))


if __name__ == '__main__':
    main()
