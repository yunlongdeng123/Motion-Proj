"""Independent CPU-only numerical review of HR-to-512 frozen-asset I/O.

Run from stdin on AutoDL with CUDA_VISIBLE_DEVICES empty; writes nothing there.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.sparse import load_npz

RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
OLD = RUN.parent / 'scene008_a6_asset_controls_20261105'
SELECTED = RUN / 'candidates_raw106/prompt2_seed3/composed.png'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def maxabs(x):
    return float(np.abs(x).max()) if np.size(x) else 0.


def sparse_maxabs(x):
    x = x.tocsr(); x.eliminate_zeros()
    return maxabs(x.data)


def binary(path):
    x = np.asarray(Image.open(path))
    assert x.ndim == 2 and np.isin(x, [0, 255]).all()
    return x != 0


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def zero_columns(A):
    A = A.tocsr(copy=True); A.eliminate_zeros()
    assert np.isfinite(A.data).all() and (A.data >= 0).all()
    return np.asarray(A.getnnz(axis=0) == 0).ravel()


old = bank(OLD / 'm2/complete_a6.npz')
base_path = RUN / 'highres/formal/complete_a6.npz'
base = bank(base_path)
build = load_npz(OLD / 'm2/design.npz')
hrbuild = load_npz(RUN / 'highres/design.npz')
oldzero, hrzero = zero_columns(build), zero_columns(hrbuild)
safe = oldzero & hrzero
stored_hr = bank(RUN / 'highres/inputs.npz')['unknown'].astype(bool)
assert np.array_equal(hrzero, stored_hr)
assert same_bytes(old['geometry_hash'], base['geometry_hash'])
geometry_fields = ('xyz', 'normal', 'sigma', 'opacity', 'thickness', 'box',
                   'knn_spacing', 'normal_radius', 'normal_confidence',
                   'generated', 'conditioning', 'normal_valid', 'out_of_box', 'conditioning_held',
                   'source_index', 'conditioning_observation_index', 'conditioning_raw_frame', 'conditioning_cloud_row')
assert all(same_bytes(old[k], base[k]) for k in geometry_fields)
assert base['canonical_sh'].shape == (18432, 16, 3)
assert (base['canonical_sh'][:, 1:] == 0).all()

frames = []
for raw in (106, 114, 129):
    packet, lowroot = RUN / f'inputs_hr/raw{raw}', RUN / f'inputs/raw{raw}'
    t, lowt = bank(packet/'transform.npz'), bank(lowroot/'transform.npz')
    s = bank(packet/'operator_samples.npz')
    xy = s['pixel_xy'].astype(np.float64)
    H = np.array([[.25, 0, -.375], [0, .25, -.375], [0, 0, 1]], np.float64)
    expected_T = lowt['original_to_generated_pixel_center'] @ H
    T = t['highres_to_generated_pixel_center']
    calculated = np.c_[xy, np.ones(len(xy))] @ expected_T.T
    mapping_error = maxabs(s['generated_xy']-calculated[:, :2])
    assert mapping_error < 1e-10 and maxabs(T-expected_T) < 1e-10
    assert maxabs(t['hr_to_low_pixel_center']-H) == 0
    assert maxabs(s['lowres_xy']-((xy+.5)/4-.5)) == 0
    assert maxabs(T @ t['generated_to_highres_pixel_center']-np.eye(3)) < 1e-10
    assert np.array_equal(xy, np.c_[s['pixels']%1920, s['pixels']//1920])
    # A missing -.375 is a measurable translation, not a harmless convention.
    naive = np.c_[xy/4, np.ones(len(xy))] @ lowt['original_to_generated_pixel_center'].T
    naive_error = maxabs(naive[:, :2]-s['generated_xy'])
    ij = np.floor(s['generated_xy']+.5).astype(int)
    assert (ij >= 0).all() and (ij < 512).all()
    known512 = binary(lowroot/'known_mask_512.png')
    inpaint512 = binary(lowroot/'inpaint_mask_512.png')
    object512 = binary(lowroot/'object_mask_512.png')
    assert not (known512 & inpaint512).any()
    assert np.array_equal(s['generated_known_nearest'], known512[ij[:, 1], ij[:, 0]])
    assert np.array_equal(s['generated_inpaint_nearest'], inpaint512[ij[:, 1], ij[:, 0]])
    A = load_npz(packet/'design_A_normalized.npz').astype(np.float64)
    W = load_npz(packet/'design_W_premultiplied.npz').astype(np.float64)
    alpha = s['alpha'].astype(np.float64)
    normalized_error = sparse_maxabs(W.multiply(1/alpha[:, None])-A)
    sum_error = maxabs(np.asarray(A.sum(1)).ravel()-1)
    alpha_error = maxabs(np.asarray(W.sum(1)).ravel()-alpha)
    straight_error = maxabs(A @ old['rgb'].astype(np.float64)-s['initial_m2_rgb'])
    premul_error = maxabs(W @ old['rgb'].astype(np.float64)-s['initial_m2_rgb']*alpha[:, None])
    assert normalized_error < 1e-6 and sum_error < 1e-5 and alpha_error < 5e-6
    assert straight_error < 1e-5 and premul_error < 1e-5
    fit = (alpha >= .9) & s['generated_inpaint_nearest'] & ~s['known'].astype(bool) & s['inpaint'].astype(bool)
    safe_fraction = np.asarray(A[:, safe].sum(1)).ravel()
    admitted = fit & (safe_fraction >= .25)
    record = {
        'raw': raw, 'pixel_rows': len(xy), 'hr_shape': [1280, 1920], 'candidate_shape': [512, 512],
        'pixel_center_matrix_maxabs_error': maxabs(T-expected_T),
        'generated_xy_maxabs_error': mapping_error,
        'inverse_maxabs_error': maxabs(T @ t['generated_to_highres_pixel_center']-np.eye(3)),
        'naive_hr_divide_by_4_error_candidate_pixels': naive_error,
        'generated_xy_range': [s['generated_xy'].min(0).tolist(), s['generated_xy'].max(0).tolist()],
        'normalized_A_vs_W_div_alpha_maxabs': normalized_error,
        'A_row_sum_maxabs': sum_error, 'W_sum_vs_alpha_maxabs': alpha_error,
        'A_rgb_vs_straight_maxabs': straight_error,
        'W_rgb_vs_premultiplied_maxabs': premul_error,
        'eligible_generated_hr_rows_before_fraction': int(fit.sum()),
        'eligible_generated_hr_rows_safe_fraction_ge_025': int(admitted.sum()),
        'generated_rows_touching_known_hr_mask': int((fit & s['known'].astype(bool)).sum()),
        'actual_HR_RGB_available_as_observation_only': True,
        'packet_hashes': {name: sha(packet/name) for name in ('transform.npz', 'operator_samples.npz', 'design_A_normalized.npz', 'design_W_premultiplied.npz')},
    }
    if raw == 106:
        rgb = np.asarray(Image.open(SELECTED)).astype(np.float64)/255
        target = np.column_stack([map_coordinates(rgb[..., c], [s['generated_xy'][:, 1], s['generated_xy'][:, 0]],
                        order=1, mode='constant', cval=0, prefilter=False) for c in range(3)])
        assert target.shape == (len(xy), 3) and np.isfinite(target).all()
        record['selected_candidate_bilinear_targets'] = {
            'finite': True, 'shape': list(target.shape),
            'admitted_target_minmax': [float(target[admitted].min()), float(target[admitted].max())],
            'target_is_straight_srgb': True, 'divide_candidate_by_alpha': False,
            'sampling': 'bilinear generated[y,x] at exact recorded generated_xy; no resize',
        }
    frames.append(record)

source_paths = [RUN/'source/fit_adapter/adapter.py', RUN/'source/fit_adapter/bridge_packet.py']


def review_adapter_output(branch):
    dest = RUN/'generated_fit'/branch
    prep = dest.with_name(dest.name+'_inputs')
    receipt = json.loads((dest/'receipt.json').read_text())
    bridge = json.loads((prep/'bridge_receipt.json').read_text())
    replay = bank(dest/'replay.npz')
    output = bank(dest/'complete_a6.npz')
    is_hr = bridge['native_full_image_hw'] == [1280, 1920]
    asset = base if is_hr else old
    permission = safe if is_hr else oldzero
    packet = RUN/('inputs_hr/raw106' if is_hr else 'inputs/raw106')
    sample = bank(packet/'operator_samples.npz')
    t = bank(packet/'transform.npz')
    T = t.get('highres_to_generated_pixel_center', t.get('original_to_generated_pixel_center'))
    xy = np.c_[sample['pixel_xy'], np.ones(len(sample['pixels']))] @ T.T
    nearest = np.floor(xy[:, :2]+.5).astype(int)
    inpaint = binary(RUN/'inputs/raw106/inpaint_mask_512.png')
    foreground = binary(RUN/'inputs/raw106/object_mask_512.png')
    nearest_inpaint = inpaint[nearest[:, 1], nearest[:, 0]]
    use = foreground[nearest[:, 1], nearest[:, 0]] & nearest_inpaint & ~sample['known'].astype(bool) & sample['inpaint'].astype(bool)
    rays = bank(prep/'operator/rays.npz')
    assert np.array_equal(rays['full_pixels'], sample['pixels'][use])
    assert np.array_equal(rays['alpha'], sample['alpha'][use])
    W = load_npz(packet/'design_W_premultiplied.npz').astype(np.float64)
    Wbridge = load_npz(prep/'operator/weights.npz').astype(np.float64)
    Werror = sparse_maxabs(Wbridge-W[use])
    assert Werror == 0
    source_candidates = [Path(p) for p in bridge['source_SHA256']
                         if str(p).endswith('.png') and 'mask' not in Path(p).name]
    assert len(source_candidates) == 1
    candidate_path = source_candidates[0]
    assert all(sha(p) == expected for p, expected in bridge['source_SHA256'].items())
    candidate = np.asarray(Image.open(candidate_path)).astype(np.float64)/255
    expected_target = np.column_stack([map_coordinates(candidate[..., c], [xy[:, 1], xy[:, 0]],
                          order=1, mode='constant', cval=0, prefilter=False) for c in range(3)])
    actual_target = bank(prep/'candidate_on_native_crop.npz')['RGB_srgb'].reshape(-1, 3)[rays['crop_pixels']]
    target_error = maxabs(actual_target-expected_target[use])
    assert target_error < 4e-8
    changed_rgb = np.any(output['rgb'] != asset['rgb'], axis=1)
    changed_sh = np.any(output['canonical_sh'] != asset['canonical_sh'], axis=(1, 2))
    changed_flags = output['colour_inferred'] != asset['colour_inferred']
    changed = changed_rgb | changed_sh | changed_flags
    assert not (changed & ~permission).any()
    protected = ~permission
    assert all(same_bytes(asset[k][protected], output[k][protected])
               for k in ('canonical_sh', 'rgb', 'colour_inferred'))
    appearance = {'canonical_sh', 'rgb', 'colour_inferred', 'metadata_json'}
    assert set(asset) == set(output)
    assert all(same_bytes(asset[k], output[k]) for k in set(asset)-appearance)
    original_prediction_error = maxabs(build @ asset['rgb']-build @ output['rgb'])
    hr_prediction_error = maxabs(hrbuild @ asset['rgb']-hrbuild @ output['rgb'])
    assert original_prediction_error == 0
    if is_hr:
        assert hr_prediction_error == 0
        assert np.array_equal(replay['permitted_zero_columns'], safe)
    Abridge = Wbridge.multiply(1/np.asarray(rays['alpha'], np.float64)[:, None]).tocsr()
    fraction = np.asarray(Abridge[:, permission].sum(1)).ravel()
    admitted = (rays['alpha'] >= receipt['identity']['min_prior_alpha']) & (fraction >= receipt['identity']['min_unknown_fraction'])
    expected_active = permission & (np.asarray(Abridge[admitted].getnnz(axis=0)).ravel() > 0)
    assert np.array_equal(replay['updated_primitive_rows'], np.flatnonzero(expected_active))
    assert int(admitted.sum()) == receipt['views'][0]['fitted_pixels']
    assert sha(dest/'complete_a6.npz') == receipt['export']['sha256']
    meta = json.loads(str(output['metadata_json']))
    inferred = meta['external_generated_appearance']
    assert 'inferred' in inferred['scope']
    assert bool(output['colour_inferred'][changed].all())
    return {
        'branch': branch, 'status': 'PASS_INDEPENDENT_CPU_BRIDGE_AND_ASSET_CHECKS',
        'candidate_path': str(candidate_path), 'candidate_sha256': sha(candidate_path),
        'asset_path': str(dest/'complete_a6.npz'), 'asset_sha256': sha(dest/'complete_a6.npz'),
        'base_asset': str(base_path if is_hr else OLD/'m2/complete_a6.npz'),
        'uses_HR1920_operator': is_hr, 'native_ray_count': int(use.sum()),
        'W_bridge_vs_packet_maxabs': Werror, 'bilinear_target_stored_float32_maxabs': target_error,
        'permitted_columns': int(permission.sum()), 'protected_union_columns': int(protected.sum()),
        'updated_rows_in_replay': int(len(replay['updated_primitive_rows'])),
        'changed_RGB_rows': int(changed_rgb.sum()), 'changed_SH_rows': int(changed_sh.sum()),
        'protected_all_16x3_SH_RGB_inferred_flags_bitwise_exact': True,
        'all_geometry_opacity_and_provenance_arrays_bitwise_exact': True,
        'original_BUILD_prediction_maxabs': original_prediction_error,
        'HR_BUILD_prediction_maxabs': hr_prediction_error,
        'HR_prediction_preservation_required_for_primary': is_hr,
        'generated_colour_inferred_flags_true': True,
        'metadata_labels_inferred_appearance': True,
        'native_render_validation_still_required': receipt['native_render_validation_required'],
        'source_hashes_verified': True,
        'receipt_sha256': sha(dest/'receipt.json'),
        'bridge_receipt_sha256': sha(prep/'bridge_receipt.json'),
    }


branches = sorted(p.parent.name for p in (RUN/'generated_fit').glob('*/receipt.json')) if (RUN/'generated_fit').exists() else []
adapters = [review_adapter_output(branch) for branch in branches]
report = {
    'status': 'CPU_PACKET_AND_AVAILABLE_ADAPTER_IO_CHECKS_PASS',
    'scope': 'Independent read-only CPU review; no GPU, renderer import, sampling, training, download or remote writes',
    'selection': {
        'candidate': 'prompt2_seed3', 'path': str(SELECTED), 'sha256': sha(SELECTED),
        'decision_by': 'root after BUILD-only candidate board inspection',
        'reason': 'seed3 more usable lower-body silhouette; seed1 block-like lower corners, seed2 bright outline, seed0 central seam',
        'held_rgb_used': False, 'hyperparameter_search_after_selection': False,
        'not_a_measurement': True,
    },
    'architecture': [
        'HR1920 native operator rays -> exact pixel-center affine -> fixed 512 proposal sampling',
        'native W and alpha -> A=W/alpha -> straight-sRGB proposal target',
        'original-zero AND HR-zero Gaussian columns -> update only inferred SH0 -> frozen geometry asset',
        'protected BUILD-contributing columns -> byte-exact SH/RGB/inferred flags -> export audit',
    ],
    'pixel_center_rule': 'low=(hr+.5)/4-.5; generated=old480_to_512_affine(low)',
    'frames': frames,
    'protected_column_contract': {
        'gaussian_count': 18432, 'original_BUILD_zero_columns': int(oldzero.sum()),
        'HR_BUILD_zero_columns': int(hrzero.sum()), 'safe_intersection_zero_columns': int(safe.sum()),
        'original_zero_but_HR_contributing_columns_to_protect': int((oldzero & ~hrzero).sum()),
        'HR_zero_but_original_contributing_columns_to_protect': int((hrzero & ~oldzero).sum()),
        'all_known_union_columns_to_protect': int((~safe).sum()),
        'preserve_all_16x3_SH_bytes_and_RGB_and_colour_inferred_relative_to_input_asset': True,
        'highres_control_asset_path': str(base_path), 'highres_control_asset_sha256': sha(base_path),
        'geometry_and_provenance_fields_original_vs_HR_control_exact': True,
        'source_asset_has_SH0_only': True,
        'required_prediction_checks': ['original_BUILD_A @ rgb exactly unchanged', 'HR_BUILD_A @ rgb exactly unchanged'],
    },
    'inspected_source_hashes': {str(p): sha(p) for p in source_paths if p.exists()},
    'adapter_final_output_checks': adapters,
    'offline_SD_selected_baseline_asset_review': 'PENDING fit adapter receipt' if not any('sd' in b.lower() for b in branches) else 'Included in adapter_final_output_checks',
    'strong_candidate': {
        'status': 'PENDING final hosted generation/candidate provenance from root',
        'classification': 'Strong pretrained image generation is an inferred material proposal, not measured RGB or verified geometry',
        'must_use_same_reviewed_affine_and_mask_contract': True,
    },
    'limitations': [
        'Candidate silhouette masking and 2D alignment do not prove multi-view 3D consistency.',
        'True HR input supplies additional observed-upper detail; it does not turn generated lower-body pixels into ground truth.',
        'Preserving original/HR BUILD operators does not claim every image pixel is unchanged outside those defined supports.',
    ],
}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
