"""Read-only CPU review of deterministic sphere output and both colour fits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.ndimage import map_coordinates
from PIL import Image

RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')


def bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def mx(x):
    return float(np.abs(x).max()) if np.size(x) else 0.


def read(path):
    return json.loads(Path(path).read_text())


paths = {
    'thin_observed_HR': RUN/'highres/formal/complete_a6.npz',
    'sphere_output_before_colour_refit': RUN/'output_adapter/sphere_sigma_asset.npz',
    'sphere_observed_HR': RUN/'output_adapter/sphere_highres/formal/complete_a6.npz',
    'sphere_generated_strong': RUN/'output_adapter/sphere_strong_v2/complete_a6.npz',
}
thin, sphere, observed, generated = [bank(p) for p in paths.values()]
geometry = ('xyz','normal','sigma','opacity','thickness','box','knn_spacing','normal_radius',
            'normal_confidence','generated','conditioning','normal_valid','out_of_box','conditioning_held',
            'source_index','conditioning_observation_index','conditioning_raw_frame','conditioning_cloud_row')
assert all(len(a['xyz']) == 18432 for a in (thin,sphere,observed,generated))
changed_geometry = [k for k in geometry if not same(thin[k],sphere[k])]
assert changed_geometry == ['thickness']
assert same(sphere['thickness'],sphere['sigma'])
assert all(same(thin[k],sphere[k]) for k in ('canonical_sh','rgb','colour_inferred','provenance_hash'))
assert all(all(same(sphere[k],a[k]) for k in geometry) for a in (observed,generated))
assert all(same(sphere['geometry_hash'],a['geometry_hash']) for a in (observed,generated))
appearance = {'canonical_sh','rgb','colour_inferred','metadata_json'}
assert all(same(observed[k],generated[k]) for k in set(observed)-appearance)

design_path = RUN/'output_adapter/sphere_highres/design.npz'
A = load_npz(design_path).tocsr();A.eliminate_zeros()
unknown = np.asarray(A.getnnz(axis=0)==0).ravel()
z = bank(RUN/'output_adapter/sphere_highres/inputs.npz')
original_z = bank(RUN/'highres/inputs.npz')
assert same(z['target'],original_z['target']) and same(z['raw'],original_z['raw'])
assert np.array_equal(unknown,z['unknown'])
protected=~unknown
assert all(same(observed[k][protected],generated[k][protected]) for k in ('canonical_sh','rgb','colour_inferred'))
prediction_error = mx(A@observed['rgb']-A@generated['rgb'])
assert prediction_error == 0
changed=np.any(observed['canonical_sh'] != generated['canonical_sh'],axis=(1,2))
assert not (changed & protected).any()
assert (generated['colour_inferred'][changed]).all()

dest=RUN/'output_adapter/sphere_strong_v2'
receipt=read(dest/'receipt.json');bridge=read(dest.with_name(dest.name+'_inputs')/'bridge_receipt.json')
replay=bank(dest/'replay.npz')
assert np.array_equal(replay['original_BUILD_zero_columns'],unknown)
assert np.array_equal(replay['permitted_zero_columns'],unknown)
assert receipt['identity']['original_BUILD_design_sha256'] == sha(design_path)
assert receipt['identity']['base_sha256'] == sha(paths['sphere_observed_HR'])
assert receipt['export']['sha256'] == sha(paths['sphere_generated_strong'])
assert receipt['formal_generated_colour_steps']==1000 and receipt['neural_training_steps']==0
assert receipt['exact_full_state_nextstep_replay']

sources=bridge['source_SHA256']
assert all(sha(p)==h for p,h in sources.items())
sample_path=next(Path(p) for p in sources if Path(p).name=='operator_samples.npz')
transform_path=next(Path(p) for p in sources if Path(p).name=='transform.npz')
candidate_path=next(Path(p) for p in sources if p.endswith('.png') and 'mask' not in Path(p).name)
s,t=bank(sample_path),bank(transform_path)
xy=np.c_[s['pixel_xy'],np.ones(len(s['pixels']))]@t['highres_to_generated_pixel_center'].T
map_error=mx(xy[:,:2]-s['generated_xy'])
assert map_error<1e-10
rgb=np.asarray(Image.open(candidate_path)).astype(np.float64)/255
sampled=np.column_stack([map_coordinates(rgb[...,c],[xy[:,1],xy[:,0]],order=1,mode='constant',cval=0,prefilter=False) for c in range(3)])
prep=dest.with_name(dest.name+'_inputs')
rays=bank(prep/'operator/rays.npz')
row=np.searchsorted(s['pixels'],rays['full_pixels'])
assert np.array_equal(s['pixels'][row],rays['full_pixels'])
actual=bank(prep/'candidate_on_native_crop.npz')['RGB_srgb'].reshape(-1,3)[rays['crop_pixels']]
target_error=mx(actual-sampled[row])
assert target_error<4e-8
W=load_npz(prep/'operator/weights.npz').astype(np.float64)
Wsource=load_npz(sample_path.parent/'design_W_premultiplied.npz').astype(np.float64)
delta=(W-Wsource[row]).tocsr();delta.eliminate_zeros()
assert delta.nnz==0
Ap=W.multiply(1/np.asarray(rays['alpha'],np.float64)[:,None]).tocsr()
fraction=np.asarray(Ap[:,unknown].sum(1)).ravel()
admit=(rays['alpha']>=receipt['identity']['min_prior_alpha'])&(fraction>=receipt['identity']['min_unknown_fraction'])
active=unknown&(np.asarray(Ap[admit].getnnz(axis=0)).ravel()>0)
assert np.array_equal(replay['updated_primitive_rows'],np.flatnonzero(active))

observed_receipt=read(RUN/'output_adapter/sphere_highres/formal/receipt.json')
smoke_receipt=read(RUN/'output_adapter/sphere_highres/smoke/receipt.json')
collection=read(RUN/'output_adapter/sphere_highres/collection.json')
assert observed_receipt['colour_fit_steps']==1000 and smoke_receipt['colour_fit_steps']==32
assert observed_receipt['neural_training_steps']==0 and observed_receipt['PVD_weight_updates']==0
assert observed_receipt['full_state_exact_resume'] and smoke_receipt['full_state_exact_resume']
assert observed_receipt['base_asset_sha256']==sha(paths['sphere_output_before_colour_refit'])
assert observed_receipt['asset']['file_sha256']==sha(paths['sphere_observed_HR'])
assert collection['target_and_raw_bitwise_identical_to_original_HR']
unknown_rgb_delta=observed['rgb'][unknown]-sphere['rgb'][unknown]
unknown_sh_delta=observed['canonical_sh'][unknown]-sphere['canonical_sh'][unknown]
unknown_rgb_error=mx(unknown_rgb_delta)
unknown_sh_error=mx(unknown_sh_delta)
assert unknown_rgb_error<=3e-8 and unknown_sh_error<=1.2e-7
assert mx(A[:,unknown]@unknown_rgb_delta)==0

result={
 'status':'PASS_SPHERE_THICKNESS_ONLY_AND_COLOUR_PREDICTION_PROTECTION',
 'review_execution':'Independent CPU-only file/array/operator comparison; no new fits, rendering, sampling or source edits',
 'asset_files':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
 'point_count':18432,
 'kernel_output_adapter':{
   'changed_geometry_fields':changed_geometry,'thickness_equals_sigma_bitwise':True,
   'xyz_normal_sigma_opacity_box_and_source_arrays_bitwise_unchanged':True,
   'pre_refit_SH_RGB_and_inferred_flags_bitwise_unchanged':True,
   'provenance_hash_unchanged':True,
   'thin_geometry_hash':str(thin['geometry_hash']),
   'sphere_geometry_hash':str(sphere['geometry_hash']),
   'interpretation':'Deterministic output-kernel change, not a new PVD sample or learned geometry correction.'},
 'observed_sphere_colour_fit':{
   'smoke_steps_separate':smoke_receipt['colour_fit_steps'],'formal_steps':observed_receipt['colour_fit_steps'],
   'neural_weight_training_steps':0,'full_state_exact_replay':True,
   'same_target_RGB_and_raw_sequence_as_thin_HR_bitwise':True,
   'observed_pixel_gate':'Sphere collector reuses the old thin native alpha>.9 gate and original eroded BUILD visible domain; no additional actual RGB admitted',
   'unknown_columns':int(unknown.sum()),
   'unobserved_column_RGB_bitwise_unchanged_by_export':bool(unknown_rgb_error==0),
   'unobserved_column_RGB_export_roundtrip_maxabs':unknown_rgb_error,
   'unobserved_column_RGB_export_roundtrip_changed_rows':int(np.any(unknown_rgb_delta!=0,axis=1).sum()),
   'unobserved_column_SH_export_roundtrip_maxabs':unknown_sh_error,
   'unobserved_column_SH_export_roundtrip_changed_rows':int(np.any(unknown_sh_delta!=0,axis=(1,2)).sum()),
   'export_roundtrip_note':'Observed-fit optimiser holds zero columns at prior RGB; export re-encodes all RGB into SH0, causing at most float32 roundtrip differences on zero-contribution columns. These are not new optimisation steps.',
   'observed_prediction_change_due_to_unknown_export_roundtrip':0.,
   'geometry_unchanged_during_colour_fit':True,
   'receipt_sha256':sha(RUN/'output_adapter/sphere_highres/formal/receipt.json')},
 'generated_sphere_colour_fit':{
   'formal_per_asset_colour_updates':receipt['formal_generated_colour_steps'],
   'neural_weight_training_steps':0,'extra_nextstep_replay_evaluations':receipt['replay_nextstep_checks_extra'],
   'full_state_exact_nextstep_replay':True,'updated_columns':int(active.sum()),
   'protected_sphere_BUILD_contributing_columns':int(protected.sum()),
   'all_16x3_SH_RGB_inferred_flags_of_protected_columns_bitwise_exact':True,
   'sphere_observed_A_times_RGB_maxabs_change':prediction_error,
   'geometry_opacity_provenance_arrays_bitwise_exact':True,
   'highres_to_generated_XY_maxabs':map_error,'bilinear_target_float32_storage_maxabs':target_error,
   'W_bridge_vs_native_packet_maxabs':0.,'candidate':str(candidate_path),'candidate_sha256':sha(candidate_path),
   'generated_colours_marked_inferred':True,'receipt_sha256':sha(dest/'receipt.json')},
 'claim_limits':[
   'This review establishes output/fit wiring invariants; it does not select sphere as the final representation or establish reference-image quality.',
   'Sphere and thin kernels differ before material fitting; equality of their whole images or operators is not claimed.',
   'Strong hosted RGB remains a material proposal. No hidden lower-body RGB ground truth is available from this test.',
   'Actual all49 rendering and cross-kernel preservation are separate parent-agent acceptance outputs.'
 ]
}
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
