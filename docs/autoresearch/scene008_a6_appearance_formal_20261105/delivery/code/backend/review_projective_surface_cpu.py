"""Independent CPU audit of the single-reference persistent 3D chart prototype."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.sparse import load_npz
from scipy.spatial import cKDTree

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
ROOT=RUN/'output_adapter/projective_surface'
PREP=RUN/'source/output_adapter/projective_surface/prepare_projective_surface.py'
EVAL=PREP.with_name('evaluate_projective_surface.py')


def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def mx(a):
    return float(np.abs(a).max()) if np.size(a) else 0.


receipt=json.loads((ROOT/'prepare_receipt.json').read_text())
chart=bank(ROOT/'reference_chart.npz')
source_path=Path(receipt['source_asset']);source=bank(source_path)
asset_path=ROOT/'projective_surface_asset.npz';asset=bank(asset_path)
proposal_path=Path(receipt['reference_rgb_proposal'])
assert sha(source_path)==receipt['source_asset_sha256']
assert sha(proposal_path)==receipt['reference_rgb_proposal_sha256']
assert sha(asset_path)==receipt['export']['file_sha256']
assert int(chart['reference_raw'])==receipt['reference_raw']==106
assert receipt['reference_split']=='BUILD' and receipt['HELD_RGB_used'] is False
assert receipt['PVD_weight_updates']==receipt['colour_optimization_updates']==0
assert receipt['depth_is_prior_prediction_not_truth'] and receipt['full360_recovery'] is False

pose=chart['reference_pose'];view=chart['reference_world_view'];full=chart['reference_full_projection']
center=chart['reference_camera_center'];xy=chart['pixel_xy'].astype(np.float64)
local=chart['local_xyz'].astype(np.float64);z=chart['expected_depth_model'].astype(np.float64)
scale=float(np.linalg.norm(pose[:3,:3],axis=0).mean())
assert abs(scale-receipt['pose_uniform_scale'])<1e-15
assert np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3)*scale*scale,atol=1e-8,rtol=0)
assert np.linalg.det(pose[:3,:3])>0
camera_model=np.c_[np.zeros((1,3)),np.ones(1)]@np.linalg.inv(view)
assert mx(camera_model[0,:3]/camera_model[0,3]-center)<1e-6
world=local@pose[:3,:3].T+pose[:3,3]
homogeneous=np.c_[world,np.ones(len(world))]@full
projected=((homogeneous[:,:2]/homogeneous[:,3,None]+1)*[1920,1280]-1)/2
reprojection=np.linalg.norm(projected-xy,axis=1)
depth_roundtrip=world@view[:3,2]+view[3,2]-z
assert reprojection.max()<.001 and mx(depth_roundtrip)<1e-6
metre_error=mx(chart['expected_depth_m']-z/receipt['meter_to_model'])
assert metre_error==0
projection=np.linalg.inv(view)@full
fx_HR=projection[0,0]*1920/2
obs=bank(RUN.parent/'scene008_instances_20261105_r1/observations/frames/106.npz')
fx_from_obs=float(obs['K'][0,0])*4
assert abs(fx_HR-fx_from_obs)<.001
sigma=(receipt['sigma_reference_HR_pixels']*z/fx_from_obs/scale).astype(np.float32)
assert mx(sigma-chart['sigma_local_m'])<1e-8

packet=RUN/'output_adapter/sphere_inputs_hr/raw106'
W=load_npz(packet/'design_W_premultiplied.npz')
samples=bank(packet/'operator_samples.npz')
assert np.array_equal(samples['pixels'],chart['pixels'])
mass=np.asarray(W.sum(0)).ravel()
assert np.array_equal(mass,chart['source_native_reference_contribution'])
retired=mass>0;retained=np.flatnonzero(~retired)
assert np.array_equal(retired,chart['source_retired']) and np.array_equal(retained,chart['source_retained'])
N=len(source['xyz']);M=len(local);cut=len(retained)
assert N==receipt['source_rows'] and M==receipt['chart_rows'] and cut==receipt['retained_source_rows']
assert len(asset['xyz'])==M+cut==receipt['export']['rows']
geometry=('xyz','normal','sigma','opacity','thickness','box','knn_spacing','normal_radius','normal_confidence',
          'generated','conditioning','normal_valid','out_of_box','conditioning_held','source_index',
          'conditioning_observation_index','conditioning_raw_frame','conditioning_cloud_row')
for k in geometry+('canonical_sh','rgb','colour_inferred'):
    if source[k].ndim and source[k].shape[0]==N:
        assert same(asset[k][:cut],source[k][retained]),k
    else:assert same(asset[k],source[k]),k
assert np.array_equal(asset['xyz'][cut:],chart['local_xyz'])
assert np.array_equal(asset['sigma'][cut:],chart['sigma_local_m'])
assert np.array_equal(asset['thickness'][cut:],chart['sigma_local_m'])
assert np.array_equal(asset['opacity'][cut:],np.full(M,.99,np.float32))
assert (asset['representation_kind'][:cut]==0).all() and (asset['representation_kind'][cut:]==1).all()
distance,nearest=cKDTree(source['xyz']).query(chart['local_xyz'],k=1)
assert np.array_equal(nearest,chart['nearest_source_row'])
assert mx(distance-chart['nearest_distance_local_m'])==0
assert np.array_equal(asset['resample_source_row'],np.r_[retained,nearest])
assert np.array_equal(asset['resample_source_parent'],np.r_[source['source_index'][retained],source['source_index'][nearest]])
assert np.array_equal(asset['resample_reference_pixel_xy'][cut:],chart['pixel_xy'])
assert (asset['resample_reference_raw'][cut:]==106).all()

low_known=np.asarray(Image.open(RUN/'inputs/raw106/known_mask_original.png'))>0
known=np.repeat(np.repeat(low_known,4,0),4,1).ravel()[chart['pixels']]
assert np.array_equal(known,chart['color_real'])
rgb_path=RUN.parent/'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137/image/image_0/0008106.png'
real=np.asarray(Image.open(rgb_path).convert('RGB')).reshape(-1,3).astype(np.float32)/255
assert np.array_equal(real[chart['pixels'][known]],chart['color_rgb'][known])
T=chart['highres_to_generated_pixel_center']
generated_xy=(np.c_[xy,np.ones(M)]@T.T)[:,:2]
assert mx(generated_xy-chart['generated_xy'])==0
nearest_xy=np.clip(np.floor(generated_xy+.5).astype(int),[0,0],[511,511])
foreground=np.asarray(Image.open(RUN/'inputs/raw106/object_mask_512.png'))>0
inpaint=np.asarray(Image.open(RUN/'inputs/raw106/inpaint_mask_512.png'))>0
generated=(~known)&foreground[nearest_xy[:,1],nearest_xy[:,0]]&inpaint[nearest_xy[:,1],nearest_xy[:,0]]
fallback=~known&~generated
assert np.array_equal(generated,chart['color_generated']) and np.array_equal(fallback,chart['color_prior_fallback'])
proposal=np.asarray(Image.open(proposal_path).convert('RGB'))
sampled=np.column_stack([map_coordinates(proposal[...,c].astype(np.float64)/255,[generated_xy[:,1],generated_xy[:,0]],
                         order=1,mode='nearest',prefilter=False) for c in range(3)]).astype(np.float32)
assert np.array_equal(sampled[generated],chart['color_rgb'][generated])
assert np.array_equal(source['rgb'][nearest[fallback]],chart['color_rgb'][fallback])
asset_rgb_roundtrip=mx(asset['rgb'][cut:]-chart['color_rgb'])
assert asset_rgb_roundtrip<=6e-8
assert np.array_equal(asset['colour_inferred'][cut:],~known)
assert not (known&generated).any() and not (known&fallback).any() and not (generated&fallback).any()
assert (known|generated|fallback).all()

prep_text=PREP.read_text();eval_text=EVAL.read_text()
tree=ast.parse(prep_text)
RAW_assignments=[ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='RAW' for t in n.targets)]
assert RAW_assignments==[106]
assert "out['depth_sum']" in prep_text and "depth_m'][" not in prep_text
assert '.backward(' not in prep_text and 'torch.optim' not in prep_text
assert "rt.scene(raw,assets['projective_surface']" in eval_text
assert "out=raster(c,rt.pipe" in eval_text and 'component_alpha_T' in eval_text
assert "image_overlay_used=False" in eval_text
evaluation_receipt=ROOT/'evaluation/receipt.json'
actual_eval=json.loads(evaluation_receipt.read_text()) if evaluation_receipt.is_file() else None
if actual_eval:
    assert actual_eval['actual_native_joint_sort'] and not actual_eval['image_overlay_used']
    assert actual_eval['PVD_weight_updates']==actual_eval['colour_fit_updates']==0

result={
 'status':'PASS_PROJECTIVE_PROTOTYPE_3D_CHART_UNITS_AND_PROVENANCE',
 'scope':'Independent CPU-only source and saved-array review; no GPU, fitting, sampling, rendering or source edits',
 'representation':'Single-reference-view conditioned persistent local3D chart, plus unchanged unreferenced source rows',
 'backprojection':{'reference_raw':106,'reference_split':'BUILD','highres_wh':[1920,1280],
   'depth_source':'Frozen sphere prior alpha-normalized expected camera-z; not LiDAR or measured hidden depth',
   'meter_to_model':receipt['meter_to_model'],'pose_uniform_scale':scale,'fx_HR':fx_from_obs,
   'expected_depth_m_vs_model_div_scale_maxabs':metre_error,
   'local_to_reference_reprojection_HR_pixels_max':float(reprojection.max()),
   'camera_z_roundtrip_model_maxabs':mx(depth_roundtrip),
   'sigma_formula':'local_sigma=.6*camera_z_model/fx_HR/actor_pose_uniform_scale',
   'sigma_formula_maxabs_m':mx(sigma-chart['sigma_local_m']),
   'pixel_center_NDC':'x_ndc=(2*x+1)/W-1; inverse full projection uses row-vector convention'},
 'persistent_3D_not_2D_overlay':{'saved_a6_local_xyz':True,'moves_with_actual_actor_pose':True,
   'native_scene_joint_depth_sort_in_source':True,'native_component_alpha_T_conservation_check_in_source':True,
   'actual_evaluation_receipt_available':actual_eval is not None,
   'actual_native_joint_sort':actual_eval.get('actual_native_joint_sort') if actual_eval else None,
   'actual_group_conservation_maxabs':actual_eval.get('native_group_conservation_maxabs') if actual_eval else None,
   'image_paste_calls_in_evaluator':'PIL strip() only lays out report panels; scene RGB comes from native raster plus transmittance-weighted environment'},
 'source_retirement':{'source_rows':N,'retired_rows_positive_native_reference_contribution':int(retired.sum()),
   'retained_rows':cut,'new_chart_rows':M,'output_rows':M+cut,
   'retirement_exactly_matches_W_column_positive_mass':True,
   'retained_original_geometry_material_flags_bytes_exact':True,
   'nearest_source_row_parent_distance_arrays_recomputed_exact':True,
   'source_file_unchanged_sha256':sha(source_path),
   'important_scope':'Original source is preserved on disk; this output representation changes point count, positions, scales and opacity for new chart rows. It is not the same frozen-geometry material-only asset.'},
 'colour_provenance':{'real_BUILD106_HR_RGB_rows':int(known.sum()),'generated_strongv2_rows':int(generated.sum()),
   'source_prior_fallback_chart_rows':int(fallback.sum()),'three_chart_colour_masks_disjoint_and_exhaustive':True,
   'real_RGB_from_original_visible_mask_nearest_lifted4':True,'real_RGB_source_file_sha256':sha(rgb_path),
   'generated_RGB_recomputed_with_exact_affine_and_bilinear_sampling':True,
   'exported_SH0_decoded_RGB_float32_roundtrip_maxabs':asset_rgb_roundtrip,
   'proposal_sha256':sha(proposal_path),'unknown_RGB_marked_inferred':True,
   'HELD_RGB_read_by_prepare':False,'reference_RAW_assignment_is_106_only':True,
   'HELD_use_in_evaluator':'Actual HELD RGB is read for fixed-domain evaluation only, never chart construction or candidate selection'},
 'budget':{'PVD_weight_updates':0,'pretrained_weight_updates':0,'prototype_colour_optimization_updates':0,
   'output_resampling_inference_seconds':receipt['seconds'],
   'prototype_budget_not_equal_to_40BUILD_observed_fits':True,
   'formal_40BUILD_observed_fit':'Separate follow-up; include its actual32 smoke+1000 formal receipts once complete, not as part of this zero-optimization prototype'},
 'files':{'prepare_source':{'path':str(PREP),'sha256':sha(PREP)},'evaluate_source':{'path':str(EVAL),'sha256':sha(EVAL)},
   'prepare_receipt':{'path':str(ROOT/'prepare_receipt.json'),'sha256':sha(ROOT/'prepare_receipt.json')},
   'chart':{'path':str(ROOT/'reference_chart.npz'),'sha256':sha(ROOT/'reference_chart.npz')},
   'asset':{'path':str(asset_path),'sha256':sha(asset_path)}},
 'claim_limits':['Not360-degree reconstruction or independently held-out geometry truth.',
   'Reference-camera texture clarity is not evidence of more accurate geometry or novel-view consistency.',
   'No assertion that this changed representation preserves the thin/sphere whole-image RGB/depth predictions.',
   'Real upper RGB and inferred lower material have explicit separate provenance.']}
formal_root=ROOT/'formal_observed'
formal_receipt_path=formal_root/'formal/receipt.json'
if formal_receipt_path.is_file():
    fr=json.loads(formal_receipt_path.read_text())
    sr=json.loads((formal_root/'smoke/receipt.json').read_text())
    collection=json.loads((formal_root/'collection.json').read_text())
    formal_path=formal_root/'formal/complete_a6.npz'
    fa=bank(formal_path);fz=bank(formal_root/'inputs.npz');original=bank(RUN/'highres/inputs.npz')
    A=load_npz(formal_root/'design.npz').tocsr();A.eliminate_zeros()
    unknown=np.asarray(A.getnnz(axis=0)==0).ravel()
    assert np.array_equal(unknown,fz['unknown']) and len(unknown)==len(asset['xyz'])
    assert same(fz['target'],original['target']) and same(fz['raw'],original['raw'])
    assert len(np.unique(fz['raw']))==40 and not(set(np.unique(fz['raw']))&{89,95,101,105,113,121,125,130,137})
    assert all(same(asset[k],fa[k]) for k in set(asset)-{'canonical_sh','rgb','colour_inferred','metadata_json'})
    assert all(same(asset[k][unknown],fa[k][unknown]) for k in ('canonical_sh','rgb','colour_inferred'))
    assert fr['base_asset_sha256']==sha(asset_path) and fr['asset']['file_sha256']==sha(formal_path)
    assert sr['colour_fit_steps']==32 and fr['colour_fit_steps']==1000
    assert sr['full_state_exact_resume'] and fr['full_state_exact_resume']
    assert fr['neural_training_steps']==fr['PVD_weight_updates']==fr['geometry_updates_during_colour_fit']==0
    fit_source=PREP.with_name('fit_projective.py')
    assert sha(fit_source)==fr['source_sha256']
    result['formal_40BUILD_observed_colour_fit']={
      'status':'PASS_FIXED_GEOMETRY_AND_EXACT_UNKNOWN_COEFFICIENT_PROTECTION',
      'base_prototype_sha256':sha(asset_path),'formal_asset_path':str(formal_path),'formal_asset_sha256':sha(formal_path),
      'source_path':str(fit_source),'source_sha256':sha(fit_source),
      'same_original_HR_target_RGB_and_raw_order_bitwise':True,'BUILD_frames':40,'HELD_in_fit_raws':False,
      'operator_shape':list(A.shape),'operator_nnz':int(A.nnz),'unknown_columns':int(unknown.sum()),
      'all_original_geometry_opacity_source_and_resample_arrays_bitwise_exact':True,
      'unknown_columns_all_16x3_SH_RGB_inferred_flags_bitwise_exact':True,
      'new_generated_colour_optimization_steps':0,
      'observed_colour_smoke_steps_separate':sr['colour_fit_steps'],'observed_colour_formal_steps':fr['colour_fit_steps'],
      'formal_fit_seconds':fr['seconds'],'neural_weight_training_updates':0,'PVD_weight_updates':0,
      'full_state_nextstep_replay_exact':True,
      'receipt_sha256':sha(formal_receipt_path),
      'scope':'Observed colour calibration on the same40BUILD real-RGB domain; no new generated-colour1000 run and no new geometry. The surface remains conditioned on a single reference view.'}
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
