"""Bind root's limited inferred-geometry review without rewriting candidates."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--r10',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    src=a.r10/'audit/local_surface_candidate_v2';gp=src/'candidate_geometry.npz'
    assert sha(gp)=='b5df7c8cb2b99723e1fabf3b2ca4c958c88003d38f4f528bb6dd8f14e97a011a'
    qp=src/'independent_limited_geometry_review.json';q=read(qp)
    assert q['complete'] and q['decision']=='PASS_FOR_LIMITED_ZERO_STEP_MODEL_INFERENCE_CONTROL'
    cp=a.r10/'audit/curvature_ridge_loo_v1/curvature_ridge_diagnostic.json';c=read(cp)
    assert c['heldout_never_used_to_select'] is True and c['physical_box_contact_prior_fitted'] is False
    plane=next(x for x in c['finalists'] if x['name']=='plane')
    assert plane['fixed_original_heldout_checks_pass'] and all(x['threshold_checks_pass'] for x in plane['heldout'])
    sp=src/'five_view_geometry_v1/five_view_geometry_receipt.json';surface=read(sp)
    for raw,total in [(86,357),(110,692)]:
        row=next(x for x in surface['rows'] if x['raw']==raw);red=row['regions']['red_diagnostic']
        assert red['fixed_pixels']==red['explicit_asset_scope_supported']==red['partition']['asset_display_visible']==total
    with np.load(gp,allow_pickle=False) as g:
        mask=g['cell_asset_mask'];xyz=g['cell_world_xyz'][mask];n=g['cell_normal_world'][mask]
        assert mask.sum()==34793 and np.isfinite(xyz).all() and np.allclose(np.linalg.norm(n,axis=-1),1)
        assert np.ptp(xyz[:,2])>.02 and (g['cell_geometry_source_kind'][mask]==2).all()
        assert not g['renderable_mask'].any()
    assert not a.output.exists();a.output.mkdir(parents=True)
    common=dict(complete=True,decision='PASS',geometry_npz_sha256=sha(gp),approval_scope='new_local_zero_step_only',
        actual_geometry_render_acceptance=False,whole_scene_quality_accepted=False,production_geometry_approved=False,
        original_vehicle_or_existing_geometry_edited=False,Gaussian_updates=0,GPU_forward=0,GPU_backward=0)
    write(a.output/'certificate.json',dict(common,schema='R10.root_limited_inferred_geometry_certificate.v1',
        geometry_npz=pin(gp),constant_z_plane_used=False,candidate_geometry_accepted_for_zero_step=True,
        approved_cell_mask_key='cell_asset_mask',spacing_m=.02,cell_center_interpolation_authorized=False,
        cell_centres_analytically_evaluated=True,independent_review=pin(qp),training_and_heldout=pin(cp),
        five_view_geometry=pin(sp),root_visual_review=dict(complete=True,views=[86,110,134,179,198],
            observed='No obvious projection mismatch or fold in the metric road-grid and raw-GT-bottom comparisons; deleted A9 photo occlusion is not new-road colour truth'),
        rationale='Local sloping plane selected only by training-frame LOO one-SE rule; held-out physical checks pass. A20 box/wheel contact is an indirect crosscheck only. This approves the declared model-inferred 34793-cell prototype, not measured ground truth or watertight stitching to the noisy parent.'))
    write(a.output/'target_coverage_gate.json',dict(common,schema='R10.root_local_zero_coverage.v1',
        local_asset_initialization_authorized=True,approved_cell_mask_key='cell_asset_mask',asset_cells=34793,
        fixed_red_coverage={'086':[357,357],'110':[692,692]},full_other_band_denominators_retained=True,
        partial_blue_coverage_is_not_full_recovery=True,evidence=pin(sp)))
    write(a.output/'model_inferred_acceptance.json',dict(common,schema='R10.root_local_model_inference_zero_gate.v1',
        accepted_as_model_inference_for_zero_step_only=True,direct_observation_claimed=False,model='training-LOO-selected local sloping plane',
        measured_witness_mask_not_changed=True,asset_all_34793_cells_inferred=True,independent_review=pin(qp),
        bootstrap_uncertainty_is_conditional_not_calibrated_probability=True))
    print(json.dumps(dict(complete=True,approval_scope='new_local_zero_step_only',asset_cells=34793)))

if __name__=='__main__':main()
