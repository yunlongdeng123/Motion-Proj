"""Root's supported-subset approval of the independently reviewed 17-frame model."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--r10',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=a.r10/'audit/timewindow17_geometry_v1';candidate=base/'candidate_reprojected_v1'
    gp=candidate/'candidate_geometry.npz';assert sha(gp)=='0d62e87abddd3017a40b06d80c485d331c0feb5c13a433be7f50f6933cda269b'
    qp=candidate/'independent_limited_geometry_review.json';q=read(qp)
    assert q['complete'] and q['decision']=='PASS_FOR_LIMITED_ZERO_STEP_MODEL_INFERENCE_CONTROL' and q['geometry_npz']['sha256']==sha(gp)
    tp=base/'training_physical_gate_summary.json';t=read(tp);assert t['complete'] and t['all17_retained']
    assert all(t['selected'][k] for k in ['context_all17','actual_all17','fitted_actual_all17'])
    hp=base/'loo_selection/expanded_curvature_diagnostic.json';h=read(hp)
    assert h['no_heldout_used_for_model_selection'] and h['no_Z_or_residual_point_filter'] and h['selected_name']=='plane'
    model=next(x for x in h['models'] if x['name']=='plane')
    assert model['fixed_expanded_heldout_pass'] and all(x['fixed_thresholds_pass'] for x in model['heldout'])
    cp=candidate/'fixed_target_coverage.json';c=read(cp);assert c['complete'] and c['all_original_pixel_denominators_preserved']
    for raw,n in [(86,357),(110,692)]:
        row=next(x for x in c['rows'] if x['raw']==raw)['regions']['red_diagnostic']
        assert row['original_pixels']==row['geometry_supported']==n
    with np.load(gp,allow_pickle=False) as z:
        m=z['cell_asset_mask'];assert int(m.sum())==175302
        assert (z['cell_geometry_source_kind'][m]==2).all() and z['cell_candidate_mask'][m].all()
        assert np.isfinite(z['cell_world_xyz'][m]).all() and np.allclose(np.linalg.norm(z['cell_normal_world'][m],axis=-1),1)
        assert not z['renderable_mask'].any()
    assert not a.output.exists();a.output.mkdir(parents=True)
    common=dict(complete=True,decision='PASS',geometry_npz_sha256=sha(gp),approval_scope='new_local_zero_step_only',
        actual_Gaussian_joint_quality_accepted=False,whole_scene_quality_accepted=False,production_geometry_approved=False,
        camera_or_existing_vehicle_or_old_road_edited=False,GPU_forward=0,GPU_backward=0,optimizer_updates=0)
    write(a.output/'certificate.json',dict(common,schema='R10.root_timewindow17_supported_geometry_zero.v1',geometry_npz=pin(gp),
        candidate_geometry_accepted_for_zero_step=True,approved_cell_mask_key='cell_asset_mask',constant_z_plane_used=False,
        height_override_used=False,model='data-fitted sloping plane selected by registered train-only LOO one-SE rule',
        spacing_m=.02,cell_centres_analytically_evaluated=True,cell_center_interpolation_authorized=False,
        independent_review=pin(qp),training_physical_gate=pin(tp),heldout_gate=pin(hp),coverage=pin(cp),
        root_visual_review=dict(complete=True,views=[86,110,134,179,198],
            observed='The five original-resolution geometry grids show no gross coordinate mismatch or fold. Original A9 photo occlusion is not material truth. Unknown supported-edge pixels remain unknown.'),
        fresh_appearance_source_gate_required=True,inherited_75_early_real_locks_NOT_certified_by_this_geometry_gate=True))
    write(a.output/'target_coverage_gate.json',dict(common,schema='R10.root_timewindow17_supported_coverage_zero.v1',
        local_asset_initialization_authorized=True,approved_cell_mask_key='cell_asset_mask',asset_cells=175302,
        full_original_denominators_preserved=True,full_target_geometry_PASS=False,raw086_unknown_full_band_pixels=62,
        raw086_blue_unknown_pixels=20,coverage=pin(cp)))
    write(a.output/'model_inferred_acceptance.json',dict(common,schema='R10.root_timewindow17_model_inference_zero.v1',
        all_asset_geometry_inferred=True,direct_observed_ground_claimed=False,
        reference_pool=list(range(90,171,5)),source_pool_selected_by_time_and_official_label_availability_not_Z_error=True,
        heldout_frames=[134,179,198],old8_pool_failures_preserved=True,old_geometry_1844_not_promoted=True,
        conditional_bootstrap_not_calibrated_model_error_probability=True,evidence=pin(qp)))
    print(json.dumps(dict(complete=True,approval_scope='new_local_zero_step_only',asset_cells=175302)))
if __name__=='__main__':main()
