"""Evidence-bound limited input recommendation after independent five-image review."""
import argparse,json
from pathlib import Path
import numpy as np
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);a=p.parse_args();C=a.campaign_root
    R=C/'road_underbody_local_20261123_r10';D=R/'audit/timewindow17_geometry_v1';P=D/'candidate_reprojected_v1';Q=P/'independent_audit_v1'
    gp=P/'candidate_geometry.npz';g=load(gp);auditpath=Q/'independent_geometry_audit.json';audit=read(auditpath)
    trainpath=D/'training_physical_gate_summary.json';train=read(trainpath);assert train['decision']=='PASS_TRAINING_HEIGHT_ONLY'
    diagpath=D/'loo_selection/expanded_curvature_diagnostic.json';diag=read(diagpath);selected=next(x for x in diag['models'] if x['name']=='plane');assert selected['fixed_expanded_heldout_pass']
    coverage=[]
    for rr in [86,110,134,179,198]:
        pr=load(Q/f'{rr:03d}/projection_and_scope.npz');f=load(R/f'fixed_eval_v1_replay1/{rr:03d}/fixed_eval.npz');regions={}
        for name in ['full_original_band','ground_band_before_support','ground_eval','red_diagnostic','blue_cyan_diagnostic','dark_context_diagnostic']:
            m=f[name];count=int(m.sum());inside=int((m&pr['asset_scope']).sum());regions[name]=dict(original_pixels=count,geometry_supported=inside,geometry_unknown_or_outside_scope=count-inside,
                supported_behind_actor_or_K=int((m&pr['asset_scope']&(pr['actor_protected']|pr['K_protected'])).sum()))
        coverage.append(dict(raw=rr,regions=regions))
    cp=P/'fixed_target_coverage.json';assert not cp.exists();write(cp,dict(complete=True,schema='R10.timewindow17_fixed_target_geometry_coverage.v1',
        candidate_geometry=pin(gp),rows=coverage,all_original_pixel_denominators_preserved=True,unknown_not_promoted_to_black_material=True,GPU_calls=0))
    notes={86:'Grid is continuous with no visible fold or gross axis mismatch.62 fixed edit pixels remain unsupported at the qualified edge. The photographed originalA9 occludes the inferred road and is not material truth.',
        110:'Smooth inferred road grid, coherent road perspective and GT bottom outlines; no abrupt height boundary visible. Ground behind the preserved vehicle requires native joint occlusion verification.',
        134:'No visible fold or gross camera mismatch. Broader qualified footprint remains continuous; original photo car body is display context, not a new road label.',
        179:'Grid remains smooth and aligned in perspective. The GT bottom crosscheck is an indirect contact reference, not a tyre-surface or hidden-road measurement.',
        198:'No visible fold or coordinate jump. Supported subset is suitable for a limited zero-step joint probe; final body/ground contact and source appearance remain separate gates.'}
    target=P/'independent_limited_geometry_review.json';assert not target.exists()
    result=dict(schema='R10.timewindow17_independent_limited_geometry_review.v1',complete=True,decision='PASS_FOR_LIMITED_ZERO_STEP_MODEL_INFERENCE_CONTROL',
        geometry_npz=pin(gp),candidate_geometry=pin(gp),geometry_source_kind=2,all_approved_geometry_model_inferred=True,
        approved_cell_mask_key='cell_asset_mask',approved_cell_mask_count=int(g['cell_asset_mask'].sum()),
        candidate_geometry_accepted_for_zero_step=True,requires_separate_root_authorization=True,
        requires_fresh_source_projection_and_acceptance=True,old_fee13_or1844_not_approved_for_new_geometry=True,
        full_requested_geometry_coverage_PASS=False,unknown_requested_cells=int(g['unsupported_requested_cell_mask'].sum()),
        full_road_geometry_final_PASS=False,whole_scene_restoration_PASS=False,
        numeric_integrity=pin(auditpath),training_physical_gate=pin(trainpath),heldout_evaluation=pin(diagpath),
        independent_pointfit_max_abs_error=audit['independent_plane_fit_max_abs'],
        globalUV_and_camera_calibrations_unchanged=True,
        visual_review=dict(complete=True,decision='PASS_FOR_SUPPORTED_GEOMETRY_SUBSET',reviewer='native_clean_photo_regions',
            root_concurred=False,frames=[86,110,134,179,198],native_image_wh=[1152,768],
            observations=[dict(raw=rr,note=notes[rr]) for rr in [86,110,134,179,198]],contactsheet=audit['contactsheet']),
        target_coverage=pin(cp),GT_contact=[dict(raw=x['raw'],contacts=x['GT_contact']) for x in audit['rows']],
        parameter_bootstrap_SD_m=stats(g['cell_height_bootstrap_SD_m'][g['cell_asset_mask']]),
        uncertainty_is_conditional_not_calibrated_probability=True,
        pool_comparison='Fixed86..198 evaluation time window, every17 available official semantic frame included; time-window change plus added observations, not count-only ablation',
        old8_pool_and_failed_models_preserved=True,heldout_not_used_to_choose_model=True,
        original_actor_camera_oldroad_and_source_RGB_unchanged=True,GPU_forward=0,backward=0,updates=0)
    write(target,result);print(json.dumps(dict(complete=True,review=pin(target),coverage=coverage)),flush=True)


if __name__=='__main__':main()
