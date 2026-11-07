"""Build compact evidence for the parent's report; no render/fit or input mutation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

B=Path(__file__).resolve().parent


def main():
    read=lambda name:json.loads((B/name).read_text(encoding='utf8'))
    registry=read('roi_registration.json');audit=read('blue_parent_audit.json')
    pair=read('n2_material_vs_n2_fixed_blue_comparison.json');total=read('n2_material_vs_R0_fixed_blue_comparison.json')
    fit=read('fit_receipt.json');pre=read('receipt.json');valid=read('linear_validation.json');contract=read('contract.json')
    z=np.load(B/'normal_equations.npz');coef=np.asarray(fit['coefficients_rgb']);views=int(z['views'])
    initial=float(np.mean([v['initial_data_RGB_MSE'] for v in pre['records']]))
    fitted=initial+(np.trace(coef.T@z['H_data']@coef)-2*np.sum(coef*z['g']))/(views*3)
    protected=float(np.trace(coef.T@z['H_protect']@coef)/(views*3))
    cards={}
    for key in ['original_road','clean_original_road','depth_qualified_clean_original_road','guard','nonroad','global_observed_road','global_protected']:
        before=pair['regions'][key]['parent'];after=pair['regions'][key]['candidate']
        cards[key]={'count_pixels':after['pixels'],
                    'n2_abs_RGB_error_sum':before.get('RGB_abs_error_sum'),
                    'n2_material_abs_RGB_error_sum':after.get('RGB_abs_error_sum'),
                    'n2_abs_RGB_error_mean':before.get('RGB_abs_error_mean'),
                    'n2_material_abs_RGB_error_mean':after.get('RGB_abs_error_mean'),
                    'abs_error_sum_definition':'Sum over pixels of mean over 3 absolute RGB channel errors; divide by count_pixels.',
                    'material_vs_n2_image_abs_change_mean':after.get('versus_parent_RGB_abs_change_mean'),
                    'material_vs_n2_image_channel_abs_change_max':after.get('versus_parent_RGB_abs_change_max')}
    result=dict(schema='BLUE_SEAM_EVIDENCE_SUMMARY_V1',status='SMALL_PHOTOMETRIC_GAIN_SEAM_UNRESOLVED',
        blue_acceptance='No seam-resolved/pass claim: fixed near-x residual-jump did not improve with the material control.',
        whole_scene_adopted=False,human_verdict=None,raw=128,bbox=registry['bbox_xyxy_halfopen'],
        bbox_provenance=registry['blue_circle_provenance'],cards=cards,
        total_R0_to_n2_material_original_road={k:total['regions']['original_road'][k]['RGB_abs_error_mean'] for k in ['parent','candidate']},
        primary_material_relative_MAE_improvement=float(1-cards['original_road']['n2_material_abs_RGB_error_mean']/cards['original_road']['n2_abs_RGB_error_mean']),
        fixed_world_band_counts=registry['bands'],
        near_x_residual_jump={v:pair['matched_tangent_residual_jumps']['near_x'][v]['equal_bin_abs_residual_jump_mean'] for v in ['parent','candidate']},
        near_x_paired_tangent_bins=6,
        unsupported=['far-x has no in-ROI supported pixels','lateral has146/75 side pixels but no fixed .2m tangent bin with>=3 pixels on both sides; no lateral residual-jump score'],
        fixed_geometry_weight_exact=pair['geometry_weight_exact_comparison'],
        parent_mechanism={'road_newW_gt_half_conditional_pixels':598,
            'weights_new_old_static':[.8156041503,.0019844559,.1819950491],
            'newroad_mass_normalized_RGB':[.421117872,.429338723,.455679148],
            'static_mass_normalized_RGB':[.061500743,.124745980,.212600023],
            'solo_alpha_mean':.993849218,'joint_over_solo_mean':.820806265,
            'interpretation':'Conditional diagnostic, not primary denominator: solo support is already high while joint static competition remains; screenshot alone does not identify a material problem.'},
        fixed_boundary_group_tables={k:audit['mask_stats'][k] for k in ['band_near_x_minus','band_near_x_plus','band_lateral_y_minus','band_lateral_y_plus']},
        cell_provenance=audit['centre_projection_provenance'],
        method={'basis_count':6,'RGB_parameter_count':18,'active_parent_rows':1270,'all_parent_rows':10098,
            'data_pixels_unique_BUILD40':714546,'data_pixel_states':1429092,
            'full_world_edges':['x=19.8','x=31.8','y=2.71'],'radius_m':.4,
            'source_proxy_bound':.15,'source_proxy_final_maxabs':fit['source_proxy_final_maxabs'],
            'colour_change_maxabs':fit['colour_delta_maxabs'],'basis_outside_exactly_frozen':True,
            'geometry_alpha_native_env_camera_updates':0,
            'ridge':contract['ridge'],'protect_weight':contract['protect_weight'],
            'fit_data':'BUILD40 original true road and frozen vehicle guards; same mask for keep/delete. No hidden/deleted road RGB.',
            'protection':'All pixels outside each fixed fitting mask preserve same n2 rendered parent RGB.',
            'fit_control':'One fixed constrained convex LS per channel; no grid, no additional colour steps, no held9 method/lambda/budget selection.',
            'development_boundary':'Raw128 is BUILD, inherited source proxy used199 views, prior scene/test already opened; same-scene development only.'},
        lifecycle={'smoke_measurement_forwards':4,'smoke_validation_forwards':4,
            'formal_measurement_forwards':80,'formal_validation_forwards':4,'own_total_forwards':92,'backwards':0,
            'normal_equations_restore':'Raw90 keep/delete saved, new process restored checksum and appended raw91 keep/delete before fit entry.',
            'formal_measurement_seconds_approx':18.15,'peak_allocated_bytes':pre['peak_allocated_bytes'],'peak_reserved_bytes':pre['peak_reserved_bytes'],
            'formal_real_render_linear_validation_maxabs':max(r['linear_prediction_render_maxabs'] for r in valid['records'])},
        predicted_BUILD_per_view_data_MSE={'before':initial,'after':float(fitted),'outside_parent_preservation_MSE':protected,'not_a_heldout_score':True},
        material_sha256=fit['material_override_sha256'],
        material_remote_path='/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1/blue/boundary_material_v1/formal/material_override.npz',
        evidence_files=['roi_registration.json','blue_parent_audit.json','n2_material_vs_n2_fixed_blue_comparison.json','n2_material_vs_R0_fixed_blue_comparison.json','contract.json','fit_receipt.json','linear_validation.json','receipt.json'],
        image_files=['blue_frozen_groups_and_bands.png','n2_material_vs_n2_fixed_blue_comparison.png','blue_material_components.png'],
        pending='Parent full49 and separate AA/frozen-coefficient-transfer assessment; no further fit authorized by this result.',
        limits=['Sparse actual scene depth / interpolated measured ground are not dense hidden-road GT',
                'centre depth/expected depth is not physical first hit; weights do not prove correct ordering',
                'No marking/shadow accuracy or learned-road-prior/generalization claim',
                'Blue and red acceptance remain separate; a small MAE gain is insufficient to declare the seam solved'])
    (B/'blue_public_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    # Simple required components diagram for this material control.
    fig,ax=plt.subplots(figsize=(13,3));ax.axis('off')
    cells=[(.01,.4,.19,.4,'Frozen parent scene\nV44 + road cellfit\nn2 geometry / alpha'),
           (.25,.4,.19,.4,'6 world-edge bases\nBUILD40 road + guard\n80 shared-sort forwards'),
           (.49,.4,.21,.4,'CPU constrained LS\n18 RGB coefficients\nsource proxy +/-0.15'),
           (.75,.4,.24,.4,'Persistent material output\nreal-render linear check\nblue / protect / red separate')]
    for x,y,w,h,t in cells:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',fc='#edf3f8',ec='#34495e'))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
    for x in [.21,.45,.71]:ax.annotate('',xy=(x+.03,.6),xytext=(x,.6),arrowprops={'arrowstyle':'->','lw':1.6})
    ax.text(.5,.14,'Geometry, opacity, static/native, environment and cameras are frozen.\nThis tests limited material compensation; residual static competition and hidden geometry remain unverified.',ha='center',va='center',fontsize=11)
    fig.savefig(B/'blue_material_components.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print(str(B/'blue_public_summary.json'))


if __name__=='__main__':main()
