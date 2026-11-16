"""Assemble completed independent audits; does not run additional checks or GPU work."""
from pathlib import Path
import argparse
import hashlib
import json


NEW = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_direct_teacher_20261116_r1')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, default=NEW)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.campaign
    sources = dict(supervision=root/'qa/supervision_r3_independent/acceptance.json',
                   lifecycle=root/'qa/lifecycle_r4_independent/acceptance.json',
                   formal=root/'qa/formal_pair_independent/acceptance.json',
                   color=root/'qa/public_color_independent/acceptance.json',
                   coverage=root/'qa/public_coverage_independent/acceptance.json',
                   final=root/'qa/public_final_independent/acceptance.json',
                   frequency=root/'qa/frequency_diagnostic_r1/diagnostic.json',
                   engineering=root/'phase0/lifecycle_r4/engineering_accounting.json')
    records = {key:read(path) for key,path in sources.items()}
    final = records['final']
    stage_records = []
    for key,label,steps in [('color','color',200),('coverage','coverage',300),('final','joint',300)]:
        value = records[key]
        stage_records.append(dict(stage=label,formal_steps_each_arm=steps,
                                  numerical_status=value['render_mechanical_status'],
                                  direct_pseudo_teacher_goal=value['direct_pseudo_teacher_20percent_goal'],
                                  support_totals=value['totals'],
                                  visual_status='failed',
                                  visual_observation='Dense horizontal stripes already visible after color200; coverage does not remove them; joint slightly increases them.'))

    teacher_rows = []
    for index,raw in enumerate((86,110)):
        row = final['frames'][index]
        source = records['supervision']['frames'][index]
        teacher_rows.append(dict(raw=raw,A_real_photo_pixels=source['A_photo_pixels'],
                                 C_common_pseudo_photo_pixels=source['C_common_photo_pixels'],
                                 C_full_any_native_footprint_pixels=source['full_C_any_footprint_pixels'],
                                 C_complete_native_footprint_pixels=source['full_C_complete_footprint_pixels'],
                                 C_full_qualified_geometry_pixels=source['qualified_C_geometry_pixels'],
                                 alpha_A_geometry_pixels=source['coverage_A_pixels'],
                                 alpha_C_geometry_pixels=source['coverage_C_pixels'],
                                 baseline=row['stages']['baseline'],direct=row['stages']['direct_C'],
                                 matched_F_control=row['stages']['matched_F_control']))

    reload_rows = []
    for role in ('direct_C','matched_F_control'):
        for stage in ('color','coverage','joint'):
            receipt = read(root/role/'formal_r1'/('stage_'+stage)/'reload.json')
            reload_rows.extend(dict(role=role,stage=stage,**row) for row in receipt['rows'])
    frequency = []
    for row in records['frequency']['frames']:
        if row['raw'] in (134,179,198):
            frequency.append(dict(raw=row['raw'],common_support_pixels=row['fixed_common_support_pixels'],
                                  direct_color200_y_second_difference_ratio=row['metrics']['direct_color']['y_ratio_to_baseline'],
                                  direct_final_y_second_difference_ratio=row['metrics']['direct_final']['y_ratio_to_baseline'],
                                  control_final_y_second_difference_ratio=row['metrics']['control_final']['y_ratio_to_baseline'],
                                  native_crop=str(root/'qa/frequency_diagnostic_r1'/f"{row['raw']:03d}_native_crop.png")))
    mechanical = (records['supervision']['status']=='pass' and records['lifecycle']['status']=='pass'
                  and records['formal']['mechanical_status']=='pass' and final['render_mechanical_status']=='pass')
    numerical = mechanical and final['direct_pseudo_teacher_20percent_goal']=='pass'
    scripts = ['qa/audit_direct_supervision_r3.py','qa/audit_lifecycle.py','qa/audit_formal_pair.py',
               'qa/compare_public_renders.py','qa/diagnose_frequency.py','qa/summarize_acceptance.py']
    result = dict(schema='module2_direct_teacher_final_independent_acceptance.v1',
                  overall_status='failed_visual_quality',mechanical_status='pass' if mechanical else 'failed',
                  numerical_goal_status='pass' if numerical else 'failed',road_visual_quality_status='failed',
                  human_verdict=None,production_promoted=False,
                  budget=dict(formal_updates_total=1600,formal_updates_per_arm=800,
                              schedule_each_arm=[200,300,300],training_frames=[86,110],
                              formal_samples_per_frame_each_arm=400,
                              engineering_updates_this_round=records['engineering']['total_engineering_updates_this_round'],
                              engineering_breakdown=records['engineering']['detail'],
                              engineering_prior_color_probe_updates=records['engineering']['prior_successful_color_probe_updates'],
                              prior_module2_600_1200_campaign='separate inherited baseline; its 1800 formal/72 engineering updates are not this-round updates',
                              new_SD_generations=0,new_199_frame_render=False),
                  fixed_domains=dict(road_evaluation_pixel_frames=38416,full_keep_pixel_frames=77266,
                                     native_image_wh=[1920,1280],render_grid_wh=[480,320],
                                     source_reduction='exact 4x4 native pixel footprint; counts below are render-grid pixel-frames',
                                     training_only=[86,110],diagnostic_only=[134,179,198],blind_B_executed=False,
                                     inherited_build_frames=199,
                                     whole_physical_hole_not_equal_common_C_photo_domain=True),
                  final_coverage=final['totals'],coverage_gain_over_current_baseline=0,
                  final_keep_joint_max_difference=max(row['stages'][role]['K_joint_max_difference'] for row in final['frames']
                                                      for role in ('direct_C','matched_F_control')),
                  numerical_thresholds=dict(C_each_teacher_relative_error_reduction_minimum=.20,
                                            A_each_frame_absolute_error_regression_maximum=.002,
                                            supported_alpha_threshold=.9,gap_alpha_threshold=.05,
                                            supported_total_minimum=37404,per_frame_support_nonregression=True,
                                            full_keep_joint_RGB_max_difference=0.,saved_loader_RGB_alpha_max_error=1e-6,
                                            source_RGB8_reload_separately_reported=True),
                  teacher_A_metrics=teacher_rows,stages=stage_records,
                  saved_public_loader=dict(max_RGB_float_error=max(row['RGB_float_max'] for row in reload_rows),
                                           max_alpha_float_error=max(row['alpha_float_max'] for row in reload_rows),
                                           max_RGB8_error=max(row['RGB8_max'] for row in reload_rows),
                                           tested_stage_frame_pairs=len(reload_rows)),
                  complete_state_restore='exact model/Adam/paramgroups/sampler/queue/trace/all RNG restoration',
                  next_update_replay=dict(bitwise=records['lifecycle']['bitwise_next_update'],
                                          tensor_max_differences=records['lifecycle']['replay_tensor_differences'],
                                          nonparameter_metadata_exact=not records['lifecycle']['replay_metadata_differences'],
                                          declared_tolerance=records['lifecycle']['replay_declared_tolerance']),
                  paired_control=dict(initial_model_exact=True,masks_q_scopes_camera_order_budget_seed_LR_equal=True,
                                      only_C_color_target_source_differs=True,
                                      control_target='immutable original R8 F projection; not a replay of the old phase2 method'),
                  visual_failure=dict(observations=['New or intensified dense horizontal stripes in diagnostic frames 134,179,198',
                                                     'Inherited marking discontinuities remain',
                                                     'Block color seams are reduced in places but the road surface is not visually accepted'],
                                      appearance_stage_attribution='color200 already has the stripes while opacity/scales are bitwise frozen; coverage does not remove them',
                                      direct_and_control_affected=True,frequency_descriptive_evidence=frequency,
                                      frequency_is_not_an_added_acceptance_threshold=True),
                  representation='direct per-Gaussian sRGB learned override; immutable prior atlas and observation ledger retained separately',
                  interpretation=dict(A='real visible training fidelity, not held-out generalization',
                                      C='frozen pseudo-teacher agreement, not hidden road ground truth',
                                      diagnostic='not world-isolated blind tests',
                                      shared_theta_scene_z_imaging_eta='not trained or introduced by this experiment',
                                      phase_or_marking_structure_truth='not independently established'),
                  source_artifacts={key:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key,path in sources.items()},
                  final_reproduction_QA_scripts=scripts,audit_GPU_forwards=0,audit_optimizer_updates=0)
    output = args.output or root/'qa/final_acceptance/acceptance.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),mechanical=result['mechanical_status'],numerical=result['numerical_goal_status'],
                          visual=result['road_visual_quality_status'],budget=result['budget'],reload=result['saved_public_loader'])))


if __name__ == '__main__':
    main()
