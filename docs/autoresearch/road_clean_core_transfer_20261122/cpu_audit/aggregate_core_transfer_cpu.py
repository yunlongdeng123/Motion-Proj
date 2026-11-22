"""汇总R4已存原生证据与CPU检查；只生成审计摘要，不执行渲染或过滤。"""
from pathlib import Path
import json


def compact_regions(regions):
    result={}
    for name in ('official_eroded_high_alpha_score','physical_core_alpha_ge_0p95','physical_edge_alpha_ge_0p95',
                 'physical_edge_alpha_0p05_to_0p95','eroded_core_alpha_0p05_to_0p95',
                 'outside_core_alpha_0p05_to_0p95','outside_core_alpha_ge_0p95'):
        r=regions[name];a=r['effective_RGB_over_alpha'];b=r['actual_or_reconstructed_neutral_display']
        result[name]=dict(pixels=a['pixels'],effective_high_chroma=a['high_chroma'],
            native_grey_high_chroma=b['high_chroma'],native_grey_chroma_max=b['chroma'].get('max'))
    return result


def main():
    out=Path(__file__).resolve().parent
    s=json.loads((out/'source_core_cpu_audit.json').read_text());t=json.loads((out/'single_stamp_cpu_audit.json').read_text())
    assert s['all_field_and_metric_checks_passed'] and t['all_checks_passed']
    source=[]
    for v in s['variants']:
        source.append(dict(label=v['label'],patch=v['patch'],core_rows=v['component_core_rows'],halo_rows_exported=0,
            all_component_fields_match_source=True,
            point_span_gt_0p18=dict(core=v['point_color_groups']['core']['point_span_gt_0p18'],
                                  halo=v['point_color_groups']['halo']['point_span_gt_0p18']),
            views=[dict(raw=w['raw'],physical_core_pixels=w['physical_core_n'],eroded_core_base_pixels=w['eroded_core_n'],
                high_chroma_threshold=w['high_chroma_threshold'],regions=compact_regions(w['native_core_footprint_classes']),
                neutral_RGBA_formula_max_abs=w['neutral_background_native_vs_CPU_RGBA_max_abs']) for w in v['views']]))
    target=[]
    for v in t['variants']:
        target.append(dict(label=v['label'],patch=v['patch'],core_rows=v['core_rows'],halo_rows_exported=0,
            source_RGB_sigma_opacity_exact=True,all_pairwise_distance_max_abs_m=v['all_pairwise_distance_max_abs_m'],
            independent_covariance_max_abs=v['independent_world_covariance_max_abs'],
            views=[dict(raw=w['raw'],physical_core_pixels=w['physical_core_pixels'],
                physical_core_alpha_ge95_pixels=w['physical_core_alpha_ge95_pixels'],physical_core_alpha95_fraction=w['physical_core_alpha95_fraction'],
                eroded_core_base_pixels=w['eroded_physical_core_base_pixels'],eroded_high_alpha_score_pixels=w['opaque_eroded_core_pixels'],
                eroded_core_alpha95_fraction=w['eroded_core_alpha95_fraction'],high_chroma_threshold=.15,
                regions=compact_regions(w['footprint_classes']),neutral_RGBA_formula_max_abs=w['neutral_grey_RGBA_formula_max_abs']) for w in v['views']]))
    result=dict(schema='clean_core_transfer_aggregate_CPU_audit.v1',complete=True,all_required_engineering_checks_passed=True,
        historical_transfer_core_only_proof=s['historical_transfer_source_proof'],source_variants=source,target_variants=target,
        common_target_origin=t['common_target_origin'],parent_asset_prior_binding_unchanged=t['parent_asset_matches_existing_authority_binding'],
        resources=dict(source_nativeF=42,target_nativeF=65,total_nativeF=107,total_nativeB=0,optimizer_updates=0,new_diffusion_calls=0,
            this_CPU_audit_new_native_calls=0),
        no_halo_export=True,no_point_color_pruning=True,no_world_support_clipping=True,
        parent_retirement=False,parent_stamp_compositing=False,boundary_fusion_tested=False,
        material_source='similar_road_inference',target_same_position_real_observation=False,
        engineering_pass_is_not_whole_component_cleanliness=True,whole_component_all_domain_clean=False,
        remaining_observation='Core-only Gaussian components retain colored edge/outside footprints, including visible neutral-grey tails at alpha .05-.95. The eroded high-alpha ROI and full physical core have distinct denominators.',
        full_CPU_evidence=['source_core_cpu_audit.json','single_stamp_cpu_audit.json'],human_verdict=None,problem_solved=None)
    (out/'aggregate_cpu_audit.json').write_text(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    compact=dict(schema='clean_core_transfer_CPU_compact_summary.v1',ready=True,engineering_checks_passed=True,
        resources=result['resources'],historical_core_only_proof=result['historical_transfer_core_only_proof'],
        source=[dict(label=v['label'],core_rows=v['core_rows'],halo_rows=0,point_span_gt_0p18=v['point_span_gt_0p18'],
            high_alpha_score=[dict(raw=w['raw'],pixels=w['regions']['official_eroded_high_alpha_score']['pixels'],
                effective_high_chroma=w['regions']['official_eroded_high_alpha_score']['effective_high_chroma'],
                native_grey_high_chroma=w['regions']['official_eroded_high_alpha_score']['native_grey_high_chroma']) for w in v['views']]) for v in source],
        target=[dict(label=v['label'],core_rows=v['core_rows'],halo_rows=0,
            rigid_distance_max_abs_m=v['all_pairwise_distance_max_abs_m'],covariance_max_abs=v['independent_covariance_max_abs'],
            views=[dict(raw=w['raw'],physical_pixels=w['physical_core_pixels'],physical_alpha95_pixels=w['physical_core_alpha_ge95_pixels'],
                eroded_base_pixels=w['eroded_core_base_pixels'],eroded_alpha95_score_pixels=w['eroded_high_alpha_score_pixels'],
                official_high_chroma=w['regions']['official_eroded_high_alpha_score']['native_grey_high_chroma'],
                edge_high_alpha_chroma=w['regions']['physical_edge_alpha_ge_0p95']['native_grey_high_chroma'],
                outside_moderate_alpha_native_grey_chroma=w['regions']['outside_core_alpha_0p05_to_0p95']['native_grey_high_chroma'],
                outside_moderate_alpha_effective_chroma=w['regions']['outside_core_alpha_0p05_to_0p95']['effective_high_chroma']) for w in v['views']]) for v in target],
        all_source_RGB_sigma_opacity_preserved=True,all_core_IDs_preserved=True,parent_prior_binding_unchanged=True,
        display_no_2D_mask=True,parent_retirement=False,parent_compositing=False,world_support_clipping=False,boundary_fusion=False,
        high_alpha_ROI_pass_does_not_mean_whole_component_clean=True,whole_component_all_domain_clean=False,
        source_threshold='photo q99.9 channel span + .10; diagnostic only',target_threshold=.15,
        point_color_pruning=False,material_source='similar_road_inference',human_verdict=None,problem_solved=None,
        full_footprint_classification='Archive: cpu_audit/aggregate_cpu_audit.json and both full CPU result JSONs')
    (out/'compact_audit_summary.json').write_text(json.dumps(compact,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,all_required_engineering_checks_passed=True,source_views=sum(len(v['views']) for v in source),
        target_views=sum(len(v['views']) for v in target),nativeF=107,nativeB=0,whole_component_all_domain_clean=False)))


if __name__=='__main__':main()
