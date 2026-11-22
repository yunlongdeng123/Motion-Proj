"""从本轮真实收据生成37件轻量Git清单及摘要；完整结果只进入archive。"""
from pathlib import Path
import argparse
import collections
import json

PREFIX='docs/autoresearch/road_source_multiframe_20261122'
RUN_ID='road_source_multiframe_20261122_r3'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,default=Path(__file__).resolve().parent.parent);args=ap.parse_args()
    root=args.run_root;out=root/'publication';payload=out/'payload'
    selected_path=out/'selected_files.json'
    if selected_path.exists() and read(selected_path).get('finalized_by_root'):
        raise RuntimeError('Preserve root-finalized selection; reopen it explicitly before rebuilding')
    receipt=read(root/'paired400_seed0/receipt.json');contract=read(root/'paired400_seed0/contract.json')
    audit=read(root/'exposure_audit/paired_cpu_audit.json');cal=read(root/'exposure_audit/shared_exposure.json')
    strict=read(root/'exposure_audit/strict_nonoverlap_validation.json');support=read(root/'exposure_audit/calibration_support_manifest.json')
    assert receipt['complete'] and audit['all_required_state_data_checks_passed']
    assert contract==receipt['contract'] and contract['steps_per_arm']==400
    assert receipt['nativeF']==1684 and receipt['nativeB']==1612
    assert receipt['formal_updates']==800 and receipt['discarded_smoke_restore_updates']==6
    assert cal['calibration_frozen'] and len(cal['records'])==12 and cal['bias_fixed']==0
    strict_main_pass=all(r['minimum8cells_pass'] and r['residual_thresholds_pass'] for r in strict['records'] if r['raw']!=86)
    assert strict_main_pass
    arms={a['label']:a for a in receipt['arms']};frames=[100,110,117,86]
    compact=[]
    for raw in frames:
        values={label:next(r for r in arm['final'] if r['raw']==raw)['raw_camera_metrics'] for label,arm in arms.items()}
        assert len({v['score_n'] for v in values.values()})==1
        compact.append(dict(raw=raw,role='LOW_RES_STRESS' if raw==86 else 'MAIN_QUERY',score_n=values['two_frames']['score_n'],
            arms={label:{k:value[k] for k in ('RGB_MAE','RGB_MSE','spectral')} for label,value in values.items()}))
    write(payload/'paired_query_metrics.json',dict(schema='paired_source_query_metrics_compact.v1',ready=True,
        run='paired400_seed0',domain='Original unresampled Brown observation; canonical render divided by common frame gain then inverse Brown warp',
        main_query_raws=[100,110,117],stress_raws=[86],default_HF_sigma_native_px=5/3,records=compact,
        main_query_equal_view_means=audit['main_query_summary'],full_all12_views_and_both_domains='Archive: paired400_seed0/receipt.json',
        human_verdict=None,problem_solved=None))
    sampling=[]
    for label,pool in contract['pools'].items():
        original=read(root/'paired400_seed0'/f'{label}_sampling.json');schedule=original['schedule']
        counts=collections.Counter(raw for batch in schedule for raw in batch)
        assert len(schedule)==400 and all(len(batch)==len(set(batch))==2 for batch in schedule)
        assert set(counts)==set(pool) and all(n==800//len(pool) for n in counts.values())
        assert not set(counts)&{86,100,110,117}
        sampling.append(dict(arm=label,pool=pool,seed=original['seed'],steps=400,frames_per_step=2,total_samples=800,
            per_frame_counts={str(k):counts[k] for k in pool},query_and_stress_sample_count=0,
            sampler='Repeated shuffled pool; complete precomputed schedule and current step saved in checkpoint',
            full_schedule_archive_path=f'paired400_seed0/{label}_sampling.json',
            online_loss_records=17,online_loss_saved_at='Step1 and every25; not a complete per-step loss trace'))
    write(payload/'sampling_summary.json',dict(schema='paired_sampling_compact.v1',ready=True,arms=sampling))
    write(payload/'strict_exposure_summary.json',dict(schema='strict_external_exposure_compact.v1',ready=True,
        calibration_id=support['calibration_id'],anchor_raw=115,gain_file='exposure_audit/shared_exposure.json',
        bias_fixed=0,gains_refit_for_strict_validation=False,source_core_samples_used=0,
        validation='Remove correspondences whose bilinear support touches any fitting pixel, in both frame and anchor',
        TRAIN8_and_main_QUERY3_pass=strict_main_pass,stress86_weaker=True,
        records=[{k:r[k] for k in ('raw','gain_unchanged','strict_validation_world_cells','strict_matching_world_queries',
            'strict_validation_raw_luma_MAE','strict_validation_canonical_luma_MAE','strict_validation_canonical_RGB_MAE',
            'minimum8cells_pass','residual_thresholds_pass')} for r in strict['records']],
        full_validation_world_cell_keys='Archive: exposure_audit/strict_nonoverlap_validation.json'))
    write(payload/'calibration_support_summary.json',dict(schema='external_exposure_support_compact.v1',ready=True,
        calibration_id=support['calibration_id'],anchor_raw=115,color_space=cal['color_space'],
        domain=cal['calibration_domain'],fit_unique_world_cells=support['fit_unique_world_cells'],
        validation_unique_world_cells=support['validation_unique_world_cells'],
        fit_validation_world_cell_keys_disjoint=True,all_masks_outside_source_core=True,
        pixel_counts_are_not_regression_denominators=True,
        records=[{k:r[k] for k in ('raw','fit_raw_bilinear_support_pixels','validation_raw_bilinear_support_pixels',
            'fit_validation_raw_support_overlap_pixels','fit_matching_world_queries','validation_matching_world_queries',
            'fit_world_cells','validation_world_cells','fit_core_overlap_pixels','validation_core_overlap_pixels')} for r in support['records']],
        original_masks_and_correspondence_table='Archive: exposure_audit/masks/ and external_correspondence_samples.npz'))
    write(payload/'cpu_audit_summary.json',dict(schema='paired_source_CPU_audit_compact.v1',ready=True,
        all_required_state_data_checks_passed=True,both_initial_core_RGB_equal_original_source=True,
        shared_initial_RGB_Adam_RNG_identical=True,source_core_RGB_parameters=2700,pool_is_only_method_difference=True,
        source_geometry_halo_and_fixed_masks_preserved=True,query_core_used_in_loss=False,
        shared_exposure_core_samples_used=0,shared_exposure_gain_values_frozen=True,
        strict_external_exposure_train_and_main_query_pass=True,stress86_exposure_weaker=True,
        native_metrics_max_recompute_delta=audit['native_metrics_max_recompute_delta'],
        fullstate_restore=audit['fullstate_restore'],
        arms=[{k:a[k] for k in ('label','pool','optimizer_step','Adam_tensor_step','formal_samples','per_frame_counts',
            'all_400_two_distinct_frame_batches','schedule_seed0_reproduced','query_and_stress_absent_from_schedule',
            'checkpoint_schedule_plus_step_represents_complete_sampler_cursor','all_rng_states_present',
            'all_non_RGB_fields_exact','halo_RGB_exact','final_core_RGB_equals_checkpoint','reload_all_float_fields_exact',
            'history_recorded_steps','all_recorded_losses_finite_and_sampled_frames_match','per_step_losses_saved')} for a in audit['arms']],
        full_CPU_result='Archive: exposure_audit/paired_cpu_audit.json'))
    write(payload/'resource_summary.json',dict(schema='source_multiframe_resources_compact.v1',ready=True,
        task='ROAD-SOURCE-MULTIFRAME-20261122',run=RUN_ID+'/paired400_seed0',
        nativeF=receipt['nativeF'],nativeB=receipt['nativeB'],formal_optimizer_updates=800,discarded_smoke_restore_updates=6,
        components=dict(data_and_common_initial=dict(nativeF=24,nativeB=0),smoke_restore=dict(nativeF=12,nativeB=12),
            formal_paired_training=dict(nativeF=1600,nativeB=1600),final_memory_and_reloaded_evaluation=dict(nativeF=48,nativeB=0)),
        this_CPU_audit_new_native_calls=0,new_diffusion_calls=0,target_migration_updates=0,
        r2_history_is_separate=True,r2_history=dict(nativeF=918,nativeB=1,RGB_channel_linear_solves=12),
        failure_ledger_refs=['V82-F06'],failure_ledger_delta='updated:V82-F06'))
    restore_path=root/'restore_manifest_compact.json'
    if restore_path.is_file():
        restore=read(restore_path);companion=restore['full_report_companion']
        write(payload/'restore_entry.json',dict(schema='source_multiframe_git_restore_entry.v1',ready=restore['complete'],
            complete=restore['complete'],run_id=RUN_ID,index_only=True,archive=restore['archive'],
            main_archive_role=restore['main_archive_role'],
            full_report_companion={k:companion[k] for k in ('filename','path','bytes','sha256','member_count',
                'members_verified','logical_restore_directory','archive_extract_parent') if k in companion},
            prior_r2_recovery=dict(base_commit='8d3b9bb81d2174115e82c100b131c31fcb444772',
                git_relative_entry='../road_source_fidelity_20261122/restore_manifest_compact.json',
                details_inherited_without_duplicate_catalog=True),
            state_recovery=restore['state_recovery'],report_restore_chain=restore['report_restore_chain'],
            restore_steps=restore['restore_steps'],resources_summary='evidence/resource_summary.json',
            full_original_restore_metadata=str(restore_path),full_original_metadata_unchanged=True,
            source_geometry_changed=False,target_migration=False,target_fusion=False,new_diffusion_calls=0))
    else:
        write(payload/'restore_entry.json',dict(schema='source_multiframe_git_restore_entry.v1',ready=False,
            complete=False,run_id=RUN_ID,awaiting='Full verified archive restore entry'))
    files=[]
    def add(source,destination=None,category='this_run_lightweight',require_ready=False):
        row=dict(source=source,destination=destination or source,category=category)
        if require_ready:row['require_ready']=True
        files.append(row)
    for name in ['README.md','selected_run.json','architecture_components.svg','restore_manifest_compact.json','local_delivery.json']:
        if name=='restore_manifest_compact.json':
            add('publication/payload/restore_entry.json',name,category='compact_recovery_entry',require_ready=True)
        else:add(name,category='final_document_or_recovery_entry')
    add('code/run_paired_source_fit.py',category='actual_GPU_execution_code')
    for name in ['audit_pool.py','retrieve_pool_inputs.py','freeze_receipt.json','pool_selection_summary.json']:
        add('pool_audit/'+name,category='photo_pool_code_or_compact_summary')
    for name in ['build_shared_exposure_cpu.py','export_exposure_support_cpu.py','exposure_interface.py',
                 'validate_exposure_nonoverlap_cpu.py','audit_paired400_cpu.py','shared_exposure.json','source_packet_equivalence.json']:
        add('exposure_audit/'+name,category='shared_exposure_and_independent_CPU_code')
    for name in ['calibration_support_summary.json','strict_exposure_summary.json','cpu_audit_summary.json',
                 'sampling_summary.json','paired_query_metrics.json','resource_summary.json']:
        add('publication/payload/'+name,'evidence/'+name,category='compact_actual_evidence',require_ready=True)
    for name in ['contract.json','initialization_equivalence.json']:
        add('paired400_seed0/'+name,'evidence/'+name,category='actual_contract_and_initial_state_equivalence')
    for name in ['archive_paired400.py','restore_manifest.py','takeback_archive.py']:
        add('recovery/'+name,category='recovery_code')
    for name in ['integrate_pair_report.py','pair_viewer.js','qa_multiframe.mjs','validate_pair.py','qa_summary.json']:
        add('report_source/'+name,'report/'+name,category='report_code_or_compact_QA_summary',require_ready=name=='qa_summary.json')
    for name in ['publish_source_multiframe.py','build_multiframe_payload.py','selected_files.json','narrative.json']:
        add('publication/'+name,category='publication_source_and_manifest')
    assert len(files)==37 and len({r['destination'] for r in files})==37
    write(selected_path,dict(schema='source_multiframe_git_selection.v1',run_id=RUN_ID,docprefix=PREFIX,
        finalized_by_root=False,expected_R2_ancestor='8d3b9bb81d2174115e82c100b131c31fcb444772',
        full_archive_is_required=True,excluded_from_Git=['PT','NPZ','PNG','full per-file hash catalogs','full400 schedules',
            'full paired CPU result','all12-view full metrics','HTML result payload'],files=files))
    print(json.dumps(dict(compact_payloads_written=7,selected_files=37,finalized_by_root=False,
        excluded_full_payloads=True,missing=[r['source'] for r in files if not (root/r['source']).is_file()]),ensure_ascii=False))


if __name__=='__main__':main()
