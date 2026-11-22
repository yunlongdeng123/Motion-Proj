"""Prepare the closed experiment phase's archive draft/estimate; never freeze or create an archive."""
from pathlib import Path
import argparse,datetime,json
import archive_r10_stage as archive

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    read=archive.read;pin=archive.pin;write=archive.write
    ledger=read(r/'docs/current_resource_ledger.json')
    review_path=r/'root_final_stage_review_v1.json';review=read(review_path)
    assert review['complete'] and review['root_actual_joint_views_reviewed']==[86,110,134,179,198]
    assert review['root_actual_joint_decision']=='FAIL' and review['no_pending_R10_GPU_or_experiment_controller']
    assert ledger['R10']['nativeF']==127 and ledger['R10']['nativeB']==76
    assert ledger['R10']['smoke_updates']==5 and ledger['R10']['formal_updates']==0 and ledger['R10']['unclassified_updates']==0
    assert ledger['all_encountered_run_counters_resolved']
    assert ledger['current_existing_representation_budget']['classified_used']==400
    actor=r/'provenance/original_A20_r3/actor.npz';assert archive.sha(actor)=='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7'
    geometry=r/'audit/timewindow17_geometry_v1/candidate_reprojected_v1/candidate_geometry.npz'
    field=r/'timewindow17_shared_requalified_v2/final_atlas.npz'
    assert archive.sha(geometry)=='0d62e87abddd3017a40b06d80c485d331c0feb5c13a433be7f50f6933cda269b'
    assert archive.sha(field)=='45b4b9d752f1584019e6687008de489de071a81907ab2c3c1b61b20964f64d32'
    base=read(r/'docs/final_method_snapshot_receipt.timewindow17.pending.json')
    required=list(base['required_complete_receipts']);critical=set(base['critical_members'])
    successful=[
        'provenance/original_A20_r3/receipt.json','provenance/original_A20_r3/local_takeback_receipt.json',
        'timewindow17_pruned_zero_v4/receipt.json','timewindow17_pruned_footprint_measure_v4/receipt.json',
        'timewindow17_component_contained_policy_v4/scope.json','timewindow17_component_replacement_bounded_zero_v4_replay1/receipt.json',
        'opacity_failed_state_replay86_v1/receipt.json','asset_intrinsic_coverage_v4_v1/receipt.json',
        'intrinsic_coverage_masks_v4_v1/receipt.json','intrinsic_coverage_masks_v4_v1/independent_partition_check.json',
        'intrinsic_coverage_footprint_measure_v4_v1/receipt.json','intrinsic_component_contained_policy_v4_v1/scope.json',
        'intrinsic_component_replacement_zero_v4_v1/receipt.json']
    for rel in successful:
        q=r/rel;assert read(q).get('complete') is True,rel
        required.append(dict(path=rel,sha256=archive.sha(q)));critical.add(rel)
    failures=[
        'edit_footprint_measure_v2/receipt.json','replacement_contained_zero_v1/receipt.json',
        'timewindow17_component_replacement_zero_v4/receipt.json','timewindow17_component_replacement_bounded_zero_v4/receipt.json',
        'opacity_gate_smoke6_v1/resources.json','opacity_gate_smoke6_coverage_v2/resources.json']
    retained_failures=[]
    for rel in failures:
        q=r/rel;d=read(q);assert d.get('complete') is False,rel
        retained_failures.append(dict(path=rel,sha256=archive.sha(q),original_complete=False,error=d.get('error')));critical.add(rel)
    # Current entire directories are critical, rather than adding per-file tests.
    directories=['provenance/original_A20_r3','timewindow17_asset_support_child_v4','asset_intrinsic_coverage_v4_v1',
        'intrinsic_coverage_masks_v4_v1','intrinsic_coverage_footprint_measure_v4_v1','intrinsic_component_contained_policy_v4_v1',
        'intrinsic_component_replacement_zero_v4_v1','opacity_gate_smoke6_v1','opacity_gate_smoke6_coverage_v2',
        'opacity_failed_state_replay86_v1','audit/intrinsic_material_fidelity_v1']
    files,skipped=archive.inventory(r)
    for q,name,fp in files:
        rel=q.relative_to(r).as_posix()
        if any(rel.startswith(x+'/') for x in directories):critical.add(rel)
        if rel.startswith('code/') and (any(x in q.name for x in ['opacity','intrinsic','component','footprint','backup_original','resource','final_stage','freeze'])):critical.add(rel)
    for rel in ['native_early_termination_source_evidence_correction_v1.json','audit/native_early_termination_source_v1.json',
                'source_label_interpretation_correction.json','root_intrinsic_component_readout_contract_v1.json','docs/current_resource_ledger.json','root_final_stage_review_v1.json']:
        assert (r/rel).is_file();critical.add(rel)
    for rel in ['docs/final_report_QA.actual.json','docs/final_report_freeze.actual.json']:
        if (r/rel).is_file():critical.add(rel)
    uniq={x['path']:x for x in required}
    base.update(complete=False,method_stage_frozen=False,overall_run_complete=False,whole_scene_restoration_accepted=False,
        pending_next_footprint_experiment=False,final_footprint_receipt=dict(path='intrinsic_coverage_footprint_measure_v4_v1/receipt.json',sha256=archive.sha(r/'intrinsic_coverage_footprint_measure_v4_v1/receipt.json')),
        final_actual_control_receipt=dict(path='intrinsic_component_replacement_zero_v4_v1/receipt.json',sha256=archive.sha(r/'intrinsic_component_replacement_zero_v4_v1/receipt.json')),
        required_complete_receipts=list(uniq.values()),retained_incomplete_failure_receipts=retained_failures,
        critical_members=sorted(critical),resources=pin(r/'docs/current_resource_ledger.json'),resources_at_preparation=ledger['R10'],
        no_further_experiments_or_training=True,stage_status='awaiting_root_five_view_and_report_QA_freeze',authorization_pending=False,
        human_verdict=None,problem_solved=None,root_all_five_final_views_reviewed=True,report_QA_passed=False,root_review=pin(review_path),
        snapshot_scope='Completed R10 diagnostic phase, all failed and retained branches, whole 17-frame source search, 75-address requalification, original selected A20 r3 full payload, all actual local geometry/source/assets/gates/readouts and optimizer states. Native/DGGT large checkpoints stay external. HTML/report bundle is a separate snapshot.')
    draft=r/'docs/final_method_snapshot_receipt.final_pending.json';write(draft,base)
    compact=dict(schema='R10.final_diagnostic_stage_compact.pending.v1',complete=True,ready=False,finalized_by_root=False,
        prepared_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),method_stage_frozen=False,whole_scene_restoration_accepted=False,
        human_verdict=None,problem_solved=None,formal_existing_budget=ledger['current_existing_representation_budget'],resources=ledger['R10'],
        resource_ledger=pin(r/'docs/current_resource_ledger.json'),original_selected_A20=pin(actor),geometry=pin(geometry),source_field=pin(field),
        whole_field_unaffected_ancestor_real_locks=6332,actual_patch_strict_new_real_vertices=0,full_source_candidate_records=2999905,
        source_qualification_note='Conservative qualification result, not proof that all photos were physically invisible; ancestry and current-XYZ certification are separate.',
        final_control=pin(r/'intrinsic_component_replacement_zero_v4_v1/receipt.json'),failed_opacity_smokes=retained_failures[-2:],
        technical_status='Root all-five whole-image review FAIL. This diagnostic phase has ended; archive freeze/delivery remains explicit. No further experiment is queued.',root_review=pin(review_path),
        full_archive_required=True,light_takeback_not_a_substitute=True,HTML_snapshot_separate=True,default_assets_not_promoted=True)
    write(r/'docs/final_stage_compact.pending.json',compact)
    status=read(r/'docs/current_stage_publication_inputs.pending.json')
    status.update(stage_status='completed_negative_pending_method_archive_freeze',blocked_technical=[],authorization_pending=False,
        resources_at_preparation=ledger['R10'],root_review=pin(review_path),whole_scene_restoration_accepted=False,human_verdict=None,problem_solved=None,
        no_further_experiments_or_training=True,method_stage_frozen=False,
        required_next_evidence=['Root explicit method freeze signal','Full R10 archive and local all-member verification','Final report bundle snapshot kept separate','Root finalized isolated-index research publication and push verification'])
    status['bindings']['final_actual_control']=pin(r/'intrinsic_component_replacement_zero_v4_v1/receipt.json')
    status['bindings']['original_selected_A20_backup']=pin(actor)
    write(r/'docs/current_stage_publication_inputs.pending.json',status)
    target=r.parent/'old_raw_teacher_joint_20261123_r9/publication/compact_result.current_stage.pending.json';write(target,compact)
    estimate=archive.estimate(r);write(r/'archives/final_phase_payload_estimate.json',estimate)
    print(json.dumps(dict(draft=str(draft),method_stage_frozen=False,required_complete_receipts=len(uniq),retained_failed_receipts=len(retained_failures),critical_members=len(critical),eligible_files=estimate['eligible_files'],payload_bytes=estimate['eligible_payload_bytes'],R10=ledger['R10'],archive_created=False)))
if __name__=='__main__':main()
