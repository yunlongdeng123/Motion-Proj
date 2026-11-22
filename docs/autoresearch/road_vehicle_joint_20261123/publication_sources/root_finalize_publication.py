"""Root-invoked final metadata and one immutable publication snapshot. No Git."""
from pathlib import Path
import argparse,json,hashlib
import prepare_inputs
from refresh_final_selection_cpu import refresh

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,d):
    p=Path(p);assert not p.exists(),str(p)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    assert a.execute,'Explicit root invocation required'
    root=a.root.resolve();pub=root/'publication';r10=root.parent/'road_underbody_local_20261123_r10'
    local=read(pub/'R10_local_takeback_receipt.json');previous=read(pub/'R8_R9_local_takeback_receipt.json')
    assert local['complete'] and local['decision']=='PASS' and local['all_member_SHA_verified']
    assert local['member_count']==2416 and local['archive']['sha256']=='4f5d4cffd49f6bd8434a05658fcd74a4f92f03a6a9fdd2a65222dfdc578358d7'
    assert previous['complete'] and previous['local_archive_and_all_1700_member_SHA_verified']
    review=read(r10/'root_final_stage_review_v1.json');ledger=read(r10/'docs/current_resource_ledger.json')
    freeze=read(r10/'docs/final_method_snapshot_receipt.json');report=read(r10/'docs/final_report_freeze.actual.json')
    qa=read(r10/'docs/final_report_QA.actual.json')
    assert review['complete'] and review['method_stage_review_complete'] and review['root_actual_joint_views_reviewed']==[86,110,134,179,198]
    assert freeze['complete'] and freeze['method_stage_frozen'] and not freeze['whole_scene_restoration_accepted']
    assert ledger['R10']['nativeF']==127 and ledger['R10']['nativeB']==76 and ledger['R10']['smoke_updates']==5
    assert ledger['R10']['formal_updates']==0 and ledger['current_existing_representation_budget']['classified_used']==400
    assert qa.get('decision')=='PASS' or qa.get('passed') or qa.get('pass') or qa.get('all_checks_passed')
    assert report['original_HTML']['sha256']=='7058cba72afc2473723786ff7a900a20cd365fbeae32149cfbd56ee5e37c7111'
    resource_keys=['R10','historical_separate','current_existing_representation_budget','DrivingEditor_historical_plus_R10','DGGT']
    compact=dict(schema='ROAD-VEHICLE-JOINT-20261123.final_finite_stage.v1',task='ROAD-VEHICLE-JOINT-20261123',
        method_stage_complete=True,overall_run_complete=False,whole_scene_restoration_accepted=False,production_asset_promoted=False,
        human_verdict=None,problem_solved=None,failure_ledger_refs=['V82-F06'],failure_ledger_delta='updated:V82-F06',
        result=review,resources={k:ledger[k] for k in resource_keys},
        resource_classification_note='Frozen ledger failed_engineering_* counts all incomplete runs, including protection stops. They are not all wiring/numerical errors. Per-run mode/exception and actual counters remain the authority.',
        incomplete_call_classification=[dict(kind='FP32,CLI and exact-readout engineering controls',nativeF=4,nativeB=1),
            dict(kind='v1 protection-readout stop before revision',nativeF=8,nativeB=2),dict(kind='v2 constrained opacity-control stop',nativeF=9,nativeB=3)],
        report=dict(url='http://127.0.0.1:18765/index.html#road-underbody-local',HTML=report['original_HTML'],QA=pin(r10/'docs/final_report_QA.actual.json'),
            freeze_metadata=pin(r10/'docs/final_report_freeze.actual.json'),only_original_public_HTML=True),
        archives=dict(R8_R9=previous,R10=local,original_vehicle_archive_member='provenance/original_A20_r3/actor.npz',
            original_vehicle_sha256='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7',original_vehicle_bytes=24031638,
            original_native_models_and_DGGT_large_weights='External immutable references remain required; see archive dependency/path-map indexes.'),
        source_00_02_correction_preserved=True,R7_NOT_ACCEPTED_preserved=True,GPU_stays_on=True)
    prepared=prepare_inputs.prepare(root,pub/'R8_R9_local_takeback_receipt.json',pub/'R10_local_takeback_receipt.json')
    selection=Path(prepared['selection']);stage=prepared['current_stage_snapshot']
    for row in read(selection)['files']:
        selected_path=root/row['source']
        if selected_path.is_file():assert selected_path.stat().st_size<300000,('Publication text budget',str(selected_path))
    write(root/'compact_result.json',compact)
    delivery=dict(schema='ROAD-VEHICLE-JOINT-20261123.explicit_root_delivery.v1',ready=True,finalized_by_root=True,
        current_stage_snapshot_sha256=stage['sha256'],delivery_scope='Finite R8/R9/R10 method/asset controls and the existing road HTML; edited whole-scene quality is not accepted.',
        method_stage_complete=True,overall_run_complete=False,whole_scene_restoration_accepted=False,production_asset_promoted=False,
        report=compact['report'],archives=compact['archives'],resources=compact['resources'],GPU_stays_on=True,
        human_verdict=None,problem_solved=None,failure_ledger_refs=['V82-F06'])
    write(pub/'final_delivery.json',delivery)
    refreshed=refresh(root,selection);final_selection=Path(refreshed['output_selection']['path'])
    narrative=dict(task='ROAD-VEHICLE-JOINT-20261123',finalized_by_root=True,selected_manifest_sha256=sha(final_selection),
        failure_ledger_refs=['V82-F06'],failure_ledger_delta='updated:V82-F06',new_failure_family=False,human_verdict=None,problem_solved=None,
        formal400_and_fresh_reviewed=True,official_DGGT_control_reviewed=True,archive_verified=True,report_QA_passed=True,takeback_verified=True,
        source_00_02_correction_preserved=True,R7_NOT_ACCEPTED_preserved=True,current_stage_snapshot_reviewed=True,
        current_stage_snapshot_sha256=stage['sha256'],R10_archive_takeback_reviewed=True,
        **{k:v for k,v in read(pub/'narrative.root_texts.json').items() if k!='task'})
    write(pub/'narrative.json',narrative)
    finalized=prepare_inputs.finalize(root,final_selection)
    result=dict(complete=True,decision='PASS',current_stage_snapshot=stage,selection=pin(pub/'selected_files.json'),
        compact=pin(root/'compact_result.json'),delivery=pin(pub/'final_delivery.json'),narrative=pin(pub/'narrative.json'),
        Git_commands_executed=False,finalized_config=finalized)
    write(pub/'root_final_publication_prepare.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
