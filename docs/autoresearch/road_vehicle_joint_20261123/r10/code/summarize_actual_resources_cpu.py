"""Sum actual top-level native-run receipts, retaining failed engineering calls and separating DGGT."""
from pathlib import Path
import argparse,datetime,hashlib,json

BASELINE_RUNS=['measure_v1','retire0_strict_v1','body_overlap_diagnostic_v1','new_local_zero_v1','new_local_background_zero_v1','edit_footprint_measure_v2','edit_footprint_measure_v3','replacement_contained_zero_v1','replacement_contained_zero_v1_replay1']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def scalar(d,keys,default=None):
    found=[(k,d[k]) for k in keys if k in d]
    if not found:return default,[]
    assert all(type(v) is int and v>=0 for k,v in found),found
    assert len(set(v for k,v in found))==1,'Conflicting counters are not auto-corrected: '+repr(found)
    return found[0][1],[k for k,v in found]

def run_documents(folder):
    receipt_path=folder/'receipt.json';resource_path=folder/'resources.json'
    if not(receipt_path.is_file() or resource_path.is_file()):return None
    receipt=read(receipt_path) if receipt_path.is_file() else None
    resources=read(resource_path) if resource_path.is_file() else None
    primary=resources if resources is not None else receipt
    primary_path=resource_path if resources is not None else receipt_path
    return dict(primary=primary,primary_path=primary_path,receipt=receipt,resources=resources,
        receipt_path=receipt_path if receipt is not None else None,resource_path=resource_path if resources is not None else None)

def actual_run(folder,docs=None):
    docs=docs or run_documents(folder)
    if docs is None:return None,None
    d=docs['primary'];result=docs['receipt'] or {};primary_path=docs['primary_path']
    if d.get('CPU_only') is True:return None,None
    counter_names=['actual_nativeF','nativeF','nativeB','actual_nativeB','native_F_attempts']
    if not any(k in d for k in counter_names):
        if docs['resources'] is not None or ('actual_optimizer_updates' in result and result.get('mode') in ['fit','smoke']):
            return None,dict(run=folder.name,reason='Authoritative resources/counters not available; no inference from logical steps or result receipt',counter_source=pin(primary_path))
        return None,None
    f,fkeys=scalar(d,['actual_nativeF','nativeF']);b,bkeys=scalar(d,['actual_nativeB','nativeB'])
    if f is None or b is None:
        return None,dict(run=folder.name,reason='Explicit completed native F/B counters missing',counter_source=pin(primary_path))
    attempts,akeys=scalar(d,['native_F_attempts']);updates,ukeys=scalar(d,['actual_optimizer_updates','updates'],0)
    mode=d.get('mode',result.get('mode'))
    explicit_formal,fmkeys=scalar(d,['formal_updates']);explicit_smoke,smkeys=scalar(d,['smoke_updates'])
    if explicit_formal is not None or explicit_smoke is not None:
        formal=explicit_formal or 0;smoke=explicit_smoke or 0
        assert formal+smoke<=updates,'Explicit update classes exceed actual_optimizer_updates'
        unclassified=updates-formal-smoke;classification='explicit_authoritative_update_counters'
    else:
        is_smoke='smoke' in folder.name.lower() or mode=='smoke'
        is_formal='formal' in folder.name.lower() or mode=='fit' or d.get('budget_classification')=='formal'
        smoke=updates if is_smoke else 0;formal=updates if is_formal and not is_smoke else 0
        unclassified=updates-formal-smoke
        classification='smoke' if is_smoke else 'formal' if is_formal else 'zero_update' if not updates else 'needs_root_classification'
    differences={}
    if docs['resources'] is not None and docs['receipt'] is not None:
        for key in ['nativeF','actual_nativeF','nativeB','actual_nativeB','actual_optimizer_updates','updates','formal_updates','smoke_updates']:
            if key in result and key in d and result[key]!=d[key]:differences[key]=dict(resources=d[key],receipt=result[key])
    exception_files=[pin(folder/name) for name in ['exception.json','failure_receipt.json','failure.json','error.json'] if (folder/name).is_file()]
    return dict(run=folder.name,schema=result.get('schema',d.get('schema')),mode=mode,complete=d.get('complete') is True,
        result_receipt_complete=result.get('complete') if result else None,
        actual_nativeF=f,actual_nativeB=b,native_F_attempts=attempts,actual_optimizer_updates=updates,
        formal_updates=formal,smoke_updates=smoke,unclassified_updates=unclassified,update_class=classification,
        logical_steps_completed=d.get('logical_steps_completed',result.get('logical_steps')),
        extraction_keys=dict(F=fkeys,B=bkeys,attempts=akeys,updates=ukeys,formal=fmkeys,smoke=smkeys),
        counter_source=pin(primary_path),counter_source_kind='resources.json' if docs['resources'] is not None else 'receipt.json',
        resources=pin(docs['resource_path']) if docs['resource_path'] else None,
        receipt=pin(docs['receipt_path']) if docs['receipt_path'] else None,
        error=d.get('error') or result.get('error'),resource_error=d.get('error'),result_error=result.get('error'),exception_files=exception_files,
        receipt_resource_counter_differences=differences,resources_authoritative_not_added_to_receipt=True),None
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--r10',type=Path,required=True);p.add_argument('--r8',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=[];skipped=[];controllers=[]
    for folder in sorted(a.r10.iterdir()):
        if not folder.is_dir():continue
        docs=run_documents(folder)
        if docs is None:continue
        d=docs['receipt'] or docs['primary'];path=docs['receipt_path'] or docs['primary_path']
        if 'controller' in str(d.get('schema','')) and isinstance(d.get('actual_GPU_jobs'),list) and isinstance(d.get('stages'),list):
            children={}
            for stage in d['stages']:
                ref=stage.get('actual_receipt')
                if not ref:continue
                child_path=Path(ref['path']);assert pin(child_path)['sha256']==ref['sha256']
                child,unresolved=actual_run(child_path.parent);assert unresolved is None,unresolved
                if child and (child['actual_nativeF'] or child['actual_nativeB']):children[str(child_path)]=dict(receipt=pin(child_path),counter_source=child['counter_source'],nativeF=child['actual_nativeF'],nativeB=child['actual_nativeB'])
            reported_f,_=scalar(docs['primary'],['actual_nativeF','nativeF'],0);reported_b,_=scalar(docs['primary'],['actual_nativeB','nativeB'],0)
            assert sum(x['nativeF'] for x in children.values())==reported_f
            assert sum(x['nativeB'] for x in children.values())==reported_b
            assert all(Path(x).parent.parent==a.r10 for x in children),'Controller child execution is not represented by canonical top-level run receipt'
            controllers.append(dict(run=folder.name,schema=d['schema'],receipt=pin(path),reported_nonadditive_nativeF=reported_f,reported_nonadditive_nativeB=reported_b,
                child_actual_receipts=list(children.values()),child_counters_match_controller=True,counted_in_R10_subtotal=False,reason='Aggregate controller index; actual child run receipts are counted once'))
            continue
        entry,unresolved=actual_run(folder,docs)
        if unresolved:skipped.append(unresolved)
        if entry:runs.append(entry)
    baseline=[r for r in runs if r['run'] in BASELINE_RUNS];assert len(baseline)==len(BASELINE_RUNS)
    assert sum(r['actual_nativeF'] for r in baseline)==36 and sum(r['actual_nativeB'] for r in baseline)==16
    totals=dict(nativeF=sum(r['actual_nativeF'] for r in runs),nativeB=sum(r['actual_nativeB'] for r in runs),
        actual_optimizer_updates=sum(r['actual_optimizer_updates'] for r in runs),
        failed_engineering_nativeF=sum(r['actual_nativeF'] for r in runs if not r['complete']),
        failed_engineering_nativeB=sum(r['actual_nativeB'] for r in runs if not r['complete']),
        formal_updates=sum(r['formal_updates'] for r in runs),
        smoke_updates=sum(r['smoke_updates'] for r in runs),
        unclassified_updates=sum(r['unclassified_updates'] for r in runs))
    source=a.r8/'forward_smoke_five_views_v1/receipt.json';recover=a.r8/'forward_smoke_five_views_recovered_v2/receipt.json';control=a.r8/'dggt_no_diff_control_v1/receipt.json'
    s,r,c=read(source),read(recover),read(control)
    assert r['inherited_completed_model_forward_calls']==s['completed_model_forward_calls']==1 and r['new_model_forward_calls']==0 and c['model_forward_calls']==0
    dggt=dict(unique_main_network_forwards=1,recovery_new_main_network_forwards=0,
        inherited_main_network_forward_not_double_counted=True,sky_auxiliary_forward_completed=r['sky_model_forward_completed'],
        sky_camera_raster_F=r['DGGT_sky_raster_F'],original_scene_camera_raster_F=c['DGGT_raster_F'],
        camera_raster_F_total=r['DGGT_sky_raster_F']+c['DGGT_raster_F'],
        sky_failed_API_attempts_retained=s['DGGT_sky_raster_api_attempts'],sky_failed_API_completed=s['DGGT_sky_raster_api_completed'],
        recovery_API_attempts=r['DGGT_sky_raster_api_attempts'],recovery_API_completed=r['DGGT_sky_raster_api_completed'],
        original_scene_API_attempts=c['DGGT_raster_api_attempts'],original_scene_API_completed=c['DGGT_raster_api_completed'],
        total_API_attempts=s['DGGT_sky_raster_api_attempts']+r['DGGT_sky_raster_api_attempts']+c['DGGT_raster_api_attempts'],
        total_API_completed=s['DGGT_sky_raster_api_completed']+r['DGGT_sky_raster_api_completed']+c['DGGT_raster_api_completed'],
        training_updates=0,new_diffusion_calls=0,DrivingEditor_nativeF_included=False,receipts=[pin(x) for x in [source,recover,control]])
    out=dict(schema='R10.live_actual_resource_ledger.v1',complete=True,prepared_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        live_ledger_not_final_freeze=True,method_stage_frozen=False,R10=totals,R10_runs=runs,aggregate_controllers_not_counted=controllers,unresolved_receipt_counter_schemas=skipped,
        pre_new17_actual_baseline=dict(nativeF=36,nativeB=16,failed_1F1B_included=True,runs=BASELINE_RUNS),
        historical_separate=dict(R7=dict(nativeF=956,nativeB=811,historical_formal_updates=800),R8=dict(nativeF=970,nativeB=833,formal_updates=400,smoke_updates=13),R9=dict(new_semantic_calls=2,new_diffusion_calls=0),
            authority='Root-provided fixed historical subtotals; preserved raw receipts/archives remain the authority. This script recomputes the current R10 subtotal only.'),
        current_existing_representation_budget=dict(prior_formal_updates=400,cap=2000,R10_classified_additional_formal_updates=totals['formal_updates'],
            classified_used=400+totals['formal_updates'],unclassified_updates_requiring_root=totals['unclassified_updates'],smoke_excluded_from_formal_cap=True,budget_not_reset=True),
        DrivingEditor_historical_plus_R10=dict(nativeF=956+970+totals['nativeF'],nativeB=811+833+totals['nativeB'],DGGT_excluded=True),DGGT=dggt,
        no_future_calls_counted=True,one_canonical_top_level_counter_source_per_run=True,resources_json_authoritative_when_present=True,
        verify_receipt_fallback_when_resources_absent=True,resource_only_failed_runs_included=True,
        all_encountered_run_counters_resolved=not skipped,aliases_validated_equal_not_summed=True,
        partial_failed_runs_counted_by_actual_counters=True,source_receipts_modified=False,GPU_calls_added_by_this_CPU_audit=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(R10=totals,runs=len(runs),aggregate_controllers_excluded=len(controllers),unresolved_schemas=len(skipped),DGGT_camera_F=dggt['camera_raster_F_total'],DGGT_main_F=1,output=str(a.output))))
if __name__=='__main__':main()
