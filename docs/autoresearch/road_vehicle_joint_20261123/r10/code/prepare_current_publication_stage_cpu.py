"""Write live pending archive/publication metadata; never freeze, archive or run Git."""
from pathlib import Path
import argparse,datetime,hashlib,json

GEOMETRY_SHA='0d62e87abddd3017a40b06d80c485d331c0feb5c13a433be7f50f6933cda269b'
SOURCE_SHA='45b4b9d752f1584019e6687008de489de071a81907ab2c3c1b61b20964f64d32'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    def ref(rel,expected=None):
        q=r/rel;h=sha(q)
        if expected:assert h==expected,rel
        return dict(path=rel,sha256=h,bytes=q.stat().st_size)
    def complete(rel):
        assert read(r/rel).get('complete') is True,rel
        return ref(rel)
    coverage=read(r/'docs/current_archive_coverage_audit.json');assert coverage['all_current_source_candidate_arrays_included']
    current=read(r/'root_current_stage.json')
    actual_new=[]
    actual_footprints=[]
    for folder in sorted(r.iterdir()):
        q=folder/'receipt.json'
        if not(folder.is_dir() and folder.name.startswith('timewindow17_') and q.is_file()):continue
        record=read(q)
        if record.get('complete') is not True:continue
        if record.get('schema')=='R10.new_local_joint_zero_step.v1':actual_new.append((q.stat().st_mtime_ns,q,record))
        if str(record.get('schema','')).startswith('R10.actual_semantic_edit_footprint_alphaT_VJP'):actual_footprints.append((q.stat().st_mtime_ns,q,record))
    latest_new=max(actual_new,default=None,key=lambda x:x[0]);latest_footprint=max(actual_footprints,default=None,key=lambda x:x[0])
    actual_asset=None
    if latest_new:
        ap=Path(latest_new[2]['asset_manifest']['path']);assert ap.is_relative_to(r)
        actual_asset=ref(ap.relative_to(r).as_posix(),latest_new[2]['asset_manifest']['sha256'])
    geometry=ref('audit/timewindow17_geometry_v1/candidate_reprojected_v1/candidate_geometry.npz',GEOMETRY_SHA)
    source=ref('timewindow17_shared_requalified_v2/final_atlas.npz',SOURCE_SHA)
    full=read(r/'audit/timewindow17_asset_real_search_v1/receipt.json')
    assert full['candidate_records']==2999905 and full['fresh_known']==0
    asof=datetime.datetime.now(datetime.timezone.utc).isoformat()
    ledger_path=r/'docs/current_resource_ledger.json';ledger=read(ledger_path) if ledger_path.is_file() else None
    resource_f=ledger['R10']['nativeF'] if ledger else current['R10_actual_nativeF']
    resource_b=ledger['R10']['nativeB'] if ledger else current['R10_actual_nativeB']
    status=dict(schema='R10.current_stage_publication_inputs.pending.v1',complete=True,ready=False,prepared_UTC=asof,
        method_stage_frozen=False,overall_run_complete=False,whole_scene_restoration_accepted=False,
        stage_status='blocked-technical',authorization_pending=False,
        blocked_technical=['Input geometry/source gates and actual joint whole-image acceptance are separate. Current actual expanded zero/footprint receipts are listed when present; this preparation does not manufacture a whole-scene PASS.',
            'Root is still evaluating residual background, footprint containment and the next bounded correction. This is a technical stage, not a request for user execution authorization.'],
        bindings=dict(geometry=geometry,shared_field=source,immutable_parent=ref('local_shared_continuous_v1/final_atlas.npz'),
            source_qualification75=complete('audit/timewindow17_real75_fresh_v1/receipt.json'),
            full_asset_photo_search=complete('audit/timewindow17_asset_real_search_v1/receipt.json'),
            full_asset_fixed_rejection_partition=complete('audit/timewindow17_asset_real_search_v1/fixed_rejection_partition.json'),
            direct_surface_review=complete('timewindow17_requalified_surface_review_v2_replay1/receipt.json'),
            new_expanded_asset=actual_asset,new_expanded_joint_zero=complete(latest_new[1].relative_to(r).as_posix()) if latest_new else None,
            new_expanded_footprint=complete(latest_footprint[1].relative_to(r).as_posix()) if latest_footprint else None,new_expanded_contained_replacement=None),
        source_qualification=dict(asset_vertices=176465,asset_cells=175302,world_area_m2=full['asset_world_surface_area_m2'],
            examined_frames=list(range(90,171,5)),candidate_records=2999905,new_strict_same_world_vertices=0,
            ancestor_real_locks=6407,unaffected_real_locks=6332,displaced_addresses_repaired=75,fresh_recertified_real=0,
            interpretation='Zero strict qualification is a conservative-certificate result, not proof of physical invisibility or absence of same-world data in all photos. GT OBB and dynamic/shadow tests retain explicit proxy status. Old source/geometry files and ancestor records remain unchanged.'),
        resources_at_preparation=dict(R10_nativeF=resource_f,R10_nativeB=resource_b,
            existing_formal_updates=current['existing_representation_formal_updates'],existing_limit=current['existing_representation_limit'],
            new_local_updates=current['new_local_optimizer_updates'],authority='current_resource_ledger.json when present, otherwise root_current_stage.json, at this preparation timestamp; recompute actual later receipts before final freeze; planned calls are not counted'),
        required_next_evidence=['Root final binding and quality decision on the exact geometry/source/asset hashes; input approval is not whole-scene acceptance',
            'Final selected actual expanded joint/footprint/contained or bounded-correction receipts, or explicit root decision to close this stage without further execution',
            'Root final method freeze receipt; full R10 archive/member verification and local takeback; original HTML QA; explicit final delivery pointer and root narrative'],
        archive_inventory_audit=ref('docs/current_archive_coverage_audit.json'),archive_created=False,Git_operations=0,GPU_calls=0)
    write(r/'docs/current_stage_publication_inputs.pending.json',status)
    receipts=[complete(rel) for rel in ['edit_footprint_measure_v3/receipt.json','replacement_contained_zero_v1_replay1/receipt.json',
        'new_local_zero_v1/receipt.json','audit/timewindow17_real75_fresh_v1/receipt.json','audit/timewindow17_asset_real_search_v1/receipt.json',
        'timewindow17_texture_seed_requalified_v2/receipt.json','timewindow17_shared_requalified_v2/receipt.json','timewindow17_requalified_surface_review_v2_replay1/receipt.json']]
    template=read(r/'docs/final_method_snapshot_receipt.template.json')
    template.update(complete=False,method_stage_frozen=False,overall_run_complete=False,whole_scene_restoration_accepted=False,
        pending_next_footprint_experiment=True,final_footprint_receipt=dict(path=None,sha256=None),
        last_completed_footprint_receipt=complete('edit_footprint_measure_v3/receipt.json'),
        last_completed_contained_zero_receipt=complete('replacement_contained_zero_v1_replay1/receipt.json'),
        required_complete_receipts=receipts,
        critical_members=coverage['expected_relative_members']+['docs/current_stage_publication_inputs.pending.json','docs/current_archive_coverage_audit.json'],
        current_input_bindings=dict(geometry=geometry,shared_field=source),
        resources_at_preparation=status['resources_at_preparation'],
        required_future_receipts_to_fill=status['required_next_evidence'],
        stage_status='blocked-technical',authorization_pending=False,
        resources_summary='Root must fill final actual counts from completed receipts; current preparation is not a freeze signal and does not count planned GPU calls.',
        snapshot_scope='All R10 regular payload including complete 17-frame/2999905-candidate source search, 75-address ancestor/failure history, all geometries and new assets/actual diagnostics when completed. R8/R9 complete archive remains external; final report HTML is separate.')
    for rel in ['timewindow17_shared_continuous_v1/final_atlas.npz','timewindow17_requalified_surface_review_v2/failure_receipt.json',
                'audit/timewindow17_real_lock_geometry_v1/locked_vertex_displacement.npz','audit/timewindow17_real75_fresh_v1/interpretation.json']:
        assert (r/rel).is_file();template['critical_members'].append(rel)
    template['critical_members']=sorted(set(template['critical_members']))
    measured=[]
    for folder in sorted(r.iterdir()):
        q=folder/'receipt.json'
        if folder.is_dir() and q.is_file():
            record=read(q)
            if record.get('complete') and str(record.get('schema','')).startswith('R10.actual_semantic_edit_footprint_alphaT_VJP'):
                measured.append(complete(q.relative_to(r).as_posix()))
    template['last_completed_footprint_receipts_at_preparation']=measured
    write(r/'docs/final_method_snapshot_receipt.timewindow17.pending.json',template)
    print(json.dumps(dict(ready=False,stage_status='blocked-technical',authorization_pending=False,current_resources=status['resources_at_preparation'],critical_members=len(template['critical_members']),actual_expanded_zero_receipt_present=bool(latest_new),actual_expanded_footprint_receipt_present=bool(latest_footprint),archive_created=False,Git_operations=0)))
if __name__=='__main__':main()
