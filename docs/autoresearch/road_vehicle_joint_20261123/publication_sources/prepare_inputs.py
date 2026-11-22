"""Prepare immutable pending inputs, or root-gated final config. Never runs Git."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import argparse,hashlib,json,shutil

LABEL='ROAD-VEHICLE-JOINT-20261123';PREFIX='docs/autoresearch/road_vehicle_joint_20261123'
ALLOWED={'.py','.json','.md','.svg','.js'};LIMIT=300000
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def write_new(path,value):
    if path.exists():raise FileExistsError('Preserve existing review input: '+str(path))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def entry(root,source,destination,ready=False):
    name=PurePosixPath(source);dest=PurePosixPath(destination)
    if name.is_absolute() or dest.is_absolute() or '..' in name.parts or '..' in dest.parts:raise ValueError('Unsafe selection path')
    if name.suffix not in ALLOWED or dest.suffix not in ALLOWED:raise ValueError('Binary payload must stay in archive')
    path=(root/Path(*name.parts)).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Selection escapes current run')
    exists=path.is_file()
    if exists and path.stat().st_size>=LIMIT:raise ValueError('Selected file exceeds lightweight budget: '+str(path))
    return dict(source=name.as_posix(),destination=dest.as_posix(),require_ready=ready,exists_at_preparation=exists,sha256=sha(path) if exists else None,bytes=path.stat().st_size if exists else None)

def prepare(root,archive_receipt=None,r10_archive_receipt=None):
    out=root/'publication';inventory=read(out/'selected_inventory.pending.json');files=[];stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for row in inventory['selected_files']:
        source='publication/'+row['snapshot'];current=entry(root,source,row['snapshot'])
        if current['sha256']!=row['sha256']:raise ValueError('Historical curated snapshot changed: '+source)
        files.append(current)
    r8=root.parent/'road_vehicle_joint_dggt_20261123_r8'
    sources=[('report',r8/'report',['build_wiring_report.py','wiring_viewer.js','joint_fit_viewer.js','dggt_control_viewer.js','qa_wiring_report.py','qa_joint_fit_report.py','qa_dggt_control_report.py','joint_fit_adapter.py','dggt_forward_adapter.py','build_joint_fit_manifest.py','r9_current_architecture_components.svg']),
             ('native_joint_RGB_sources',r8/'code',['fit_joint_rgb.py','joint_rgb_runtime.py','joint_rgb_common.py','joint_replay_state_cpu.py','summarize_joint_rgb.py','render_joint_colour_lineage.py'])]
    for category,base,names in sources:
        for name in names:
            source=base/name
            if not source.is_file():continue
            if source.stat().st_size>=LIMIT:raise ValueError('Oversized current source: '+str(source))
            snapshot=out/'current_sources'/stamp/category/name;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,snapshot)
            if sha(source)!=sha(snapshot):raise ValueError('Source changed while snapshotting: '+str(source))
            destination=('draft_sources/report/' if category=='report' else category+'/')+name
            files=[row for row in files if row['destination']!=destination]
            files.append(entry(root,snapshot.relative_to(root).as_posix(),destination))
    from prepare_current_stage_snapshot import prepare as snapshot_current
    current=snapshot_current(root,archive_receipt=archive_receipt,r10_archive_receipt=r10_archive_receipt)
    for row in current['files']:
        files=[old for old in files if old['destination']!=row['destination']]
        files.append(row)
    for name in ['publish_road_vehicle_joint.py','root_finalize_publication.py','prepare_inputs.py','prepare_current_stage_snapshot.py','qa_publication_current_snapshot.py','isolated_publication_base.py','isolated_base_derivation.json','bootstrap_publication_base.py','qa_publication_preparation.py','status_tail_replacement.py','origin_failure_index_cpu.py','qa_status_replacement_cpu.py','prepare_failure_index_preflight_cpu.py','refresh_final_selection_cpu.py','qa_refresh_final_selection_cpu.py']:
        files.append(entry(root,'publication/'+name,'publication_sources/'+name))
    for name in ['status_replacement_CPU_QA.json','failure_index_preflight_v1/actual_rebuild/receipt.json','failure_index_preflight_v1/actual_rebuild/shared_checkout_readonly_check.json']:
        if (root/'publication'/name).is_file():files.append(entry(root,'publication/'+name,'publication_preflight/'+name))
    files.extend([entry(root,'README.md','README.md'),entry(root,'compact_result.json','compact_result.json'),entry(root,'publication/architecture_components.svg','architecture_components.svg'),entry(root,'publication/final_delivery.json','final_delivery.json',True)])
    destinations=[row['destination'] for row in files]
    if len(destinations)!=len(set(destinations)):raise ValueError('Duplicate publication destination')
    path=out/'selected_files.pending.json'
    if path.exists():path=out/f'selected_files.pending.{stamp}.json'
    snapshot=dict(source=Path(current['status']['path']).relative_to(root).as_posix(),sha256=current['status']['sha256'])
    selection=dict(run_id=root.name,docprefix=PREFIX,task=LABEL,finalized_by_root=False,files=files,current_stage_snapshot=snapshot,requires_current_stage_review=True,source_00_02_correction_required=True,R7_NOT_ACCEPTED_required=True,Git_operations_executed=False)
    write_new(path,selection)
    narrative_path=out/'narrative.pending.json'
    if not narrative_path.exists():
        narrative=dict(task=LABEL,finalized_by_root=False,selected_manifest_sha256=sha(path),failure_ledger_refs=['V82-F06'],failure_ledger_delta='updated:V82-F06',new_failure_family=False,human_verdict=None,problem_solved=None,
            formal400_and_fresh_reviewed=False,official_DGGT_control_reviewed=False,archive_verified=False,report_QA_passed=False,takeback_verified=False,source_00_02_correction_preserved=True,R7_NOT_ACCEPTED_preserved=True,status_replacement='',experiment_entry='',failure_append='')
        write_new(narrative_path,narrative)
    result=dict(selection=str(path),selection_sha256=sha(path),files=len(files),current_stage_snapshot=snapshot,missing_final_inputs=[row['source'] for row in files if not row['exists_at_preparation']],finalized_by_root=False,ready=False,Git_operations_executed=False)
    write_new(out/f'prepare_inputs.pending.{stamp}.json',result);return result

def finalize(root,path):
    from publish_road_vehicle_joint import final_narrative,current_stage_review
    out=root/'publication';narrative=final_narrative(read(out/'narrative.json'))
    if narrative.get('selected_manifest_sha256')!=sha(path):raise ValueError('Root narrative binds another pending selection')
    value=read(path);current_stage_review(root,narrative,value)
    for row in value['files']:
        current=entry(root,row['source'],row['destination'],row.get('require_ready',False))
        if not row.get('sha256') or current['sha256']!=row['sha256']:raise ValueError('Refresh and root-review final inputs before finalization: '+row['source'])
        if row.get('require_ready') and read(root/row['source']).get('ready') is not True:raise ValueError('Final delivery is not ready')
    value['finalized_by_root']=True;value['root_narrative_sha256']=sha(out/'narrative.json')
    write_new(out/'selected_files.json',value)
    return dict(finalized_by_root=True,selection=str(out/'selected_files.json'),Git_operations_executed=False,stage_executed=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--finalize-config',action='store_true');parser.add_argument('--selection',type=Path);parser.add_argument('--archive-receipt',type=Path);parser.add_argument('--r10-archive-receipt',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.finalize_config:
        if not args.selection:parser.error('--finalize-config requires a root-reviewed --selection')
        result=finalize(root,args.selection)
    else:result=prepare(root,args.archive_receipt,args.r10_archive_receipt)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
