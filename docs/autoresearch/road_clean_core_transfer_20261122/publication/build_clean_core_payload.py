"""R4 Git只记录代码、关键摘要与恢复入口；完整组件/视图在archive。"""
from pathlib import Path
import json

RUN='road_clean_core_transfer_20261122_r4'
PREFIX='docs/autoresearch/road_clean_core_transfer_20261122'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    root=Path(__file__).resolve().parent.parent;out=root/'publication';payload=out/'payload'
    selected=out/'selected_files.json'
    if selected.exists() and read(selected).get('finalized_by_root'):
        raise RuntimeError('Preserve root-finalized selection; reopen explicitly before rebuilding')
    source=read(root/'source_core_probe/receipt.json');target=read(root/'single_stamp_zero/receipt.json')
    assert source['complete'] and target['complete'] and (source['nativeF'],target['nativeF'])==(42,65)
    assert not target['parent_asset_modified'] and not target['parent_retirement'] and not target['parent_stamp_compositing']
    source_summary=[];target_summary=[]
    for s in source['variants']:
        source_summary.append(dict(label=s['label'],patch=s['patch'],core_rows=s['core_rows'],halo_rows=0,
            material_input=s['material_input'],raw_frames=[v['raw'] for v in s['views']],
            reload_RGB_max_abs=max(v['reload_RGB_max_abs'] for v in s['views']),
            original_parent_core_equivalence_max_abs=max([v['component_vs_parent_core_RGB_max_abs'] for v in s['views'] if v['component_vs_parent_core_RGB_max_abs'] is not None],default=None)))
    for t in target['variants']:
        target_summary.append(dict(label=t['label'],patch=t['patch'],core_rows=t['core_rows'],halo_rows=0,
            source_RGB_sigma_opacity_exact=t['source_RGB_sigma_opacity_exact'],rigid_internal_distances=t['internal_pairwise_distances_rigid'],
            covariance_reconstruction_max_abs=t['covariance_reconstruction_max_abs'],raw_frames=[v['raw'] for v in t['views']],
            reload_RGB_max_abs=max(v['reload_RGB_max_abs'] for v in t['views'])))
    write(payload/'component_summary.json',dict(schema='clean_core_component_summary.v1',ready=True,
        source=source_summary,target=target_summary,source_view_count=12,target_view_count=20,
        source_black_grey_are_actual_native_RGBA=True,display_pixel_mask_applied=False,
        metric_domain='Projected physical core eroded3px AND alpha>=.95; this score domain is not the whole physical footprint.',
        color_regions='Full physical and saved-crop outside/edge regions are separately summarized in cpu_audit_summary.json; RGB/alpha and native-grey chroma are distinct.',
        parent_retirement=False,parent_stamp_compositing=False,target_fusion_executed=False,
        material_source='Similar-road inference; not same-position target observed material',human_verdict=None,problem_solved=None))
    chosen=read(root/'target_audit/target_candidates.json')['recommended']
    raster=read(root/'target_audit/manual_target_spec.json')['exact_raster_footprint_checks_1152']
    write(payload/'target_summary.json',dict(schema='manual_same_center_target_summary.v1',ready=True,
        target_raw=chosen['target_frame_raw'],origin=chosen['target_frame']['origin'],frame=chosen['target_frame']['frame'],
        frame_radius_m=chosen['default_road_frame_radius_m'],source_core_sizes_m=[.8,.6],scale=1.,
        primary_exact_1152_footprints={side:raster['110'][side] for side in ['0.8','0.6','1.0']},
        old_transfer_already_core_only=True,halo_historically_used_for_source_gallery=True,
        existing_rigid_transport_unchanged=True,full_target_and_trace='Archive: target_audit/ and single_stamp_zero/'))
    write(payload/'resource_summary.json',dict(schema='clean_core_resources_compact.v1',ready=True,
        task='ROAD-CLEAN-CORE-TRANSFER-20261122',run=RUN,nativeF=107,nativeB=0,optimizer_updates=0,new_diffusion_calls=0,
        source_stage=dict(nativeF=42,views=12,components=4),target_stage=dict(nativeF=65,views=20,components=4),
        parent_asset_modified=False,parent_retirement=False,parent_stamp_compositing=False,target_fusion=False,
        prior_r2_and_r3_resources_counted_separately=True))
    gate=root/'cpu_audit/compact_audit_summary.json'
    if gate.exists():
        value=read(gate)
        assert value['ready'] and value['engineering_checks_passed']
        write(payload/'cpu_audit_summary.json',value)
    else:write(payload/'cpu_audit_summary.json',dict(ready=False,awaiting='Independent CPU aggregate'))
    restore=root/'restore_manifest_compact.json'
    if restore.exists():
        r=read(restore);companion=r['full_report_companion']
        write(payload/'restore_entry.json',dict(schema='clean_core_git_restore_entry.v1',ready=r['complete'],complete=r['complete'],
            run_id=RUN,index_only=True,archive=r['archive'],main_archive_role=r['main_archive_role'],
            prior_r3_git_entry='../road_source_multiframe_20261122/restore_manifest_compact.json',
            full_report_companion={k:companion[k] for k in ['filename','bytes','sha256','member_count','logical_restore_directory'] if k in companion},
            state_recovery=r['state_recovery'],report_restore_chain=r['report_restore_chain'],restore_steps=r['restore_steps'],
            no_duplicated_parent_model_catalog=True,parent_retirement=False,parent_stamp_compositing=False,target_fusion=False))
    else:write(payload/'restore_entry.json',dict(ready=False,awaiting='Verified full archive'))
    rows=[]
    def add(src,dst=None,ready=False):
        row=dict(source=src,destination=dst or src)
        if ready:row['require_ready']=True
        rows.append(row)
    for name in ['README.md','selected_run.json','architecture_components.svg','local_delivery.json']:add(name)
    add('publication/payload/restore_entry.json','restore_manifest_compact.json',True)
    for name in ['probe_core_only.py','transfer_single_core.py']:add('code/'+name)
    add('target_audit/choose_target_cpu.py')
    for name in ['component_summary.json','target_summary.json','resource_summary.json','cpu_audit_summary.json']:
        add('publication/payload/'+name,'evidence/'+name,True)
    for name in ['source_core_probe','single_stamp_zero']:add(name+'/contract.json','evidence/'+name+'_contract.json')
    for p in sorted((root/'cpu_audit').glob('*.py')):add('cpu_audit/'+p.name)
    for name in ['archive_core_transfer.py','restore_manifest.py','takeback_archive.py']:add('recovery/'+name)
    sync=root/'report_source/report_source_sync_manifest.json'
    report_complete=sync.exists()
    if report_complete:
        info=read(sync)
        for item in info.get('lightweight_source_files',[]):
            name=item['relative'] if isinstance(item,dict) else item
            if Path(name).suffix in ['.py','.js','.mjs']:add('report_source/'+name,'report/'+name)
    add('report_source/qa_summary.json','report/qa_summary.json',True)
    for name in ['publish_clean_core.py','build_clean_core_payload.py','selected_files.json','narrative.json']:add('publication/'+name)
    assert len({r['destination'] for r in rows})==len(rows)
    write(selected,dict(schema='clean_core_git_selection.v1',run_id=RUN,docprefix=PREFIX,finalized_by_root=False,
        expected_R3_ancestor='32a52f69f75ba0e5bbb713c5eae601914bb43c07',report_scripts_finalized=report_complete,
        excluded=['PNG','NPZ','fullfloat','full per-file hashes','full metrics/CPU traces','model weights'],files=rows))
    print(json.dumps(dict(prepared=True,selected_files=len(rows),report_scripts_finalized=report_complete,
        missing=[r['source'] for r in rows if not (root/r['source']).is_file()])))


if __name__=='__main__':main()
