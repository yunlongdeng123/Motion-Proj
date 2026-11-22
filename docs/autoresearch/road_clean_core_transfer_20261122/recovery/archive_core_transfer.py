"""R4独立组件诊断归档；保留真实三维数据，不把单组件展示称为融合。"""
from pathlib import Path
import argparse
import datetime
import gzip
import json
import tarfile
import numpy as np
from restore_manifest import sha,verify_archive

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_clean_core_transfer_20261122_r4')
C=ROOT.parent
R2=C/'road_source_fidelity_20261122_r2'
R3=C/'road_source_multiframe_20261122_r3'
LABELS=['source00_original','source00_camera137','source02_original','source02_eight400']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)


def validation(root):
    source=read(root/'source_core_probe/receipt.json');target=read(root/'single_stamp_zero/receipt.json')
    assert source['complete'] and target['complete']
    assert (source['nativeF'],target['nativeF'])==(42,65)
    assert source['nativeB']==target['nativeB']==0
    assert source['optimizer_updates']==target['optimizer_updates']==0
    assert source['new_diffusion_calls']==target['new_diffusion_calls']==0
    assert not source['original_asset_modified'] and not target['parent_asset_modified']
    assert not target['parent_retirement'] and not target['parent_stamp_compositing']
    events=[json.loads(x) for x in (root/'runtime/native_session_log.jsonl').read_text().splitlines() if x.strip()]
    assert len(events)==2 and [r['native_context_F'] for r in events]==[42,65]
    assert not read(root/'gpu_lease.json').get('active',False)
    s={v['label']:v for v in source['variants']};t={v['label']:v for v in target['variants']}
    assert set(s)==set(t)==set(LABELS)
    matrix=[];states=[];required=[]
    for label in LABELS:
        assert s[label]['halo_rows_exported']==t[label]['halo_rows']==0
        assert s[label]['core_rows']==t[label]['core_rows'] in (900,1598)
        assert t[label]['source_RGB_sigma_opacity_exact'] and t[label]['internal_pairwise_distances_rigid']
        source_base=root/'source_core_probe'/label;target_base=root/'single_stamp_zero'/label
        required.extend([source_base/'road.npz',target_base/'road.npz',target_base/'transfer_trace.npz',target_base/'provenance.json'])
        for phase,variant,base,float_name,pngs in [
            ('source',s[label],source_base,'core_float.npz',['photo_pinhole.png','core_black.png','core_grey.png','alpha.png','metric_mask.png']),
            ('target',t[label],target_base,'target_float.npz',['target_black.png','target_grey.png','alpha.png','parent_road_context.png','target_full_grey.png','parent_footprint_locator.png','metric_mask.png'])]:
            if phase=='target':assert [v['raw'] for v in variant['views']]==[86,110,134,179,198]
            for v in variant['views']:
                assert v['reload_RGB_max_abs']==0
                folder=base/f"{v['raw']:03d}";required.append(folder/float_name);required.extend(folder/n for n in pngs)
                matrix.append(dict(stage=phase,variant=label,patch=variant['patch'],view=v))
        states.append(dict(variant=label,source_patch=s[label]['patch'],core_rows=s[label]['core_rows'],halo_rows=0,
            material_input=s[label]['material_input'],source_component=f'source_core_probe/{label}/road.npz',
            target_component=f'single_stamp_zero/{label}/road.npz',trace=f'single_stamp_zero/{label}/transfer_trace.npz',
            provenance=f'single_stamp_zero/{label}/provenance.json',RGB_sigma_opacity_preserved=True,
            rigid_internal_distances_preserved=True,covariance_reconstruction_max_abs=t[label]['covariance_reconstruction_max_abs']))
    assert len(matrix)==32
    for p in required:assert p.is_file() and p.stat().st_size>0,p
    ledger=dict(schema='clean_core_transfer_resources.v1',complete=True,run_id=root.name,
        totals=dict(nativeF=107,nativeB=0,optimizer_updates=0,new_diffusion_calls=0,target_training_updates=0,
            native_contexts=2,power_operations=0,native_context_wall_seconds=sum(e['ended_at']-e['started_at'] for e in events)),
        stages=[dict(name='source_core_probe',nativeF=42,nativeB=0,views=12,standalone_components=4),
                dict(name='single_stamp_zero',nativeF=65,nativeB=0,views=20,standalone_components=4)],
        forward_accounting=dict(source_black_grey_reload=36,source_parent_core_equivalence=6,target_parent_context=5,target_black_grey_reload=60),
        session_events=events,latest_process_snapshot=read(root/'resource_ledger.json'),latest_snapshot_added_to_total=False,
        parent_asset_modified=False,parent_retirement=False,parent_stamp_compositing=False,
        target_fusion_executed=False,component_rendering_is_not_parent_scene_fusion=True)
    state=dict(complete=True,format='Standard RoadRows-compatible independent components; not a full persistent scene',states=states,
        target_origin=target['target_origin'],transport='Existing locally rigid mean/covariance transport',scale=1.,
        material_source='Similar-road inference, not same-position target observation',
        reload_evidence='Actual second RoadRows native load, grey RGB and alpha asserted exact; equal duplicate pixels are represented by the saved grey result and receipt zero-delta.')
    return ledger,dict(complete=True,source_views=12,target_views=20,rows=matrix,
        score_domain='Projected physical core eroded3 pixels AND alpha>=.95',
        whole_physical_and_edge_statistics='run/cpu_audit/compact_audit_summary.json',
        outside_statistics_domain='Saved native crop: physical quad bbox expanded20 pixels; not full-screen tail counts.',
        effective_RGB_divided_by_alpha_and_native_grey_chroma_are_distinct=True),state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    root=ROOT;recovery=root/'recovery';recovery.mkdir(parents=True,exist_ok=True)
    ledger,matrix,states=validation(root)
    for name,value in [('resource_ledger_cumulative.json',ledger),('complete_result_matrix.json',matrix),('packet_state_audit.json',states)]:write(recovery/name,value)
    required_final=['README.md','selected_run.json','architecture_components.svg','cpu_audit/aggregate_cpu_audit.json','report_source/report_source_sync_manifest.json']
    issues=['Missing final artifact: '+name for name in required_final if not (root/name).is_file()]
    gate_path=root/'cpu_audit/aggregate_cpu_audit.json'
    if gate_path.is_file():
        gate=read(gate_path)
        accepted=[gate[k] for k in ['all_checks_passed','all_required_checks_passed','all_required_state_data_checks_passed','all_required_engineering_checks_passed','ready','passed'] if k in gate]
        if not accepted or not all(x is True for x in accepted):issues.append('Independent CPU aggregate gate has not passed')
    files={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or '__pycache__' in p.parts or rel.parts[0] in ('archives','takeback_parts') or 'download_parts' in rel.parts or 'prior_mask_inputs' in rel.parts or p.suffix in ('.pyc','.tmp'):continue
        if rel.as_posix() in ('recovery/restore_manifest_prepared.json','recovery/archive_receipt.json','restore_manifest_compact.json','local_delivery.json'):continue
        files['run/'+rel.as_posix()]=p
    dependencies=[(R3/'restore_manifest_compact.json','dependencies/r3_restore_manifest_compact.json'),
        (R3/'publication/git_receipt.json','dependencies/r3_git_receipt.json'),
        (R2/'code/probe_source_chain.py','dependencies/r2_code/probe_source_chain.py'),
        (R2/'code/camera_corrected_source.py','dependencies/r2_code/camera_corrected_source.py')]
    for name in ['rigid_patch.py','gaussian_patch_transport.py','fixed_surface.py','build_transfer.py']:
        dependencies.append((C/'road_gaussian_transfer_20261122_r2_isometric/code'/name,'dependencies/transport/'+name))
    for p,arc in dependencies:files[arc]=p
    # Exact camera/ego inputs and untouched photographs for the executed nine
    # source/target cameras; large parent scene weights remain in the prior chain.
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    bank_receipt=read(bank/'receipt.json');identifiers={int(x['raw']):x['id'] for x in bank_receipt['frames']}
    camera_raws=[86,100,110,115,117,134,137,179,198]
    files['inputs/source_bank/receipt.json']=bank/'receipt.json'
    for raw in camera_raws:
        obs=read(bank/f'observations/{raw:03d}.json')
        for rel in [f'observations/{raw:03d}.json',f'calib/{identifiers[raw]}.txt',obs['image_path']]:
            files['inputs/source_bank/'+rel]=bank/rel
    mask_folder=recovery/'prior_mask_inputs';mask_folder.mkdir(exist_ok=True)
    for raw in [86,110,134,179,198]:
        with np.load(C/f'road_common1152_20261121/maps/{raw:03d}.npz',allow_pickle=False) as z:
            record={k:z[k] for k in ['camera_K','world_to_camera','K']}
        domains=C/f'road_surface_residual_20261121_r2/proposal/domains/{raw:03d}.npz'
        if domains.exists():
            with np.load(domains,allow_pickle=False) as z:
                record.update({k:z[k] for k in ['appearance_C','appearance_S']})
        record['mask_resolution_wh']=np.array([1152,768]);p=mask_folder/f'{raw:03d}.npz'
        np.savez_compressed(p,**record);files[f'inputs/prior_mask_grid1152/{raw:03d}.npz']=p
    runtime=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit')
    for name in ['v44_factory.py','v44_manifest.py','v44_scene.py']:
        files['dependencies/runtime/'+name]=runtime/name
    inventory=[]
    for arc,p in sorted(files.items()):
        before=p.stat();digest=sha(p);after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Changing payload: '+str(p))
        inventory.append(dict(archive_path=arc,original_path=str(p),bytes=after.st_size,sha256=digest,mtime_ns=after.st_mtime_ns))
    prior=read(R3/'restore_manifest_compact.json')
    manifest=dict(schema='clean_core_transfer_curated_restore.v1',complete=not issues,run_id=root.name,original_root=str(root),
        readiness_issues=issues,files=inventory,payload_files=len(inventory),payload_bytes=sum(r['bytes'] for r in inventory),
        resources=ledger['totals'],state_recovery=states,prior_r3_recovery=prior,full_report_companion=prior['full_report_companion'],
        original_photograph_and_camera_raws=camera_raws,
        prior_mask_grid='Exact existing1152 K/C/S inputs are included; native1920 rendering/calibration remains explicit and is not relabeled as1152.',
        report_restore_chain=['Verified r2 full report companion',
            'road_source_fidelity_20261122_r2 curated run/report_source/publication patch',
            'road_source_multiframe_20261122_r3 curated run/report_source/publication patch',
            'road_clean_core_transfer_20261122_r4 run/report_source/publication patch'],
        report_policy='Continue the same original HTML/HTTP; no new full report companion or viewing entry.',
        old_transport_was_already_core_only=True,new_test_scope='Native core-only component verification plus manually selected same-center single rigid stamp and updated source RGB variants',
        parent_asset_modified=False,parent_retirement=False,parent_stamp_compositing=False,target_fusion_executed=False,
        render_reload_scope='Black/grey RGBA, native second-load equality receipts, all32 float views, source/target component NPZ and transport traces.',
        parent_scene_engine_external='Use verified r3/r2/prior recovery chain; original models and engine are not duplicated.',
        restore_steps=['Verify every archive member before extracting to a new empty directory.',
            'Restore original parent scene/engine through prior_r3_recovery and preserve all archived original code. Map paths in a separate working copy.',
            'Load each source_core_probe/VARIANT/road.npz as an independent RoadRows component; IDs are only original core rows, no gallery halo.',
            'Load each single_stamp_zero/VARIANT/road.npz independently; transfer_trace.npz carries original IDs, XYZ, rigidR, origins/frames and world covariance.',
            'These components did not retire or overlay the parent scene and are not proof of completed fusion. No optimizer state exists because this run has0 updates.',
            'Restore the full r2 report companion, then r2, r3 and this report patch over the same logical report directory.'],
        GPU_calls_by_archiver=0,Git_operations=0,power_operations=0)
    write(recovery/'restore_manifest_prepared.json',manifest)
    if not args.archive:
        print(json.dumps(dict(prepared=True,compressed=False,ready=not issues,issues=issues,files=len(inventory),bytes=manifest['payload_bytes'])));return
    if issues:raise RuntimeError(json.dumps(issues))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive=root/'archives'/f'{root.name}_curated_{stamp}.tar.gz'
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=5,dereference=True) as tar:
        for row in inventory:
            p=Path(row['original_path'])
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise RuntimeError('Payload changed: '+str(p))
            tar.add(p,arcname=row['archive_path'],recursive=False)
        tar.add(recovery/'restore_manifest_prepared.json',arcname='restore_manifest.json',recursive=False)
    verify_archive(archive)
    with gzip.open(archive,'rb') as stream:
        for _ in iter(lambda:stream.read(8*1024*1024),b''):pass
    receipt=dict(complete=True,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),payload_files=len(inventory),verified_every_member=True,GPU_calls=0,Git_operations=0,power_operations=0)
    write(recovery/'archive_receipt.json',receipt)
    compact=dict(schema='clean_core_transfer_compact_restore.v1',complete=True,run_id=root.name,
        archive=dict(path=str(archive),filename=archive.name,bytes=receipt['archive_bytes'],sha256=receipt['archive_sha256'],payload_files=len(inventory),verified_every_member=True),
        main_archive_role='R4八个源/目标独立RoadRows组件、刚体covariance/rowID追踪、32视图原生黑灰底/alpha/PNG/float及完整CPU审计和同页增量。',
        full_report_companion=prior['full_report_companion'],report_companion_role='复用r2完整原报告备份，再依次覆盖r2/r3/r4报告增量。',
        prior_r3_recovery=prior,resources=ledger['totals'],state_recovery=states,report_restore_chain=manifest['report_restore_chain'],
        restore_steps=manifest['restore_steps'],old_transfer_already_core_only=True,parent_retirement=False,parent_stamp_compositing=False,
        target_fusion_executed=False,optimizer_updates=0,new_diffusion_calls=0)
    write(root/'restore_manifest_compact.json',compact);print(json.dumps(receipt))


if __name__=='__main__':main()
