"""归档02两帧/八帧400步配对；完整状态与报告增量，默认只prepare。"""
from pathlib import Path
import argparse
import datetime
import gzip
import json
import tarfile
from restore_manifest import sha, verify_archive

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_source_multiframe_20261122_r3')
R2 = ROOT.parent / 'road_source_fidelity_20261122_r2'
FRAMES = [86,88,90,92,94,96,99,100,104,110,115,117]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)


def validate(root):
    folder=root/'paired400_seed0';receipt=read(folder/'receipt.json');contract=read(folder/'contract.json')
    assert receipt['complete'] and receipt['formal_updates']==800 and receipt['discarded_smoke_restore_updates']==6
    assert contract['steps_per_arm']==400 and contract['batch_frames']==2
    assert receipt['source_geometry_unchanged'] and not receipt['target_migration'] and receipt['new_diffusion_calls']==0
    assert receipt['resume_next_update_RGB_max_abs']<=1e-6
    events=[json.loads(x) for x in (root/'runtime/native_session_log.jsonl').read_text().splitlines() if x.strip()]
    assert len(events)==1 and sum(e['native_context_F'] for e in events)==receipt['nativeF']==1684
    assert receipt['nativeB']==(800+6)*2==1612
    assert receipt['nativeF']==receipt['nativeB']+12+12+2*2*12
    assert not read(root/'gpu_lease.json').get('active',False)
    pool=read(root/'pool_audit/pool_plan.json')
    assert pool['frozen_after_native_visual_review']
    assert set(pool['query_raws']).isdisjoint(pool['train_raws'])
    gate=read(root/'exposure_audit/aggregate_cpu_audit.json')
    audit=read(root/'exposure_audit/paired_cpu_audit.json')
    assert audit['all_required_state_data_checks_passed']
    # Aggregate schemas retain the detailed audit rather than duplicating it.
    if 'all_checks_passed' in gate: assert gate['all_checks_passed']
    matrix=[];states=[];required=[folder/'smoke_checkpoint.pt']
    for row in receipt['initial']:
        matrix.append(dict(arm='shared_initial',state='initial',raw=row['raw'],role=row['role'],
                           raw_camera_metrics=row['raw_camera_metrics'],canonical_metrics=row['canonical_metrics']))
    assert sorted(r['raw'] for r in receipt['initial'])==FRAMES
    for arm in receipt['arms']:
        label=arm['label'];base=folder/label
        assert arm['steps']==400 and arm['reload_render_max_abs']==0
        assert arm['non_RGB_fields_exact'] and arm['halo_RGB_exact']
        sampling=read(folder/(label+'_sampling.json'))
        assert len(sampling['schedule'])==400 and all(len(set(x))==2 for x in sampling['schedule'])
        assert set(sum(sampling['schedule'],[]))==set(arm['pool'])
        assert set(sum(sampling['schedule'],[])).isdisjoint(pool['query_raws'])
        expected_count=800//len(arm['pool']);assert all(x==expected_count for x in sampling['counts'].values())
        for name in ('initial_state.pt','checkpoint.pt','history.json','source_patch_02.npz'):
            required.append(base/name)
        for key,state in [('in_memory','final_in_memory'),('final','final_reloaded')]:
            assert sorted(r['raw'] for r in arm[key])==FRAMES
            for row in arm[key]:
                role='QUERY_MAIN' if row['raw'] in [100,110,117] else 'LOW_RES_STRESS' if row['raw']==86 else 'TRAIN' if row['raw'] in arm['pool'] else 'UNUSED_POOL_REFERENCE'
                matrix.append(dict(arm=label,state=state,raw=row['raw'],role=role,
                    raw_camera_metrics=row['raw_camera_metrics'],canonical_metrics=row['canonical_metrics'],alpha_exact=row['alpha_exact']))
        states.append(dict(arm=label,pool=arm['pool'],initial_state=f'paired400_seed0/{label}/initial_state.pt',
            checkpoint=f'paired400_seed0/{label}/checkpoint.pt',source_packet=f'paired400_seed0/{label}/source_patch_02.npz',
            initial_step=0,final_step=400,optimizer='Adam',full_state_keys=['RGB','optimizer','step','schedule','python_rng','numpy_rng','torch_rng','cuda_rng'],
            RGB_parameters=arm['source_core_RGB_parameters'],reload_render_max_abs=arm['reload_render_max_abs'],
            non_RGB_fields_exact=True,halo_RGB_exact=True,sampling_counts=sampling['counts']))
    for group in [folder/'initial']+[folder/a/s for a in ['two_frames','eight_frames'] for s in ['final_in_memory','final_reloaded']]:
        required.append(group/'receipt.json')
        for raw in FRAMES:
            for name in ['source_float.npz','observation_raw.png','observation_pinhole_canonical.png','render_raw.png','render_pinhole_canonical.png','alpha.png','score_mask_pinhole.png','score_mask_raw.png']:
                required.append(group/f'{raw:03d}'/name)
    for p in required:
        assert p.is_file() and p.stat().st_size>0,p
    resources=dict(schema='source_multiframe_resources.v1',complete=True,run_id=root.name,
        totals=dict(nativeF=1684,nativeB=1612,formal_optimizer_updates=800,discarded_smoke_restore_updates=6,
            actual_optimizer_updates=806,contexts=1,formal_updates_per_arm=400,new_diffusion_calls=0,
            target_training_updates=0,target_migration_updates=0,target_fusion_updates=0,power_operations=0,
            native_context_wall_seconds=sum(e['ended_at']-e['started_at'] for e in events)),
        session_events=events,latest_snapshot=read(root/'resource_ledger.json'),
        latest_snapshot_added_to_total=False,
        forward_accounting=dict(optimizer_and_smoke=1612,initial_alpha_setup=12,initial_evaluation=12,final_in_memory_and_reloaded=48),
        backward_semantics='Two native views per RGB optimizer update; 806 actual updates =1612 participating-view backward calls.',
        smoke=dict(saved_step=4,executed_updates=6,next_update_RGB_max_abs=receipt['resume_next_update_RGB_max_abs'],tolerance=1e-6),
        loss_history='17 sparse observations per arm; full 400x2 sampling schedule included. No full per-step loss trace claim.')
    return resources,dict(complete=True,rows=matrix,initial_views=12,final_views=48),dict(complete=True,states=states,
        smoke_checkpoint='paired400_seed0/smoke_checkpoint.pt',independent_CPU_gate='exposure_audit/paired_cpu_audit.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    root=ROOT;recovery=root/'recovery';recovery.mkdir(parents=True,exist_ok=True)
    resources,matrix,states=validate(root)
    for name,value in [('resource_ledger_cumulative.json',resources),('complete_result_matrix.json',matrix),('packet_state_audit.json',states)]:write(recovery/name,value)
    required_final=['README.md','selected_run.json','architecture_components.svg','report_source/report_source_sync_manifest.json']
    issues=['Missing final artifact: '+p for p in required_final if not (root/p).is_file()]
    files={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or '__pycache__' in p.parts or rel.parts[0] in ('archives','takeback_parts') or 'download_parts' in rel.parts or p.suffix in ('.pyc','.tmp'):
            continue
        if rel.as_posix() in ('recovery/restore_manifest_prepared.json','recovery/archive_receipt.json','restore_manifest_compact.json','local_delivery.json'):
            continue
        files['run/'+rel.as_posix()]=p
    for name in ['probe_source_chain.py','camera_corrected_source.py']:
        files['dependencies/r2_code/'+name]=R2/'code'/name
    files['dependencies/r2_restore_manifest_compact.json']=R2/'restore_manifest_compact.json'
    prior_git=read(R2/'publication/git_receipt.json') if (R2/'publication/git_receipt.json').exists() else None
    if prior_git is not None:files['dependencies/r2_git_receipt.json']=R2/'publication/git_receipt.json'
    original=Path(read(root/'paired400_seed0/initialization_equivalence.json')['source_packet'])
    files['dependencies/original_source_patch_02.npz']=original
    inventory=[]
    for arc,p in sorted(files.items()):
        before=p.stat();digest=sha(p);after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Changing input: '+str(p))
        inventory.append(dict(archive_path=arc,original_path=str(p),bytes=after.st_size,sha256=digest,mtime_ns=after.st_mtime_ns))
    prior=read(R2/'restore_manifest_compact.json')
    manifest=dict(schema='source_multiframe_curated_restore.v1',complete=not issues,run_id=root.name,
        original_root=str(root),readiness_issues=issues,files=inventory,payload_files=len(inventory),
        payload_bytes=sum(r['bytes'] for r in inventory),resources=resources['totals'],
        prior_r2_recovery=prior,full_report_companion=prior['full_report_companion'],
        prior_r2_Git_publication=prior_git,
        report_restore_chain=['Verified full r2 report companion','r2 curated run/report_source/publication patch','this run/report_source/publication patch'],
        report_policy='Same original HTML/HTTP; no new full report companion or viewing entry.',
        state_recovery=states,full_sampling_and_Adam_RNG_included=True,
        parent_scene_and_engine_external='Use prior_r2_recovery; do not duplicate large original parent/static/dynamic/environment/engine assets.',
        path_prefix_map=[[str(root),'run'],[str(R2/'code'),'dependencies/r2_code']],
        restore_steps=['Verify all archive members using recovery/restore_manifest.py before extracting to a new empty directory.',
            'Restore full parent persistent scene and engine from prior_r2_recovery; map the explicit original path prefixes in a separate working copy.',
            'Restore source packet row mapping and load the chosen arm checkpoint RGB, Adam optimizer, step400, schedule and all Python/NumPy/Torch/CUDA RNG states.',
            'Smoke checkpoint stores step4; independent next-update replay was exercised during the GPU smoke and its numeric tolerance/result are recorded in resource_ledger_cumulative.json.',
            'Both arms start from identical original RGB. The archive contains initial12 and all final in-memory/reloaded48 float/PNG views with frozen masks, gains and alpha.',
            'Restore report companion, then r2 publication patch, then this publication patch over the same logical report directory.'],
        GPU_calls_by_archiver=0,Git_operations=0,power_operations=0)
    write(recovery/'restore_manifest_prepared.json',manifest)
    if not args.archive:
        print(json.dumps(dict(prepared=True,compressed=False,ready=not issues,issues=issues,files=len(inventory),bytes=manifest['payload_bytes'])));return
    if issues:raise RuntimeError(json.dumps(issues))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=root/'archives'/f'{root.name}_curated_{stamp}.tar.gz';archive.parent.mkdir(exist_ok=True)
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
    receipt=dict(complete=True,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
        payload_files=len(inventory),verified_every_member=True,GPU_calls=0,Git_operations=0,power_operations=0)
    write(recovery/'archive_receipt.json',receipt)
    compact=dict(schema='source_multiframe_compact_restore.v1',complete=True,run_id=root.name,
        archive=dict(path=str(archive),filename=archive.name,bytes=receipt['archive_bytes'],sha256=receipt['archive_sha256'],payload_files=len(inventory),verified_every_member=True),
        main_archive_role='r3配对400步两完整Adam/RNG状态、smoke、两源包、60视图浮点/PNG、池/标定/采样/CPU审计及同页报告增量。',
        full_report_companion=prior['full_report_companion'],report_companion_role='复用已验证r2完整原报告备份，再依次覆盖r2及r3报告增量。',
        prior_r2_recovery=prior,resources=resources['totals'],state_recovery=states,
        report_restore_chain=manifest['report_restore_chain'],restore_steps=manifest['restore_steps'],
        source_geometry_changed=False,target_migration=False,target_fusion=False,new_diffusion_calls=0)
    write(root/'restore_manifest_compact.json',compact);print(json.dumps(receipt))


if __name__=='__main__':main()
