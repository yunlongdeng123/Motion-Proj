"""显式白名单生成器：仅封装实际源码、轻量记录及报告；不操作共享Git。"""
from pathlib import Path
import json
import zipfile
from archive_common import ARCHIVE, ALLOWED_ROLES, LINK_UPDATES, dump_json, read_json, digest, lightweight_text, validate_navigation_delta, validate_training_summary

PUB=Path(__file__).resolve().parent
WORK=PUB.parent
BASE=WORK.parents[1]
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/a20_joint_reconstruction_20261122_r1'
TASK='A20-JOINT-RECONSTRUCTION-20261122'
REV='final02'


def main():
    final=read_json(WORK/'final_iteration_receipt.json')
    verification=read_json(WORK/'assets/takeback_verification.json')
    qa=read_json(WORK/'report_qa/receipt.json')
    result=read_json(WORK/'result.json')
    if not qa.get('passed') or result.get('current_group')!='r3_ablation' or not result.get('root_report_complete') or result.get('scene_qualified') is not False:
        raise ValueError('报告未处于最终真实收尾状态')
    actor=final['selected_actor']
    if verification['sha256']!=actor['sha256']:
        raise ValueError('本地资产验收SHA与最终指针不符')
    counts={}
    render_counts={}
    ledgers={}
    for trial in ['r1','r2','r3']:
        for key,n in [('smoke_geometry_updates',10),('replay_updates',2),('discarded_appearance_updates',1),('discarded_joint_updates',1),('formal_geometry_updates',40)]:
            counts[trial+'_'+key]=n
        for stage in ['smoke','geometry']:
            rel='takeback/execution_'+trial+'/'+stage+'/resource_ledger.json'
            ledger=read_json(WORK/rel)
            ledgers[trial+'_'+stage]=dict(relative=rel,sha256=digest((WORK/rel).read_bytes()),**ledger)
            render_counts[trial+'_'+stage+'_native_calls']=ledger['native_calls']
    counts.update(postprocess_optimizer_updates=0,formal_appearance_updates=0,formal_joint_updates=0,road_updates=0,other_actor_updates=0,background_updates=0)
    post=read_json(WORK/'takeback/execution_r3/post_geometry_ablation/resource_ledger.json')
    render_counts['postprocess_native_calls']=post['new_native_calls']
    render_counts['postprocess_counterfactual_calls_subset']=post['counterfactual_native_calls']
    ledger=dict(schema='a20.actual_execution_ledger.v1',actual_stage_counts=counts,
        total_optimizer_updates_all_independent_trials=sum(counts.values()),formal_geometry_updates_all_independent_trials=120,
        selected_asset_formal_updates=40,selected_asset_postprocess_updates=0,actual_render_counts=render_counts,
        rendering_count_note='counterfactual2属于postprocess native54的子集；初始导出的调用未另计，不能把本表相加称全程GPU调用总数',
        ledgers=ledgers,postprocess=post,scene_qualified=False,root_report_complete=True)
    dump_json(WORK/'actual_execution_ledger.json',ledger)
    restore=dict(schema='a20.joint_reconstruction.restore_index.v1',task_id=TASK,
        local_workspace=str(WORK),remote_task_root=REMOTE,local_report=str(BASE/'outputs/a20_joint_reconstruction_20261122/index.html'),
        report_source='report_source/build_report.py',report_registry='report_source/report_groups.json',
        source_paths={'initialization':'initialization/build_joint_initialization.py','current_runtime':'runtime/run_joint_reconstruction.py',
                      'postprocess':'runtime/post_geometry_ablation.py','independent_trials':'execution_source/r1 and execution_source/r2; r3 is runtime'},
        local_asset=dict(actor),
        full_state_checkpoint=final['full_state_checkpoint'],
        asset_verification='receipts/assets/takeback_verification.json',final_receipt='receipts/final_iteration_receipt.json',
        final_review='receipts/root_geometry_review.json',actual_ledger='actual_execution_ledger.json',
        restore_proof='receipts/execution_r3/post_geometry_ablation/geometry_ablation_reload_proof.json',
        ablation_proof='receipts/execution_r3/post_geometry_ablation/proof.json',
        exact_G40_Adam_restore=final['exact_G40_Adam_restore'],saved_checkpoint_exact_reload=final['saved_checkpoint_exact_reload'],
        no_resampling=True,scene_qualified=False,root_report_complete=True,
        final_render_source=REMOTE+'/execution_r3/post_geometry_ablation/evaluation/geometry_ablation',
        pure_teacher_render_source=REMOTE+'/execution_r3/geometry/teacher_cache/novel_00..05',
        local_render_root=str(WORK/'takeback/execution_r3/post_geometry_ablation/evaluation/geometry_ablation'),
        image_restore='直接取回remote任务下各evaluation目录；report_source中的work/takeback相对链接在原本地workspace解析。Git不含图片。',
        report_rebuild='原workspace: python assemble_result.py && python build_report.py；Git归档保留生成器与来源SHA，先按本索引恢复workspace相对目录。',
        prior_dependency_archive='docs/autoresearch/a20_sd15_sameview_20261122/restore_index.json',
        legacy_fusion_remote='/root/autodl-tmp/drivingeditor_repro/campaign/a20_native_gaussian_fusion_20261122_r1',
        old_route_parameter_updates=0,model_images_npz_checkpoint_in_git=False)
    # 资产有两个位置：本地取回路径与远端canonical；不覆盖其中任何一个。
    restore['local_asset']['path']=str(WORK/'assets/a20_geometry_candidate.npz')
    restore['local_asset']['remote_path']=actor['path']
    dump_json(WORK/'restore_index.json',restore)
    out=PUB/('local_final_inputs_'+REV)
    if out.exists():raise FileExistsError('保留已有封包；修改revision后重建')
    remote_upload=REMOTE+'/publication/'+out.name
    entries=[];sources=[];seen=set()
    def local(src,dest,role):
        p=Path(src).resolve();data=lightweight_text(p)
        if dest in seen:raise ValueError('重复路径 '+dest)
        seen.add(dest);rel='payload/'+str(len(sources)).zfill(3)+p.suffix;target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        rec=dict(source=remote_upload+'/'+rel,destination=dest,role=role,local_origin=dict(path=str(p),sha256=digest(data)))
        sources.append(rec);entries.append(dict(origin='local',source=str(p),destination=dest,role=role,sha256=digest(data)))
    def remote(rel,dest,role):
        if dest in seen:raise ValueError('重复路径 '+dest)
        seen.add(dest);sources.append(dict(source=REMOTE+'/'+rel,destination=dest,role=role))
        entries.append(dict(origin='remote',source=REMOTE+'/'+rel,destination=dest,role=role))
    for name in ['assemble_result.py','build_report.py','append_report_group.py','qa_report.py','report_ui.js','report_style.css','report_groups.json','result.json']:
        local(WORK/name,ARCHIVE+'/report_source/'+name,'report_source')
    local(BASE/'outputs/a20_joint_reconstruction_20261122/index.html',ARCHIVE+'/index.html','report_source')
    for name,role in [('report.md','record'),('architecture.svg','architecture_source'),('restore_index.json','provenance'),('actual_execution_ledger.json','receipt')]:
        local(WORK/name,ARCHIVE+'/'+name,role)
    for name in ['build_joint_initialization.py','manifest.json','cpu_exact_source_verification.json','interface_proposal.json','README.md']:
        local(WORK/'initialization'/name,ARCHIVE+'/initialization/'+name,'run_source' if name.endswith('.py') else 'provenance')
    for name in ['native_bindings.json','context_plan.json','protocol.json','launch_controller.py','launch_ablation.py','make_geometry_views.py','finalize_receipt.py']:
        local(WORK/name,ARCHIVE+'/execution_source/'+name,'run_source' if name.endswith('.py') else 'configuration')
    for name in ['joint_core.py','joint_gpu.py','run_joint_reconstruction.py','remote_path_map.json','training_contract.frozen.json']:
        remote('execution_r3/source/'+name,ARCHIVE+'/runtime/'+name,'run_source' if name.endswith('.py') else 'contract')
    for name in ['README.md','post_geometry_ablation.py']:
        remote('runtime/'+name,ARCHIVE+'/runtime/'+name,'run_source')
    for trial in ['r1','r2']:
        for name in ['joint_core.py','joint_gpu.py','training_contract.frozen.json']:
            remote('execution_'+trial+'/source/'+name,ARCHIVE+'/execution_source/'+trial+'/'+name,'run_source' if name.endswith('.py') else 'contract')
    receipt_paths=['execution_initial/initial/evaluation/fresh_process/metrics.json','execution_initial/initial/fresh_process_reload_proof.json']
    for trial in ['r1','r2','r3']:
        receipt_paths += ['execution_'+trial+'/'+suffix for suffix in ['smoke/smoke_proof.json','smoke/resource_ledger.json','geometry/completion.json','geometry/resource_ledger.json','geometry/fresh_reload_proof.json','geometry/evaluation/geometry_final/metrics.json']]
    receipt_paths += ['execution_r3/post_geometry_ablation/'+x for x in ['proof.json','postprocess_contract.json','resource_ledger.json','geometry_ablation_reload_proof.json','evaluation/geometry_ablation/metrics.json']]
    for rel in receipt_paths:
        local(WORK/'takeback'/rel,ARCHIVE+'/receipts/'+rel,'metrics' if rel.endswith('metrics.json') else 'receipt')
    for name in ['final_iteration_receipt.json','root_geometry_review.json','assets/takeback_verification.json','report_qa/receipt.json']:
        local(WORK/name,ARCHIVE+'/receipts/'+name,'receipt')
    helpers=['archive_common.py','prepare_archive.py','publish_selected.py','validate_archive.py','verify_published.py','verify_takeback.py','capture_preflight.py','run_prepare_stage.py','build_final_archive.py']
    for name in helpers:local(PUB/name,ARCHIVE+'/publication_tools/'+name,'publication_tool')
    nav_paths={'docs/autoresearch/a20_sd15_sameview_20261122/report_source/build_report.py':BASE/'work/a20_sd15_sameview_20261122/build_report.py',
               'docs/autoresearch/a20_sd15_sameview_20261122/index.html':BASE/'outputs/a20_sd15_sameview_20261122/index.html'}
    for dest,src in nav_paths.items():
        validate_navigation_delta(dest,(PUB/'preflight_final01/prior_navigation_base'/dest).read_bytes(),src.read_bytes())
        local(src,dest,'report_source')
    dump_json(PUB/'whitelist.final.json',dict(task_id=TASK,finalized=True,entries=entries,raw_media_models_weights_excluded=True))
    local(PUB/'whitelist.final.json',ARCHIVE+'/publication_tools/whitelist.final.json','provenance')
    manifest=read_json(PUB/'manifest.template.json')
    manifest.update(publication_ready=True,root_report_complete=True,actual_result_kind='GEOMETRY_REJECTED_NO_FORMAL_APPEARANCE_OR_JOINT',actual_training_performed=True,actual_stage_counts=counts,actual_render_counts=render_counts,sources=sources)
    for name in ['status_snapshot.md','experiments_append.md','failure_append.md']:
        p=out/'support'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(lightweight_text(PUB/name))
    manifest['status_update'].update(old_source=REMOTE+'/publication/preflight_final01/status_replace_before.md',source=remote_upload+'/support/status_snapshot.md')
    manifest['experiments_append']['source']=remote_upload+'/support/experiments_append.md'
    manifest['failure_append']=dict(source=remote_upload+'/support/failure_append.md',task_marker=TASK)
    validate_training_summary(manifest)
    dump_json(PUB/'manifest.final01.json',manifest)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    dump_json(PUB/'local_upload_receipt.final01.json',dict(path=str(archive),sha256=digest(archive.read_bytes()),bytes=archive.stat().st_size,source_files=len(sources),model_images_npz_checkpoint_included=False))
    dump_json(PUB/'final_ready.json',dict(task_id=TASK,publication_ready=True,root_report_complete=True,scene_qualified=False,root_approval='explicit parent instruction: finalize report and publish using isolated index; final sources frozen',actual_stage_counts=counts))
    print(json.dumps(dict(files=len(sources),upload_bytes=archive.stat().st_size,total_updates=sum(counts.values()),selected_asset_updates=40,postprocess_updates=0,ready=True)))

if __name__=='__main__':main()
