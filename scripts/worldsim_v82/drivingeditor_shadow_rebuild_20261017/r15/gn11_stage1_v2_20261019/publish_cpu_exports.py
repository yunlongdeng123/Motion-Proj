"""Publish actual CPU-exported images/videos and explicit pending GPU scope."""
from pathlib import Path
import datetime,json,re
from publish_ablation_nodes import W,O,REPO,DOC,CODE,read,copy,write

def main():
    media=read(O/'cpu_export_media.json');checks=read(O/'cpu_export_validation.json');sources=read(O/'cpu_export_sources.json')
    assert media['new_videos']==22 and media['reused_videos']==12 and checks['decoded_frames']==1666
    assert media['independent_completed']==['full','real_only'] and media['independent_pending']==['prior_only','ownership_only']
    assert media['training_updates']==0 and media['export_GPU_used'] is False
    assert sources['PNG_count']==1078 and not sources['remote_files_written']
    runtime=read(W/'ablation_runtime.json');assert runtime['gpu']=='NO_GPU_DEVICE_NODES'
    assert all(a['status']['status']=='complete' and a['status']['iteration']==30000 and len(a['audited_steps'])==6 for a in runtime['arms'].values())
    card=REPO/'docs/research_failures/entries/V82-F06.md';old=card.read_text(encoding='utf-8')
    heading='## 2026-10-19：四组30K完成与无卡CPU视频导出'
    if heading in old:raise RuntimeError('CPU export milestone already published')
    linkbase=DOC.relative_to('docs').as_posix()
    status={
      'status':'FOUR_TRAININGS_COMPLETE_CPU_DELIVERY_READY_TWO_INDEPENDENT_RENDERS_PENDING_GPU',
      'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_four_actual_optimizer_counts':30000,
      'all_24_checkpoints_CPU_audited':True,'current_device_state':'NO_GPU_DEVICE_NODES',
      'independent_completed':{a:{'completed_utc':sources['sources'][a]['receipt']['completed_utc'],'frames':49,'PNG_count':343,
        'parameter_updates':0,'source_checkpoints_unchanged':True,'actual_model_Adam_RNG_sampler_SGCM_restore':True} for a in ['full','real_only']},
      'independent_pending':{'prior_only':'SSH disconnected during user no-GPU reboot; partial 0PNG, no completion/restore receipt; preserved',
        'ownership_only':'not started; no independent result'},
      'first_full_attempt':'stopped on CUDA/CPU comparison of native SGCM lineage metadata; no optimizer update; diagnostic directory preserved',
      'exporter_fix':'exact value/shape/dtype comparison normalizes only native SGCM CPU-captured metadata device; no value tolerance or tensor dropped',
      'exporter_actual_GPU_validation':['full actual success','real_only actual success'],'native_state_device_normalizations':sources['sources']['full']['restore_receipt']['native_state_device_normalizations'],
      'no_training_restart_or_replay':True,'source_checkpoints_modified':False,'CPU_only_export_GPU_used':False,
      'quality_accepted':False,'human_verdict':None}
    write(O/'independent_export_status.json',json.dumps(status,ensure_ascii=False,indent=2))
    record=read(O/'task_record.json');record['state']=status['status'];record['updated_utc']=status['updated_utc']
    record['cpu_export_delivery']={'native_four_page':'native30k_four_compare.html','independent_two_page':'independent_pair_compare.html',
      'new_videos':22,'reused_videos':12,'decoded_frames':1666,'training_updates':0,'GPU_used':False,'independent_pending':['prior_only','ownership_only']}
    record['independent_export_status']=status
    write(O/'task_record.json',json.dumps(record,ensure_ascii=False,indent=2))
    current=REPO/'docs/RESEARCH_STATUS.md';text=current.read_text(encoding='utf-8');marker='## 已完成的原A/B基线与历史证据';top,history=text.split(marker,1)
    paragraphs=top.split('\n\n')
    for i,p in enumerate(paragraphs):
        if p.startswith('当前实际在训：'):
            paragraphs[i]='当前实际在训：无。四组static/dynamic/environment实际optimizer.step各30000，全部24个5K节点CPU只读审计完成。训练控制器已在独立评估前退出并释放锁；当前用户已切换无卡开机，旧PID仅为历史记录，不启动训练、恢复或GPU导出。'
        elif p.startswith('已完成训练：'):
            paragraphs[i]='已完成训练：完整版、real_only、prior_only、ownership_only各fresh第一阶段30000、第二阶段0。训练及原生49节点输出齐全；独立zero-update49已完成full和real_only，prior_only/ownership_only待GPU，不能称四组最终独立评估完成。'
        elif p.startswith('资源采样：'):
            paragraphs[i]=f'资源采样：{runtime["collected_utc"]}，无/dev/nvidia设备节点，nvidia-smi权限拒绝；md0 {runtime["md0"]["array_state"]}/degraded{runtime["md0"]["degraded"]}/sync{runtime["md0"]["sync_action"]}，磁盘空{runtime["disk_free_bytes"]/1e9:.3f}GB。无卡状态继续CPU只读导出，不尝试新GPU任务。独立评估与完整质量报告仍待完成，当前不关机；automation-2保持ACTIVE，普通无变化安静。'
    top='\n\n'.join(paragraphs)
    top=re.sub(r'更新：[^。]+。',f'更新：{status["updated_utc"]}。',top,count=1)
    delivery=f'\n\n本轮无卡导出：[四组30K完整同步视频与原生A对照]({linkbase}/native30k_four_compare.html)，统一使用四组各自原生节点的gt/composite/static/dynamic完整49帧；[已完成独立渲染的两组七通道]({linkbase}/independent_pair_compare.html)，包含真实独立delete_all及alpha。新增22视频、复用12视频，均480×320/49帧/10fps全I帧；全部34视频1666帧实际解码、对应源PNG顺序/编码误差和JS控制fixture通过，非浏览器UI。保留原两组视频页。\n\n实际静态全49复核：四组共196张完整静态PNG已查看。完整版仍有矩形道路纹理、暗条与接缝；real_only有黄白亮块/蓝边，prior_only有黑块，ownership_only仍有黑色车状轮廓。均不判干净静态通过；动态/完整重建/alpha/道路来源区域与阴影的全量质量对照仍未完成。[实际观看范围]({linkbase}/native_static_review.json)。human_verdict=null。\n\n'
    write(current,top+delivery+marker+history)
    entry=f'\n\n{heading}\n\n四组实际更新各30000，所有24保存节点CPU审计通过；无新增训练、恢复或重放。full独立49于{status["independent_completed"]["full"]["completed_utc"]}完成，real_only于{status["independent_completed"]["real_only"]["completed_utc"]}完成，各343张PNG/0更新且源checkpoint未变。首次full导出遇SGCM谱系元数据的CUDA/CPU比较错误；修复仅对原生capture固定放CPU的两项元数据做严格值/shape/dtype比较，随后两组实际成功。没有修改训练源码或checkpoints。\n\n用户切无卡重启时prior独立导出中断，0PNG、无receipt/restore，partial保留；owner独立导出未启动。不能把本轮四组原生静态视频冒充四组独立删车完成。当前/dev无nvidia，CPU导出不调用GPU。\n\nCPU只读stdout打包实际1078PNG＋6JSON，仅下载181906875B归档，SHA256 {read(O/"cpu_export_download.json")["archive_sha256"]}；远端无新写文件，模型/大NPZ未下载。原392PNG归档复用，四组native逐帧GT hash相同；独立两组GT彼此hash一致，但与native量化方法有最大1/255差异，未冒称跨方法文件hash相同。新增22完整视频、复用12视频，共1666帧实际解码和源顺序检查通过，7模式4通道＋1模式7通道实际生成JS fixture通过，非浏览器UI。\n\n[四组原生30K对比](../../{linkbase}/native30k_four_compare.html) · [两组独立七通道](../../{linkbase}/independent_pair_compare.html) · [真实独立状态](../../{linkbase}/independent_export_status.json)。\n\n全49静态实际观看发现完整版仍有道路暗条/接缝，real有黄白亮块、prior有黑块、owner有车状轮廓；未作质量通过或内部单约束因果归因。余下两组独立渲染需GPU，完整多通道质量对照待完成。human_verdict=null；当前不关机。\n'
    write(card,old+entry)
    files=['native30k_four_compare.html','independent_pair_compare.html','cpu_export_media.json','cpu_export_sources.json','cpu_export_download.json',
      'cpu_export_validation.json','native_static_review.json','independent_export_status.json','index.html','candidate_source.zip','local_delivery_checks.json','task_record.json',
      'runtime_progress.json','final_evaluation_preflight.json','ablations/index.html','ablations/runtime.json','ablations/local_delivery_checks.json']
    for p in sorted((O/'independent_receipts').rglob('*.json')):files.append(p.relative_to(O).as_posix())
    # Publish only the 22 explicitly completed new media files and their posters.
    for key,item in media['videos'].items():
        if item['path'].startswith('cpu_export_videos/'):
            files.extend([item['path'],item['poster']])
    paths=['docs/RESEARCH_STATUS.md','docs/research_failures/entries/V82-F06.md']
    for name in dict.fromkeys(files):
        rel=DOC/name;copy(O/name,REPO/rel);paths.append(rel.as_posix())
    for name in ['collect_cpu_exports.py','build_cpu_comparisons.py','check_cpu_export_player.js','verify_cpu_comparisons.py','verify_ablation_delivery.py',
      'cpu_delivery_banner.py','collect_runtime.py','collect_ablations.py','build_report.py','build_ablation_report.py','enrich_runtime_report.py',
      'evaluate_v2_full49.py','preflight_v2_final.py','build_native_review_sheets.py','publish_cpu_exports.py','final_full49_player.js']:
        rel=CODE/name;copy(W/name,REPO/rel);paths.append(rel.as_posix())
    (W/'publish_cpu_exports_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print(json.dumps({'paths':len(paths),'new_video_files':22,'independent_pending':status['independent_pending']}))

if __name__=='__main__':main()
