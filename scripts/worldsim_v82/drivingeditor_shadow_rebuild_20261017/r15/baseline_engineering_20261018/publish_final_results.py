"""Publish only actual completed A/B results and an observed visual verdict."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess,zipfile
import publish_checkpoint_milestone as p
from publish_direct_plan import referenced_current_media

def main():
 assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=p.P).strip()
 plan=json.loads((p.O/'direct_plan.json').read_text(encoding='utf-8'));assert plan['report']['phase']['complete']
 q=json.loads((p.W/'parallel_queue_status.json').read_text());assert q['state']=='PARALLEL_A_B_FULL49_COMPLETE_AWAITING_REPORT_GIT'
 review=json.loads((p.O/'final_quality_review.json').read_text(encoding='utf-8'));assert review['status']=='COMPLETED_ACTUAL_FULLFRAME_QUALITY_REVIEW'
 assert review['human_verdict'] is None
 archive=p.W/'direct_implementation.zip';manifest=json.loads((p.W/'direct_source_manifest.json').read_text());assert len(manifest)==13
 with zipfile.ZipFile(archive) as z:
  assert all(hashlib.sha256(z.read(name)).hexdigest()==value for name,value in manifest.items())
  assert {'LICENSE','LICENSE_gaussian_splatting.md'}<=set(z.namelist())
 for name in ['index.html','results.html','results_status.json','direct_architecture.svg','direct_plan.json','direct_validation.json','player_controls_validation.json','live_direct_instance_status.json','live_B_native_status.json','A_recovery_queue_status.json','parallel_queue_status.json','parallel_group_guard.json','all_vehicle_train_queue_status.json','all_actor_gpu_claim.json','remaining_disk_forecast.json','runtime_io_health.json','runtime_io_recovery_receipt.json','final_quality_review.json','final_full49_delivery.json','final_delivery_collection.json']:
  src=p.O/name if (p.O/name).exists() else p.W/name
  p.cp(src,p.D/name)
  if name.endswith('.html'):
   dest=p.target(p.D/name);dest.write_text(dest.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
 for root in ['direct_full49','all_vehicle_full49']:
  for name in ['receipt.json','restore_receipt.json','progress.json']:
   if (p.W/root/name).exists():p.cp(p.W/root/name,p.D/root/name)
 for src in p.O.glob('parallel_[AB]_*_durability_receipt.json'):p.cp(src,p.D/src.name)
 for src in (p.O/'final_quality_frames').rglob('*'):
  if src.is_file():p.cp(src,p.D/'final_quality_frames'/src.relative_to(p.O/'final_quality_frames'))
 media=referenced_current_media();assert len(media)==46
 for name in media:p.cp(p.O/name,p.D/name)
 for name in ['audit_parallel_checkpoint_node.py','audit_remaining_disk.py','package_final_full49.py','collect_final_delivery.py','prepare_final_visual_review.py','record_final_quality_review.py','build_direct_report.py','build_direct_player.py','check_player_identity_controls.js','validate_direct_artifacts.py','publish_final_results.py','publish_direct_plan.py','remote_shutdown_preflight.py','shutdown_after_delivery.py']:
  p.cp(p.W/name,p.S/name)
 for name in ['final_visual_review_manifest.json','report_phase_validation.json']:
  if (p.W/name).exists():p.cp(p.W/name,p.D/name)
 summary=review['summary'];at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
 text=f'''# 当前研究状态

更新：{at}。当前 task=WS-V82-R15-BASELINE-ENGINEERING-20261018；两项授权实验及最终49帧评估完成，质量结论单列。

A同一批a6/a37/a5/a26/a18完成fresh GN11逻辑30K＋fresh实例30K。B按最新范围完成LiDAR∩FRONT相机20身份原生第一阶段30K，第二阶段0。两项均保留原49帧完整RGB；13纯相机轨迹只作上下文，B删除全部纳入车辆不等于全画面无车。原45K暂停，所有历史输入/checkpoint保留；未加入窗外训练帧、生成补全或新消融，human_verdict=null。

{summary}

技术完成凭据：A真正30K完整单文件3047836844B，91项模型/Adam张量有限、RNG/采样器30000与终点策略合同已CPU核验，终点检查明确不授权30001步；最终评估另实际CUDA恢复全部模型/Adam/RNG/采样状态精确相等，零更新。A完整49帧×5身份共245记录、1568 PNG；每ID删除目标可见alpha精确0，静态分支逐像素不变。B四组件30K/Adam/RNG/SGCM/sampler及native49评估已CPU核验，最终另49帧343 PNG删除纳入动态集合，独立static回放通过。两项最终评估不改checkpoint。177558879B仅PNG/JSON归档已一次下载并逐文件SHA核对，未下载checkpoint/NPZ。46路当前完整480×320/49帧/10fps/all-I视频已发布（新A32、新B7、A第一阶段7），默认新A，可逐身份选择；全部显示/导出完整画面，无ROI。技术和文件通过不代表背景质量通过。

原GN11曾在25847读取未完成CUDA事件中断，用户仅授权5846必要重放；该历史额外预算单列。后续所有原生/实例完整状态每5K保存；原不存在的GN11 5/15K不补造。实例1999完整保存后切换同预算共享准入、从2000精确恢复，额外重放0。后来宿主机md0写入等待时保留原进程，02:45UTC确认同PID恢复继续，额外重放0；具体平台根因和恢复时刻未获确认。30K只读检查脚本首次遇到终点续训限制和终态精简字段，现用独立终点合同/真实最后4000字节日志解决；未修改训练实现、目标、上限或已保存模型。

两阶段身份/mask/init输入绑定已验证。实例适配为GT刚体、原孤立raw-body100＋visible-ID100、SH2000、49无放回、原生static与保owner空间clone/split/prune，staticLR.1/环境冻结、第二阶段无新SGCM；不复现PVG时间split。静态原生时间位移与衰减正则不一致的继承问题、缺道路观测与外观阴影证据分别保留，不能将所有残留归因于补道路，也未对具体残留Gaussian完成因果归因。学习率10倍误判和短测body域误标撤回继续保留。

[完整画面同步播放器与质量报告](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html) · [原始输入覆盖](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/intersection_inputs.html)。文件/PNG/实际视频解码与JS合同验证已完成；file URL安全限制未绕过，未宣称浏览器UI交互验收。13份真实训练源码原字节ZIP和licenses哈希核对，源ZIP SHA256={hashlib.sha256(archive.read_bytes()).hexdigest()}。failure_ledger_refs=[V82-F06]，不新增ID。当前交付后按授权执行全任务preflight/sync/AutoDL关机并记录SSH关闭观察，不声称核实云计费。

'''
 rel=Path('docs/RESEARCH_STATUS.md');dest=p.target(rel);old=dest.read_text(encoding='utf-8');dest.write_text(text+old[old.index('五身份分支同步视频已交付：'):],encoding='utf-8');p.paths.append(rel.as_posix())
 rel=Path('docs/EXPERIMENTS.md');dest=p.target(rel);old=dest.read_text(encoding='utf-8');new=re.sub(r'^\| WS-V82-R15-BASELINE-ENGINEERING-20261018 \|.*$', '| WS-V82-R15-BASELINE-ENGINEERING-20261018 | A五身份30K＋30K与B交集20身份30K完成；完整画面结果与质量结论 | [最终报告与同步视频](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html) |',old,flags=re.M);assert new!=old;dest.write_text(new,encoding='utf-8');p.paths.append(rel.as_posix())
 rel=Path('docs/research_failures/entries/V82-F06.md');dest=p.target(rel);old=dest.read_text(encoding='utf-8');heading='## 2026-10-19：A双阶段与B单阶段完成及完整画面复核';assert heading not in old
 dest.write_text((old+'\n\n'+heading+'\n\n'+text.split('\n\n',2)[-1].replace('](autoresearch/','](../../autoresearch/')).rstrip()+'\n',encoding='utf-8');p.paths.append(rel.as_posix())
 subprocess.run(['python','scripts/build_research_failure_index.py'],cwd=p.P,check=True,capture_output=True)
 (p.W/'direct_publish_paths.json').write_text(json.dumps(sorted(set(p.paths)),indent=2))
 print(json.dumps(dict(scoped_files=len(set(p.paths)),actual_media=len(media),training_updates=0)))
if __name__=='__main__':main()
