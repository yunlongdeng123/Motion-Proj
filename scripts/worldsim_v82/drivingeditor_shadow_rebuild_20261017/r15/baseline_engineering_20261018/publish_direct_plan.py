"""Publish only this task's code, current evidence, and full-frame input review."""
from pathlib import Path
import json,os,shutil,re,zipfile,hashlib,datetime
from PIL import Image
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo';O=W.parents[1]/'outputs/baseline_engineering_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018')
S=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/baseline_engineering_20261018');paths=[]
def long(p):return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p
def cp(src,rel):
 dest=long(P/rel);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);paths.append(rel.as_posix());return dest
def main():
 for n in ['index.html','intersection_inputs.html','engineering_history_before_direct_plan.html','direct_architecture.svg','direct_plan.json','direct_CPU_gate.json','direct_plan_cancellation_receipt.json','direct_validation.json','A_recovery_queue_status.json','live_GN11_recovery_status.json','GN11_resume_GPU_receipt.json','GN11_resume_sequence_receipt.json','GN11_recovery_audit.json','GN11_timing_fix_GPU_receipt.json','native_runner_timing_GPU_receipt.json','native_cadence_CPU_receipt.json','A_replay_authorization.json','intersection_input_passed.json','intersection_input_review_receipt.json','lidar_camera_intersection_input_receipt.json','lidar_camera_intersection_selection.json','GN11_25k_durability_receipt.json']:
  p=cp(O/n,D/n)
  if p.suffix=='.html':p.write_text(p.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
 for src in sorted((O/'intersection_input_frames').glob('*.jpg')):
  rel=D/'intersection_input_frames'/src.name;dest=long(P/rel);dest.parent.mkdir(parents=True,exist_ok=True)
  with Image.open(src) as im:
   assert im.size==(1920,2560);im.resize((960,1280),Image.Resampling.LANCZOS).save(dest,quality=84)
  paths.append(rel.as_posix())
 code=['apply_direct_plan.py','formal_support.py','rigid_spatial_topology.py','test_rigid_spatial_topology.py','build_direct_implementation.py','direct_cpu_gate.py','run_direct_30k30k.py','evaluate_direct.py','start_direct_plan.py','audit_all_vehicle_coverage.py','infer_all_vehicles_native.py','run_all_vehicle_preparation.py','start_all_vehicle_preparation.py','run_all_vehicle30k_queue.py','evaluate_all_vehicle_stage1.py','start_all_vehicle_queue.py','prepare_fullframe_report.py','fullframe_player_template.html','build_direct_report.py','build_direct_player.py','validate_direct_artifacts.py','publish_direct_plan.py',
 'review_all_vehicle_association.py','infer_vehicle_geometric_prompts.py','start_vehicle_geometric_prompts.py','infer_vehicle_instance_prompts.py','start_vehicle_instance_prompts.py','audit_vehicle_instance_masks.py','audit_full_segment_vehicle_measurements.py','infer_vehicle_camera_prompts.py','start_vehicle_camera_prompts.py','assemble_camera_vehicle_candidates.py','audit_zero_seed_sources.py',
 'prepare_lidar_camera_intersection.py','finalize_intersection_cpu_metadata.py','render_intersection_input_review.py','close_intersection_input_review.py','build_intersection_input_page.py','restart_intersection_queue.py','run_native_timing_safe.py','check_native_cadence_wrapper.py','audit_gn11_recovery.py','gn11_train_timing_safe.py','check_cuda_event_timing.py','resume_gn11_worker.py','direct_after_recovery.py','run_A_recovery_pipeline.py','start_A_recovery.py','collect_recovery_progress.py','audit_remaining_disk.py','audit_25k_checkpoint.py']
 for n in code:cp(W/n,S/n)
 archive=W/'direct_implementation.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in (W/'direct_implementation').glob('*.py'):z.write(p,p.name)
  for n in ['LICENSE','LICENSE_gaussian_splatting.md']:z.write(W/'source'/n,n)
 cp(archive,S/archive.name);cp(W/'direct_source_manifest.json',S/'direct_source_manifest.json')
 for n in ['index.html','sync_player.html']:
  p=cp(O.parent/'identity_comparison_20261018'/n,D.parent/'identity_20261018'/n)
  p.write_text(p.read_text(encoding='utf-8').replace('../r15_repair_4k_20261018/','../repair_4k_20261018/'),encoding='utf-8')
 cp(W.parent/'identity_completion/sync_player_template.html',S.parent/'identity_completion_20261018/sync_player_template.html')
 live=json.loads((W/'live_GN11_recovery_status.json').read_text());step=live['iteration']
 paragraph=f'''用户当前授权：A完成两阶段身份一致的五身份fresh GN11 30K＋新实例30K，两个30K之间不做消融、不在1K等质量批准。B已按最新用户要求改为LiDAR与FRONT相机共同命中的20身份，只跑原生第一阶段30K；不再追补13条相机独有远车。两项实验均保留完整49帧RGB，最终视频、显示、导出全部完整画面，无ROI或局部放大。B仅验收“删除全部纳入车辆”，排除身份可留作静态RGB上下文，不能称全景无车。failure_ledger_refs=[V82-F06]；无subagent、新聊天、R16/R17，原45K暂停，历史资产保留，human_verdict=null。

A原GN11在25847前后向后、优化器更新前读取未完成CUDA事件的耗时而中断，已更新25846；最近完整状态20K，丢失5846步。用户明确授权这次必要重放，并要求后续每5K保存。计时读前同步结束事件的修复通过20次真实GPU检查；新controller52748从20K完整模型/Adam/RNG/采样器/SGCM状态恢复，实际GPU张量与状态逐项相等，前100次重放的帧、时间偏移、学习率一致，未声称CUDA逐位确定性。当前恢复后{step}/30000；只改保存节奏和计时同步，不改目标或优化器。新run后缀GN11_resume20k。原失败5153/6099/39526不可重复启动。

恢复GN11到逻辑30K后自动绑定实际最终背景、环境和点数，fresh五实例30K、完整49帧动态/静态/独立资产/逐ID与全部删除。第二阶段孤立raw-body100＋visible-ID100，SH2000、49相机无放回采样、原生静态与保身份刚体空间拓扑已登记。GT刚体无PVG时间split，第二阶段不新增SGCM；原生append重置屏幕半径次序保留，不冒称持续screen大点保证。新实例保存1/5K/10K/15K/20K/25K/30K；B完整模型/Adam/RNG/采样状态每5K保存。原生test_iterations仍10K/20K/30K，但保留的每checkpoint评估开关会在新增5K节点附带49视图零更新评估，不是中间消融。历史未保存的5K/15K不补造。配置、实际args及剩余磁盘节点计数已检查。

B的20身份为a1/a5/a6/a9/a10/a12/a16/a18/a19/a20/a24/a25/a26/a28/a30/a36/a37/a39/a41/a42。49帧、20个真实非零LiDAR先验、身份清单和980项逐身份原生掩码已核对；实际CPU相机/.1米尺度通过，全部49完整画面已复核并另看raw128。部分真实相机矩形提示原生分割已披露，未填矩形为body，未按GT框逐点裁造，未用额外观察帧。原2×2像素/.5m筛选、聚类和整帧几何QA保留。a41只有19个种子点，输入准入不代表像素或资产质量通过。intersection_input_passed已保存，B controller52417仍0更新；A整条训练及评估完成后再释放all_vehicle_ready，避免阶段交接抢GPU。尚未证明两套完整训练可同时容纳，当前独占排队。

取消的中间消融/贡献追踪不恢复；five_raw_body短测域误标及学习率10倍错误初判撤回继续保留。已有修复R15 4K质量仍未通过，a6白带和a37静态车形不因输入门槛或框外率下降而视为解决。未观测道路不加生成补全。

[当前实验、计时修复与恢复证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html) · [完整49帧20身份输入复核](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/intersection_inputs.html)。全部A+B训练/评估、会启动工作的controller、下载、报告与Git交付完成才按授权关机；automation-2每30分钟跟进，无变化保持安静。
'''
 paragraph=paragraph.replace('取消的中间消融/贡献追踪不恢复；','新增25K完整检查点已实际落盘，静态/动态/环境/训练状态四组件与完成事件相符，Adam张量有限，RNG/采样器/SGCM/5K配置及严格续训合同CPU只读检查通过，49视图评估完成。此处未对25K做GPU恢复或增加训练更新。\n\n取消的中间消融/贡献追踪不恢复；')
 p=P/'docs/RESEARCH_STATUS.md';text=p.read_text(encoding='utf-8');end=text.index('五身份分支同步视频已交付：')
 updated=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
 p.write_text('# 当前研究状态\n\n更新：'+updated+'。当前 task=WS-V82-R15-BASELINE-ENGINEERING-20261018。\n\n'+paragraph+'\n'+text[end:],encoding='utf-8');paths.append('docs/RESEARCH_STATUS.md')
 p=P/'docs/EXPERIMENTS.md';text=p.read_text(encoding='utf-8')
 text=re.sub(r'^\| WS-V82-R15-BASELINE-ENGINEERING-20261018 \|.*$', '| WS-V82-R15-BASELINE-ENGINEERING-20261018 | 五身份GN11 30K＋实例30K；LiDAR相机交集20身份一阶段30K；每5K保存、完整画面删除验收 | [当前实验与实际证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html) |',text,flags=re.M)
 p.write_text(text,encoding='utf-8');paths.append('docs/EXPERIMENTS.md')
 p=P/'docs/research_failures/entries/V82-F06.md';text=p.read_text(encoding='utf-8');heading='## 2026-10-18：CUDA计时中断修复、授权重放和20身份交集范围'
 evidence=f'''原GN11在25847读取尚未完成的CUDA事件耗时中断，25846更新已完成而最近完整checkpoint为20K；5846未持久化步数丢失。用户授权必要重放，并要求后续5K完整保存。事件同步修复20次真实GPU检查通过；实际20K静态/动态/环境及Adam、RNG、采样器、SGCM恢复通过，重放前100步相机/时间偏移/学习率一致，未宣称GPU逐位确定性。新实例与B的实际args/保存节点和磁盘计数已核对；原生每checkpoint评估保持开启，新增5K保存节点也附带零更新评估，不是消融。A目前{step}/30000，未宣布完整训练或质量通过。

用户将B从全可见车辆改为LiDAR与相机共同命中20身份；13相机独有轨迹作为上下文，不再追补。原49帧完整RGB、980项逐ID原生mask、20个非零真实先验及统一清单核对，49完整画面已看。相机GT矩形提示分割明确披露，无矩形填body或逐点框裁。稀疏a41=19点，不保证资产质量。B仍0更新、独占排队等A全链路结束，不能以输入验收声称全景无车或道路已补全。human_verdict=null。

[恢复、准入和完整画面证据](../../autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html)。历史错误结论及撤回原文保留；不新增failure ID。
'''
 evidence+='\n首个新增25K完整四组件、Adam有限性、RNG/采样器/SGCM和严格续训合同已CPU只读核对，49视图评估完成；未进行25K GPU恢复或新增更新。实际凭据随报告保存。\n'
 if heading not in text:text+='\n\n'+heading+'\n\n'+evidence
 else:text=re.sub(re.escape(heading)+r'\n.*?(?=\n## |\Z)',lambda m:heading+'\n\n'+evidence,text,flags=re.S)
 p.write_text(text,encoding='utf-8')
 paths.append('docs/research_failures/entries/V82-F06.md')
 (W/'direct_publish_paths.json').write_text(json.dumps(sorted(set(paths)),indent=2),encoding='utf-8')
 print(json.dumps(dict(scoped_files=len(set(paths)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
if __name__=='__main__':main()
