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

def referenced_current_media():
 page=(O/'results.html').read_text(encoding='utf-8')
 match=re.search(r'<script type="application/json" id="data">(.*?)</script>',page,re.S)
 assert match,'actual player data missing'
 found=set()
 def walk(x):
  if isinstance(x,dict):
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
  elif isinstance(x,str) and x.startswith('branch_videos/'):
   assert re.fullmatch(r'branch_videos/(?:gn11five|matched5|allvehicles)_[A-Za-z0-9_]+\.mp4',x),x
   assert (O/x).is_file(),x
   found.add(x)
 walk(json.loads(match.group(1)))
 return sorted(found)
def main():
 for n in ['index.html','intersection_inputs.html','engineering_history_before_direct_plan.html','direct_architecture.svg','direct_plan.json','direct_CPU_gate.json','direct_plan_cancellation_receipt.json','direct_validation.json','A_recovery_queue_status.json','live_GN11_recovery_status.json','GN11_resume_GPU_receipt.json','GN11_resume_sequence_receipt.json','GN11_recovery_audit.json','GN11_timing_fix_GPU_receipt.json','native_runner_timing_GPU_receipt.json','native_cadence_CPU_receipt.json','A_replay_authorization.json','intersection_input_passed.json','intersection_input_review_receipt.json','lidar_camera_intersection_input_receipt.json','lidar_camera_intersection_selection.json','GN11_25k_durability_receipt.json']:
  p=cp(O/n,D/n)
  if p.suffix=='.html':p.write_text(p.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
 for src in sorted((O/'intersection_input_frames').glob('*.jpg')):
  rel=D/'intersection_input_frames'/src.name;dest=long(P/rel);dest.parent.mkdir(parents=True,exist_ok=True)
  with Image.open(src) as im:
   assert im.size==(1920,2560);im.resize((960,1280),Image.Resampling.LANCZOS).save(dest,quality=84)
  paths.append(rel.as_posix())
 for n in ['live_direct_instance_status.json','live_B_native_status.json','direct_initial_checkpoint_receipt.json','video_encoding_CPU_receipt.json']:
  if (O/n).exists():cp(O/n,D/n)
 for n in ['results.html','results_status.json','player_controls_validation.json','parallel_cutover_receipt.json','parallel_capacity_receipt.json','parallel_A_runtime_restore.json','direct_active_run.json','parallel_5k_durability_receipt.json','remaining_disk_forecast.json']:
  if (O/n).exists():
   dest=cp(O/n,D/n)
   if n=='results.html':dest.write_text(dest.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
 for src in O.glob('parallel_[AB]_*_durability_receipt.json'):cp(src,D/src.name)
 if (O/'branch_videos').exists():
  # Only actual player references; include finalA/B and exclude interrupted encodes.
  for rel in referenced_current_media():cp(O/rel,D/rel)
 if (O/'road_observability_20261019').exists():
  for src in (O/'road_observability_20261019').rglob('*'):
   if src.is_file():cp(src,D/'road_observability_20261019'/src.relative_to(O/'road_observability_20261019'))
 if (O/'gn11_static_duplicate_20261019').exists():
  for src in (O/'gn11_static_duplicate_20261019').rglob('*'):
   if src.is_file():cp(src,D/'gn11_static_duplicate_20261019'/src.relative_to(O/'gn11_static_duplicate_20261019'))
 code=['apply_direct_plan.py','formal_support.py','rigid_spatial_topology.py','test_rigid_spatial_topology.py','build_direct_implementation.py','direct_cpu_gate.py','run_direct_30k30k.py','evaluate_direct.py','start_direct_plan.py','audit_all_vehicle_coverage.py','infer_all_vehicles_native.py','run_all_vehicle_preparation.py','start_all_vehicle_preparation.py','run_all_vehicle30k_queue.py','evaluate_all_vehicle_stage1.py','start_all_vehicle_queue.py','prepare_fullframe_report.py','fullframe_player_template.html','build_direct_report.py','build_direct_player.py','validate_direct_artifacts.py','publish_direct_plan.py',
 'review_all_vehicle_association.py','infer_vehicle_geometric_prompts.py','start_vehicle_geometric_prompts.py','infer_vehicle_instance_prompts.py','start_vehicle_instance_prompts.py','audit_vehicle_instance_masks.py','audit_full_segment_vehicle_measurements.py','infer_vehicle_camera_prompts.py','start_vehicle_camera_prompts.py','assemble_camera_vehicle_candidates.py','audit_zero_seed_sources.py',
 'prepare_lidar_camera_intersection.py','finalize_intersection_cpu_metadata.py','render_intersection_input_review.py','close_intersection_input_review.py','build_intersection_input_page.py','restart_intersection_queue.py','run_native_timing_safe.py','check_native_cadence_wrapper.py','audit_gn11_recovery.py','gn11_train_timing_safe.py','check_cuda_event_timing.py','resume_gn11_worker.py','direct_after_recovery.py','run_A_recovery_pipeline.py','start_A_recovery.py','collect_recovery_progress.py','audit_remaining_disk.py','audit_25k_checkpoint.py','audit_direct_initial_checkpoint.py','check_final_video_encoding.py']
 code+=['check_player_identity_controls.js']
 code+=['audit_road_observability.py','audit_road_silhouette_and_source.py','build_road_observability_report.py','validate_road_observability.py']
 code+=['audit_parallel_5k_durability.py','audit_parallel_checkpoint_node.py']
 code+=['check_report_phase_contract.py']
 code+=['package_final_full49.py']
 code+=['audit_gn11_static_duplicate.py','audit_gn11_static_motion_features.py','build_gn11_static_duplicate_report.py','validate_gn11_static_duplicate.py']
 for n in code:cp(W/n,S/n)
 for n in ['export_GN11_stage1_existing.py','probe_parallel_capacity_and_yield.py','shared_gpu_policy.py','run_A_native_shared.py','run_B_native_shared.py','run_parallel_campaign.py','start_parallel_campaign.py']:cp(W/n,S/n)
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
 inst=json.loads((W/'live_direct_instance_status.json').read_text()) if (W/'live_direct_instance_status.json').exists() else {}
 paragraph=f'''用户当前授权：A完成两阶段身份一致的五身份fresh GN11 30K＋新实例30K，两个30K之间不做消融、不在1K等质量批准。B已按最新用户要求改为LiDAR与FRONT相机共同命中的20身份，只跑原生第一阶段30K；不再追补13条相机独有远车。两项实验均保留完整49帧RGB，最终视频、显示、导出全部完整画面，无ROI或局部放大。B仅验收“删除全部纳入车辆”，排除身份可留作静态RGB上下文，不能称全景无车。failure_ledger_refs=[V82-F06]；无subagent、新聊天、R16/R17，原45K暂停，历史资产保留，human_verdict=null。

A原GN11在25847前后向后、优化器更新前读取未完成CUDA事件的耗时而中断，已更新25846；最近完整状态20K，丢失5846步。用户明确授权这次必要重放，并要求后续每5K保存。计时读前同步结束事件的修复通过20次真实GPU检查；新controller52748从20K完整模型/Adam/RNG/采样器/SGCM状态恢复，实际GPU张量与状态逐项相等，前100次重放的帧、时间偏移、学习率一致，未声称CUDA逐位确定性。当前恢复后{step}/30000；只改保存节奏和计时同步，不改目标或优化器。新run后缀GN11_resume20k。原失败5153/6099/39526不可重复启动。

恢复GN11到逻辑30K后自动绑定实际最终背景、环境和点数，fresh五实例30K、完整49帧动态/静态/独立资产/逐ID与全部删除。第二阶段孤立raw-body100＋visible-ID100，SH2000、49相机无放回采样、原生静态与保身份刚体空间拓扑已登记。GT刚体无PVG时间split，第二阶段不新增SGCM；原生append重置屏幕半径次序保留，不冒称持续screen大点保证。新实例保存1/5K/10K/15K/20K/25K/30K；B完整模型/Adam/RNG/采样状态每5K保存。原生test_iterations仍10K/20K/30K，但保留的每checkpoint评估开关会在新增5K节点附带49视图零更新评估，不是中间消融。历史未保存的5K/15K不补造。配置、实际args及剩余磁盘节点计数已检查。

B的20身份为a1/a5/a6/a9/a10/a12/a16/a18/a19/a20/a24/a25/a26/a28/a30/a36/a37/a39/a41/a42。49帧、20个真实非零LiDAR先验、身份清单和980项逐身份原生掩码已核对；实际CPU相机/.1米尺度通过，全部49完整画面已复核并另看raw128。部分真实相机矩形提示原生分割已披露，未填矩形为body，未按GT框逐点裁造，未用额外观察帧。原2×2像素/.5m筛选、聚类和整帧几何QA保留。a41只有19个种子点，输入准入不代表像素或资产质量通过。intersection_input_passed已保存，B controller52417仍0更新；A整条训练及评估完成后再释放all_vehicle_ready，避免阶段交接抢GPU。尚未证明两套完整训练可同时容纳，当前独占排队。

取消的中间消融/贡献追踪不恢复；five_raw_body短测域误标及学习率10倍错误初判撤回继续保留。已有修复R15 4K质量仍未通过，a6白带和a37静态车形不因输入门槛或框外率下降而视为解决。未观测道路不加生成补全。

[当前实验、计时修复与恢复证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html) · [完整49帧20身份输入复核](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/intersection_inputs.html)。全部A+B训练/评估、会启动工作的controller、下载、报告与Git交付完成才按授权关机；automation-2每30分钟跟进，无变化保持安静。
'''
 paragraph=paragraph.replace('取消的中间消融/贡献追踪不恢复；','新增25K完整检查点已实际落盘，静态/动态/环境/训练状态四组件与完成事件相符，Adam张量有限，RNG/采样器/SGCM/5K配置及严格续训合同CPU只读检查通过，49视图评估完成。此处未对25K做GPU恢复或增加训练更新。\n\n取消的中间消融/贡献追踪不恢复；')
 if inst:
  paragraph+=f'''\nA第一阶段GN11实际complete30K；同预算fresh实例30K已自动启动，当前{inst.get('step')}/30000（{inst.get('state')}）。首步完整checkpoint2028722412B已保存，严格续训字段/策略、91项模型Adam tensor有限性、RNG/49相机采样状态已CPU只读检查；未对首步做GPU恢复或额外训练。实际静态初始化1988336点全保留，按框删除0点；新30K背景/环境、同五身份、新鲜测量种子绑定已检查。\n\nB运行器补入首更新前六个5K节点的磁盘预估，保存时再按剩余节点检查；不是等到第一5K才发现不能保存。实际wrapper CPU fixture通过，未占训练GPU。最终播放器补全A全部删除入口、A/B不同集合标识，以及原生模型不具备物理阴影开关的提示。媒体编码先写临时文件，核对49完整帧/10fps/all-I后发布；CPU合成验证通过，不冒充真实模型质量。\n'''
 if (W/'parallel_cutover_receipt.json').exists():
  paragraph=re.sub(r'intersection_input_passed已保存，B controller52417.*?当前独占排队。','intersection_input_passed已通过；用户最新要求现显存足够即并行。实测与A并存的额外13GiB分配成功，A/B分配器预算16/13GiB加2GiB余量，总31GiB小于Torch实际31.475GiB。未来峰值仍需监测。原等待B52417在0更新时取消；新共享GPU守护controller79622持全局锁，A79624、B79625均已真实训练。A在1999步完整保存后从2000继续，模型/Adam/RNG/采样器实际GPU逐项相等，无丢步或重放。',paragraph,flags=re.S)
  paragraph+='\n已将新五身份GN11第一阶段30K的真实49帧完整原图/合成/动态/静态/α及删除动态集合视频加入报告，7路480×320、10fps、全I帧逐一验证；主入口嵌入同步播放器，默认raw128。实际全49静态复核仍有车形与亮白/彩色条带，不能认定干净背景，原因仍需区分监督盲区与剩余实现问题，当前A第二阶段和B继续授权训练。\n'
 if (O/'road_observability_20261019/index.html').exists():
  paragraph+='\n新增只读道路可观测性核查：49帧跨度4.800秒，自车仅移动2.04m，34帧距初始位置不足0.1m；左侧对向车辆移动约40—45m。raw128车身轮廓背后的有效平面路面查询，a26始终无曝光机会0.07%、a6为100%、a37为78.54%；是几何近似机会而非干净道路GT。原199帧只读3D框检查发现更多窗外候选，未加入训练；已有曝光部分仍有误差，不能把全部问题等同于身份错分或纯补全。A/B输入和预算未改。详见[左右车道观测与原始序列核查](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/road_observability_20261019/index.html)。\n'
 if (O/'parallel_5k_durability_receipt.json').exists():
  paragraph+='\nA/B首个5K完整检查点均已实际CPU只读核验：A单文件3162838956B，91项模型/Adam张量有限、严格续训合同及5K采样/RNG通过；B四组件与完成事件相符，Adam/RNG/SGCM/5K采样状态通过，49视图零更新评估已完成。未进行5K GPU恢复或增加更新。存储预测按真实未保存5K节点、A整体临时文件及实际导出节奏修正，不再重复计已完成A第一阶段、缺失历史step1和B已保存5K，未来仍逐次核对并保留全部资产。\n'
 if (O/'gn11_static_duplicate_20261019/receipt.json').exists():
  paragraph+='\nraw129静态车形进一步核查：两车真实动态mask已接入，未发现静态/动态视频混用；但静态body直接RGB输出梯度为0，动态α不对静态排车，原生静态仍使用时间位移和时间显隐。实际1195277个静态点49时间中心跨度>0.1m，而衰减后正则速度特征<1e-6。相关getter与归档上游AST一致，是继承的表示/约束漏洞；道路缺观测不能替代分离正确性证明，不能再称主体已彻底分离。全局计数不等于标注车形的Gaussian贡献归因；B也共享此路径，当前原配置和预算未改，无新消融。详见[实际参数、梯度和完整画面](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/gn11_static_duplicate_20261019/index.html)。\n'
 phase=json.loads((O/'direct_plan.json').read_text(encoding='utf-8')).get('report',{}).get('phase',{})
 if phase.get('complete'):
  paragraph=paragraph.replace('同预算fresh实例30K已自动启动，当前30000/30000（complete）','同预算fresh实例30K已完成30000/30000')
  paragraph=paragraph.replace('当前A第二阶段和B继续授权训练。','A第二阶段和B均已完成训练与最终49帧删除评估。')
  paragraph+='\n两项正式训练和最终49帧评估均有真实完成状态、逐帧凭据及已发布视频；质量是否通过仍以实际完整画面复核为准，不由训练完成状态代替。\n'
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
 if inst:evidence+=f'\n第一阶段已实际完成30K，fresh实例训练当前{inst.get("step")}/30000；首步完整检查点已保存并CPU核对，背景按框删除0点。同预算直接训练，无中间消融；质量仍待完整49帧评估。\n'
 if (W/'parallel_cutover_receipt.json').exists():evidence+='\n用户要求现显存够即并行：13GiB额外分配实测成功，A利用完整1999步抢占保存从2000精确GPU恢复，B原0更新等待队列改同预算并行，两者实际运行。新GN11第一阶段30K已交付7路真实49帧全画面同步视频，全49静态复核仍见车形和亮色条带，特别raw128；未将其判为干净背景，也不据此定性原算法固有限制。\n'
 if heading not in text:text+='\n\n'+heading+'\n\n'+evidence
 else:text=re.sub(re.escape(heading)+r'\n.*?(?=\n## |\Z)',lambda m:heading+'\n\n'+evidence,text,flags=re.S)
 p.write_text(text,encoding='utf-8')
 paths.append('docs/research_failures/entries/V82-F06.md')
 (W/'direct_publish_paths.json').write_text(json.dumps(sorted(set(paths)),indent=2),encoding='utf-8')
 print(json.dumps(dict(scoped_files=len(set(paths)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
if __name__=='__main__':main()
