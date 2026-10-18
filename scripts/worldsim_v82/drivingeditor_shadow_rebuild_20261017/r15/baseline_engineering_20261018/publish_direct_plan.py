"""Scoped update for direct30K+30K and full-vehicle stage1, no stale queues."""
from pathlib import Path
import json,os,shutil,re,zipfile,hashlib
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo';O=W.parents[1]/'outputs/baseline_engineering_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018')
S=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/baseline_engineering_20261018')
paths=[]
def long(p):return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p
def cp(src,rel):
 dest=long(P/rel);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);paths.append(rel.as_posix());return dest
def main():
 for n in ['index.html','engineering_history_before_direct_plan.html','direct_architecture.svg','direct_plan.json','direct_CPU_gate.json','direct_queue_status.json','direct_plan_cancellation_receipt.json','all_actor_queue_status.json','all_vehicle_coverage.json','all_vehicle_raw128.png','direct_validation.json']:
  p=cp(O/n,D/n)
  if p.suffix=='.html':p.write_text(p.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
 for n in ['all_vehicle_inference_receipt.json','all_vehicle_fullres_coverage.json','all_vehicle_train_queue_status.json','direct_launch.json','all_vehicle30k_launch.json']:
  if (W/n).exists():cp(W/n,D/n)
 code=['apply_direct_plan.py','formal_support.py','rigid_spatial_topology.py','test_rigid_spatial_topology.py','build_direct_implementation.py','direct_cpu_gate.py','run_direct_30k30k.py','evaluate_direct.py','start_direct_plan.py','audit_all_vehicle_coverage.py','infer_all_vehicles_native.py','run_all_vehicle_preparation.py','start_all_vehicle_preparation.py','run_all_vehicle30k_queue.py','evaluate_all_vehicle_stage1.py','start_all_vehicle_queue.py','prepare_fullframe_report.py','fullframe_player_template.html','build_direct_report.py','build_direct_player.py','validate_direct_artifacts.py','publish_direct_plan.py']
 for n in code:cp(W/n,S/n)
 archive=W/'direct_implementation.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in (W/'direct_implementation').glob('*.py'):z.write(p,p.name)
  for n in ['LICENSE','LICENSE_gaussian_splatting.md']:z.write(W/'source'/n,n)
 cp(archive,S/archive.name);cp(W/'direct_source_manifest.json',S/'direct_source_manifest.json')
 for n in ['index.html','sync_player.html']:
  p=cp(O.parent/'identity_comparison_20261018'/n,D.parent/'identity_20261018'/n)
  text=p.read_text(encoding='utf-8').replace('../r15_repair_4k_20261018/','../repair_4k_20261018/')
  p.write_text(text,encoding='utf-8')
 cp(W.parent/'identity_completion/sync_player_template.html',S.parent/'identity_completion_20261018/sync_player_template.html')
 live=json.loads((W/'live_GN11_status.json').read_text());step=live['iteration']
 paragraph=f'''用户最新要求：直接完成两阶段身份一致的五身份 GN11 30K＋新实例30K，取消两阶段之间的消融；额外补齐所有可见车辆（含停驶车辆）的掩码、真实初始化和统一身份表后，自动跑第一阶段30K。完整RGB保留，车辆由动态分支重建。本轮全车辆实验不跑第二阶段；最终同时检查逐ID删除和删除全部车辆后的完整静态街景。所有视频及显示使用完整画面，无ROI裁剪。failure_ledger_refs=[V82-F06]，不增加R16/R17，原45K暂停，历史资产保留。

五身份GN11实际继续训练，最近快照{step}/30000。等待中的中间消融controller18165及贡献追踪27418已核对无子进程后取消，未消费新增训练步数；旧4×300保留，原five_raw_body误标更正继续有效。direct controller39526已登记：等待当前GN11真正30000后绑定实际背景/环境及静态点数，直接fresh五实例30000，然后实际完整49帧分支、独立资产、逐ID与全部删除。没有1K质量停等或新的中间对照。第二阶段恢复孤立raw-body alpha100，同时保留visible-ID100，按2000步SH/49相机无放回采样，并接入原生静态及保身份刚体空间clone/split/prune。空间参考9项和epoch sampler完整恢复/未来150次序列通过CPU；GPU训练仍待执行。刚体不采用PVG时间split，SGCM沿用第一阶段结果；原生append重置屏幕半径历史的次序被忠实保留，不冒称有效的持续屏幕大点约束。

全车辆输入CPU清查49帧共45条车辆轨迹，其中20条在本相机有投影LiDAR，32条按中位速度<0.5m/s计停驶，这些统计不代表45辆均可见。现有R15的43身份不能等于完整覆盖。已用原始1920×1280完整49帧运行原生SAM3 vehicle预测，推理峰值11.2704GiB，在14GiB自身分配器上限与核实主进程下并行完成；未改旧mask。低分辨率旧缓存和新全分辨率提议仍有远处车辆关联/覆盖缺口，不能把未知当背景。all-vehicle30K controller41828已登记，等待all_vehicle_ready.json的实际覆盖/初始化/同一身份集验收，然后自动30K和完整49帧删除全部车辆评估。输入准备及峰值准入通过才并行，未证明显存足够时排队；未观测遮挡区域没有新补全算法，不声称自动恢复。

[双实验组件、当前状态和完整画面说明](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html)。human_verdict=null。全部两项实验、评估、报告及Git交付完成后才按授权关机，automation-2每30分钟跟进。
'''
 p=P/'docs/RESEARCH_STATUS.md';text=p.read_text(encoding='utf-8')
 end=text.index('五身份分支同步视频已交付：')
 text='# 当前研究状态\n\n更新：2026-10-18。当前 task=WS-V82-R15-BASELINE-ENGINEERING-20261018；无subagent，两个正式实验直接执行。\n\n'+paragraph+'\n'+text[end:]
 p.write_text(text,encoding='utf-8');paths.append('docs/RESEARCH_STATUS.md')
 p=P/'docs/EXPERIMENTS.md';text=p.read_text(encoding='utf-8')
 text=re.sub(r'^\| WS-V82-R15-BASELINE-ENGINEERING-20261018 \|.*$', '| WS-V82-R15-BASELINE-ENGINEERING-20261018 | 五身份直接GN11 30K＋实例30K；另补全可见车辆输入后第一阶段30K；完整画面同步删除验收；中间消融取消 | [当前实验安排](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html) |',text,flags=re.M)
 p.write_text(text,encoding='utf-8');paths.append('docs/EXPERIMENTS.md')
 p=P/'docs/research_failures/entries/V82-F06.md';text=p.read_text(encoding='utf-8');heading='## 2026-10-18：用户收敛为直接30K＋30K及全车辆单阶段'
 if heading not in text:p.write_text(text+'\n\n'+heading+'\n\n'+paragraph.replace('(autoresearch/','(../../autoresearch/'),encoding='utf-8')
 paths.append('docs/research_failures/entries/V82-F06.md')
 (W/'direct_publish_paths.json').write_text(json.dumps(sorted(set(paths)),indent=2),encoding='utf-8')
 print(json.dumps(dict(scoped_files=len(set(paths)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
if __name__=='__main__':main()
