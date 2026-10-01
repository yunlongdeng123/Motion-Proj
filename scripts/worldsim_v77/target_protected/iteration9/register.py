from pathlib import Path
import shutil,json
P=Path('/root/autodl-tmp/motion_proj_v77');T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r9';E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r9';E.mkdir(parents=True,exist_ok=True)
B=O/'docs_before_r9';B.mkdir(exist_ok=True)
for n in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
 dst=B/Path(n).name
 if not dst.exists():shutil.copy2(P/n,dst)
for n in ['plan.md','temporal_audit.json','real_temporal_audit.json']:
 src=O/n
 if src.exists():shutil.copy2(src,E/n)
status='''# 当前研究状态

2026-10-01，v77；用户将下一阶段调整为badcase运动—遮挡—显露过程覆盖控制，默认主机wm-3090-1001。唯一新run `WS-V77-TARGET-PROTECTED-20260929/r9`；r7/r8已收口，不重复启动，最终集不使用，无电源/自动化操作。

实际r7的68训练窗口、r8的50训练例均0个B归一化遮挡中心跨度≥0.20的扫过样本；r8已有12个静止A＋运动ego且图像变化例，不把该类说成完全缺失。r9原70例真实几何盘点13个静止A＋运动ego、28个后车GT包络扫过代理；真实车身像素/其他帧真实纹理证据UNKNOWN。A022固定窗口ego静止但A移动，不能把全部失败归因缺时序。

新工厂保持真实Y、已有两种mesh、完整合成RGB先遮再编码和原物理门槛，固定世界静止/固定世界速度，不随B逐帧固定偏移。首轮46候选只有7扫过/5scene，隔离验证仅scene-0289；在新模型输出前冻结唯一16来源扩展，25锚点/三速度，缺额则停本轮训练。实际时序代理、全帧覆盖/空间与独立gpt-6-sol xhigh抽帧AI2共同准入，人工留空。

目标约50train/至少20scene，补8个扫过train/至少3scene、3个扫过val/至少2个隔离scene；同80张量/160步/原loss、官方106目标encoder、同推理条件，不加模块/步数/加权。新增过程隔离GT评测及冻结r8合成7/真实DEV8；先数据准入后训练。

前轮r8对r7洞MAE仅−1.05%、保护车+4.78%，未观察稳定真实迁移，不推广；原/r7/r8与全部反例保留。当前63GB可用，复用素材不批量下载。参见[r9预案与图](autoresearch/worldsim_v77/target_protected_20260929/r9/plan.md)。failure_ledger_refs: [V77-F02]；本轮结果待实验。
'''
(P/'docs/RESEARCH_STATUS.md').write_text(status)
idx=P/'docs/EXPERIMENTS.md';s=idx.read_text();tag='WS-V77-TARGET-PROTECTED-20260929/r9'
if tag not in s and tag.replace('/r9',' / r9') not in s:s=s.replace('|---|---|---|','|---|---|---|\n| '+tag.replace('/r9',' / r9')+' | running：盘点＋有界数据工厂；尚未训练 | [计划](autoresearch/worldsim_v77/target_protected_20260929/r9/plan.md) |');idx.write_text(s)
entry={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r9','host':'wm-3090-1001','stage':'temporal_audit_and_bounded_data_factory','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','architecture_unchanged':True,'power_authorization':None,'human_verdict':None,'data_GT':'real nuScenes Y; synthetic occluder appearance fully erased','query_inputs':'masked X / H only; target Y never supplied as condition','seed_training':6201,'seed_inference':42,'finetune_steps':160,'modules':'same80 spatial attention tensors','resource_budget':'one reuse search plus one16-source expansion; no threshold/seed/module sweep','train_before_quality':False}
(O/'run.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n');shutil.copy2(O/'run.json',E/'run.json')
print('registered',tag)
