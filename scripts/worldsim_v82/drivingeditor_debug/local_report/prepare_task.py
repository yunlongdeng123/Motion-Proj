import json,tomllib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
saved=tomllib.loads(Path('C:/Users/dengyunlong/.codex/automations/drivingeditor-50k/automation.toml').read_text(encoding='utf-8-sig'))
prompt='''用户最新明确：现在只排查vGPU版DrivingEditor的bug，直到算法与效果正常，并尽量完整复现2026正式论文。暂停其它官网场景、oncoming/KITTI/镜像检索训练队列，旧50k完成后转队列的全部历史优先级已被取代。用户允许受控实验与其它debug，不需要常规授权再次确认。每小时检查并持续推进本debug；正常变化安静，仅确定新结论、实验完成、失败或需要用户处理时汇报。不要以GPU利用率为由启动不相关任务。保留所有已有模型、数据与失败结果，不关闭实例。

只连接SSH wm-vgpu-1011。BASE=/root/autodl-tmp/drivingeditor_repro；ENV=/root/autodl-tmp/envs/drivingeditor/bin/python；GPU实际RTX4080SUPER vGPU32760MiB，实时核对。唯一debug工作目录BASE/campaign/vgpu_bugfix_20261012（子实验独立输出）；local work/vgpu_bugfix_20261012，用户可见报告outputs/drivingeditor_vgpu_debug/index.html。每次先读task_manifest.json、各实验status/exit/PID/日志、最近handoff、gpu_slot.lock和nvidia-smi；避免重复启动、覆盖checkpoint或热改运行源码。任一时刻仅一个GPU任务。任何新subagent必须显式model=gpt-6.1-sol, reasoning_effort=max, fork_turns=none或正整数；不要复活Astra。

对照定义：2024官方代码在3090运行的原始30k eval有32.7914dB，是static opacity reset前的清晰历史输出；用户指定它作为旧基线。对应old actual saved30k是reset后的另一快照，重渲染约9.98dB，不得冒充用户基线。正确主对比outputs/drivingeditor_30k_edit_comparison/index.html，旧saved-vs-saved保留diagnostic/，均已交付。原始旧run BASE/runs/0017085_demo_000_020_dynamic_30000，21FRONT raw0..20，旧repoDrivingEditor-run。当前论文run parent BASE/runs/scene0017_000_048_paper_50000_seed0 seed0从零50k；迁移child BASE/runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline 从同horizon10k严格恢复。child于35900被用户主动SIGINT，exit130，不是新OOM/完成；仅parent10k和child20/30k完整。正式frozen源码BASE/campaign/resume_vgpu_baseline_20261011/frozen_repository，与child code_snapshot/config/sourcepins绑定。新repoDrivingEditor-paper不能假定等同frozen。旧3090不用访问。

初始证据：新30k四件checkpoint和相机/配置已核验；196PNG重渲染与保存时评估逐像素一致，故不支持简单导出错文件假说，但仍须审计训练与导出合成语义。20k composite35.2485dB，30k10.4835dB；30k保存的2234298静态点全部opacity<=.01，晚期static opacity reset每3000仍在15k densify结束后执行；SGCM在14950停止。20k的static道路已经大面积彩噪、dynamic补上道路，说明分支分离退化早于30k瞬时reset，不能只用global PSNR恢复宣称正常。30k dynamic通道支撑共同21帧99.88058%全图。训练期render_d含共享env残差，eval导出actor通道不含env；论文明确动态分支共享env，因此不能把这项语义差直接称bug。

排查依用户四类：A错误env版本或高斯checkpoint混用；B导出渲染/训练参数或合成不一致；C模型配对正确但env/static/dynamic优化退化；D道路确未观测的方法能力边界。第一阶段从10/20/30k文件完整性、env/opacity/SH/尺度统计、真实固定相机及分支/环境诊断检查A/B；可在明确标注的隔离diagnostic中跨20/30k单组件对照，不能把混配诊断当正常训练成果。随后基于原始完整训练loss/全部optimizer/RNG/sampler严格恢复的短continuation测30k重置后恢复曲线，再逐项明确更改的受控消融定位SGCM/NPS/opacity/depth/learning-rate原因。原版自身重复作为GPU浮点波动参考；不换相机/数据/seed/精度来挑好结果。修复需要证据、独立源码版本、同条件before/after与复验。

论文在C:/Users/dengyunlong/Downloads/DrivingEditor.pdf（SHA4158fa7a6057d7173fd878973f101364ea0821ba3c433e917c1156c279f3e98e），提取text work/paper_audit/DrivingEditor.txt，原页work/paper_audit/pages。既有审计outputs/drivingeditor_paper_alignment及outputs/drivingeditor_campaign/edit30k_compare_20261012/evidence/{branch_constraint_source_cpu.json,paper_loss_domain_review.json,loss_review_handoff.md}。Eq10未列单独staticRGB，不可直接指控遗漏mandatory论文项；Eq13的mask/求和域不明确，new前景mask是显式复现选择，old100x全图MSE并非最终论文公式；both dynamic RGB mask。作者未给细节、论文自身歧义、我们的确定实现bug必须分别记录。不能为了好看盲目套回旧loss后声称完全论文复现。独立CPU审计分工paper_loss_bug_audit_solmax与sgcm_data_bug_audit_solmax，证据在debug子目录；检查其实际handoff状态，避免重复。

遵守无精度损失工程原则：FP32、原图/帧/点数支持与seed保持；禁止AMP、降低分辨率/点数上限/帧数避开问题，禁止分支分阶段反传/CPU反向暂存。允许用户本次授权下明确的算法纠错与受控诊断，但必须区分paper修复、假设变体和工程优化；每个实验冻结完整配置/源码/数据哈希，必要恢复只能同run完整四组件加RNG/sampler/全部optimizer，任何horizon变化必须是从零新实验而不是假装原轨迹。compact和renderer cleanup尚未部署，不自动采用。采样/训练步数用于短诊断时明确上限，不假称50k完成。

验收应覆盖：完整固定视角重建、背景/道路static清晰度、动态branch对目标局部隔离、移除/平移不带走大块已观测背景、组件正确和训练/导出一致、数值有限、不同seed或场景的必要复验，以及论文所需SGCM/NPS/目标函数/50k调度等逐条对齐。没有无遮挡GT不声称验证FID或去除区域真实性。未实现shadow/Mdetector/作者未公开细节，必须补齐或如实保留缺口；不得仅达到旧PSNR便宣称完整复现论文算法/四场景效果。每轮更新深色HTMLdebug报告，含简单输入→模块→输出架构图、受控实验矩阵、真实视频/图像、指标/分支统计、证据和剩余问题。只在目标真实达到且必要验证完成后交付结论并暂停本heartbeat；研究假说未解决时继续排查并清楚记录。

oncoming旁支已因用户此最新目标暂停。它曾完成sky49/DEVA49，仍有v1 raw29地面LiDAR污染几何失败及v2候选，均不能当training ready；请读其pause/handoff但不继续改数据或开训。'''
payload={'mode':'update','id':saved['id'],'kind':'heartbeat','name':'DrivingEditor vGPU 论文版排错','status':'ACTIVE','rrule':saved['rrule'],'targetThreadId':saved['target_thread_id'],'prompt':prompt}
if 'notification_policy' in saved:payload['notificationPolicy']=saved['notification_policy']
(HERE/'automation_vgpu_debug.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest={'schema_version':1,'objective':'Only diagnose and fix vGPU final-paper DrivingEditor until algorithm and effects verified; other scenes paused',
 'status':'diagnosing','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'host':'wm-vgpu-1011','frozen_repository':'/root/autodl-tmp/drivingeditor_repro/campaign/resume_vgpu_baseline_20261011/frozen_repository',
 'reference_run':'/root/autodl-tmp/drivingeditor_repro/runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline',
 'gpu_owner':'root_pending_engineering_diagnostic','other_scene_queue':'paused_by_user',
 'hypotheses':['component mismatch','train-export mismatch','optimization or algorithm implementation','unobserved region'],
 'first_experiment':'20k/30k checkpoint/scene/env/opacity statistics and fixed-camera component diagnostics',
 'loss_audit':'CPU paper_loss_bug_audit_solmax','sgcm_data_audit':'CPU sgcm_data_bug_audit_solmax',
 'acceptance_not_yet_met':True,'paper_complete_reproduction_claimed':False}
(HERE/'task_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Prepared focused debug manifest and existing hourly heartbeat update')
