# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r6`已完成有界数据准入与微调对照。用户授权GPU/CPU自主迭代及技术AI2分后训练，历史人工分保持null。52case/10个receiver scene，48train/4validation；29背景、23单保护车、密集0。新的4例精确mask仅C012通过，原阈值未放宽。

原DriveEditor结构不变，160步更新既有80个空间attention张量。latent loss0.544→0.165，但4个共享一个场景的合成留出例保护车MAE均变差（均值+35.7%），视频变模糊；本轮权重不升为默认，原权重保留。A041空首窗不计删除成功，按输入非空规则补跑有效10–19帧，两权重相同条件，保留旧空窗。8个有效任务+1个空窗全部在本地r6报告，人工verdict留空；215视频/2750帧解码通过。

下一步优先原分辨率训练/推理一致性控制，同数据同更新范围，尚未执行；之后才判断保护区监督是否需要变化。密集类及来源场景多样性仍欠缺，不把这次单场景负结果否定整个路线。GPU当前允许；无本轮关机或自动化授权，作业结束不关机。

见[报告与组件图](v77/TARGET_PROTECTED_FINETUNE_R6.md)、[证据](autoresearch/worldsim_v77/target_protected_20260929/r6/pilot_summary.json)。run `/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r6`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
