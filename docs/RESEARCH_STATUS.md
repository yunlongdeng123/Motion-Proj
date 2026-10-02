# 当前研究状态

2026-10-02，wm-3090-1001，v77。用户授权Temporal reveal＋projected actor-state先验证简单条件价值，surfel后置；允许清可恢复的不重要旧物，目前约50GB空余，无删除。当前实际任务r23为CPU新长窗来源准备：冻结80scene各一条、61例通过长窗输入门槛，按需补603文件；尚非50个已质检训练case。输入源ID别名缺上下文已按八个真实keyframe token修复，不重新抽样。

r21完成8例完整SAM入口/16个同条件原模型-r7对照，A022固定f5旧两臂补车、新两臂均删车，收益属于输入修复。A013/A041/A048等仍失败，A042首帧身份不确定。30帧2.9秒单次前后向和3步推理通过、训练0步，峰值14.19/21.45GiB。旧目录去重为46train/10合成DEV-val，原目录保留。

r22完成遮后4轨迹SAM与3例合法状态条件，L001/L009 O精度高但被遮B覆盖18%/37%，N很稀疏；独立QA uncertain/pass-visible-only/uncertain。局部地面片采样没有各例一致收益，保持默认点条件。当前优先补真正显露过程；新条件尚未接模型训练，A/B/C同预算训练与surfel都未启动。原模型、r7/r14/r18、失败资产全部保留，final未用，无新自动化，当前未满足关机收口条件。

交付与证据：[r21/r22报告](v77/TARGET_PROTECTED_INPUT_STATE_R21_R22.md)、[r23预案](autoresearch/worldsim_v77/target_protected_20260929/r23/plan.md)，failure_ledger_refs [V77-F02]。人工分数空。
