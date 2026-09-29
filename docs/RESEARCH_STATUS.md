# 当前研究状态

更新：2026-09-29，v77。当前唯一任务`WS-V77-TARGET-PROTECTED-20260929/r1`，CPU准备约50例Target + Protected Actors数据。用户开启无卡实例，明确要求到批量推理前停止等GPU；本轮不启动SAM2、DriveEditor、Ω/GLB或正式训练，不恢复旧自动化和关机流程。

上一阶段70例CSV已原样归档，human_score 0/1/2/未填=26/26/15/3。未填A043/A052/A060和两列评分差异保留。V77-F02更新用户视频证据，原单帧助手记录不覆盖。见[用户评审](v77/DELETE_AUDIT_USER_REVIEW.md)。旧9例与已曝光audit不作新训练素材，25个final scene保持隔离。

新数据只来自nuScenes train：真实视频Y保持不变，在输入X增加一个真实donor track作为待删A，围绕纯背景、单个真实actor、密集车列三类。架构不变，protected metadata先用于数据检查/监督，不增加condition通道。制定逐帧硬门槛和subagent每例抽帧，再由用户在HTML逐帧全检；源素材通过不等于合成通过。见[质量协议与组件图](v77/TARGET_PROTECTED_QUALITY.md)。

远端run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r1`。当前正在CPU元数据筛选；无卡资源为0.5 CPU/2GiB内存，数据盘600GB、可用约150GB。按最多两个公共分片建立小规模试产素材池，选择依据是元数据可用性和质量，不是模型输出；不声称完整train分布代表性。后续检查RGB、几何和逐例素材，再停在GPU分割之前。尚无合格合成样本或微调结果。

failure_ledger_refs: [V77-F02]。failure_ledger_delta: updated V77-F02。
