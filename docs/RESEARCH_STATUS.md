# 当前研究状态

2026-09-29，v77；唯一任务`WS-V77-TARGET-PROTECTED-20260929/r1`。CPU素材准备收口，按用户要求**等待用户开启GPU后再运行SAM2**。当前没有模型推理/训练，没有旧定时任务或关机授权在执行。

70例用户CSV已原样归档并推送，human_score 0/1/2/未填=26/26/15/3；两列差异不补写。下一阶段聚焦well-observed Target + Protected Actors，真实nuScenes train视频作Y，只添加遮挡A构成X；原DriveEditor架构不变。旧DEV/audit不用于新训练，隔离final保持不动。

64个初选来源经时序、文件与几何检查，40段RGB可读且独立subagent全部抽帧。Receiver来源29通过，donor来源26通过，29段进入SAM2来源队列；拒绝与待定保留。2段RGB在索引指向的分片缺失并排除；3段标注间隔不合格。来源主要集中Boston少数日志，外观重复，不声称泛化。

**尚无合格合成对，约50例是下一步目标。** 本地`outputs/v77-target-protected/index.html`是来源预审，不是最终合成审核。GPU分割后完成放置/遮挡/边缘/模糊/模型条件检查，独立subagent每例抽帧，仅合格例进入用户逐帧全检；人工verdict仍为空。次要actor尚未自动准入。

远端run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r1`。CPU配额0.5核/2GiB，磁盘可用约149GiB，本批无需扩容。4项CPU合同测试、页面资源和JS语法核验已完成，GPU调用尚未验证。见[CPU结果与组件图](v77/TARGET_PROTECTED_CPU.md)、[质量协议](v77/TARGET_PROTECTED_QUALITY.md)、[接口合同](v77/TARGET_PROTECTED_CONDITION.md)。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
