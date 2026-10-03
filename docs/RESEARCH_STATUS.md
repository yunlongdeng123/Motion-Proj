# 当前研究状态

更新：2026-10-03。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

用户要求先修关键帧时间边界bug，再做小条件分支实验。当前唯一run `WS-V77-TARGET-PROTECTED-20260929/r46`。
统一直接使用官方NuScenes.get_boxes：关键帧取关联sample，非关键帧前后sample插值。旧r45保留160步权重、全部条件及32窗结果，因条件工程错误中止，不能作为完整条件的负结果。

24例240帧重建与CPU合同通过；11/12训练条件、10/12评测条件实际变化，因此同预算重训160步、主干冻结。复用11个不依赖条件的原模型窗口，其余25窗重跑，总比较36窗。没有扩数据/结构/训练步数，人工verdict空，全部是DEV。

修复后训练正在运行；尚无新收益结论。GPU结束就停下通知，不自动开新一轮。

细节与组件图见 [r46报告](v77/TARGET_PROTECTED_ONUQ_TIME_FIX_R46.md)，沿用 [V77-F02](research_failures/entries/V77-F02.md)。N仍稀疏，A022首帧洞内无背景返回仍应U。
