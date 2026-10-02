# 当前研究状态

2026-10-02，wm-3090-1001，v77。用户授权先修工程问题，再验证Temporal reveal＋projected actor-state条件，surfel后置；可清可恢复旧缓存，但当前约49GB可用，本轮无删除。

最新实际执行r28：复核r23原有30scene/73条因未核验动态实例拒绝的轨迹，保持原位置/速度/阈值，只补实例标签。CPU准备在跑；完整Y仅质量评价，target guard擦除后的X方法SAM分目录。无自动训练、无新自动化。输入准备的NumPy bool序列化错误已修复，原代码/日志保留，不是数据失败。

r26已完成8真实DEV×原模型/r7共16窗：直接保留可见邻车的洞裁减，A048/A034/A061固定f5出现明确车形/保护结构退化，不推广，继续r21 sam_full_v2入口。A022输入相同，20帧新旧逐像素一致；A013/A041/A042关键问题未解。全部48视频/480帧本地解码、链接与JS通过；人工分数留空，非全视频质量认证。

r23固定61scene/1830RGB、77 SAM轨迹仅1普通背景候选，reveal与密集类0，未形成50条训练配方。r25三例输入独立QA2不代表生成效果；r27车道连接/实际地面控制仍未产出额外合格样本，停止同组枚举。

r24空间Adapter30帧真实前后向通过。新发现最近邻使2869个含背景证据特征格仅91个保留，已改证据比例汇聚并保留未知占比；修复后网络输出零初始化等价、8连接梯度与10合同通过，优化0步。新A/B/C同预算训练与身份分支尚未启动，条件收益仍未证明，surfel不启动。

r21完整目标覆盖对A022的实际入口收益保留；原模型、r7/r14/r18、所有失败资产保留，final未用。当前未达到真实DELETE保护车＋无车背景的跨场景新方法收益，关机收口条件未满足。

交付：[r23–r27报告与组件图](v77/TARGET_PROTECTED_VISIBLE_INPUT_R23_R27.md)、[r26收口](autoresearch/worldsim_v77/target_protected_20260929/r26/closeout.json)、[r28登记](autoresearch/worldsim_v77/target_protected_20260929/r28/run.json)。本地outputs/v77-target-protected-r26/index.html；failure_ledger_refs [V77-F02]。
