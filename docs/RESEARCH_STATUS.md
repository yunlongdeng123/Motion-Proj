# 当前研究状态

更新：2026-10-04。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

用户更新评分：当前真实DELETE明确的是r21完整SAM修复有收益，微调/r46 Adapter未建立明确收益。当前基线固定为r46官方原始权重+r21完整SAM；8个真实例7个1分，A022=2分，原表与5张内嵌GPT静态例已归档。

当前唯一run `WS-V77-TARGET-PROTECTED-20260929/r47`，研发扩大为DriveEditor条件接口重构。新增独立DeletionRequest、参考RGB交叉注意力、BEV CNN与投影2D控制，官方主干冻结、原9通道保留。不是把旧四通道Adapter换名，也不是仅外部后处理。

CPU阶段完成：12训练/4合成DEV/8真实DEV的240帧输入合同，真实bank46–50候选/例；191个额外RGB/LiDAR提取解码约41MB；389856参数支路CPU零初始化/梯度/路径检查通过。公开审核 `outputs/v77-priors-r47/index.html`，旧r46视频明确标注，人工新实验verdict空。约42GB数据盘空闲，不需要扩盘。

按用户要求停在GPU边界，无训练/新推理/自动GPU任务或关机操作。下一步先GPU验证额外参考的A剔除与实际官方模型前后向/显存，通过后固定320步只训新支路，并作12例5臂对照（12基线复用+48新窗）。输入消融训练有25%先验dropout，不逐例改查询SAM/seed，不根据真实结果选checkpoint。

当前训练参考仍单相机，GT框是包络，LiDAR稀疏返回不是稠密背景，完整自然语言/state encoder/surfel均未实现；CPU通过不代表真实补景收益。细节与组件图见 [r47报告](v77/TARGET_PROTECTED_MULTI_PRIOR_R47.md)，沿用 [V77-F02](research_failures/entries/V77-F02.md)。旧r46及A022覆盖诊断全部保留。
