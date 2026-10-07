# 当前研究状态

更新：2026-10-07。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

唯一run `WS-V77-TARGET-PROTECTED-20260929/r49` CPU准备完成，停在等待用户开启GPU。训练0步、新模型采样0窗，无后台等卡控制器或定时任务；数据盘约余92GB，无需清理。

r49仅给原RGB cross-attention加实例/局部位置软偏置，389856参数/state_dict兼容，关闭绑定与全U严格重现r47路径。复用4train+3真实DEV+2合成DEV，原RGB、H、α、参考和几何不变，Y不进条件。9例90帧和受控激活验证通过；CPU审核HTML已准备。

对应覆盖有限：A034/A061 f05的主B绑定粗query为3/30与8/72；低尺度/混合格仍U，N没有纯背景patch。M013无可绑定参考patch，不能称4例都有有效路由监督；不用扩框/降门槛伪造覆盖。这是输入与接口准备，尚未证明薄膜减少或身份绑定收益。

GPU阶段先冻结r47采样9窗并直接review；若未达标且输入/工程有效，最多一次64步与11窗。同数据/预算r48_64保留控制；A034/A061需要保护结构与清残影同时收益，A022和M003/M006检查回退。零训练与64步间有review节点，不自动增加训练。

r48_128明显回退、r48_64大多持平的旧结论保留；默认仍r46官方原权重+r21完整SAM，不推广新权重。人工verdict只由用户填写。完整范围、组件图、CPU证据与GPU入口见[r49报告](v77/TARGET_PROTECTED_SPATIAL_BINDING_R49.md)，同一失败卡[V77-F02](research_failures/entries/V77-F02.md)。
