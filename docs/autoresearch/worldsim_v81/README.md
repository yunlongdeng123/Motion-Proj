# V8.1 CPU证据索引

WS-V81-CPU-01 / 20260913-cpu-r2；CPU_COMPLETE_WAIT_GPU。完整run路径见asset_index.json，仓库只存轻量证据。原始数据、完整权重、逐ROI参考及大表均在仓库外。

27日志、34场景、68窗口、6120 ROI；1527几何候选；49复核、36混杂排除；完整四格匹配组=0。三个官方模型参数合同和真实输入预处理通过，7项检查通过，模型前向=0。

status.json为当前状态，spot_checks.json为assistant逐图观察（不是human verdict），matched_cohorts.json为空数组。gpu_pilot_commands.json默认dry，仅开GPU后显式执行。自然H1仍需新增干净对照；V8.2证据未就绪。
