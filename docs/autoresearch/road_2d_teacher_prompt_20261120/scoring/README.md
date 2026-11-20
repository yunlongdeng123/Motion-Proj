# 冻结60组提示词对照的独立评分

`score_next_round.py`只做整组提示词与seed分布的诊断数据，不选最佳seed，不调权，不提供新的无参考评分器。`metrics_pilot_frozen.py`是旧48组评分源码的原字节副本；只复用其颜色、Lab、HP、PSD函数，不调用旧48组gate，也不修改历史源码。

```text
python scoring/score_next_round.py --predictions GPU_formal/predictions.json --manifest prepared/plan.json --inference-entry code/infer_next_round.py --truth-config private_scoring/truth_config.json --out scoring_run_r1/scores.json
```

先调用计划对应的`infer_next_round.freeze_gate`，核查60个实际原始PNG、逆变换、W/K像素与12个历史重放。全部通过后才打开独立truth配置和GT图像。运行中、不完整、改写范围错误或未通过历史重放的结果不会读取GT。本程序不连接SSH、不加载GPU，线程为1。

结果schema为`road_2d_teacher_prompt_round_postfreeze_CPU_score.v1`。`jobs`保留`case_id,prompt_id,arm,seed,view,metric_role,native,same_view`；`prompt_id`为`legacy/material_match_v1`，arm固定`full1152_T0`，seed固定101/202/303/404/505。`gate.jobs=gate.completed_jobs=60`、`gate.historical_replays=12`。颜色、HP和PSD内部结构沿用旧评分；PSD额外明确共同前三带（0.025–0.0625、0.0625–0.125、0.125–0.2周期/原图像素），各自所有可用频带总分仅作描述。

诊断支持域沿用旧三个视图的共同有效窗口及固定参考规则，使12个历史重放的指标分母可核对。这不是把新完整1152画幅的所有像素都纳入评价。预测RGB不选择mask或频谱窗口。真实两帧只评已知可见道路，绝不生成遮挡道路误差；四个合成任务只有两张底图。输出不含自动排序、最佳候选或显著性结论。

CPU最小验收：`python scoring/test_score_next_round_cpu.py`。只使用人工构造并显式标记的CPU fixture；覆盖缺第60组、真实W外像素错误即使更新pin也阻止GT、全60实际像素gate及prompt接线、共同频带和冻结指标恒等。`--allow-cpu-mock`只允许此类标记测试，生产默认拒绝mock；测试不冒充真实GPU或真实数据画质证据。
