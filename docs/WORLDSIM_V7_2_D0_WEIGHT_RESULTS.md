# WorldSim V7.2 D0 W0--W4 回波权重结果

日期：2026-09-07  
数据角色：`legacy_diagnostic`  
几何：冻结 G0 raw fusion 与 G2 legacy M8  
读出：同一 categorical Gaussian reader  
结论状态：D0 权重机制筛选完成；D1 仍冻结

## 1. 冻结协议

两组实验使用相同 593 train / 66 holdout Actors。G0/G2 的表面中心与尺度在训练中冻结；W0--W2 只读取 build 证据，W3/W4 使用相同 18 维 build-only primitive 特征和相同 primitive/context MLP 主干。W3 参数量为 13,697，W4 为 13,827，相差 `0.95%`。

W3 与 `W4-response` 只使用相同 categorical response NLL + depth L1；`W4-aux` 是单独配方，在同一 response loss 外增加 F/O/U 软标签交叉熵。训练 12 epochs、单种子；训练期间不读 holdout。holdout 的权重在挂载 target 前一次性生成，之后才进入同一 evaluator。`source_test`、`external_test` 均未读取。

| Arm | 权重来源 | 是否学习 | 额外监督 |
|---|---|---:|---|
| W0 | 单位权重 | 否 | 无 |
| W1 | build-only occupied support rate | 否 | 无 |
| W2 | W1 加局部密度与采样机会归一化 | 否 | 无 |
| W3 | 单标量 MLP | 是 | response only |
| W4-response | F/O/U MLP 的 O 分量 | 是 | response only |
| W4-aux | F/O/U MLP 的 O 分量 | 是 | response + F/O/U soft labels |

这里的 W0 categorical 指标与 `WORLDSIM_V7_2_D0_GPU_RESULTS.md` 中 literal point-surface 指标使用不同读出，不能跨表直接比较绝对 early/hit；本表只比较同一 reader 内的权重差异。

## 2. 同算子结果

| 几何 | Arm | early↓ (%) | hit↑ (%) | 深度 MAE↓ (mm) | Δearly vs W0 (pp) | Δhit vs W0 (pp) | ΔMAE vs W0 (mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| G0 | W0 | 12.142 | 78.520 | 189.354 | 0.000 | 0.000 | 0.000 |
| G0 | W1 | 15.802 | 73.366 | 240.795 | +3.660 | -5.154 | +51.441 |
| G0 | W2 | 15.504 | 73.147 | 244.755 | +3.362 | -5.373 | +55.400 |
| G0 | W3 | 12.421 | 79.264 | 182.159 | +0.279 | +0.744 | -7.195 |
| G0 | W4-response | 12.654 | 79.266 | **181.839** | +0.512 | +0.746 | **-7.515** |
| G0 | W4-aux | **12.086** | 79.143 | 185.327 | **-0.056** | +0.623 | -4.027 |
| G2 | W0 | **17.110** | 61.623 | 299.253 | 0.000 | 0.000 | 0.000 |
| G2 | W1 | 17.711 | 64.222 | 282.778 | +0.602 | +2.599 | -16.476 |
| G2 | W2 | 17.158 | 64.056 | 283.343 | **+0.048** | +2.433 | -15.910 |
| G2 | W3 | 17.837 | **67.523** | 259.666 | +0.728 | **+5.900** | -39.588 |
| G2 | W4-response | 17.706 | 67.499 | **258.317** | +0.597 | +5.876 | **-40.936** |
| G2 | W4-aux | 18.050 | 66.070 | 270.016 | +0.940 | +4.447 | -29.238 |

## 3. 结论

1. **W3 与 W4-response 等价到当前分辨率。** G0 上 hit 只差 `0.002pp`，G2 上差 `0.024pp`；early 的差分别为 `0.233pp` 与 `-0.131pp`。同输入、同 response loss 下，三输出结构没有稳定优于单标量。
2. **F/O/U 辅助监督不跨几何稳定。** G0 上 W4-aux 相对 W0 为 early `-0.056pp`、hit `+0.623pp`，是小幅双赢；G2 上相对 W4-response 却为 early `+0.344pp`、hit `-1.428pp`。因此不能把 F/O/U 结构或辅助监督写成当前核心贡献。
3. **标量学习的稳定信号是 hit/深度改善，同时付出 early。** W3 在 G0/G2 分别提高 hit `0.744/5.900pp`、降低 MAE `7.20/39.59mm`，early 同时增加 `0.279/0.728pp`。该方向可作为后续观测约束研究的真实缺口，但不是已解决结果。
4. **简单支持率不是通用答案。** W1/W2 在 G0 明显退化；W2 在 G2 几乎不增加 early（`+0.048pp`）而提升 hit `2.433pp`，说明它是几何相关的强简单控制，后续方法必须报告跨几何结果。
5. **D0 的原“普遍三态表征缺陷”主张不成立。** 当前证据更具体：回波权重能改变 hit--early 权衡，单标量已复现主要学习收益；额外三态监督没有跨几何稳定增量。

## 4. 正式运行与资源

| 几何 | canonical run | wall | 峰值 GPU | 备注 |
|---|---|---:|---:|---|
| G0 | `20260906T185900Z__w0-w4-g0-s7205-r3` | 134.45 s | 1.60 GiB | 1/659 capped Actor 使用显式最近点映射，最大距离 0.392m；见 V71-F60 |
| G2 | `20260906T190200Z__w0-w4-g2-s7206-r1` | 132.09 s | 1.29 GiB | 无映射 fallback |

两次 run 均保存 resolved config、manifest、fingerprint、三份模型、训练曲线、66 Actor rows、逐 log rows、bootstrap summary 与 status。G0 r1/r2 在任何训练或 holdout 指标前失败并原样保留；canonical r3 为同一冻结协议的修复运行。

下一步不再扩展 W5/W6。需要先建立干净 dev/route-select 数据，并完成路线 B 的原生 capability，之后才能按 D1 规则选择对象补全或 full neural LiDAR 主任务。
