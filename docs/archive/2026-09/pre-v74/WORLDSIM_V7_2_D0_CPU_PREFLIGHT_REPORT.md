> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# WorldSim V7.2 D0 CPU 前置报告

日期：2026-09-06

Canonical runs：

- G0：`run://worldsim_v72/WS-V72-D0-G0-RAW-FUSION-01/20260906T153519Z__g0-raw-fusion-cpu-r1`
- G1：`run://worldsim_v72/WS-V72-D0-G1-ACTOR-TSDF-01/20260906T160409Z__g1-actor-tsdf-cpu-r3`
- G3 adapter：`run://worldsim_v72/WS-V72-P0-G3-DATA-ADAPTER-01/20260906T154442Z__adapointr-legacy-export-r1`

本报告只使用 V7.1 已暴露的 66-Actor holdout，角色为 `legacy_diagnostic`。G0/G1 使用完全相同的 Actor identity、34 个日志级组、99,208 条正回波射线、密度预算和 evaluator。它用于检验任务、密度混杂与统一评测管线，不能用于 D1 选路或独立源域主张。

## G0/G1 同协议结果

CD 为 Actor mean 对称 CD-L1；early/hit/observable 使用 0.20 m beam tube 与 0.20 m depth tolerance，射线分母固定。所有 FPS 下采样只读取候选 surface，不读取 target。

| 预算 | G0/G1 实际点数 | G0 CD↓ | G1 CD↓ | G0/G1 F-score↑ | G0/G1 Early↓ | G0/G1 Hit↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 60.9 / 64.0 | 209.62 mm | 230.32 mm | 59.98% / 48.65% | 18.60% / 17.07% | 44.49% / 36.09% |
| 128 | 112.1 / 126.5 | 183.02 mm | 193.10 mm | 73.93% / 67.74% | 29.04% / 27.53% | 56.38% / 53.62% |
| 256 | 196.1 / 241.9 | 167.55 mm | 168.76 mm | 79.49% / 77.70% | 37.07% / 36.22% | 56.50% / 58.70% |
| 512 | 310.6 / 437.7 | 159.87 mm | 154.26 mm | 80.93% / 81.00% | 41.84% / 42.03% | 53.33% / 56.36% |
| native | 453.1 / 1043.9 | 156.13 mm | 141.24 mm | 81.22% / 82.94% | 44.27% / 47.65% | 51.04% / 51.12% |

G1 相对 G0 的同预算差值如下。native 的实际点数相差较大，只能作为各自上限诊断。

| 预算 | ΔCD↓ | ΔF-score↑ | ΔEarly↓ | ΔHit↑ | 判断 |
|---:|---:|---:|---:|---:|---|
| 64 | +20.70 mm | -11.33 pp | -1.53 pp | -8.39 pp | G0 明显更强，G1 仅 early 较低 |
| 128 | +10.08 mm | -6.18 pp | -1.50 pp | -2.76 pp | G0 更强 |
| 256 | +1.21 mm | -1.79 pp | -0.85 pp | +2.21 pp | G1 用小幅几何损失换更少 early 与更高 hit |
| 512 | -5.61 mm | +0.07 pp | +0.20 pp | +3.04 pp | G1 改善 CD/hit，early 基本持平但略差 |
| native | -14.89 mm | +1.72 pp | +3.38 pp | +0.08 pp | 不匹配密度；只显示 TSDF 的高密度几何上限 |

这组结果把 D0 的难点定位得更清楚：密度显著影响几何和首返回，G0 与 G1 在中等预算上形成非支配简单基线前沿。学习方法必须在同点数条件下同时越过这两条曲线，单独改善 CD 或增加表面点数不能构成方法贡献。

## 稳定性、来源与失败恢复

G1 的 66 Actors 来自 63 scenes：44 Actors/43 scenes 可由 raw nuScenes 恢复，22 Actors/20 scenes 必须走 DriveStudio processed recovery。r1 错误假设全部 raw 可用，在任何 metric 前失败并登记 `V71-F54`。修复后拒绝同 scene 混合来源、缺 root 和缺 Actor，不做 fallback 或删样本。r2 完成但 manifest 的 failure delta 不完整；r3 按相同协议重跑，r2/r3 的 Actor rows、log rows 与全部 metrics 完全一致，r3 为 canonical。

日志级 bootstrap 显示异质性。G0 native 的 log-macro CD-L1 为 153.41 mm，95% CI `[120.48, 191.82]` mm；early 为 38.89%，95% CI `[33.16%, 44.68%]`。G1 native 的 log-macro CD-L1 为 136.90 mm，95% CI `[117.73, 159.43]` mm；early 为 46.06%，95% CI `[41.95%, 50.41%]`。主判断因此应保留日志级不确定性，而不能只看 66 个 Actor 的均值。

V7.2 NumPy 低内存算子与 V7.1 Torch 算子在真实 Actor smoke 上得到完全相同的 early/hit/observable 计数，CD 绝对差 `2.24e-8 m`。G0 峰值 RSS=`0.061 GiB`、wall=`17.34 s`；G1 峰值 RSS=`0.717 GiB`、wall=`274.80 s`。两者正式 artifact 均含 resolved config、manifest、fingerprint、330 条 Actor-budget rows、170 条 log-budget rows 和 summary。

## 与旧 M8 的边界比较

同一 66 Actors 的历史 M8 native 输出为 CD-L1 `231.46 mm`、early `25.70%`、hit `50.43%`。它未按本轮密度预算重建，且旧 M43 描述性 point-surface 曾存在 literal/categorical 字段覆盖问题。因此旧 M8 数值只证明 G2 需要重跑，不能与 G0/G1 作胜负主张。

## AdaPoinTr 开卡前状态

- 官方源码固定为 `4603257ed3db9e7dad349b712e1b2fe0da207015`。
- 官方 PCN checkpoint 已下载，`389,745,620` bytes，SHA-256=`f58a5650b348ca3d68fc8e01e3e88f57d5a889bbad5521fec7a2760eda664fa1`；checkpoint epoch=353，含 335 个模型 tensors。
- 独立 adapter cache 已生成：593 train / 66 holdout、54/34 logs、512 input / 4096 target、Actor box half-extent normalization，约 15.6 MiB；输入采样不读取 target。
- adapter canonical run 峰值 RSS=`0.037 GiB`，wall=`43.71 s`。

当前 GPU 门槛为硬阻塞：PyTorch=`2.4.1+cu121`，但 CUDA available=false、`nvidia-smi` 无权限、`nvcc` 不存在；PointNet++ 与 Chamfer 扩展未安装。开卡后的第一步是核对 driver/toolkit 后建立隔离环境并做单 batch checkpoint capability，再冻结训练预算。

## 当前结论边界

- 已支持：几何完整性与首返回一致性存在可测权衡；输出密度是强混杂因素；G0/G1 都是不可省略的强简单基线；target-free 评测和日志级统计可以在低内存 CPU 上复现。
- 尚不支持：V7.2 新方法优于成熟补全、F/O/U 必需、路线 A 或 B 已胜出、任何独立源域或跨传感器泛化。
- GPU 阶段未完成：G2/G3 matched density；W0--W4 同信息/容量权重对照；至少一个完整 neural LiDAR capability run。
