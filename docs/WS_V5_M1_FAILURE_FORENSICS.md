# WorldSim V5 M1 Failure Forensics

- Task：`WS-V5-M1-D0-BAYES-FORENSICS-01`
- 日期：`2026-08-14`
- 状态：`running`
- 范围：只读 V4 historical diagnostic；未读 fresh nuScenes / KITTI quality，未训练、未调参

## 1. 已冻结事实

V4 M1 r200 的完整 validation denominator 是 `3 evaluable + 3 abstain = 6`，directional support=`0/6`。相对 V3.3 O1，scene-balanced Boundary F1/Brier/ECE/FN semantic mass delta=`-0.0664623346/+0.0024487362/+0.0024972500/+0.0083741268`；r201 已把任务冻结为 `rejected`，禁止继续扩 feature。

本 forensic 不把该结果解释为“SAM supervision 无效”，也不把 V5 graph 当作既定答案。当前问题是：V4 canonical artifact 是否足以区分 observation reliability、Beta aggregation、calibration 与 topology 根因。

## 2. Canonical 绑定

- r200=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T204156Z__m1-validation-six-scene-confirmation-s0-r200`
- r200 summary/metrics/manifest SHA=`57d732ba13e46cd57758d5a272d1fbb2d4e21c8a85e41a2b59a849ee975d0309 / 0ae07b58822d7be37e5220a781d951725e802404ce52881c1281bb7974e1d504 / 43af8f881667da578ebd567bca2c9dd17492fc270d8be1a5cfe1d849af05ede5`
- r201=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T210150Z__m1-validation-rejection-audit-s0-r201`
- r201 summary SHA=`470338f225a0ea22e4d5df75d71e86a516c142bd766d1a3ed08009a55fe8fec2`

## 3. 现有 per-Gaussian state 盘点

定义：`O1 target` 仅表示 `hard_instance_id == actor_instance_id` 的 V3.3 O1 membership proxy，不是真实 Gaussian GT；`extreme` 表示 posterior `<=0.01` 或 `>=0.99`；`unobserved` 表示 `positive_count + negative_count == 0`。pixel Boundary F1 与 Gaussian proxy 不在同一统计单位，下面只做并列诊断，不作因果回归。

| Scene / role | Gaussian | O1 target | posterior≥0.5 | O1 target recall | extreme | uncertainty≤1e-3 | unobserved | mixed +/- views | ΔBoundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene-0071/high_support` | 1,083,930 | 26,779 | 27,470 | 88.47% | 81.56% | 86.00% | 30.68% | 19.99% | -0.112514 |
| `scene-0317/boundary_support` | 1,420,017 | 13,351 | 6,292 | 45.14% | 94.75% | 94.71% | 27.06% | 4.80% | -0.070040 |
| `scene-0317/high_support` | 1,420,017 | 14,227 | 12,996 | 91.09% | 91.53% | 93.27% | 18.82% | 8.95% | -0.070040 |
| `scene-0450/high_support` | 754,161 | 26,567 | 21,520 | 67.10% | 88.54% | 89.60% | 30.25% | 11.32% | -0.016834 |

逐场 rendered delta：

| Scene | ΔBoundary F1 | ΔBrier | ΔECE | ΔFN semantic mass | directional |
|---|---:|---:|---:|---:|---|
| `scene-0071` | -0.1125135123 | +0.0079068264 | +0.0069264891 | -0.0005531649 | no |
| `scene-0317` | -0.0700398466 | +0.0000496447 | -0.0000385040 | -0.0047589359 | no |
| `scene-0450` | -0.0168336449 | -0.0006102625 | +0.0006037649 | +0.0304344813 | no |

## 4. 可得结论与不可得结论

已观察到：

1. 最弱的 O1-proxy target recall 出现在唯一 `boundary_support` state（45.14%）；scene-0450 high-support 也只有 67.10%，并伴随 rendered FN mass `+0.03043`。
2. 四个 state 有 `18.82%–30.68%` Gaussian 没有正/负 observation count；Beta state 仍会由 O1 prior 与聚合权重给出 posterior。
3. 全体 Gaussian 的 posterior/uncertainty 大量位于极端区间，但背景 Gaussian 占绝大多数，不能仅凭全体 extreme ratio 宣称“错误过度自信”。需在 boundary/interior、actor/background、observed/unobserved 分层后再判断。
4. canonical state 只保存聚合后的 positive/negative count 与 `mask*visibility*depth*lidar` 乘积权重。它不保存原始 per-view observation 或各 observation 的独立可靠性。

当前不可判定：

- SAM 本身错误，还是 view visibility/depth reliability 失配；
- boundary 与 interior 的 posterior/calibration 差异，因为 state 没有 projected boundary distance；
- posterior 与 local topology disagreement，因为 state 没有 center/covariance/neighborhood；
- graph 是否会改善，或是否会把 actor evidence 泄漏到 road；
- 30-scene 历史 state 上得到的机制是否能在 fresh development 复现。

## 5. D0 缺失字段合同

后续 run-local per-Gaussian table 至少持久化：

```text
scene / role / gaussian_id / base_model / base_index
center / covariance / normal_proxy
view_id / frame_id / projected_pixel / visibility
sam_probability / mask_boundary_distance
depth_residual / lidar_support / view_angle
positive_observation / negative_observation
effective_evidence_count
prior / unary_posterior / unary_uncertainty
knn_ids / edge_affinity / boundary_barrier
local_topology_mean / topology_disagreement
```

大表写入不可变 run 目录的 Parquet/NPZ，不把百万行表提交到 `docs/`；文档只记录 schema、hash、聚合结果与 evidence path。

## 6. 下一门禁

1. 先恢复或重新物化 per-view provenance，并验证每个字段可从 V4 historical input 确定性重建；无法恢复的字段显式 `blocked_evidence_missing`。
2. 在 V4 historical scenes 上只做 diagnostic 分层：boundary/interior、observed/unobserved、O1-target/non-target、LiDAR support、view disagreement。
3. fresh cohort 冻结后，先比较 V4 Beta 与 reliability-aware unary；unary 未改善前禁止扩大 graph。
4. graph 只在 unary 可解释后启动，并必须先报告 cross-boundary leakage；不得直接加入 Transformer 或解锁 semantic split。
