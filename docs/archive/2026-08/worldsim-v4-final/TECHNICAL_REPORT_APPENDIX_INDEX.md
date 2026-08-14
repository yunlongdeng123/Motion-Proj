# WorldSim V4 技术报告附录索引

本索引把 V4 的正结果、失败结论和实验记录映射到可直接引用的轻量证据。正文应先陈述结论边界，附录再展开逐 scene、逐 request 和协议细节。

## 建议附录结构

| 附录 | 建议标题 | 主要材料 | 必报字段 |
|---|---|---|---|
| A | V4 预注册协议与数据冻结 | `snapshots/WORLDSIM_V4_EVIDELTA_GS_PLAN_EXECUTED.md`、`snapshots/WS_V4_P0_SCOPE.md`、`snapshots/WS_V4_D0_NUSCENES_COHORT.md` | 6/6/18 split、scene-disjoint、test freeze、单 RTX 3090 |
| B | M1 scene-disjoint 泛化失败 | `evidence/m1-validation-r200/summary.json`、`metrics.json`、`evidence/m1-rejection-r201/summary.json` | `3 evaluable + 3 abstain`、directional support=`0/6`、Boundary F1/Brier/ECE delta、无 validation 重搜 |
| C | M2 selective routing 与 geometry 反例 | `evidence/m2-development-r212/`、`evidence/m2-validation-r222/summary.json`、`artifacts/router_decisions.json`、`artifacts/selective_risk_curve.json` | `83/154`、coverage、risk separation、hole PSNR、hole geometry `2.1435→5.5343 m` |
| D | M3 validation 与 frozen test | `evidence/m3-validation-r238/`、`evidence/m3-test-r335/summary.json`、`scene_metrics.jsonl`、`paired_metrics.jsonl` | `12 evaluable + 6 abstain`、warp L1、temporal LPIPS、scene-level CI/paired statistics、rollback |
| E | exact-once 与可追溯性 | `evidence/m3-test-r335/source_snapshot/V4_TEST_FREEZE.json`、`evidence/closeout-r336/` | freeze/source commit、freeze SHA、attempt/completion=`18/18`、无 test-time search、无 source reread |
| F | 失败与防重复规则 | `snapshots/RESEARCH_FAILURES_V4_FINAL_SNAPSHOT.md` | `V4-F30`–`V4-F45`，特别是 F35、F38、F43、F45 |
| G | KITTI 外部数据 blocker | `snapshots/WS_V4_KITTI_AUDIT.md`、`snapshots/KITTI_LAYOUT_AUDIT.md` | V4 未执行 cross-domain；数据后来到达不能倒写历史结论 |

## 三条主结论的推荐表述

### M1 负结果

V4 的独立 Gaussian、Beta-only evidence field 在 development 上通过，但在 scene-disjoint validation 上被拒绝：六场中三场可评、三场 abstain，六场方向支持为 `0/6`；可评场景均值 Boundary F1 下降 `0.06646`，Brier/ECE 分别恶化 `0.002449/0.002497`。这支持“建模假设泛化不足”，不支持“SAM supervision 无效”。

### M2 tradeoff

冻结的 `uncertainty_forward@1.0` router 在 154 个 validation 请求中接受 83 个、abstain 71 个；abstain 的 counterfactual error 比 accepted 高 `0.12413`，说明 selective risk 排序具有信息量。与此同时，hole PSNR 提高 `3.17978 dB`，但 hole geometry MAE 从 `2.14350 m` 增至 `5.53431 m`。因此只能声明 selective routing 和 appearance 收益，不能声明物理或几何支配。

### M3 正结果

冻结的 4-control-point SE(3) temporal delta 在 exact-once 18-scene test 上完成 `18/18` 次预注册尝试，其中 `12` 场可评、`6` 场 abstain。可评场景聚合的 warp L1 与 temporal LPIPS 分别相对改善 `34.3943%` 和 `16.3656%`，rollback exact fraction=`1.0`。完整分母、abstain 原因和非目标区域指标必须同时附上。

## 图表取数位置

- M1 calibration 与 scene breakdown：`evidence/m1-validation-r200/calibration.json`、`metrics.json`；
- M2 risk-coverage 曲线：`evidence/m2-validation-r222/artifacts/selective_risk_curve.json`；
- M2 request/candidate 明细：`evidence/m2-validation-r222/artifacts/router_decisions.json`、`matched_repair_table.json`；
- M3 scene table：`evidence/m3-test-r335/scene_metrics.jsonl`；
- M3 paired statistics：`evidence/m3-test-r335/paired_metrics.jsonl` 与 `summary.json`；
- 终局 machine-readable verdict：`evidence/closeout-r336/summary.json`。

## 禁止的附录写法

- 删除 abstain 后把 M1 写成 `3/3`、把 M3 写成 `12/12`，或把 attempt=`18/18` 写成质量成功=`18/18`；
- 只报告 M2 hole PSNR，不报告 `+3.3908096237 m` geometry 退化；
- 把 M3 的时序确认写成 M2 geometry 问题已解决；
- 把 V4 KITTI adapter 的历史 blocker 写成 cross-domain 失败或成功；
- 用 V4 已读取 test scene 作为 V5 confirmatory test。
