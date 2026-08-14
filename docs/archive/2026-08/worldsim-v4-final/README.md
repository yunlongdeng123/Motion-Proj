# WorldSim V4 终局归档

- 归档日期：2026-08-14
- 路线：`WorldSim V4 / EviDelta-GS`
- 冻结分支：`research/worldsim-v4-evidelta`
- canonical closeout HEAD：`403c5703a755c999d42a5ec3eb063db6cc751761`
- 终态：`M1 rejected / M2 done with geometry caveat / M3 confirmed`
- 权威 closeout audit：`evidence/closeout-r336/`

本目录是 V4 的终局文档与轻量实验凭证包，面向技术报告正文、附录、失败分析和后续 V5 机制诊断。它不会授权重跑 V4，也不会把历史已读 test scene 重新解释为 V5 confirmatory data。

## 一页结论

| 模块 | 终态 | 核心证据 | 必须同时报告的边界 |
|---|---|---|---|
| M1 Bayesian evidence field | `rejected` | validation `3 evaluable + 3 abstain`，directional support=`0/6` | development 正结果不能覆盖 scene-disjoint validation 失败；禁止把 M2/M3 成功倒写为 M1 成功 |
| M2 risk-aware repair router | `done` | `83/154` accepted，coverage=`0.5389610390`，abstain-error minus accepted-error=`+0.1241311528` | hole geometry MAE 从 `2.1435024986 m` 退化到 `5.5343121223 m`，即 `+3.3908096237 m`；不能宣称 geometry dominance |
| M3 SE(3) temporal delta | `confirmed` | exact-once test=`18/18`，`12 evaluable + 6 abstain`；warp L1/temporal LPIPS relative improvement=`34.3943%/16.3656%` | 结论只覆盖冻结 nuScenes 18 scenes、三前向相机、2–4 s clips 与单 RTX 3090；不是长时序、KITTI 或闭环安全结论 |

M1 validation 的聚合变化为 Boundary F1/Brier/ECE=`-0.0664623346/+0.0024487362/+0.0024972500`。M2 同时得到 hole PSNR=`+3.1797798583 dB`，但 appearance 改善不能抵消 geometry 失败。M3 test 的 baseline/candidate warp L1=`0.0618690015/0.0405895766`，temporal LPIPS=`0.0263505519/0.0220381359`，rollback exact fraction=`1.0`。

## 恢复和引用顺序

1. `TECHNICAL_REPORT_APPENDIX_INDEX.md`：附录写作入口与建议表格；
2. `evidence/closeout-r336/summary.json`：M1/M2/M3 终态与 exact-once 总审计；
3. `evidence/m1-validation-r200/` 与 `evidence/m1-rejection-r201/`：M1 负结果；
4. `evidence/m2-development-r212/` 与 `evidence/m2-validation-r222/`：M2 冻结选择、selective risk 和 geometry caveat；
5. `evidence/m3-validation-r238/` 与 `evidence/m3-test-r335/`：M3 validation、test、scene-level 分母和配对指标；
6. `snapshots/RESEARCH_FAILURES_V4_FINAL_SNAPSHOT.md`：禁止重复项与复开条件；
7. `snapshots/EXPERIMENTS_V4_FINAL_SNAPSHOT.md`、`snapshots/RESEARCH_STATUS_V4_FINAL_SNAPSHOT.md`：完整历史账本；
8. `SHA256SUMS`：归档内文件完整性。

## 目录说明

- `snapshots/`：V4 closeout 时的三份研究事实源、执行计划和 P0/D0/B0/KITTI/文献审计快照；
- `evidence/`：从 canonical run 复制的轻量 JSON/JSONL/YAML/源码快照，不含 checkpoint、渲染帧和训练缓存；
- `ARCHIVE_MANIFEST.json`：来源 commit、canonical run 路径和保留策略；
- `SHA256SUMS`：相对路径内容校验清单。

原始 canonical run 仍驻留于 `/root/autodl-tmp/runs/worldsim_v4/`。本归档保存足以复核结论、分母、冻结参数、manifest 和主要逐场指标的轻量副本；大型产物的保留继续服从仓库资产策略。

## 结论冻结规则

- M1 保持 `rejected`，除非新任务提出新的结构假设、fresh split 和预注册门；不得在 V4 validation 上继续调阈值或扩 feature。
- M2 的 selective routing 结论与 `+3.3908096237 m` geometry 退化必须成对引用。
- M3 test 的 6 个 abstain 必须保留在 18-scene denominator；不得写成 `18/18` 质量成功。
- V4 已读取的 30 个 scene 只允许用于 V5 failure diagnosis、regression 和 retrospective visualization。
- KITTI 在 V4 期间为外部数据缺失 blocker；2026-08-13 到达的新压缩包属于 V5 数据预检输入，不能倒写 V4 cross-dataset 结果。
