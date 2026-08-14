# WorldSim V5 M2 技术报告附录索引

- Task：`WS-V5-M2-GEOMETRY-FIRST-REPAIR-01`
- 状态：`running`
- 冻结范围：r001–r015；surface/Gaussianization/cross-view sequence closed，M2 正式 rejected

## 附录建议表

| 附录项 | 证据入口 | 用途 |
|---|---|---|
| M2-A：历史几何根因 | [`../../../WS_V5_M2_GEOMETRY_FORENSICS.md`](../../../WS_V5_M2_GEOMETRY_FORENSICS.md) | V4 risk saturation、oracle regret、154-request denominator |
| M2-B：staged/per-actor development | [`../../../WS_V5_M2_GEOMETRY_FIRST_DEVELOPMENT.md`](../../../WS_V5_M2_GEOMETRY_FIRST_DEVELOPMENT.md) | r001–r011 时间线、协议修正、失败/机制结论、哈希 |
| M2-C：surface 机器元数据 | [`M2_R001_R009_METADATA.json`](M2_R001_R009_METADATA.json) | surface run path、source、terminal、主要指标与 SHA-256 |
| M2-D：Gaussianization 元数据 | [`M2_R010_R011_GAUSSIANIZATION_METADATA.json`](M2_R010_R011_GAUSSIANIZATION_METADATA.json) | launcher failure、四因子臂、factorial contrast、完整性 SHA-256 |
| M2-D2：cross-view / closeout 元数据 | [`M2_R012_R015_CROSS_VIEW_CLOSEOUT_METADATA.json`](M2_R012_R015_CROSS_VIEW_CLOSEOUT_METADATA.json) | G4/G5、coverage、blocked terminal、r015 rejection ledger 与 SHA-256 |
| M2-E：实验总账 | [`../../../EXPERIMENTS.md`](../../../EXPERIMENTS.md) | 与全项目 canonical run 对齐 |
| M2-F：负结论 | [`../../../RESEARCH_FAILURES.md`](../../../RESEARCH_FAILURES.md) | 禁止重复项、claim 边界与复开条件 |

## 推荐正文引用

1. 先说明 r002/r003 的 actor-union request-unit 错误以及 r004 的逐像素 union replay；
2. surface 主表只使用 r005/r006/r008/r009 的 `22 evaluable + 1 abstain` 逐 actor denominator；Gaussianization 表使用 r011 同一 denominator；
3. G0 的 absolute fail 与 G1/G2/G3 的 relative rejection 分开写；
4. 所有 MAE 必须同时标注 reference=`base background model proxy` 和低 confidence；
5. r001/r007/r010/r012 只进入工程失败附录，不进入方法均值；
6. r011 的 DENSE 支持必须写成 model-proxy mechanism evidence，不能写成方法选择、validation 或真实道路 GT 改善。
7. r014 的相对改善必须与 raw/post absolute-safe=`1/22,0/22` 成对报告；r015 task status=`rejected`，不得改写为 M2 成功。

大型 NPZ、checkpoint 与 render 不复制进 Git。正式 run 目录是权威 payload；本目录保存轻量可检索索引。
