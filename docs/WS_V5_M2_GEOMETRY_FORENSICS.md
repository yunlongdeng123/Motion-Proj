# WorldSim V5 M2 Geometry Forensics

- Task：`WS-V5-M2-D0-GEOMETRY-FORENSICS-01`
- 日期：`2026-08-14`
- 状态：`running`
- 范围：只读 V4 r212/r219–r222 historical diagnostic；未改 V4 artifact，未调 router

## 1. 问题与 canonical 绑定

V4 r222 相对 frozen TELEA 的 scene-balanced hole geometry MAE 从 `2.1435024986 m` 变为 `5.5343121223 m`，policy delta=`+3.3908096237 m`。V5 不删除这个 caveat，但必须区分 accepted candidate、risk abstain、role-asset blocked 与 metric/reference。

- r212 summary SHA=`93e166d9bed748fcd96adb94ff314b73059caa05d32d7941a6b490d7246430a9`
- r222 summary/manifest SHA=`6bfeb3c6a1e8f1905da936d4e83c93828c030a301ee9d4bedae081c7cc6b1a95 / 702cdb487643bbe633a164d24b9664f35bebc754186fb0845cec4b46250447dd`
- r222 router decisions SHA=`e59ce5f17e6c3271825875c057e0d777ad9aca95f0ef4b8e07b74d946022caf3`
- r222 matched table SHA=`c7bcadfef2b23b3889c0d130eaf634c3faeca3f4c17abd76898152517b0a86fb`

Denominator：`154 requests = 130 measured with candidates + 24 role-asset blocked`；candidate=`214`，arm count=`TELEA 130 / OBSERVED 76 / ROADPATCH 8`。router=`83 accepted + 71 abstain`，其中 `47` 为 risk-threshold abstain，`24` 为无 role asset。

## 2. Risk saturation 已确认

V4 mapping：

```text
geometry_risk = clip(hole_geometry_mae_m / 0.5, 0, 1)
```

| Arm | Candidate | geometry risk=1 | saturation | unique risk |
|---|---:|---:|---:|---:|
| OBSERVED | 76 | 60 | 78.95% | 17 |
| ROADPATCH | 8 | 3 | 37.50% | 6 |
| TELEA | 130 | 129 | 99.23% | 2 |
| total | 214 | 192 | 89.72% | 23 |

- total saturation=`192/214=89.72%`。
- MAE `>=0.5 m`：`192/192` risk=1；`>=1 m`：`161/161`；`>=2 m`：`63/63`；`>=5 m`：`14/14`。
- `57/130` measured requests 存在“candidate rendered MAE 不同，但 request 内所有 geometry risk 相同”的排序碰撞。
- 最坏样例 `scene-0071__high_support__f007__c0`：TELEA/OBSERVED MAE=`8.7745/1.0184 m`，二者 geometry risk 都是 1，router 最终选 TELEA，geometry regret=`7.7561 m`。

结论：geometry term 在主要分布上退化为常数。单纯把 geometry weight 从 0.2 提高，仍无法恢复 1 m、2 m、5 m tail 的相对顺序。

## 3. Geometry oracle / accepted routing

对每个有候选 request，以最小 rendered `hole_geometry_mae_m` 定义 retrospective oracle；这只是 diagnostic oracle，不是新训练目标。

| 项 | 结果 |
|---|---:|
| measured request | 130 |
| accepted request | 83 |
| accepted exact geometry-oracle | 62 |
| accepted positive geometry regret | 21 |
| accepted mean regret | 0.3083979811 m |
| accepted median regret | 0 m |
| accepted max regret | 7.7560878992 m |
| accepted regret sum | 25.5970324315 m |

130 个 oracle arm 分布=`TELEA 69 / OBSERVED 54 / ROADPATCH 7`。21 个 positive-regret selection 中，selected arm=`OBSERVED 19 / TELEA 2`。这证明 routing/risk representation 是可观测根因之一，但不证明 candidate 与 reference 没有问题。

## 4. `+3.3908 m` denominator 分解

| Group | n | Router MAE request mean | TELEA same-request mean | request delta | Router/TELEA scene-balanced delta |
|---|---:|---:|---:|---:|---:|
| accepted | 83 | 1.658543 | 2.033483 | -0.374940 | -0.391581 |
| risk abstain | 47 | 16.582834 | 2.608174 | +13.974660 | +11.572890 |
| role-asset blocked | 24 | 1.029191 | 1.029191 | 0 | 0 |
| full denominator | 154 | 6.115278 | 2.052362 | +4.062916 | +3.390810 |

按 154-request mean 加性分解：accepted selection 贡献 `-0.202078 m`，47 个 risk abstain 的 atomic no-op 贡献 `+4.264994 m`，24 个 blocked 贡献 0。也就是说：

1. accepted-only repair 并未相对同请求 TELEA 退化；
2. full policy 的 geometry 仍显著退化，因为 abstain 保留在 denominator 且 no-op 在 remove-hole background reference 下误差很高；
3. V5 必须同时优化 geometry feasibility 与 valid edit yield，不能靠删除 abstain 获得好看的 accepted-only 指标。

## 5. 四类根因的当前判定

| 类别 | 当前状态 | 证据 |
|---|---|---|
| routing failure | observed | 21/83 accepted 非 geometry oracle；risk saturation 严重 |
| candidate failure | unresolved | 尚无结果前 absolute feasibility threshold；oracle 好坏不能只靠相对排名定义 |
| Gaussianization/rendering failure | blocked evidence missing | r222 未保存 raw surface error、pre-Gaussian error 与 post-render error 三段值 |
| metric/reference failure | unresolved high risk | `hole_geometry_mae_m` 比较 candidate rendered depth 与 base renderer `Background_depth`，不是 same-view hidden-background GT |

当前 V4 reference 的物理含义是“与 immutable base 的 Background model depth 一致”，它对 maintenance 很有价值，但不能自动写成真实道路 geometry GT。后续必须用可观测 LiDAR、multi-view depth support、reference confidence 与 occlusion/extrapolation 标志分层。

## 6. 下一轮 per-request 合同

每个 candidate 必须持久化：

```text
request_id / scene / frame / camera / role
reference_source / reference_confidence / observed_reference_pixels
raw_surface_model / raw_geometry_error
pre_gaussianization_geometry_error
post_gaussianization_render_error
hole_depth_mae / median / relative / p90 / p95
photo_risk / raw_geometry_error / normalized_geometry_risk
geometry_support / lidar_support / multi_view_depth_support
surface_fit_residual / extrapolation_distance / occlusion_uncertainty
temporal_risk / uncertainty / compute_cost
selected_candidate / geometry_oracle_candidate / oracle_regret
accepted / abstain_reason / coverage_denominator
```

## 7. 下一门禁

1. 先修 forensic/evaluator schema，使 reference 与 raw→Gaussian→render 三段误差可分；禁止先改 router。
2. 在 historical diagnostic 上重算 non-saturating mapping 只用于辨识，不在 V4 validation 上选择 R1/R2/R3/R4。
3. fresh development 冻结后才拟合 mapping 与 feasibility threshold；validation/test 只读。
4. 主表同时报告 accepted geometry、abstain geometry、full-denominator geometry、coverage 与 valid edit yield。
5. 若 candidate 全部不满足可信 reference 下的 geometry gate，先做 geometry-first candidate；只有存在 safe candidate 仍选错时才改 feasibility-first router。

## 8. Machine-reproducible freeze contract

本页的 saturation、oracle regret 与 denominator decomposition 由下列冻结入口重算：

- resolved config：`configs/worldsim_v5/m2_forensics_v1.yaml`
- fail-closed runner：`scripts/run_worldsim_v5_m2_forensics.py`
- 共享正式产物协议：`scripts/worldsim_v5_forensics_common.py`

runner 必须逐文件校验 r222 terminal artifacts 与 6 个 scene summary SHA，验证 V4 `clip(MAE/0.5, 0, 1)` 映射，重建 154-request/214-candidate 分母、risk collision、retrospective geometry oracle、accepted/risk-abstain/role-asset-blocked 分解，并把 reference 与 raw→pre-Gaussian→post-render 缺失字段写入 run-local `artifacts/geometry_audit.json`。它只做 historical diagnosis，不重拟合 router、不删 abstain、不读取 fresh/test quality，也不授权 full M2。
