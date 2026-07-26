# N1 receiver-centric cut-in final：最终报告

> **最终状态**：`REJECTED / stop_nuscenes_cutin_mining_too_sparse`
> **正式分母**：675/675 evaluation scenes
> **strict 结果**：1 PASS / 1 scene
> **N2 授权**：`false`

## 结论

用户开放 120 GiB 容器内存后，final strict v2 已在 clean commit `beee1de`、seed 0、冻结研究规则下完成全部
675 scenes。K4 regression、raw-only 和 Resource Contract V2 都通过，但 prospective population 只有 1 个
strict PASS，且只覆盖 1 scene，低于预注册的 3 candidates / 3 scenes machine-readiness。因此本计划按
P7.5C 以稀疏拒绝关闭，不再继续阈值优化或 nuScenes cut-in mining。

这不是“nuScenes 不存在 cut-in”，也不是对唯一 PASS 的人工真伪裁决；准确结论是：当前冻结 strict v2
无法在 scene-disjoint official train split 上形成达到最低规模的 prospective 候选池。

## 最终 formal

```text
/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/
v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142941598714Z__883fae9a
```

| 项目 | 值 |
|---|---|
| code | `beee1ded758e8970b57a07dce8c685b44a1d9e90`；clean |
| config | `configs/resim/event_first_n1_cutin_final_resource_v2.yaml` |
| config fingerprint | `883fae9a6514c0bff5bba8bcaf81a22c79e6d719586221596a7d4b5364c337da` |
| data fingerprint | `9c516bd005562b6afd758f5f46667492a7266f5b860aaf4e52bb0756f50138b3` |
| calibration/evaluation | 42 / 675 scenes；intersection=0 |
| strict status | `ABSTAIN=1,556`、`FAIL=200`、`PASS=1` |
| strict modes | parallel lane change=208、receiver branch merge=1,549 |
| PASS coverage | 1 candidate / 1 scene |
| machine readiness | candidate=false、scene=false、raw-only=true、resource=true、K4=true |
| peak batch RSS | `337,154,048` bytes |
| peak cgroup current | `4,556,898,304` bytes |
| strict pool canonical SHA256 | `c151351d7eff06f02588a5e96578304631ab2c13bfc9ba6c3be1057862e721a6` |
| artifact-set SHA256 | `51b3f122b120c056198ad878b3129ea1f3d1715f84694dc96854839069d50ba7` |
| terminal | `REJECTED / stop_nuscenes_cutin_mining_too_sparse` |

primary reason 分布为 `UNSUPPORTED_BRANCH_MERGE_MODE=1,549`、
`SOURCE_TARGET_NOT_PARALLEL=149`、`NO_RAW_LATERAL_ENTRY=51`、
`INSUFFICIENT_RAW_SUPPORT=4`、`BOUNDARY_RAW_ENTRY_EVIDENCE=3`。这些是冻结 verifier 的 first-failure
统计，不能被事后提升为人工 FP/TP。

## 两个工程失败如何处理

Resource Contract V1 的 2 GiB 启动失败和独立资源拒绝裁决保持不可变。用户改变外部资源前提后，V2 使用新
config fingerprint 和新 run ID 恢复；V1 历史没有被覆盖。

V2 首次 clean run：

```text
.../v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142634031503Z__5c8c65d7
```

在 K4 后、任何 evaluation scene 前发现 `675 != 669`。官方 train=700，42 个 calibration scenes 中仅
25 个属于 train、另 17 个属于 val，所以集合差确定为 675。commit `beee1de` 只修正 expected-count assertion；
scene identity、strict gate、K4、seed、抽样和人工门槛均未改变。

## 完整人工审核包

```text
/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-SPARSE-AUDIT-01/
v71_n1-event-cutin-final-sparse-audit-01__human-review-v1__s0__20260726T145456566329Z__d3ceeef5
```

包内包含：

- 1 条 `primary_pass`：唯一 strict PASS；
- 3 条 `diagnostic_abstain`：覆盖边界 raw entry、raw support 不足和不支持的 branch merge；
- blind/debug HTML、4 个 evidence JSON、14 张 topdown/占位 PNG、4 张 signal PNG；
- `review_template.jsonl`、空白 `review_working.jsonl`、完整 `HUMAN_REVIEW_PROMPT.md`；
- parent/source SHA256、package manifest 和 validator 精确命令。

immutable set SHA256：
`949aed9405721643613a72f9947cbea1a47e94caec4f8f14bc5e1d491b41ec7a`。
全部 source/immutable hash、18 张 PNG 解码、盲法字段和代表性视觉布局已复核；空白 validator 按预期在
`F1-P-001` component verdict 处 fail closed。两个 first-failure diagnostic 没有 raw per-frame payload，
包内以明确 `TOPDOWN UNAVAILABLE` 占位与原因展示，不能被误作车辆几何证据。

最终包另行封存 `build_sparse_review_package.py`，SHA256 为
`1e799660117d81f0d61f505929d64cb65f4ff4f35e695ba6b8ac997e0246212c`，避免把一次性包装逻辑隐藏在
clean builder commit 之外。

人工 verdict 只能由用户或指定评审者填写。即使唯一 primary 被判为 TP，也只有 1 TP / 1 scene，仍低于
sparse 最低 3/3；人工结果不能改变 parent `REJECTED`、恢复挖掘器、生成正式 seed pool或授权 N2。

## 最终边界

- 不把 K4 calibration、单例 PASS 或 diagnostic ABSTAIN 写成总体 precision/recall；
- 不把 machine PASS 未经人工确认写成真值；
- 不为增加数量恢复 receiver branch merge、放宽 parallel/raw/receiver 时序或改变 scene split；
- 不启动 N2/N3、渲染、训练、传感器下载或下一轮 cut-in 规则迭代；
- 未来若提出不同事件 taxonomy 或新数据资产，必须是新的研究问题、任务 ID、预注册和 scene-disjoint 评估。
