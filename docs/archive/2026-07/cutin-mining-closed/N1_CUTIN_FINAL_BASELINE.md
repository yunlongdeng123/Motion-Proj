# N1 receiver-centric cut-in final：冻结基线

> **状态**：`REJECTED / stop_nuscenes_cutin_mining_too_sparse`
> **日期**：2026-07-26
> **N2 授权**：`false`

本文冻结 final v2 的输入、资源复开、split 计数修复与最终产物。Resource Contract V1 的历史失败保持有效，
但最终研究拒绝原因已经是 prospective pool 过稀。

## 当前冻结输入

| 项目 | 值 |
|---|---|
| final config | `configs/resim/event_first_n1_cutin_final_resource_v2.yaml` |
| resource authorization | `memory.max=128,849,018,880` bytes（120 GiB） |
| start / process stop / cgroup stop | 8 / 8 / 24 GiB |
| formal 代码要求 | clean git、seed 0、不得截断 evaluation、不得跳过 K4 或 audit |
| hard evidence | nuScenes 原始 2 Hz annotation / ego pose / vector-map centerline；插值不得通过 hard gate |
| calibration | 67 labels / 42 scenes；其中 train 25、val 17 |
| evaluation | official train 700 减 train calibration 25 = 675 scenes；intersection=0 |
| machine readiness | ≥3 strict PASS candidates、≥3 PASS scenes；K4/raw/resource 全通过 |
| N2 | 此计划永不自动授权 |

关键提交：

- `7f35a9c`：strict final mining、流式 worker A、K4 replay；
- `7ef2d00`：审核 V2、逐 raw-frame topdown/signal、blind/debug 分离；
- `7104f5c`：runtime 按需导入；
- `d88d5e2`：Resource Contract V1 的独立拒绝裁决；
- `3a548c2`：用户授权的 Resource Contract V2；
- `beee1de`：把错误的 669 scene assertion 修正为确定的 675，不改变实际 split。

## 历史资源失败

V1 clean formal 记录 `memory.current=1,523,929,088 > 1,350,000,000` bytes，在任何 evaluation scene 前
失败；独立裁决为 `REJECTED/stop_nuscenes_cutin_mining`。用户随后开放 120 GiB 并授权继续，因此 V2 使用
新配置和新 run ID；旧 parent/rejection 不覆盖，也不再作为最终方法拒绝原因。

## 最终 formal 与哈希

```text
/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/
v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142941598714Z__883fae9a
```

| 文件/字段 | SHA256 / 值 |
|---|---|
| code commit | `beee1ded758e8970b57a07dce8c685b44a1d9e90` |
| resolved config | `883fae9a6514c0bff5bba8bcaf81a22c79e6d719586221596a7d4b5364c337da` |
| data fingerprint | `9c516bd005562b6afd758f5f46667492a7266f5b860aaf4e52bb0756f50138b3` |
| strict event pool canonical | `c151351d7eff06f02588a5e96578304631ab2c13bfc9ba6c3be1057862e721a6` |
| strict event pool file | `b0b251d599898080ec018694f81a1f219c3136dd38a8fd246068712c20573011` |
| artifact set | `51b3f122b120c056198ad878b3129ea1f3d1715f84694dc96854839069d50ba7` |
| summary file | `70bf037ccca21ac1145564624a423e0f29e32246e279154392944c14c2699829` |
| terminal | `REJECTED / stop_nuscenes_cutin_mining_too_sparse` |

结果：675/675 scenes；`ABSTAIN=1,556`、`FAIL=200`、`PASS=1`；PASS scenes=1；peak batch RSS
`337,154,048` bytes，peak cgroup current `4,556,898,304` bytes。

## 人工包基线

```text
/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-SPARSE-AUDIT-01/
v71_n1-event-cutin-final-sparse-audit-01__human-review-v1__s0__20260726T145456566329Z__d3ceeef5
```

| 项目 | 值 |
|---|---|
| purpose | `sparse_terminal_diagnostic_only` |
| population | 1 primary + 3 diagnostic |
| PNG | 18；全部解码，代表图视觉 QA 通过 |
| package spec | `d3ceeef53f690739ee105dd9b964d4157c8a94b3993cb767e45fefff51379b57` |
| package generator | `1e799660117d81f0d61f505929d64cb65f4ff4f35e695ba6b8ac997e0246212c` |
| audit manifest | `c6393970526cfae84e9c34618ad5dce0544a55868ab1b599333c8ba8ce7623e0` |
| immutable set | `949aed9405721643613a72f9947cbea1a47e94caec4f8f14bc5e1d491b41ec7a` |
| mutable file | `audit/review_working.jsonl`；当前空白 |
| package terminal | `AWAITING_HUMAN_REVIEW` |
| parent terminal | 保持 `REJECTED` |

人工填写只复核唯一 PASS 与 diagnostics，不改变数量门、parent 终态或 N2 授权。
