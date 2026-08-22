# WorldSim V6 Selector 研究族收口

记录时间：server UTC `2026-08-22T06:50:19Z`。

## 决策

Selector 研究族在 R140 后冻结。R141 未执行。本研究族不再探索 threshold 13 或 45、新 actor、新编辑方向或新的 selector 机制。这是基于当前证据已经充分而做出的治理收口，不是 R141 rejected 结果。

## R140 recovery 权威

H-R140-001 与 H-R140-002 是不可变的基础设施失败 V6-F97 与 V6-F98。两者写出了相同的科学 certificate，但 Python dictionary 中的小写 JSON boolean 令正式 closeout 失败。H-R140-003 只把剩余的 `false` 改为 Python `False`、更新 hypothesis binding，并从干净且已推送的源提交 `a13759ba8db03e1f740ad93e246ca24f0ff2d7fa` 运行。

Canonical run：

```text
run://worldsim_v6/WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01/20260822T063937Z__end-to-end-utility-s20260821-r1
```

| Artifact | SHA256 |
| --- | --- |
| certificate | `913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363` |
| gate | `ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51` |
| summary | `50900ff99736055a10c32f4362176b7fc87862ae84667591077d6c17024e635b` |
| manifest | `1cc753b3c0a9489ced2a58b23035466ee26cba963d2abc56c84ebd4d057e5a62` |
| resource audit | `06c110236591529d5fef5f4178bfed696b6c0ad0cfbce94c497896dc92230265` |
| terminal | `be263ba010cdb936fbd01dbfa0fe294b8022101aae348c95030d2a42d45fdb77` |

Scientific certificate 与 H001/H002 的 partial certificate byte-identical。只有 H003 拥有完整的 resource、manifest 与 terminal 权威。

## 端到端 wall-time account

Shared sensor time 同时计入 full 与 selective 总成本。

| 条件 | Reduction fraction |
| --- | ---: |
| StreetGS selective perception | 0.13533665047667254 |
| AD-GS development exact-input reuse | 0.11143415340582441 |
| AD-GS exact-once confirmation | 0.016636471392706964 |
| Macro | 0.08780242509173464 |
| Worst condition | 0.016636471392706964 |

三个条件均为 0 reconstruction errors。该 account 证明已观测 utility 为正，同时显示 sensor rendering 主导 AD-GS confirmation 路径。

## 冻结时保留的证据

- R134 因 1 个 AD-GS false negative 拒绝全局 threshold-13 transfer（V6-F94）。
- R136 因 1 个 heldout conservative false positive 拒绝 threshold-1 exact classification（V6-F95）。
- R137 在 development 上接受 identity-only guard：调用减少 16.56%，0 false reuse，628 个输出 hash 全部精确。
- R138 在 sensor 输出前因 argparse 基础设施错误被消耗，不存在方法结论（V6-F96）。
- R139 接受唯一 orthogonal exact-once condition：调用减少 17.95%，0 false reuse，156 个输出 hash 全部精确。
- R140 H001/H002 继续记为失败（V6-F97/V6-F98）；H003 是 recovery 与 wall-time 权威。

## Claim boundary

本收口只支持冻结实验中的 operational selector behavior、exact reconstruction checks 与已观测端到端成本核算。它不证明任意编辑或 frontend 的 generalization、semantic correctness、physical validity、planning quality、safety、statistical significance 或 production throughput。

## 最终状态

- Active selector hypothesis：none。
- Selector confirmation/test partitions：locked。
- 新 selector research：frozen。
- Canonical repository branch：`main`。
- Failure authority：`docs/RESEARCH_FAILURES.md` through V6-F98。
