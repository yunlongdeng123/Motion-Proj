# Research Status

## WorldSim V6.1 P0 范围冻结已预注册（2026-08-22）

状态：`p0_infrastructure_recovery_pre_registered / formal_retry_pending`

当前 active hypothesis=`WS-V61-H-P0-002`，task=`WS-V61-P0-SCOPE-FREEZE-01`。V6 selector 研究族继续冻结，
V6.1 转向 Occupancy-authoritative、Gaussian-rendered、task-verifiable 的四维世界编译器，不再继续阈值、selector、
2D inpainting 或 per-case generator 混选。

P0 精确绑定：

- V6.1 plan SHA-256=`8ac58801...38be`；
- R10 28-case baseline=`3 ACCEPT / 7 ABSTAIN / 18 REJECT`、false-safe=`0`、accepted mask pixels=`107807`；
- scene mapping=`scene-0048 -> processed 045`、`scene-0242 -> processed 191`；
- `O_method` 与 `O_eval` 使用不重叠的 raw LiDAR sweep 路径，confirmation 保持锁定；
- failure refs=`V6-F25/V6-F26/V6-F65/V6-F71/V6-F78/V6-F79`。

H-P0-001 在创建 run 或读取任何科学输入前因新 namespace 不存在而触发 `FileNotFoundError`；GPU/训练/生成器均未启动，
没有方法结论，登记为 `V61-F01`。H-P0-002 只在磁盘审计前创建精确 `runs/worldsim_v61` 路径，其余科学合同不变。

下一步：从干净提交运行 P0；通过后立即启动 `WS-V61-ME0-OCCIR-01`，实现 `T_dst_src`、oriented volume、
actor identity/lifecycle、FREE/OCCUPIED/UNKNOWN 与 method/eval evidence disjointness。ME-1 oracle 若不能达到
`5/28`，则停止 Hunyuan3D/learned Occupancy 接入并修 compiler/evaluator。

## WorldSim V6 收口：selector 研究族已冻结（2026-08-22）

状态：`selector_research_family_frozen_closeout_complete`

当前没有 active hypothesis。R141 未执行。按照最终研究决策，本研究族不再继续 threshold 13/45、新 actor、新编辑方向，也不引入新的 selector 机制。

### R140 recovery

R140 H001 与 H002 已完成科学计算，但由于 Python 源码使用小写 JSON boolean，在正式 closeout 阶段失败；它们继续作为 V6-F97 与 V6-F98 不可变保留。H003 只把剩余的 `false` 改为 `False`，所有科学输入、公式与 gate 均保持不变，并从干净且已推送的源提交 `a13759ba8db03e1f740ad93e246ca24f0ff2d7fa` 完成。

Canonical run：

```text
run://worldsim_v6/WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01/20260822T063937Z__end-to-end-utility-s20260821-r1
```

| 条件 | End-to-end reduction | Reconstruction errors |
| --- | ---: | ---: |
| StreetGS | 0.13533665047667254 | 0 |
| AD-GS development | 0.11143415340582441 | 0 |
| AD-GS exact-once confirmation | 0.016636471392706964 | 0 |
| Macro | 0.08780242509173464 | 0 |
| Worst | 0.016636471392706964 | 0 |

Full 与 selective 路径以相同方式计入 sensor time。这些数值是单次已观测 artifact cost，不是 replicated performance estimate。

Artifacts：

- certificate `913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363`
- gate `ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51`
- summary `50900ff99736055a10c32f4362176b7fc87862ae84667591077d6c17024e635b`
- manifest `1cc753b3c0a9489ced2a58b23035466ee26cba963d2abc56c84ebd4d057e5a62`
- resource audit `06c110236591529d5fef5f4178bfed696b6c0ad0cfbce94c497896dc92230265`
- terminal `be263ba010cdb936fbd01dbfa0fe294b8022101aae348c95030d2a42d45fdb77`

### Selector 最终证据

| 实验 | 状态 | 保留结论 |
| --- | --- | --- |
| R134 | rejected / V6-F94 | threshold 13 漏检 AD-GS frame 13（RGB 1、label 1）。 |
| R136 | rejected / V6-F95 | 冻结 threshold 1 在 heldout frame 14 出现 1 个 FP；精确分类声明失败。 |
| R137 | accepted development | 157 个 AD-GS 帧，调用减少 16.56%，0 false reuse，628 个 hash 全部精确。 |
| R138 | failed consumed / V6-F96 | 负数 CLI 参数在 sensor 输出前失败；不存在方法结论。 |
| R139 | accepted exact-once | 39 个 AD-GS 帧，调用减少 17.95%，0 false reuse，156 个 hash 全部精确。 |
| R140 | V6-F97/F98 recovery 后 accepted analysis | Macro 端到端 reduction 8.78%，worst 1.66%，0 reconstruction errors。 |

### 治理状态

- Failure ledger 的当前权威边界是 V6-F98；recovery 注记不删除或重分类失败 attempt。
- Selector 研究族在 R140 后冻结。R141 明确为未执行，不是 rejected，也不是 accepted。
- Confirmation 与 test 分区继续锁定。
- Claim boundary 只覆盖 operational equivalence 与已观测 wall-time accounting；不声明 semantic、physics、planning 或 safety correctness。
- 仓库收敛目标为唯一远端分支 `main`，指向本次 closeout。

详见 [selector 研究族收口](autoresearch/worldsim_v6/SELECTOR_RESEARCH_FAMILY_CLOSEOUT.md)、[failure ledger](RESEARCH_FAILURES.md) 与 [V6 plan](WORLDSIM_V6_VERIFIABLE_WORLD_COMPILER_AUTORESEARCH_PLAN.md)。
