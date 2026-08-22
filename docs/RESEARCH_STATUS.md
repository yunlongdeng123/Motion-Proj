# Research Status

## WorldSim V6.1 ME-0 完成；ME-1 oracle upper bound 已预注册（2026-08-22）

状态：`me0_done / me1_pre_registered / formal_run_pending`

当前 active hypothesis=`WS-V61-H-ME1-002`，task=`WS-V61-ME1-ORACLE-OCC-PROPOSAL-01`。V6 selector 研究族继续冻结，
V6.1 转向 Occupancy-authoritative、Gaussian-rendered、task-verifiable 的四维世界编译器，不再继续阈值、selector、
2D inpainting 或 per-case generator 混选。

P0 精确绑定：

- V6.1 plan SHA-256=`8ac58801...38be`；
- R10 28-case baseline=`3 ACCEPT / 7 ABSTAIN / 18 REJECT`、false-safe=`0`、accepted mask pixels=`107807`；
- scene mapping=`scene-0048 -> processed 045`、`scene-0242 -> processed 191`；
- `O_method` 与 `O_eval` 使用不重叠的 raw LiDAR sweep 路径，confirmation 保持锁定；
- failure refs=`V6-F25/V6-F26/V6-F65/V6-F71/V6-F78/V6-F79`。

H-P0-001 在创建 run 或读取任何科学输入前因新 namespace 不存在而触发 `FileNotFoundError`；GPU/训练/生成器均未启动，
没有方法结论，登记为 `V61-F01`。H-P0-002 只创建精确 run namespace 后正式通过，`V61-F01` 已 resolved。

P0 canonical：

```text
run://worldsim_v61/WS-V61-P0-SCOPE-FREEZE-01/20260822T100812Z__scope-freeze-s20260822-r1
```

source=`6247fd89068615f791b428c3296faf945e713c75`；gate/summary/manifest=`fb2a416a...ae40 / e53a86f2...907c /
2ed96578...7593`。全部 gate PASS；R10=`3/28`、false-safe=`0`、case identity 与 scene mapping exact，
method/eval source paths disjoint。

ME-0 canonical：

```text
run://worldsim_v61/WS-V61-ME0-OCCIR-01/20260822T101817Z__occir-s20260822-r1
```

source=`5a3bc42eb68cfcda673df3c32d81479373b1bff3`；4 scene/frame units、8 truth tiers、28 case bindings 全部
通过。`O_method/O_eval` 的 raw LiDAR path 与 payload hash 全局互斥；每格 UNKNOWN/FREE/OCCUPIED 非零；
oriented actor volume、identity/lifecycle、source-removal→UNKNOWN、fresh-process content exact 与
`<=2.14e-14m` round-trip 均通过。gate/summary/manifest=`1e818074...8bb7 / 6e50644b...b14f /
386d99ab...59ec`；wall=`10.57s`，4 CPU workers，无训练/生成器/confirmation read。

ME-1 预注册固定五臂：冻结 Big-LaMa 的 `B0-2D`、冻结 R10 的 `B1-R10`、不增 coverage 的 `O1-GATE`、
主臂 `O2-OCC-GEOMETRY` 与带 native trajectory/lifecycle/swept OBB collision 的 `O3-OCC-4D`。编译只读
`O_method`，先固化 method decisions，再让 `O_eval` 只计算 hidden truth/false-safe。阈值来自既有合同：
0.2m voxel、0.1m ray step、R9 的 50% coverage 与 20% depth consistency；没有 case 特判或 threshold sweep。
一次结构审计显示 10 个 P1-ACCEPT case 的 method mask coverage=`73.65%..94.78%`，故直接进入正式 run。
若 O2 不能达到 `>=5/28`、false-safe=`0`、保留原3例并新增 actor+static/disocclusion，则停止模型接入。

H-ME1-001 在创建 run directory 或启动 GPU 前读取 ME-0 gate 时误把 authority 从 `checks.passed` 当成顶层
`passed`，触发 `KeyError`；无 run、无方法结果，登记为 `V61-F02`。H-ME1-002 只修正该 schema 路径并增加回归测试，
所有科学输入、arms、thresholds、预算与 stop rule 不变。

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
