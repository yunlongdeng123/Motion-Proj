# Motion-Proj 当前研究状态

> **文档职责**：唯一当前状态与执行授权入口。
> **最后更新**：2026-07-23
> **当前阶段**：`V7.1 / H1-11D_rejected / autoresearch_stopped`
> **证据基线**：`9722fa2`（V7 feasibility 收口提交）
> **当前决策**：`reject_occgs_method_claim`
> **当前计划**：[`OCCGS_RESIM_AUTORESEARCH_PLAN_V7.1.md`](OCCGS_RESIM_AUTORESEARCH_PLAN_V7.1.md)
> **当前任务**：无 running task；11D 预注册 pilot 已同时拒绝 H1-CERT 与 H1-PROJ，按停止规则不进入
> H2/H3/scale。保留 object-centric GS、WorldState、typed-label 和 run-contract 基础设施。
> **执行授权**：用户于 2026-07-23 授权持续 Auto Research；按 gate 自动推进，直到 research reject、必须人工审核、
> 新外部授权缺失或硬阻塞。
> **启动代码状态**：commit `b48130d1e48b3964875def1800ae4ccbf7da161c`；dirty fingerprint
> `37827824e789f55ec066f0c0e807762f451afba2fe7f6c1a1d45f09fef9fe414`；启动时唯一未跟踪文件为本计划，
> SHA256 `add6c69da67e7e6eac192ee069d5765fe5b5c310e7409b60de33fd655af18bbb`。
> **EV-10 实现提交**：`3590558cd1ef3644f10c1b981366c3ccce9cd580`；证据索引
> `/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json`；contract smoke
> `v71_v7-ev-10__smoke__s0__20260723T141019751134Z__7d97212f`。
> **H1-11A 正式 run**：`v71_v7-h1-11a__pilot-3__s0__20260723T144155452295Z__0ff143d9`；
> code `766f2287e79b3cdfc877eb175776482c79c3f98c`；engineering gate `PASS`，hypothesis 未评估。
> **H1-11B 正式 run**：`v71_v7-h1-11b__pilot-3-calibration__s0__20260723T145956893820Z__b8349bc0`；
> code `002bbb499e2bf967a0b16e19c09088cef2e60ef5`；engineering/calibration gate `PASS`，hypothesis 未评估。
> **H1-11C 正式 run**：`v71_v7-h1-11c__pilot-3-interface__s0__20260723T152729207694Z__8429e9a5`；
> code `9780e391fca689c1df033b0aa09611404af583a6`；engineering gate `PASS`，hypothesis 未评估。
> **H1-11D 正式 run**：`v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc`；
> code `304407b94350ddfd17a9d4f29e43b7d1b789a326`；engineering gate `PASS`，H1-CERT/H1-PROJ
> 均为 `REJECTED`，terminal marker 为唯一 `REJECTED`。

正式数值以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 和实际产物为准；历史路线与完整旧账本见
[`archive/2026-07/README.md`](archive/2026-07/README.md)。

## 1. 一句话结论

V7/V7.1 已证明 object-centric GS、统一 WorldState、同步 typed label 与 matched evaluation 的工程闭环；
但冻结的 30-proposal pilot 中，D1 precision 仅 `0.75 < 0.80`，D2 拒绝 `30/30`、usable yield 为 `0`。
因此 occupancy certificate/repair 方法主张被预注册拒绝，不进入 H2/H3/scale，也不能把 feasibility 写成论文假设通过。

## 2. V7 当前进展

| Gate | 状态 | 已有证据 | 准确边界 |
|---|---|---|---|
| `E0-ENV-01` | done | DriveStudio、gsplat、PyTorch3D、nvdiffrast 单卡 smoke | 环境可运行；阶段清单已归档 |
| `G0-THIRDPARTY-00` | done | DriveStudio/Occ3D/SplatAD 代码、license 与接口审计 | 不代表方法新颖性成立 |
| `D0-DATA-02` | done | mini 003/004/005，3 前向相机，8 秒训练窗，10 Hz 处理 | 本机完整 sweep 只覆盖 mini 10 scenes，外部有效性受限 |
| `B0-RECON-03` | done | 3/3 StreetGS 训练完成；test PSNR 为 25.60 / 20.18 / 25.37 | object-centric 重建可行；S1 held-out 质量偏弱；无正式 user review |
| `O0-OCC-04` | artifact_done | 三场景 LiDAR+box occupancy，unknown 保留 | occupancy 尚未接入 S0 约束、C0 可见性或 L0 mask，H1 未验证 |
| `S0-EDIT-05` | prototype_done | raised-cosine 横向编辑；V4 极端负例被运动学/距离规则拒绝 | 当前是几何启发式 editor，不是 occupancy-certified editor |
| `C0-CF-06` | machine_screen_done | 3 scenes 渲染；全部可见 case 46/62 合法；按效应选出的 top-24 为 24/24 | 结论来自机器规则；没有用户人工 verdict；未完整重生 semantic/instance/box |
| `L0-COMP-07` | feasibility_done | Telea + hard composition，12 帧 outside-mask L1=0 | 0 泄漏由构造保证；mask 来自 RGB 差分而非 occupancy/ray visibility；H2 未验证 |
| `U0-UTILITY-08` | partial | 3-scene 约束/渲染 signal proxy | naive V4 是极端无效负例；未跑 detector/event task；H3 未验证 |
| `D1-DECIDE-09` | done | feasibility 轮次收口 | 决策为先改方法再扩规模 |

阶段详情见 [`OCCGS_FINAL_REPORT.md`](OCCGS_FINAL_REPORT.md)；整理前长计划和逐 Gate 报告已移至
[`archive/2026-07/v7-feasibility/`](archive/2026-07/v7-feasibility/)。

## 3. 核心假设状态

| 假设 | 状态 | 还缺什么 |
|---|---|---|
| H1：occupancy anchor 提高 actor edit 合法性 | rejected | H1-CERT precision 未达门槛；H1-PROJ 以 30/30 拒绝、0 usable yield 失败 |
| H2：显式 disocclusion mask 使局部补全既局部又有效 | not_triggered_after_H1_reject | 11D 未通过方法前置门禁；不继续实例化方法实验 |
| H3：合法反事实数据带来下游收益 | not_triggered_after_H1_reject | proposal bank 无 0→1 positive/same-actor pair，且 H1 已拒绝 |

## 4. 下一步顺序

| ID | 状态 | 目标 | 解锁条件 |
|---|---|---|---|
| `V7-EV-10` | done | 建立 V7 retrospective evidence index，明确缺失 manifest/terminal marker，不伪造历史 provenance；为所有新 run 接入正式 run contract | 1,610 个旧文件已逐文件索引；25 项测试与正式 smoke 通过 |
| `V7-H1-11` | rejected | 11A/11B/11C 工程 gate 通过；11D H1-CERT/H1-PROJ 均拒绝 | 不解锁方法 claim |
| `V7-H2-12` | not_triggered | 11D primary gate 已触发立即停止；高成本 render/recovery 不实例化 | 需要新的研究决策与重新预注册，不属于本轮自动授权 |
| `V7-H3-13` | not_triggered | H1 已拒绝且冻结 proposal bank 无合格 positive pair | 需要新路线而非继续当前 OccGS 配方 |
| `V7-SCALE-14` | blocked | 扩 scene、baseline 与 seed | 仅在 H1 与 H3 通过、瓶颈确认为吞吐后解锁 |

详细协议、停止条件和单卡预算见当前 V7.1 计划。本轮 Auto Research 已在预注册 reject 终点停止；不得自动
通过调低 coverage、删除 S1、改 proposal 或把拒绝样本算作零违规来重开。

## 5. 当前硬边界

- V1–V6 已拒绝路线不因归档而重开；尤其 `RF-18` 的 ReSim `exp0_no_carla` action-response 结论仍有效。
- C0 的机器筛选不得写成用户人工评测；需要人审时必须先交付完整提示词和独立材料，再由用户填写 verdict。
- 当前 O0 产物存在不等于 occupancy 已进入方法；未做 matched ablation 前不得使用“occupancy improves legality”表述。
- hard composition 的 outside-mask L1=0 是实现不变量，不是补全质量收益。
- U0 proxy 不等于 detector mAP、事件召回或训练数据效用。
- V7 既有 run 缺少正式 `manifest.json`、`resolved.yaml` 与终态标记；只能记为 retrospective evidence。
- 后续新 run 必须使用唯一 run ID，并保存 config、fingerprint、metrics、summary、checkpoint 与终态标记。
- V7.1 新 run 必须通过 `configs/resim/v71/run_contract.yaml`；EV-10 smoke 的 `COMPLETE` 只表示工程合同通过，
  不表示 H1/H2/H3 hypothesis supported。
- H1-11C 的 `COMPLETE` 同样只表示 state→renderer→typed-label 接口通过；H1-CERT/H1-PROJ 必须由 11D
  的外部 evaluator 与十条 pilot gate 分别裁决。
- 持续 Auto Research 授权不包含 push、双卡、全量数据或大型权重下载；这些仍服从计划中的独立 gate。

## 6. 事实源优先级

发生冲突时按以下顺序处理：

1. 实际 run 产物、配置、checkpoint 与原始指标；
2. [`EXPERIMENTS.md`](EXPERIMENTS.md) 的 V7 登记；
3. 本文件的当前状态与执行边界；
4. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 的风险和防重复条件；
5. 当前 V7 计划中尚未执行的设计；
6. `docs/archive/` 中的历史计划、报告和提示词。

## 7. 当前证据根

- reconstruction：`/root/autodl-tmp/runs/occgs_resim/b0_recon/occgs_b0/`
- occupancy：`/root/autodl-tmp/data/occgs/occupancy/{003,004,005}/`
- edits：`/root/autodl-tmp/data/occgs/scene_specs/s0_edits/`
- counterfactual：`/root/autodl-tmp/runs/occgs_resim/c0_cf/`
- legality screen：`/root/autodl-tmp/data/occgs/reviews/c0_legality/c0_legality_screen.json`
- completion：`/root/autodl-tmp/runs/occgs_resim/l0_comp/`
- utility proxy：`/root/autodl-tmp/runs/occgs_resim/u0_screen/u0_proxy_v1.json`
