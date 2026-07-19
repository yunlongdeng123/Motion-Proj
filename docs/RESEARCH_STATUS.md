# Motion-Proj 当前研究状态

> **文档职责**：唯一当前状态与执行授权入口。
> **最后更新**：2026-07-19
> **研究基线**：`43eda43878b5104cd043c4d8fee2ab177a356858`（V6 起草时 HEAD；V5 证据基线仍见终报）
> **当前计划**：[`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md)
> **归档计划**：[`MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md`](MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md)（`done`，不得恢复已拒绝任务）
> **当前状态**：`running`
> **当前任务**：V6 `C1B-00` smoke/shape/确定性正在执行；实现 commit `1477e54`；正式证据根
> `/root/autodl-tmp/runs/resim_c1_v6/C1B-00/`；下一步仅在该 gate 通过后进入 `C1B-01`
> **最终决策**：`C1`；执行入口为 V6 的 `C1A → C1B → C1P → C1S`（单卡）
> **硬件**：单张 RTX 4090 24 GB；数据盘 128G 不可扩容

本文只写当前决策、执行边界和稳定里程碑。正式数值以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 与对应
run 为准；为什么不能重复旧尝试见 [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)。

## 1. 当前研究问题

已拒绝的两条旧路线是：

1. **Explicit dynamics projection / endpoint distillation**：当前 synthetic target、RGB/VAE
   counterfactual 和 shared temporal LoRA 组合未通过动力学、target legality 与 locality 门禁；
2. **SVD internal sibling physics preference**：旧 P-UNC forced-binary 标签器未通过人工可信性复核；
   common-support selective partial order 虽消除了校准集 false-strict，但候选 strict yield 过低，
   唯一 earlier-fork fallback 又触发首帧/质量门禁。

V5 不恢复这两条路线，而检验两个独立的新问题：

- **Route A**：真实训练视频中的 ego-induced motion 与 actor residual motion 是否在冻结 SVD 表示中可辨，
  并能否通过 training-only auxiliary supervision 稳定进入 temporal LoRA；
- **Route B**：冻结 SVD 的自然独立 rollout 分布是否已经包含人工与独立 evaluator 都认可的更优 motion
  sample，从而为后续 condition-relative AWR + real SFT 提供 support。

Route A 的 A1 scan 已给出分解结论：冻结表示中的 sparse ego-flow signal 可泛化，但当前 compact linear
probe 无法从 driving-specific ego/actor entanglement 中可靠恢复 actor residual，并严重违反 stationary
safeguard。因此 actor 路线已拒绝，只保留 ego-only representation baseline；A1-CONFIRM 与 A2 不执行。

Route B 也已关闭：128 个独立 natural rollouts 中，严格 anti-collapse 下仅 `1/16` conditions 有两条合法
候选；P-UNC-best 对 random/Base 的 CoTracker win-credit 均为 `41.67%`，并出现 low-motion/catastrophic
selection。18 dB checker 虽过严，但只读 sensitivity 证明移除它后仍仅 `4/16` diverse；Route B 的拒绝稳健。

Route C 只读迁移审计已经完成。ReSim `exp0_no_carla` 是唯一同时具备公开 checkpoint、显式 future ego
trajectory、历史帧预测、且不要求 future actor boxes 的候选，因此选择 `C1`；VISTA 为 fallback，OpenDWM 与
MagicDrive-V2 只作 layout/geometry baseline。完整边界见
[`BACKBONE_MIGRATION_AUDIT.md`](BACKBONE_MIGRATION_AUDIT.md)。

C1 源码在 `/root/autodl-tmp/third_party/ReSim`（`bf13dff...`）；独立环境
`/root/autodl-tmp/envs/resim`（`torch 2.4.0+cu121`）；权重在
`.../checkpoints/CogVideoX-2b-sat`（含 EMA、VAE、由 CogVideoX 合成的 T5）。`motionproj` 环境保留给
evaluator。尚未通过单卡 smoke、尚未训练。磁盘约 43 GiB 可用，正式候选/训练前必须按 V6 §1.3 预留峰值。
细节见 [`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md)。

## 2. 已关闭里程碑

| ID | 状态 | 已完成事实 | 当前决策 |
|---|---|---|---|
| `P2-V1-TUNE-01` | rejected | 16 个 100-step 与 4 个 300-step synthetic projection trial | 不继续旧 cache、Optuna、t10-800 或同配方调参 |
| `P2-V2-COND-00` | done / branch rejected | future-GT ego mismatch 已确认；self-estimated static V1 未过人工门槛 | 未条件化 SVD 禁用 future-GT static target |
| `P2-V2-REPLAY-05` | done | object-only generated-track replay 的 schema 与人工合理性通过 | 只保留基础设施，不外推训练收益 |
| `P2-V2-PILOT-03` | blocked | C/D/E capacity 与 single-pair locality 诊断完成 | shared temporal LoRA endpoint 不进入 rollout 或长训 |
| `F0/F1/P1` | rejected | endpoint preserve、旧 raw-feature probe、RGB/VAE target legality 完成 | 不绕过 target legality 启动旧 feature head 或生成器训练 |
| `PA0-REVIEW-00` | done | P-UNC 与 E0 既有人工 review 聚合完成 | 仅完成基础设施可信度，不产生偏好标签 |
| `PA1-BRANCH-02` | done | common-prefix siblings 通过 same-scene 结构盲审 | 只证明结构合法，不证明 physics winner 可辨 |
| `PA2-PAIR-03` | rejected | 120 conditions、53 machine pairs、48-case 人工复核完成 | 旧 P-UNC forced-binary recipe 禁止训练 |
| `PA2-UPO-03B` | done / yield blocked | tie holdout、shortcut、cycle 与 bootstrap 门禁通过 | `2/96` strict 不足以进入训练 |
| `PA2-CAND-03D` | rejected | 唯一 8-condition earlier-fork fallback 完成 | 不筛唯一 strict，不再搜索 fork/rho |
| 旧 `PA3`–`PA8` | rejected / not run | 上游没有合法且足量的 preference 数据 | 不恢复旧 DrivePO trainer、screening 或双卡计划 |
| `RP-R0-00` | done | 当前仓库、单张 4090、磁盘、权重、nuScenes、RAFT、Depth、CoTracker 与 scene split 均已核验；baseline 为 208 passed | 单卡资产足以继续 R1/A0/A1/B0；保持 30 GB 磁盘安全线 |
| `RP-LIT-01` | done | [`ROUTE_PIVOT_LITERATURE_MATRIX.md`](ROUTE_PIVOT_LITERATURE_MATRIX.md) 完成一手核查，并补入 WMReward、SARA 等最近邻 | BoN 只作 B0 ceiling；主张收紧为 driving ego–actor decomposition + uncertainty/local safeguard |
| `RP-R1-02` | done | 32 个真实 clips 确认中位 `2.0 Hz`；48 个 SVD fps 对照与 16 个 paired groups 完整；[`ROUTE_PIVOT_TEMPORAL_AUDIT.md`](ROUTE_PIVOT_TEMPORAL_AUDIT.md) 已固化 | `fps=2/4` 虽显著增大 motion，但未通过画质/轨迹/加速度 safeguard；后续冻结 `generation.fps=7`，32 个盲审 pair 仍待标注、仅作补充诊断并受保护 |
| `RP-A0-03` | machine pass / awaiting reviews | 16 scenes 上 392 个可局部化 pairs、89 tracks；AUC `0.8600`、velocity 方向 `0.9725`、ego 相关 `0.2226`、背景方向 `0.9870`；[`ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md`](ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md) 已固化 | 只解锁 A1 machine probe；12 个 panel 的人工 target legality 仍 pending、受保护，且不因 Route A machine reject 自动代填或删除 |
| `RP-A1-SCAN-04A` | rejected | 24/8 scene-disjoint clips、21 个 layer/sigma 配置；ego baseline 改善 `17.86%–25.01%`，actor 对 zero baseline 全为负且 stationary ratio `3.292–5.062`；[`ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md`](ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md) 已固化 | 保留 ego-only diagnostic；actor hypothesis、A1-CONFIRM 与 A2 停止，不调 probe 追门槛 |
| `RP-B0-05` | rejected | 16 conditions × 最多 8 natural seeds；N=8 仅 `1/16` diverse，P-UNC 对 random/Base CoTracker win-credit 均 `41.67%`；[`ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md`](ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md) 已固化 | 不做人审、不扩 N/改 CFG/降 anti-collapse，不进入 AWR/SFT；解锁 Route C 迁移审计 |
| `RP-C0-07` | done | 固定 5 个官方 repo HEAD 与 VLA-World 项目页，审计 6 个候选；ReSim 的官方 nuScenes schema 与本机 raw data 匹配；[`BACKBONE_MIGRATION_AUDIT.md`](BACKBONE_MIGRATION_AUDIT.md) 已固化 | 选择 `C1`，但只晋级下一阶段单卡 feasibility；未下载/推理/训练 |
| `RP-D0-08` | done | [`ROUTE_PIVOT_FINAL_REPORT.md`](ROUTE_PIVOT_FINAL_REPORT.md) 汇总全部门禁、review pending、停止项与最多 3 个后续实验 | V5 关闭；新动作必须由下一份预注册计划授权 |

## 3. V5 稳定任务表

| ID | 当前状态 | Gate | 通过后的动作 |
|---|---|---|---|
| `RP-R0-00` | done | 仓库、环境、资产与文档基线 | 已解锁一手文献、R1、A0、B0 |
| `RP-LIT-01` | done | 最近邻一手文献与创新边界 | 已收紧 A/B novelty，不单独晋级方法 |
| `RP-R1-02` | done | 真实时间采样与 SVD fps audit | 已冻结后续 `generation.fps=7`；review material awaiting，不阻塞 A0 |
| `RP-A0-03` | awaiting_reviews | 真实 ego–actor target legality | machine pass 已解锁 A1；human gate 并行 pending |
| `RP-A1-SCAN-04A` | rejected | 24/8 clips frozen feature scan | 0 个合法候选；不进入 confirm |
| `RP-A1-CONFIRM-04B` | rejected / not run | 64/16/16 scene-disjoint confirm | scan dependency 未满足 |
| `RP-B0-05` | rejected | natural-rollout best-of-N ceiling | machine gate failed；不生成人审、不自动长训 |
| `RP-A2-06` | rejected / not run | auxiliary-alignment capacity | A1 confirm dependency 未满足 |
| `RP-C0-07` | done | action-conditioned backbone 迁移审计 | `C1 = ReSim exp0_no_carla feasibility` |
| `RP-D0-08` | done | 最终路线决策与报告 | V5 已固化并关闭 |

Route A 与 Route B 的机器任务相互独立。一条路线失败后仍继续另一条。人工 review 不阻塞独立机器 gate；
若最终只缺人工 verdict，则相应任务标为 `awaiting_reviews` 并一次性交付完整提示词与材料。

## 4. V5 已完成执行边界

V5 已按下列固定顺序执行完毕；该顺序现在是证据记录，不是新的任务队列：

```text
R0 + literature
→ R1 temporal/fps audit
→ A0 target legality
→ A1 scan/confirm（按门禁）
→ B0 natural rollout ceiling（与 A 独立）
→ A2（仅 A1 pass）或 Route C（仅 A/B rejected）
→ D0 final report
```

V6 授权范围：按 [`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md)
执行 `C1B-00`…`C1B-03`；`C1P` 依赖 C1B 机器与人工 gate，`C1S` 依赖 C1P 机器与人工 gate。
允许为 C1B 建立独立 `resim` conda 环境；本轮所有阶段都限单张 4090。

本轮明确禁止：

- 继续 denoising-prefix sibling、fork、rho、CFG branch 或 candidate 数搜索；
- 使用旧 53 pairs、旧 local labels、UPO 的 2 个 strict 或 fallback 唯一 strict 训练；
- 实现旧 DrivePO tube-DPO、vanilla DPO、在线 PPO/GRPO 或大型 reward model；
- 对完整采样链反传；
- future actor boxes/tracks 进入条件或自由 rollout 正式 evaluator；
- 把 image-plane acceleration 称为真实世界加速度；
- 把 `exp0_no_carla` 称为含 CARLA 非专家能力的完整 ReSim；
- 自动填写人工 verdict、覆盖正式 run、自动 push 或在任何阶段切换双卡；
- C1B 未过时启动 C1P 数据生成或任何 adapter/LoRA。

允许在真实训练视频的 representation target/probe 中使用 nuScenes ego pose、3D annotation 与 LiDAR；这些
信号不得进入 generated-rollout 正式 evaluator，也不得描述为 inference-time future condition。

## 5. 研究门禁原则

V5 必须逐项遵守 [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：

1. 先证明 target/preference 存在、合法、可观察且 scene-disjoint，再训练；
2. 真实时间戳参与所有速度/加速度归一化，SVD `fps` 作为版本化 micro-conditioning；
3. generated rollout 的训练侧 scorer 与正式 CoTracker3 evaluator 隔离；
4. low-motion、time-slow、track dropout、camera nuisance、画质与首帧损坏 fail closed；
5. single-pair、machine pass 或 feature probe pass 只解锁下一门禁，不产生 rollout/论文结论；
6. localized auxiliary loss 必须实测 gradient、outside、frame-0、motion amount 与 held-out transfer；
7. 工程失败使用新 run ID 修复；只有通过预注册检查后的结果才能标为 research rejected。

## 6. 可复用资产

| 资产 | 已验证范围 | 不得解释为 |
|---|---|---|
| official SVD generation parity | matched inputs 下与 Diffusers pipeline exact | preference 或训练收益 |
| scene-level split 与 provenance | scene/clip 无泄漏、fingerprint 可追溯 | 当前数据足以训练 |
| real nuScenes geometry/annotations | 真实训练视频上的 target/probe | generated rollout 的 future condition 或 evaluator truth |
| generated point tracks / P-UNC | point-space support、visibility 与部分运动不变量 | 合法 RGB target 或可靠 winner |
| CoTracker3 evaluator | 当前协议内 rerun 与扰动排序稳定 | 绝对物理标定 |
| common-support UPO oracle | 旧 tie holdout 上低 false-strict、shortcut reaudit 通过 | 足量 preference yield |
| manifest / fingerprint / atomic runtime | 正式 run 可追溯与 fail-closed | 方法结论本身成立 |

## 7. 事实源优先级

发生冲突时按以下顺序处理：

1. 正式 run 中不可变的 `manifest.json`、`resolved.yaml`、指标、人工 verdict 与终止标记；
2. [`EXPERIMENTS.md`](EXPERIMENTS.md) 的实验登记；
3. 本文件的当前状态与授权；
4. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 的跨实验解释与重开条件；
5. 下一份正式计划中尚未执行的设计；
6. `docs/archive/` 中的历史计划、报告和提示词。

每个正式 gate 完成后必须更新本文件与 `EXPERIMENTS.md`；新负结论或重开边界同时更新
`RESEARCH_FAILURES.md`。归档材料不得覆盖本文件。

## 8. 当前唯一合法入口（V6）

完整协议见 [`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md)：

```text
C1B-00 smoke/shape/确定性  →  C1B-01 proxy 校准  →  C1B-02 10-context action screen  →  C1B-03 人工盲审
    ↓ pass
C1P preference support（依赖 C1B 人工 pass）
    ↓ machine + human pass
C1S single-GPU learning（仍限单张 4090）
```

权重下载完成后的第一项正式动作是 `C1B-00`，不是推理或训练。任何上游 fail 都直接停止，不靠扩卡、
降 safeguard、打开 future boxes 或增加 candidate 数挽救。

## 9. 存储维护状态

`STORAGE-RETENTION-20260719` 是已完成的基础设施维护，不是新研究任务。清理清单由 commit `2d52056`
先行固化；随后精确删除 75 个目标，逻辑大小 `46,525,314,508` 字节，文件系统实际回收
`46,541,172,736` 字节。128G 数据盘清理完成时可用 `91,591,008,256` 字节（85.3 GiB）。此后 ReSim
权重下载会降低可用空间；执行前以实时 `df` 为准并守 30 GB 安全线。推理/训练须通过 V6 对应 gate，
不得因“已经在下权重”跳过 `C1B-00`。

以下材料明确受保护：

- `autoresearch-pa2-pair-expanded-s20260715-v1` 的 48 条正式 `reviews.jsonl` 与完整 review package；
- R1 的 32 个待标注 pair，以及 A0 v3 的 12 个待标注 panel；
- 所有 manifest、resolved YAML、metrics、summary、result、人工 verdict 和终止标记；
- UPO v1/v2 的 paired tracks、common support、bootstrap、graph 与 stress 证据；
- nuScenes、本项目环境、CoTracker3、RAFT/Depth/evaluator 资产与 C0 迁移证据。

已删除的 SVD 权重、历史 checkpoint、candidate 视频、adapter 和 feature tensor 只是二进制载荷；其清理不改变
`RF-01`、`RF-06`、`RF-09`–`RF-12`、`RF-14`、`RF-15` 的负结论、适用范围或重开条件。清理后
75/75 目标不存在，保护哈希逐项不变；JSON/JSONL/YAML 全部可解析，逐文件隔离的完整测试为 247 passed。
历史 SVD asset check 现明确报告 `non-resident`。
