# Motion-Proj 当前研究状态

> **文档职责**：唯一当前状态与执行授权入口。
> **最后更新**：2026-07-23
> **当前阶段**：`V7.1 / evidence_contract_done / world_state_pending`
> **证据基线**：`9722fa2`（V7 feasibility 收口提交）
> **当前决策**：`modify_method_then_scale`
> **当前计划**：[`OCCGS_RESIM_AUTORESEARCH_PLAN_V7.1.md`](OCCGS_RESIM_AUTORESEARCH_PLAN_V7.1.md)
> **当前任务**：`V7-H1-11A` pending；EV-10 gate 已完成，下一轮从坐标、schema 与 actor registry 开始。
> **执行授权**：用户于 2026-07-23 明确指令执行 V7.1；本轮按启动合同只完成 EV-10，现已解锁 H1-11A。
> **启动代码状态**：commit `b48130d1e48b3964875def1800ae4ccbf7da161c`；dirty fingerprint
> `37827824e789f55ec066f0c0e807762f451afba2fe7f6c1a1d45f09fef9fe414`；启动时唯一未跟踪文件为本计划，
> SHA256 `add6c69da67e7e6eac192ee069d5765fe5b5c310e7409b60de33fd655af18bbb`。
> **EV-10 实现提交**：`3590558cd1ef3644f10c1b981366c3ccce9cd580`；证据索引
> `/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json`；contract smoke
> `v71_v7-ev-10__smoke__s0__20260723T141019751134Z__7d97212f`。

正式数值以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 和实际产物为准；历史路线与完整旧账本见
[`archive/2026-07/README.md`](archive/2026-07/README.md)。

## 1. 一句话结论

V7 已证明单张 4090 上可以完成“nuScenes 三场景预处理 → StreetGS object-centric 重建 → actor 轨迹改写 →
反事实 RGB/depth 渲染 → 局部 hard composition”的工程闭环；但尚未证明 occupancy 带来方法增益、局部补全优于
弱基线，也未证明合成数据有下游收益。因此当前不能 scale，也不能把 feasibility 写成论文假设通过。

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
| H1：occupancy anchor 提高 actor edit 合法性 | open | occupancy 真正进入编辑/可见性/标签链；与 kinematic-only、naive transform 做 matched ablation |
| H2：显式 disocclusion mask 使局部补全既局部又有效 | open | 几何生成 mask；在伪缺失可测区域比较 no-completion/Telea/局部生成，并测时序、深度与 identity |
| H3：合法反事实数据带来下游收益 | open | scene-disjoint 的 R / R+naive / R+OccGS / R+OccGS+completion 对照和多 seed 任务指标 |

## 4. 下一步顺序

| ID | 状态 | 目标 | 解锁条件 |
|---|---|---|---|
| `V7-EV-10` | done | 建立 V7 retrospective evidence index，明确缺失 manifest/terminal marker，不伪造历史 provenance；为所有新 run 接入正式 run contract | 1,610 个旧文件已逐文件索引；25 项测试与正式 smoke 通过 |
| `V7-H1-11` | pending | occupancy 接入约束、可见性与同步标签；在相同 scene/actor/轨迹幅度上做公平消融 | occupancy 方法相对 matched baselines 有预注册、非循环的合法性收益 |
| `V7-H2-12` | pending | 用 geometry/ray visibility 构建 disocclusion mask，验证局部 completion | 不只 outside=0，还要在可测 pseudo-hole 上改善 inside quality、temporal、depth/identity |
| `V7-H3-13` | pending | 先完成下游 smoke，再做 scene-disjoint utility 实验 | OccGS 在等样本量、多 seed 下优于 real-only 与 matched naive GS |
| `V7-SCALE-14` | blocked | 扩 scene、baseline 与 seed | 仅在 H1 与 H3 通过、瓶颈确认为吞吐后解锁 |

详细协议、停止条件和单卡预算见当前 V7.1 计划。当前最高信息增益路径是 `H1-11A → H1-11B`，不是先增强
Telea、扩大 scene 数或切换双卡。

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
- 未经新的明确执行指令，不自动启动实验、不 push、不扩到双卡。

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
