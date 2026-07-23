# OccGS-Resim V7 当前研究计划

> **归档状态**：`SUPERSEDED_BY_V7.1 / H1_REJECTED`
> **归档日期**：2026-07-24
> **权威当前入口**：[`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md)
> **说明**：本文是 V7 feasibility 后的过渡计划，已先被 V7.1 取代，随后 H1-11D 正式 rejected。
> 其中 `planned`、`current task` 和执行授权均已失效。

> **工作名称**：Occupancy-Anchored Object-Centric Gaussian Resimulation
> **计划版本**：V7 feasibility 后续整理版
> **最后更新**：2026-07-22
> **状态**：`planned`；后续实验尚未启动
> **当前决策**：`modify_method_then_scale`
> **硬件目标**：先在单张 RTX 4090 24 GB 上验证；不默认双卡
> **当前任务**：`V7-EV-10`，随后为 `V7-H1-11`
> **历史执行计划**：[`archive/2026-07/v7-feasibility/OCCGS_RESIM_AUTORESEARCH_PLAN_V7_EXECUTED.md`](archive/2026-07/v7-feasibility/OCCGS_RESIM_AUTORESEARCH_PLAN_V7_EXECUTED.md)

当前状态与执行授权只看 [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)。本文件定义后续研究问题、实验顺序和门禁，
不把 feasibility 产物预先写成方法结论。

## 1. Executive decision

V7 第一轮已打通以下工程闭环：

```text
nuScenes scene preparation
→ object-centric StreetGS reconstruction
→ actor trajectory rewrite
→ counterfactual RGB/depth render
→ local hard composition
```

这足以保留 OccGS-Resim 路线，但还不足以形成论文主张。当前最重要的三条事实是：

1. occupancy 已构建，但尚未进入 editor、visibility 或 completion mask；
2. C0 只有机器规则筛选，没有用户人工 verdict，也没有完整标签重生；
3. U0 只有约束/渲染 proxy，没有下游任务收益。

因此下一轮不扩规模，先验证“occupancy 是否真的改变方法结果”。

## 2. 研究问题与假设

> 对真实驾驶日志进行 actor 轨迹编辑时，能否把 occupancy、object-centric Gaussian nodes、visibility 和同步标签
> 绑定为同一显式世界状态，生成比 matched naive GS edit 更合法、更可补全、且对下游训练更有用的反事实数据？

| 假设 | 当前证据 | 相关约束 | 状态 |
|---|---|---|---|
| H1：occupancy anchor 提高编辑合法性与标签一致性 | 只有独立 occupancy artifact 和 kinematic editor；未做集成/消融 | `RF-05/08/16`，`V7-RISK-01/02/07` | open |
| H2：geometry-derived disocclusion mask 让局部补全既局部又有效 | 只有 RGB-diff mask + Telea；outside=0 由 hard composition 保证 | `RF-05/06`，`V7-RISK-03` | open |
| H3：合法反事实数据提高长尾下游任务 | 只有 3-scene proxy；无 detector/event 训练 | `RF-09/16`，`V7-RISK-04/06` | open |

H1 不是重跑旧 2D point/RGB target：它把 occupancy、actor node、visibility 与 label writer 绑定为同一显式状态；
H2 不再调 RGB-diff mask 或 shared diffusion loss，而使用几何可见性和可测 pseudo-hole；H3 不再用 scorer、像素差或
accept rate 代理收益，而运行 matched、scene-disjoint 的真实下游任务。这些结构变化是重开相关风险的必要条件，
但仍必须通过下述门禁，不能仅凭方法名称变化晋级。

论文创新边界保持不变：如果最终只有 StreetGS 复现和手工移动车辆，应作为基础设施结果，而不是 OccGS 方法贡献。

## 3. 已完成的 feasibility 基线

| 模块 | 冻结事实 | 后续不得误写成 |
|---|---|---|
| 数据 | mini 003/004/005；3 前向相机；8 秒；10 Hz 插值 | 大规模或 scene-disjoint 结论 |
| B0 | 3/3 StreetGS 训练完成；S0/S1/S2 test PSNR 25.60/20.18/25.37 | 方法优于其他重建框架 |
| O0 | 200×200×16 的 LiDAR+box occupancy；unknown 保留 | occupancy 已参与编辑决策 |
| S0 | V1/V2=0.8 m、V3=1.6 m 横向 edit；V4 极端负例被规则拒绝 | occupancy-certified edit |
| C0 | 机器可见 case 46/62；按效应排序 top-24 为 24/24 | 用户人工 24/24 |
| L0 | Telea + hard composition；12 帧 outside-mask L1=0 | geometry-guided completion 提升 |
| U0 | accepted edit 有 RGB signal，极端 V4 被拒绝 | 下游数据收益 |

完整 feasibility 审计见 [`OCCGS_FINAL_REPORT.md`](OCCGS_FINAL_REPORT.md)，实验数值见
[`EXPERIMENTS.md`](EXPERIMENTS.md)。

## 4. `V7-EV-10` — 证据与运行契约修复

### 4.1 目标

既有 V7 run 是可定位的 retrospective evidence，但缺少本仓库协议要求的 `manifest.json`、`resolved.yaml` 和
终态标记。不得事后伪造原始 provenance；先建立诚实的 evidence index，并保证后续 run 全部 fail-closed。

### 4.2 输出

```text
runs/occgs_resim/V7_EVIDENCE_INDEX.json
configs/resim/v7_run_contract.yaml
resim/runtime helpers or wrappers
tests for unique run ID / required artifacts / terminal marker
```

evidence index 对每个既有目录记录：

- `evidence_mode=retrospective`；
- 可用的 config、metrics、checkpoint、报告与文件 hash；
- 缺失字段明确为 `missing`，不得推断 seed/fingerprint；
- 代码近似基线 `9722fa2`，并注明不是运行开始时的 immutable manifest。

### 4.3 Gate

- 既有 B0/O0/S0/C0/L0/U0 每项都可从 index 定位到原始证据；
- 新 run 若缺少 config、fingerprint、metrics/summary 或唯一终态标记则测试失败；
- 不修改既有数值、不补写人工 verdict、不伪造完成时间。

`EV-10` 对应 `V7-RISK-05`，只修复证据可信度，不产生 H1/H2/H3 结果。

## 5. `V7-H1-11` — Occupancy 真正进入方法

### 5.1 研究目标

把 O0 从“旁路产物”变成 editor 与 renderer 的显式约束源，并在完全 matched 的轨迹提案上证明增益。

### 5.2 最小实现

1. 建立 `actor true_id ↔ RigidNodes model_idx ↔ occupancy instance_id` 的版本化映射；
2. 对每帧 edited box 查询 `unknown/free/static/dynamic`，输出 collision、ground、road/free-space 与 visibility certificate；
3. 同步重生 edited 3D box、projected 2D box、instance mask、depth visibility 与 occupancy instance grid；
4. C0 渲染读取同一 world-state record，禁止 editor、renderer、label writer 各自维护不同 pose；
5. 所有插值状态保留 `provenance=interpolated`。

### 5.3 公平消融

使用同一 scene、actor、时间窗、轨迹形状和幅度，禁止再用故意撞向 ego 的 V4 充当唯一 naive baseline：

| 组别 | 约束 |
|---|---|
| A | naive rigid transform；只保证 pose 可写入 |
| B | kinematic-only；复现当前 S0 规则 |
| C | kinematic + pairwise distance |
| D | C + occupancy/free-space/visibility certificate（OccGS） |

首轮单卡 pilot：3 scenes，至少 2 个可映射 vehicle/scene，5 个冻结 edit magnitudes，形成至少 30 个 matched
trajectory proposals。若 actor 覆盖不足，状态记 `blocked`，不得改成只挑成功 actor。

### 5.4 指标

Primary：

- collision / occupied-space violation rate；
- road/free-space support 与 unknown intrusion；
- edited pose adherence；
- RGB-depth-instance-box 共位误差；
- valid edit yield 与 per-scene worst case。

Safeguards：

- 所有组使用相同 proposal 和 render budget；
- evaluator 不以“是否被本方法 accept”作为唯一标签，避免循环论证；
- 单独报告 unknown，不把 unknown 当 free；
- 机器 screen 只解锁后续人审，不替代用户 verdict。

### 5.5 Gate

只有同时满足以下条件才将 H1 标为 supported：

- D 相对 B/C 在 matched proposals 上显著降低几何/可见性违例，且 95% bootstrap CI 不跨 0；
- valid edit yield 不因只拒绝困难案例而坍缩，coverage 至少为 matched pool 的 70%；
- 同步标签共位不劣于 B/C，所有场景 worst case 均可报告；
- 若需要人工 legality 结论，先生成完整盲审包和提示词，由用户填写 verdict。

失败则记录新的 V7 research 结论；不得通过更换极端负例、降低 coverage 或只报 top-k 挽救。

## 6. `V7-H2-12` — Geometry-derived localized completion

### 6.1 前置条件

仅在 `H1-11` 至少完成 world-state/visibility/label chain 后开始。当前 `|V-V0|` RGB mask 不再作为主方法 mask。

### 6.2 方法与基线

- 由 edited/source depth、actor footprint、ray visibility 与 unknown state 生成 disocclusion mask；
- 比较 no completion、Telea、单帧 local inpaint、geometry-conditioned local video completion；
- 所有方法最终都使用同一 hard composition，因此 outside-mask=0 只作实现检查；
- 在真实已知区域构造 pseudo-hole，使 inside reconstruction 有可计算真值。

### 6.3 Primary endpoints

- pseudo-hole LPIPS/PSNR 与边界 seam；
- depth ordering、instance consistency、identity 与 temporal flicker；
- mask precision/recall 和 unknown/free/occupied 分层结果；
- 每种方法的显存、耗时与失败率。

### 6.4 Gate

局部生成必须相对 no-completion/Telea 在 inside quality 和 temporal endpoint 上有一致收益，且 depth/instance
不劣、outside exact。只通过 outside=0 不算 H2 pass。

## 7. `V7-H3-13` — 下游数据效用

### 7.1 两阶段策略

1. **U0-A pipeline smoke**：在当前三场景上验证标签格式、训练入口和评估闭环，只产生工程结论；
2. **U0-B formal utility**：补足 scene-disjoint 数据覆盖后，运行等样本量、多 seed 的正式对照。

正式组别固定为：

```text
R: real only
R+N: real + matched naive GS edits
R+O: real + occupancy-certified OccGS edits
R+O+C: real + OccGS + validated completion
```

优先选择能明确覆盖 cut-in/merge strata 的轻量事件分类或 camera 3D detection 任务。不得用当前约束 accept rate 或
RGB max difference 代替下游任务。

### 7.2 Gate

- scene-disjoint train/val/test，组间 real data、合成数量与优化预算一致；
- 至少 3 seeds，报告均值、置信区间、per-stratum 和最坏场景；
- `R+O` 必须优于 R 与 `R+N`；completion 的增量贡献单独比较；
- 若现有 mini sweep 覆盖不足以形成有效 split，标 `blocked` 并先补数据，不在 3 scenes 上宣称 H3。

## 8. `V7-SCALE-14` — 扩规模触发条件

只有以下全部满足才扩到 10+ scenes 或考虑多卡并行：

- H1 matched ablation 通过；
- H3 至少有一个正式下游 endpoint 通过；
- 单卡瓶颈确认为吞吐而非显存或方法失败；
- 数据 sweep/annotation 覆盖和磁盘预算已审计；
- 新规模仍使用 scene-disjoint split 和完整 run contract。

双卡优先用于独立 scene/baseline/seed 并行，不默认 DDP，也不能用来掩盖单卡 OOM 或低 candidate yield。

## 9. 单卡资源与停止规则

### 9.1 单卡预算

- 首轮保持 3 scenes、3 cameras、8 seconds、vehicles only；
- 单 run 峰值显存目标 `<22 GB`，硬顶 24 GB；
- 大写盘前保留至少 30 GiB 可用空间；
- 未通过 H1 前不下载全量 Occ3D、Waymo/PandaSet 或大视频生成权重。

### 9.2 停止条件

- world-state 映射不能稳定覆盖至少 2 actors/scene；
- occupancy 只通过拒绝绝大多数 proposal 提高表面合法率；
- 标签共位无法从同一 pose/state 重生；
- geometry mask 相对 RGB-diff 无可测精度，或 completion 只满足构造性的 outside=0；
- 下游正式 split 不足，或 `R+O` 不优于 matched `R+N`；
- 任何方法需要删除 hard safeguards、只报 top-k 或偷用 future actor truth 才能通过。

## 10. Run 与文档协议

每个正式 run 必须包含：

```text
manifest.json
resolved.yaml
code / third-party / data fingerprints
scene IDs, actor IDs, camera/time range, seed
metrics.jsonl
summary.json
checkpoint（如适用）
唯一终态标记：COMPLETE / FAILED / REJECTED / BLOCKED
```

状态只使用 `pending/running/blocked/done/rejected`。每个 gate 后依次更新：

1. [`EXPERIMENTS.md`](EXPERIMENTS.md)；
2. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)；
3. 若出现新负结论或重开边界，更新 [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)；
4. 研究 commit 正文写明 task ID、split、seed、fingerprint、证据路径与验证命令。

人工 verdict 只能由用户或其指定评审者填写。机器规则、agent 目视检查和 top-k screen 均不得写成 human pass。

## 11. 与历史路线的边界

- 不重开 `RF-01`–`RF-18`；完整旧账本见
  [`archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md`](archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md)。
- 不回到 2D diffusion latent 作为唯一世界状态；
- 不用 future actor boxes 定义“自由生成”物理正确性；
- 不把 layout adherence、像素差、PSNR 或 machine legality 单独写成下游收益；
- 不从归档计划中的“下一步”恢复旧训练。

## 12. 决策路径

```text
V7-EV-10 evidence contract
→ V7-H1-11 occupancy integration + matched ablation
   → fail: 记录结论，停止 scale
   → pass: V7-H2-12 geometry-derived completion
          + V7-H3-13 downstream utility
             → H3 fail: 保留基础设施，不形成数据生成主张
             → H3 pass: V7-SCALE-14
```

当前优先级是 `EV-10 → H1-11`。增强 completion、训练下游、扩 scene 和多卡均不得越过这一顺序。
