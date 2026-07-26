# OccGS-Resim V7 feasibility 收口报告

> **轮次结束时间**：2026-07-21
> **审计整理时间**：2026-07-22
> **证据基线**：`9722fa2`
> **轮次决策**：`modify_method_then_scale`
> **当前后续计划**：[`OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md`](OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md)

## 1. 结论

V7 第一轮完成了单卡 object-centric Gaussian resimulation 的工程可行性验证，路线不应整体 reject；但三条核心
论文假设 H1/H2/H3 均未完成验证，当前不应 scale，也不应写成 occupancy、completion 或 synthetic utility 已经
优于 baseline。

准确决策是：

```text
保留 OccGS-Resim 路线
→ 修复证据与方法耦合
→ 先做 occupancy matched ablation
→ 再验证 completion 和 downstream utility
→ 通过后才扩规模
```

## 2. 已完成事实与证据等级

| Gate | 状态 | 已完成事实 | 证据等级 / 限制 |
|---|---|---|---|
| E0 | done | DriveStudio 单卡环境与 CUDA extensions smoke | engineering；原清单已归档 |
| G0 | done | 主框架、license、数据接口与 scene 限制审计 | engineering / compliance |
| D0 | done | mini 003/004/005 三场景数据完整，冻结 3 cameras × 8 seconds | feasibility only |
| B0 | done | 3/3 StreetGS 重建完成；test PSNR 25.60/20.18/25.37 | retrospective run；无正式 user review |
| O0 | artifact_done | LiDAR+box occupancy 构建，unknown 保留 | 尚未接入 editor/render/completion |
| S0 | prototype_done | 运动学/距离约束的 actor 轨迹编辑可生成 | 不是 occupancy-certified editor |
| C0 | machine_screen_done | 三场景 counterfactual render；46/62 machine legal；effect top-24 为 24/24 | machine-only；无用户 verdict；标签链未完整 |
| L0 | feasibility_done | Telea + hard composition；12 帧 outside-mask L1=0 | 构造不变量；RGB-diff mask；未证明质量收益 |
| U0 | partial | accepted edits 有 RGB signal；极端 V4 被拒绝 | 非 matched baseline；无下游任务 |
| D1 | done | 得出 `modify_method_then_scale` | 本报告 |

原始阶段报告与长计划见
[`archive/2026-07/v7-feasibility/`](archive/2026-07/v7-feasibility/)，V7 数值事实见
[`EXPERIMENTS.md`](EXPERIMENTS.md)。

## 3. 关键数字

### B0 reconstruction

| Scene | test PSNR | test SSIM | test LPIPS | vehicle test PSNR |
|---|---:|---:|---:|---:|
| 003 / S0 | 25.60 | 0.799 | 0.142 | 23.48 |
| 005 / S1 | 20.18 | 0.472 | 0.325 | 18.26 |
| 004 / S2 | 25.37 | 0.697 | 0.142 | 22.11 |

### C0 / L0 / U0

- C0：全可见 machine legal `46/62`；按 mean edit effect 选出的 top-24 为 `24/24`。
- L0：s0/s2 各 6 帧，outside-mask L1 `0.0`；mask 占比约 `1.27% / 1.99%`。
- U0：V1/V2 accept rate `1.0`，V3 `0.667`，极端 V4 `0.0`；`u0_full_map_pass=false`。

## 4. 三条核心假设的实际状态

### H1：occupancy 提高 actor edit 合法性 — open

O0 occupancy 与 S0/C0 代码路径当前独立；editor 没有查询 occupancy，C0 也没有基于 occupancy 生成 visibility
或完整同步标签。V4 是故意无效的极端负例，不是公平 naive GS baseline。

### H2：显式 disocclusion mask 改善局部补全 — open

现有 L0 用 V0/edited RGB 差分生成 mask，Telea 只作弱 baseline。outside-mask 为 0 由 hard composition 直接保证，
没有 inside ground truth、时序、depth、identity 或用户人工质量结论。

### H3：合成数据有下游收益 — open

未运行 camera 3D detection、occupancy prediction 或 event classifier。当前 proxy 不能替代 mAP/recall，也不能
证明 OccGS 优于 real-only 或 matched naive GS。

## 5. 证据审计发现

1. `runs/occgs_resim/` 的既有 V7 目录缺正式 `manifest.json`、`resolved.yaml` 和唯一终态标记；只能作为
   retrospective evidence，不能事后伪造 seed/fingerprint。
2. `s0_edit_summary.json` 会被单次脚本执行覆盖，当前只汇总 scene 004；三场景事实应读取独立 edit JSON。
3. C0 的 `reviews/` 是机器 screen/material，不是用户 human review。
4. O0、S0、C0、L0 目前是相邻模块，不是由同一 world-state record 驱动的闭环方法。
5. mini 三场景不足以支持外部有效性或正式下游 utility 结论，且 S1 held-out 质量明显偏弱。

这些问题不抹去已有 feasibility 结果，但会限制其论文解释。对应防重复条件见
[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)。

## 6. 下一步

| 优先级 | ID | 动作 | 为什么现在做 |
|---:|---|---|---|
| 1 | `V7-EV-10` | 建立 retrospective evidence index 与新 run contract | 先让后续结果可追溯，且不伪造旧 provenance |
| 2 | `V7-H1-11` | occupancy 接入 editor/visibility/label regeneration；做 matched ablation | 直接检验项目最核心、目前完全未证的贡献 |
| 3 | `V7-H2-12` | geometry-derived mask + completion 基线 | 避免继续把构造性 outside=0 当质量提升 |
| 4 | `V7-H3-13` | scene-disjoint downstream utility | 决定路线是否具有数据生成价值 |
| 5 | `V7-SCALE-14` | 扩 scene/seed/多卡 | 仅在 H1 和 H3 通过后解锁 |

下一轮仍以单张 4090 为首要验证环境。具体样本量、组别、指标、门禁和停止条件以当前 V7 计划为准。

## 7. 历史边界

- V1–V6 旧路线和 `RF-01`–`RF-18` 不因本轮成功重建而自动重开；
- 不回到通用 2D 视频扩散 latent 作为唯一世界状态；
- 不把 machine top-k、PSNR、RGB difference 或 constraint accept rate 写成 human validity / downstream utility；
- 不从归档报告中的“下一步”直接启动实验；当前授权只看 `RESEARCH_STATUS.md`。
