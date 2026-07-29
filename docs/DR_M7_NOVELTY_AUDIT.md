# DR-M7 唯一假设与 novelty 审计

- 审计日期：2026-07-29
- 输入 M6：`/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`
- 正式 M7：`/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`
- 机器执行状态：`done`
- 研究裁决：`rejected`

## M6 稳定失败

六个官方场景的冻结 SAM pseudo ID 最长支持帧分别为：

```text
scene-0230 1
scene-0242 6
scene-0255 1
scene-0295 1
scene-0518 2
scene-0749 1
```

全部低于预注册的 `≥20/60` 门槛；processed scene 没有训练前冻结的持久 vehicle track artifact。与此同时，
六个 AD-GS 60k checkpoint 的 `point_cloud.ply` 都只保留 `obj∈{0,1}`，没有实例 ID。稳定失败因此是：

> `persistent_object_identity_unavailable`

它在 `6/6` scenes 重复，满足 M7 的“至少 3 scenes”前置门禁。M6 的对象编辑、pseudo-hole 与噪声端点全部按
协议记为 `ABSTAIN`，没有从 coverage 分母删除。事后几何重关联没有回填 M6 baseline。

## 唯一候选假设

按照权威计划第 13 节决策表，只考察 A：

> 从 SAM pseudo masks 恢复跨时间实例身份，将 Gaussian 绑定到持久 actor，并以置信度/ABSTAIN 约束轨迹编辑。

没有同时注册 B/C/D，也没有在看到 proposed 结果后选择 endpoint。

## 官方 novelty 对照

| 工作 | 已有能力 | 与候选 A 的关系 |
|---|---|---|
| [InstDrive](https://arxiv.org/abs/2508.12015) | SAM pseudo masks、2D/3D instance identity、voxel consistency、离散 codebook、动态驾驶交互编辑 | 直接覆盖身份恢复与 3D Gaussian instance binding |
| [Director](https://arxiv.org/abs/2604.01678) | 时间对齐 instance masks、4D Gaussian identity consistency、flow 稳定运动 | 直接覆盖时序身份一致性与运动绑定 |
| [OmniRe](https://openreview.net/forum?id=9cwxZxJixB) | actor scene graph、canonical vehicle representations、动态对象仿真 | 直接覆盖持久 actor nodes 与对象中心表示 |
| [HorizonForge](https://arxiv.org/abs/2602.21333) | editable Gaussian Splats/Meshes、任意车辆轨迹与对象操作、编辑 benchmark | 直接覆盖轨迹编辑与车辆操控 claim |
| [G²Editor](https://arxiv.org/abs/2508.20471) | reposition/insert/delete、3D layout 引导的遮挡区重建 | 直接覆盖对象编辑、删除与遮挡补全 claim |

官方来源于 2026-07-29 重新核对。详细逐项矩阵和冻结 URL 在 M7 run 的 `novelty_matrix.json` 与
`metrics.jsonl`。

## 裁决

novelty gate 不通过。候选 A 的主机制已被直接覆盖，剩余差异只是把这些能力适配到 AD-GS 当前 checkpoint，属于
重要工程/负结果，不足以注册一个新的方法主张。候选里的 confidence/ABSTAIN 是评测与安全护栏，不作为独立
技术 novelty 补回已重合的身份、actor binding、时序一致性和轨迹编辑核心。因此：

- M7：`rejected`；
- 不注册事后 primary endpoint 或 effect size；
- M8：`rejected / not authorized`，0 seeds、0 ablation；
- M9：`rejected / not triggered`，0 blind samples，human verdict 保持 `null`；
- 不把 M6 的 0 coverage 伪装成方法改善，也不把已有 instance-aware/editing 机制重命名。

可保留的贡献边界只有：AD-GS 六场景 exact reproduction、DGGT upstream 对照证据，以及 AD-GS pseudo identity /
checkpoint identity collapse 的跨场景负结果。
