# Route Pivot B0 自然 Rollout Ceiling 审计

> **任务**：`RP-B0-05`  
> **结论**：`rejected`  
> **正式 run**：`route-pivot-b0-natural-rollout-s20260718-v1`  
> **代码 commit**：`06dc211`  
> **配置 / split / result 指纹**：`7fcdff809aa0` / `e69c3a08b8e4` / `29fc4036a7bd`

## 1. 结论先行

冻结 SVD-XT 的自然 independent seeds 能产生明显不同的视频与 P-UNC 能量，但这些差异大多与首帧损坏、
运动量偏移、track survival、flicker/锐度或 P-UNC support failure 纠缠。扩到协议上限 8 samples/condition
后，16 个 conditions 中只有 1 个存在两条合法且非重复候选；P-UNC-best 也未被独立 CoTracker 认可。

因此当前 Base 分布不存在足量、可用于 AWR/SFT 的安全 preference support。失败对象是
`SVD-XT + official natural seed sampling + 当前 anti-collapse`，不是“所有 best-of-N 都无效”；但增加 N、
换 scorer 或放松安全门不能构成重开。

## 2. 冻结协议

| 项目 | 固定值 |
|---|---|
| conditions | nuScenes val 首帧，16 scenes 严格不交叉 |
| sampling | `svd_official_v1`，fps7，8 frames，25 steps，冻结 Base |
| 总预算 | 先 4/condition；不足 12 diverse 才扩到 8；硬上限 `16×8=128` |
| Base | candidate index 0，固定 seed，只作对照，不进入 best-of-N selection pool |
| training-side rank | RAFT + P-UNC；独立 generic RAFT smoothness 对照 |
| independent evaluation | 官方 offline CoTracker3；不参与 P-UNC-best 选择 |
| anti-collapse | first frame、dynamic floor/ceiling、survival、coverage、sharpness、flicker、saturation |
| future GT | generation、candidate rank、CoTracker evaluation 均为 false |

P-UNC-best、generic-best、CoTracker-best 分别落盘；CoTracker-best 只作 oracle upper bound。所有 raw frames、
MP4、condition noise、initial latent、模型与 scorer checkpoint 均有指纹。

## 3. Candidate support

N=4 时 diverse conditions 为 `0/16`，故按预注册自动扩到 N=8。最终 112 个 selection candidates 只有
7 个 eligible，分布在 6 个 conditions；仅 1 个 condition 有两条 eligible candidates 并同时满足 RGB
nonduplicate 与 motion/P-UNC spread。

| eligibility check | 失败 candidates / 112 |
|---|---:|
| motion floor / ceiling | 59 |
| first-frame absolute | 51 |
| first-frame relative to fixed Base | 33 |
| survival | 30 |
| flicker | 29 |
| P-UNC projection support | 24 |
| sharpness | 21 |
| P-UNC valid | 20 |
| track length | 8 |
| primary-track support | 6 |

这些失败会重叠，不能相加解释为 candidate 总数。关键事实是：自然多样性主要来自与质量/活动度共同变化的
seed effect，而不是可单独利用的 motion factor。

## 4. Independent evaluator gate

严格 eligibility 下只有 6 个 conditions 能形成 P-UNC-best 对照：

| 项目 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| diverse conditions | `1/16` | `>=12/16` | fail |
| valid P-UNC vs random | `6` | `>=12` | fail |
| P-UNC vs random win-credit | `41.67%` | `>=60%` | fail |
| valid P-UNC vs fixed Base | `6` | `>=12` | fail |
| P-UNC vs Base win-credit | `41.67%` | `>=55%` | fail |
| selected low-motion / catastrophic | `1 / 3` | `0 / 0` | fail |
| positive-improvement conditions | `0` | `>=8` | fail |
| max winner seed fraction | `33.33%` | `<=50%` | pass |
| median survival delta | `+0.0046` | `>=-0.05` | pass |

win-credit 将 tie 计为 0.5，并没有把 invalid 填成 tie。机器 gate 失败后未生成 24-case review；人工 verdict
不能把 machine-rejected 路线晋级，因而这里不需要用户标注。

## 5. 18 dB checker sensitivity

R1 的 16 个合法 fps7 Base 首帧 PSNR 为 `12.64–22.14 dB`，其中 11/16 低于 B0 v1 的 18 dB absolute
floor；该 absolute floor 标为 miscalibrated。只读 run `route-pivot-b0-sensitivity-s20260718-v1` 不修改 source、
不重生成、不重算 scorer，只移除指定 eligibility checks：

| sensitivity | eligible candidates | 至少两条 eligible 的 condition 上界 | 实际 diverse |
|---|---:|---:|---:|
| strict | 7 | 1 | 1 |
| 忽略 absolute floor | 11 | 4 | 4 |
| 忽略全部 first-frame checks | 16 | 6 | 6 |
| 再忽略 motion floor/ceiling | 44 | 10 | 10 |
| 剥掉 first-frame + motion + flicker + sharpness + survival | 84 | 16 | 16 |

前三种仍无法达到 12/16；只有删除整套 anti-collapse 才能通过 support。故 checker 有错，但研究 rejection
不依赖该错误。sensitivity commit/config/result 为 `d9ac65d` / `631757e943f6` / `d52b5f71da38`。

## 6. Reviewer 预判

1. **“为什么不再增大 N？”** 8 已是预注册上限，且 112 条 selection candidates 只有 7 条合法；继续抽 seed
   是低效的 winner's-curse 搜索，并不能修复 support 与 quality/motion 的纠缠。
2. **“P-UNC 是否只是选错 scorer？”** Route 在 scorer 前就因合法 support 失败；仅 1/16 有两条候选。
   另外 P-UNC-best 在可比子集对 CoTracker 仅 `41.67%`，换排序器不能增加合法候选数量。
3. **“能否让人工看完再决定？”** 人审只负责最终可信性，不能覆盖 machine anti-collapse。机器失败后做人审
   会造成从少数幸存样本选择性报告。
4. **“18 dB 过严是否让结论无效？”** 它确实过严，已明确标错；但移除它后仅 `4/16`，移除全部首帧门也
   仅 `6/16`。结论由多组独立 safeguards 共同支持。
5. **“这是否否定 reward-guided sampling？”** 只否定当前 frozen SVD natural distribution。若显式
   action/trajectory condition 改变 support，必须以新 backbone、新预算和同样 safeguards 重开。

## 7. 决策

- `RP-B0-05 = rejected`；
- 不生成人工 review，不进入 condition-relative AWR/SFT；
- 不搜索更多 seeds、CFG、scheduler、fps、motion bucket 或 scorer；
- Route A 与 B 均 rejected，按 V5 解锁 `RP-C0-07` 的 action-conditioned backbone 只读迁移审计。
