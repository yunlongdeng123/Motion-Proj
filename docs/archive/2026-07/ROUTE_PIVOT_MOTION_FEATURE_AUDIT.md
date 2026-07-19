# Route Pivot A1 冻结运动表征审计

> **任务**：`RP-A1-SCAN-04A`  
> **结论**：`rejected`  
> **正式 run**：`route-pivot-a1-feature-scan-s20260718-v2`  
> **代码 commit**：`b27fb5a`  
> **配置 / split / result 指纹**：`982b13b356cf5` / `acb5edcab582` / `467a50bd546f`

## 1. 结论先行

冻结 SVD-XT feature 对真实驾驶背景的 ego-induced flow 有稳定、跨 scene 的线性可读信号，但当前
ego-centered local cost probe 不能可靠恢复独立 actor residual：所有 21 个 layer/sigma 配置都输给
zero-residual baseline，并在 stationary actor 上产生比 moving target 更大的假 residual。A1-SCAN 因此没有
top-2，A1-CONFIRM 与 A2 不解锁。

这给出一个比“feature 好/坏”更具体的负结论：当前表示主要暴露 camera/ego motion，却没有在 compact
local query 中把 actor-independent motion 安全分离出来。它是 V5 所要求的 driving-specific motion
entanglement 证据，不是低运动 winner，也不能用较弱的 absolute-position baseline 掩盖。

## 2. 冻结协议

| 项目 | 固定值 |
|---|---|
| 数据 | nuScenes train，24 train / 8 dev，32 个 scenes 严格不交叉 |
| 输入 | 8 帧真实视频 latent，真实 timestamp，`svd_official_v1`，`fps=7` |
| 主干 | SVD-XT、VAE、image encoder 全冻结；无 LoRA、diffusion 或 refiner 训练 |
| hooks | `down_s8/down_s16/down_s32/mid_s64/up_s32/up_s16/up_s8` |
| sigma | `0.05 / 0.2 / 1.0` |
| compact record | 固定 96-d random projection、5×5 local correlation、position、真实 delta-t |
| probe | matched-capacity linear ridge；训练侧固定 `regularization=1.0` |
| actor support | actual_t、actual_t+1、static_t+1 三点共同图内；moving/stationary 平衡抽取 |
| ego support | 两帧 GT boxes 外的 sparse LiDAR ego-flow anchors |

正式 cache 只保存 sampled records，不保存完整 feature map。backbone、target builder、projection、query set、
scene split 与 conditioning noise 均有指纹。real-video future geometry 只用于 representation probe，不进入
任何 generated-rollout evaluator。

## 3. 正式结果

| 指标 | 结果 | SCAN 门槛 | 判定 |
|---|---:|---:|---|
| train/dev actor queries | `567 / 176` | `>=256 / >=64` | pass |
| train/dev ego queries | `2304 / 768` | `>=1024 / >=256` | pass |
| ego 对最佳 zero/mean 改善 | `17.86%–25.01%` | `>=10%` | 21/21 具备 signal |
| actor A-RES 对 zero 改善 | `-213.80%–-120.35%` | `>=7.5%` | 21/21 fail |
| A-RES 对 matched A-ABS 改善 | `35.94%–65.81%` | `>=5%` | pass，但不足以晋级 |
| stationary / moving magnitude ratio | `3.292–5.062` | `<=0.75` | 21/21 fail |
| primary candidates / stable layers | `0 / 0` | top-2，layer 至少 2 sigma | fail |

最佳 ego 配置是 `up_s16 / sigma=0.05`：EPE `24.820 px`，zero/mean 为
`33.098/33.185 px`，相对最佳 baseline 改善 `25.01%`；time-shuffled feature 使 EPE 恶化 `30.14%`。

最佳 actor-zero 配置是 `up_s32 / sigma=0.05`：moving A-RES EPE `5.862 px`，zero residual 为
`2.660 px`，即改善 `-120.35%`。它对 A-ABS 改善 `65.81%`，time/instance-target shuffle 分别恶化
`34.91%/81.52%`，说明 probe 学到统计关联；但 stationary ratio `4.348` 表明该关联不是安全的独立 actor
motion。主门必须优先于“相对更弱 baseline 有改善”。

因为 primary candidate 为 0，预注册的 expensive top-2 single-frame 与 future-reversed 重跑没有执行。
这是 fail-closed 分支，不是缺失结果；所有 21 个配置仍完成了 feature shuffle 与 instance-target shuffle。

## 4. Reviewer 预判与边界

1. **“ego positive 是否来自 future GT 泄漏？”** 这里评价的对象是真实视频 representation，future pose/LiDAR
   只构造 training-side target；manifest 明确 `uses_future_gt_for_generated_evaluation=false`。它不构成自由
   rollout 的物理分数。
2. **“A-RES 明明优于 A-ABS，为何拒绝？”** A-ABS 的绝对坐标任务受 scene/scale 分布影响，是较弱 generic
   baseline。真实 driving safeguard 是能否超过零残差并保持 stationary；两项均大幅失败。
3. **“actor residual 很小，zero baseline 是否过强？”** moving target median 约 `1.454 px` 正是低运动偏置
   风险所在。若方法不能在此 baseline 上证明收益，训练最容易学成少动或伪 residual，不能降低门槛。
4. **“96 维 projection 是否丢失信息？”** 可能，因此结论限定于当前预注册 compact probe，不声称所有 SVD
   feature 都不含 actor motion。但 V5 的 scan 不能在看到结果后扩维或换 MLP；这类改变必须作为新假设，
   重新预注册并说明为何能解决 stationary false motion。
5. **“为何不直接训练 auxiliary LoRA 看 rollout？”** target 可读性与 stationary safety 尚未通过，直接训练会
   把不可辨识关联写回共享 temporal 参数，重复旧路线的局部监督泄漏问题。

## 5. 决策

- `RP-A1-SCAN-04A = rejected`；
- `RP-A1-CONFIRM-04B = rejected / not run by dependency`；
- `RP-A2-06 = rejected / not run by dependency`；
- ego-only frozen representation 作为诊断资产保留，不命名为 EgoActor-Align；
- 继续与 Route A 独立的 `RP-B0-05`；若 B0 也失败，按 V5 进入 action/trajectory-conditioned backbone 的
  只读迁移审计。

## 6. 工程失败旁注

`route-pivot-a1-feature-scan-s20260718-v1` 在首个 clip、任何研究指标产生前因 fp32 preprocess 与 bf16 VAE
dtype mismatch 失败。修复仅给 official conditioning 加入 C0 parity 已验证的 autocast，并换用 v2 run ID；
数据、split、layer、sigma、probe、seed 与门槛未改变。v1 不覆盖、不计入 research rejection。
