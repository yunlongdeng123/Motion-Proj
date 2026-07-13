# Motion-Proj CVPR 2027 可持续研发计划：P2-V2 结构性诊断

> 本文档是当前阶段的唯一研发计划与决策源。Coding Agent 必须按照任务依赖顺序执行，不得跳过阻塞门槛、静默修改阈值或用更长训练掩盖失败。

* 最后更新：2026-07-13
* 计划基线 commit：`3cb8445`
* 当前阶段：`P2-V2-GEN-04 running / provider implementation awaiting Base-panel review`
* 当前开发骨干：Stable Video Diffusion XT
* 状态词：`pending / running / blocked / done / rejected`
* 当前主问题：H0 已确认，SVD future GT ego static target 禁用；self-estimated static V1 人工合理率 66.67%，static replay branch blocked

---

## 0. Coding Agent 执行约束

### 0.1 总体原则

Coding Agent 必须遵守：

1. 先复核仓库当前实现，再修改代码。
2. 先完成条件有效性、代数正确性和单步可学习性，再构建正式 replay cache。
3. 未通过当前 milestone 的验收门槛，不得继续后续训练。
4. 不运行新的 Optuna，不启动 300/800-step 长训练。
5. 不在 dirty worktree 上构建正式 cache、训练正式模型或生成正式比较表。
6. 每个正式 run 必须保存：

   * `resolved.yaml`
   * `manifest.json`
   * cache/dataset fingerprint
   * Git commit 和 dirty 状态
   * generation、training、evaluation seed
   * `metrics.jsonl`
   * `summary.json`
   * checkpoint
   * `COMPLETE` 或明确失败状态
7. 所有新阈值必须进入配置和 manifest，不允许仅存在于代码常量中。
8. 所有新实验必须具有唯一 run ID；不得覆盖或改变已有 run 的语义。
9. 每完成一个 milestone：

   * 运行对应测试；
   * 更新 `docs/EXPERIMENTS.md`；
   * 更新本计划状态；
   * 形成独立 commit；
   * 不自动 push。

### 0.2 本阶段禁止事项

在 `P2-V2-PROMOTE-09` 通过前，禁止：

* 启动 P3 五相机泛化；
* 启动 P4 OpenDWM 正式迁移；
* 把 synthetic P1 target 的合理性外推为 rollout 改善；
* 使用 V1 最佳 trial 继续挖 replay；
* 使用未来 GT 轨迹审计未被该轨迹条件化的 SVD rollout；
* 把无有效轨迹的 clip 记为 acceleration 等于 0；
* 在 32 个或更少 clip 上以 FVD 作为筛选或晋级指标；
* 通过增大训练步数替代扩大 replay 分布。

---

# 1. 核心研究决策

## 1.1 V1 结论

`P2-V1-TUNE-01：rejected`

立即停止：

* 不再继续 V1 Optuna 搜索；
* 不运行 `t10-800`；
* 不从现有 32-clip synthetic mini cache 延长任何 trial；
* 不把 LPIPS 改善解释为动力学改善；
* 不在当前 V1 loss 结构中继续扩大 `lambda_proj`；
* 不在 V2 通过前启动 P3/P4。

拒绝范围严格限定为：

> 当前“synthetic corruption → low-noise absolute x0 projection regression + full-image anchor + spatial/temporal mixed LoRA”的 V1 配方，在当前 mini 协议下不值得继续调参或长训。

V1 的失败不直接否定：

* no-grad dynamics projector；
* offline projection cache；
* 几何或轨迹诱导的 dense correction；
* projection distillation 总体方向。

## 1.2 V2 研究问题

V2 只检验以下更窄、可证伪的命题：

> 当投影目标与生成条件相容、训练样本来自冻结 Base 的真实 rollout、监督参数化与 SVD 原始模型输出一致、修正区域与保持区域解耦时，projected replay 是否能够改善完整 rollout 的驾驶动力学，同时保持预训练视觉先验？

在回答该问题前，不宣称 V2 方法成立。

---

# 2. V1 实验事实与结论边界

## 2.1 已确认事实

证据根目录：

```text
/root/autodl-tmp/runs/p2-tune-mini
```

运行结果：

* 16 个 100-step synthetic trial 全部正常结束；
* 排名前 4 个重新训练至 300 step；
* 未启动 800-step；
* 无 NaN、OOM、Traceback 或异常恢复；
* 所有 100-step trial 的动力学综合分数均为负；
* 300-step 续训没有反转趋势。

| 指标                   |   Base |     t10 100-step |      t10 300-step |
| -------------------- | -----: | ---------------: | ----------------: |
| static drift ↓       | 8.2095 | 9.8610（恶化 20.1%） | 12.2478（恶化 49.2%） |
| track acceleration ↓ | 4.3953 | 5.7143（恶化 30.0%） |  6.5092（恶化 48.1%） |
| LPIPS ↓              | 0.5088 | 0.4453（改善 12.5%） |  0.4502（改善 11.5%） |
| eligibility ↑        | 85.51% |           86.37% |            76.46% |
| tune score ↑         |      — |          -0.2506 |           -0.4864 |

## 2.2 当前解释

已观察到的稳定现象是：

> 当前优化路径能够更快降低重建类损失和 LPIPS，却同时破坏静态背景和目标轨迹动力学。

当前证据支持工程停损，但不支持论文级统计结论，因为：

* 只有 4 个 validation clip；
* 其中一个 clip 的 track acceleration 无效；
* 推理只使用 8 denoising steps；
* 当前评估可能与训练 auditor 部分同源；
* validation 数量不足以稳定比较相近 trial。

## 2.3 重复暴露问题

定义：

[
\text{average replay exposure}
==============================

\frac{
\text{optimizer steps}
\times
\text{micro batch}
\times
\text{gradient accumulation}
}{
\text{replay sample count}
}
]

在 32 个 cache clip、`micro_batch=1`、`grad_accum=8` 下：

* 100 optimizer steps：平均约 25 次暴露；
* 300 optimizer steps：平均约 75 次暴露。

这与高学习率下 LPIPS 快速改善、动力学和 eligibility 持续下降的现象一致。

---

# 3. 当前代码事实与首要阻塞项

## 3.1 当前 V1 loss 路径

当前训练器执行：

```text
projection_loss(y, x_dagger)
+ real_loss(clean)
+ full-image anchor(z, sigma)
```

当前 `projection_loss`：

* 从 `y` 独立采样 sigma 和 noise；
* 预测 `x0_hat`；
* 在 mask 内回归 `x_dagger`；
* 默认不启用 EDM weighting。

当前 `real_loss` 独立采样另一套 sigma 和 noise。

当前 anchor 在 projection branch 的相同 noisy latent 上，要求学生预测靠近关闭 LoRA 后的 Base 输出。

## 3.2 当前 SVD 参数化

当前 SVD backbone 的 UNet 原始输出记为 (v_\theta)，其干净 latent 预测为：

[
\hat x_0
========

## \frac{z_\sigma}{1+\sigma^2}

v_\theta
\frac{\sigma}{\sqrt{1+\sigma^2}}.
]

因此：

[
\frac{\partial\hat x_0}{\partial v_\theta}
==========================================

-\frac{\sigma}{\sqrt{1+\sigma^2}}.
]

在极低噪声区域，未加权 `x0` loss 对原始 UNet 输出的有效梯度会衰减。

## 3.3 当前 replay 不是严格 Base replay

当前 `motion_proj/replay/mine.py`：

* 强制接受 `--adapter`；
* 加载 synthetic checkpoint adapter；
* 用该 adapter 生成 replay parent；
* 因而当前 replay 并非纯 frozen Base rollout。

V2 必须支持：

```yaml
replay:
  parent_kind: base
  parent_adapter: null
```

且 V2 正式 cache 的默认值必须是 `parent_kind=base`。

## 3.4 当前 generated replay 没有 object track

当前 generated rollout 审计会设置：

```python
boxes = [[] for _ in frames]
```

因此当前 replay：

* 没有 generated object track；
* `num_tracks=0`；
* 只能检验 static component；
* 无法给 object-track acceleration 提供真实 replay 监督。

在 generated track provider 完成前，禁止宣称 replay 同时覆盖静态背景和动态目标。

## 3.5 首要阻塞项：未来 ego 条件不匹配

SVD 是 image-to-video 模型，不接收未来 ego pose、action 或 layout condition。

但当前 generated rollout auditor 会复用源 nuScenes clip 的：

```text
intrinsics
cam2ego
future ego2global
```

这会使用未提供给 SVD 的未来自车轨迹定义 expected static flow。

因此必须首先回答：

> SVD 自由生成的相机运动是否与该 nuScenes future ego trajectory 一致？如果不一致，当前 static drift 究竟是在测物理错误，还是在测未条件化的未来分支差异？

在该问题关闭前，不允许用 `GT ego-induced static flow` 构建正式 SVD replay target。

---

# 4. 待验证假设

| ID | 假设                                                | 当前证据                                                          | 证伪方式                              |
| -- | ------------------------------------------------- | ------------------------------------------------------------- | --------------------------------- |
| H0 | 当前 SVD static auditor 存在未来 ego 条件不匹配              | 已确认：16-case GT-ego residual 19.2887，identity 2.1164，self-estimated 0.9320；GT-ego 人工 16/16 无效 | `P2-V2-COND-00` 已关闭            |
| H1 | synthetic corruption 与 Base rollout error 分布不匹配   | synthetic cache 32 个；旧 replay 仅 1/8 kept                      | synthetic 与 Base replay 因果对照      |
| H2 | low-noise unweighted x0 loss 对 SVD 原始 (v) 输出梯度过弱  | 当前 loss 默认不加 EDM weight                                       | x0 / weighted-x0 / direct-v 对照    |
| H3 | full-image anchor 抵消 mask 内 projection correction | projection 与 anchor 在同一 (z_\sigma) 上追逐不同 target               | 梯度 cosine 和 outside-mask preserve |
| H4 | real/replay 独立 noise scale 导致目标失衡                 | 当前两个 branch 独立采样                                              | shared sigma/noise 对照             |
| H5 | mixed spatial/temporal LoRA 优先拟合外观                | 当前按通用 `to_q/to_k/to_v/to_out` 匹配                              | temporal-only / spatial-only 梯度审计 |
| H6 | replay cache 太小导致快速记忆                             | 300 step 平均约 75 次暴露                                           | 实际 sample exposure 统计             |
| H7 | endpoint correction 可降低单步误差，但无法影响完整 rollout       | 尚未验证                                                          | one-step 与 25-step rollout 联合门槛   |
| H8 | 训练 auditor 存在 metric hacking 风险                   | train/eval 部分同源                                               | 独立 flow/tracker 与人工评审             |

这些均是待检验假设，不得提前写成论文结论。

---

# 5. P2-V2-COND-00：条件有效性门槛

状态：`done`（2026-07-12；commit `fff5ccb`；证据：`/root/autodl-tmp/runs/p2-v2-condition/p2-v2-cond16-s20260712-fff5ccb-97d2d05d`；下一步 `P2-V2-API-01`）

这是所有 V2 replay 工作的前置阻塞项。

## 5.1 实现 audit geometry mode

为 generated rollout auditor 增加显式配置：

```yaml
auditor:
  generated_geometry_mode: gt_ego_debug
```

支持以下枚举：

```text
gt_ego_debug
identity_ego
estimated_background_motion
controlled_ego
```

定义：

### `gt_ego_debug`

* 复用 dataset future ego pose；
* 只允许用于：

  * synthetic 单元测试；
  * GT-aligned reconstruction；
  * projector debug；
* 禁止用于正式 SVD replay 训练和正式 SVD 指标。

### `identity_ego`

* 假设相机近似静止；
* 只允许用于预注册的 near-static ego subset；
* subset 阈值必须基于 nuScenes 自车位姿预注册。

### `estimated_background_motion`

* 从 generated RGB 自身估计背景运动；
* 不读取 future GT ego pose；
* 初始版本可使用：

  * RAFT flow；
  * 前景/高残差区域剔除；
  * robust homography、affine flow 或低秩背景流拟合；
* 输出背景流、置信度和拟合残差。

这是 SVD static replay 的正式候选模式。

### `controlled_ego`

* expected static flow 来自模型实际接收到的 ego/action/control condition；
* 仅为 OpenDWM 或其他可控 backbone 保留；
* 不在当前 SVD 上启用。

## 5.2 条件有效性诊断

在 16 个固定 SVD Base rollout 上，同时运行：

1. `gt_ego_debug`
2. `identity_ego`
3. `estimated_background_motion`

输出：

```text
condition_validity.jsonl
condition_validity_summary.json
condition_validity_panel/
```

每个样本记录：

* clip ID；
* generation seed；
* dataset ego translation/rotation；
* generated global-flow estimate；
* GT-ego expected flow；
* GT-ego residual；
* self-estimated residual；
* 两种 static mask；
* 两种 proposed correction；
* frame-by-frame visual panel。

## 5.3 人工检查

至少人工检查 12 个样本，每个样本比较：

```text
Base rollout
GT-ego correction
self-estimated correction
flow/mask visualization
```

标注：

```text
gt_ego_valid: yes / no / uncertain
self_estimated_valid: yes / no / uncertain
failure_reason
```

## 5.4 验收标准

V2 SVD 正式 replay 必须满足：

* manifest 中 `uses_future_gt_ego=false`；
* Base generation 不加载任何 adapter；
* self-estimated correction 人工合理率不低于 70%；
* frame 0 不被 projector 修改；
* mask 与 correction 均 finite；
* 没有 future GT box、track 或 pose 泄漏。

如果 self-estimated static projector 未通过：

* SVD V2 暂时只执行 generated point-track/object branch；
* static geometry branch 保留为 synthetic debug；
* static control-consistent 主结论推迟到 OpenDWM。

## 5.5 实验结论

固定 16 个 nuScenes val clip、冻结 SVD Base、25 inference steps 的正式 run 已完成：

```text
run_id: p2-v2-cond16-s20260712-fff5ccb-97d2d05d
commit: fff5ccb
config fingerprint: 97d2d05d
generation seeds: 20260712–20260727
adapter_loaded: false
```

结果：

* GT-ego、identity、self-estimated residual 均值分别为 `19.2887 / 2.1164 / 0.9320`；
* formal candidate 的 `uses_future_gt_ego=false`，所有 target/mask finite；
* 16/16 首帧完全冻结且首帧 mask 为零；
* 人工复核 16/16：self-estimated 为 8 yes、4 no、4 uncertain；
* decisive 合理率为 `8/12 = 66.67%`，低于预注册 `70%` 门槛；
* 失败主要是高覆盖 background mask 传播车辆或 Base 既有伪影，形成路面色块和拖影。

决策：

```text
H0 future ego mismatch: confirmed
SVD GT-ego static target: rejected except synthetic/debug
SVD self-estimated static replay V1: blocked
SVD generated point-track branch: unlocked for later P2-V2-GEN-04
```

不得通过事后调整 review 或覆盖该 run 改变结论。

---

# 6. V2 方法定义

工作名称：

> Scale-Aligned Teacher-Relative Projected Replay Distillation

## 6.1 Base rollout

冻结 Base：

[
X^b = G_{\theta_0}(\xi,c).
]

其中必须保存：

* conditioning frame；
* condition metadata；
* initial noise seed；
* sampler；
* inference steps；
* guidance/motion bucket/fps；
* Base model fingerprint；
* 是否加载 adapter，正式 V2 必须为 false。

## 6.2 Projector 输出

no-grad auditor/projector 输出：

[
\Gamma(X^b)
===========

\left(
X^\dagger,
M_s,M_o,
w_s,w_o,
e_{\text{before}},
e_{\text{after}}
\right).
]

其中：

* (M_s)：static/background correction mask；
* (M_o)：object/point-track correction mask；
* (w_s,w_o\in[0,1])：component-level confidence；
* (X^\dagger)：projected RGB；
* frame 0 强制保持不变；
* 所有 projector 输出均 stop-gradient。

初始允许某一 component 为空，但必须在 manifest 中明确记录。

## 6.3 确定性 latent 编码

使用 deterministic VAE posterior mode：

[
x^b=E(X^b),\qquad
x^\dagger=E(X^\dagger).
]

定义：

[
d=\operatorname{sg}[x^\dagger-x^b].
]

要求：

* Base 和 projected RGB 使用完全相同的 resize、normalization 和 VAE；
* 不直接把 RGB displacement 当作 latent displacement；
* 不混用 sampler final latent 与重新编码的 projected latent；
* cache 保存 VAE fingerprint 和 scaling factor；
* `mask[:,0]=0`；
* `d[:,0]=0`。

## 6.4 Forward diffusion

对 Base rollout latent 加噪：

[
z_\sigma=x^b+\sigma\epsilon.
]

禁止再以 synthetic corruption 作为 V2 主训练输入。

## 6.5 暴露 SVD 原始模型输出

为 backbone 增加：

```python
predict_model_output(z, sigma, cond) -> raw_v
anchor_predict_model_output(z, sigma, cond) -> raw_v_base
x0_from_model_output(z, sigma, raw_v) -> x0
model_output_from_x0(z, sigma, x0) -> raw_v
```

冻结 Base teacher：

[
v_0=
\operatorname{sg}
\left[
F_{\theta_0}(z_\sigma,\sigma,c)
\right].
]

## 6.6 Teacher-relative residual-v target

因为：

[
x_0
===

## \frac{z_\sigma}{1+\sigma^2}

v
\frac{\sigma}{\sqrt{1+\sigma^2}},
]

希望在 Base teacher endpoint 上加入 (+\eta Md)，对应：

[
v^{\text{tar}}
==============

\operatorname{sg}
\left[
v_0
---

\eta_{\text{eff}}
\frac{\sqrt{1+\sigma^2}}{\sigma}
M\odot d
\right].
]

其中：

[
M=\operatorname{clip}(M_s+M_o,0,1).
]

初始只比较：

[
\eta\in{0.25,0.5,1.0}.
]

不做连续搜索。

## 6.7 连续 trust-region scaling

不再优先使用整样本 binary gate。定义：

[
r=
\operatorname{RMS}_{M}(d).
]

采用：

[
\eta_{\text{eff}}
=================

\min
\left(
\eta,
\frac{B(\sigma+\varepsilon)}{r+\varepsilon}
\right).
]

目的：

* 防止低 sigma 下 direct-v target 爆炸；
* 保持 correction-to-noise ratio 有界；
* 避免因为单个 component 超界而丢弃整个 clip；
* 保存 `eta_eff`、correction RMS 和 clipping fraction。

binary gate 仅保留为消融。

## 6.8 Correction loss

[
\mathcal L_{\text{corr}}
========================

\frac{
\sum
wM,
\rho
\left(
v_\theta-v^{\text{tar}}
\right)
}{
\sum wM+\varepsilon
}.
]

其中：

* (\rho) 默认 Huber；
* (w) 由 (w_s,w_o) 合成；
* static 和 object component 必须分别记录 loss；
* 禁止只有 union loss 而无法定位 component。

## 6.9 Outside-mask preserve

全图 anchor 替换为同参数化的 outside-mask preserve：

[
\mathcal L_{\text{pres}}
========================

\frac{
\sum
w_{\bar M}\bar M
\left|
v_\theta-v_0
\right|*2^2
}{
\sum w*{\bar M}\bar M+\varepsilon
},
]

其中：

[
\bar M=1-\operatorname{dilate}(M).
]

要求：

* mask 边缘使用 soft dilation/feather；
* mask 内不得施加 Base imitation；
* preserve 与 correction 使用同一个 (z_\sigma,\sigma,\epsilon)；
* 记录 mask 内、边缘和 mask 外的 teacher drift。

## 6.10 Noise-aligned real SFT

修改 `real_loss`，允许传入：

```python
real_loss(
    backbone,
    clean,
    cond,
    sigma=shared_sigma,
    noise=shared_noise,
)
```

combined step 内：

* real branch 与 replay branch 使用同一 sigma；
* 使用同一标准高斯 noise tensor；
* clean endpoint 可以不同；
* 保留 `independent_noise` 作为明确消融。

总损失：

[
\mathcal L
==========

\lambda_{\text{corr}}\mathcal L_{\text{corr}}
+
\lambda_{\text{pres}}\mathcal L_{\text{pres}}
+
\lambda_{\text{real}}\mathcal L_{\text{real}}.
]

V2 诊断阶段不允许额外加入 flow loss、DPO 或其他新目标。

---

# 7. LoRA 参数隔离

## 7.1 当前问题

当前通用 target name：

```yaml
[to_q, to_k, to_v, to_out.0]
```

可能同时匹配 spatial 和 temporal attention。

## 7.2 必须增加模块选择模式

配置：

```yaml
model:
  lora:
    scope: temporal_only
```

支持：

```text
temporal_only
spatial_only
all_attention
```

要求：

* 根据完整 module path 分类；
* 不只根据叶子节点名称分类；
* 启动时打印和保存全部 selected module names；
* manifest 保存：

  * temporal module count；
  * spatial module count；
  * trainable tensor count；
  * trainable parameter count；
* `temporal_only` 下发现任何 spatial LoRA 时直接 fail closed；
* 未匹配到 temporal module 时直接报错，不得静默训练空 adapter。

V2 诊断期固定：

```yaml
rank: 16
scope: temporal_only
```

只有 temporal-only 已产生稳定动力学收益后，才允许加入 spatial adapter 作为画质恢复消融。

---

# 8. Generated object/point-track replay

状态：`pending`（条件门槛已关闭；static branch blocked，仅推进无 future GT 的 generated point-track）

## 8.1 目标

当前 replay boxes 为空，必须建立不依赖 future GT track 的 generated motion provider。

## 8.2 初始实现

新增统一接口：

```python
class GeneratedTrackProvider:
    def track(frames) -> GeneratedTrackState:
        ...
```

至少支持：

```text
raft_chain
cotracker3
```

### `raft_chain`

* 使用当前已有 RAFT；
* 选择高置信 query points；
* 逐帧前向传播；
* 做 forward-backward consistency；
* 对失效、出界和遮挡点降权；
* 作为单卡开发和训练 auditor。

### `cotracker3`

* 作为独立长时点跟踪 evaluator；
* 初始可只用于评估；
* 不与训练 auditor 共享同一套轨迹。

## 8.3 Query point 选择

不得均匀地让背景点完全支配目标。

至少分层采样：

```text
background points
dynamic-residual points
foreground candidate points
```

动态候选可来自：

* observed flow 与 robust background flow 的残差；
* detector 在 generated first frame 上的结果；
* segmentation/dynamic mask。

禁止使用 future GT box/track。

## 8.4 Point-track 动力学

对有效轨迹 (p_{i,t})：

1. 移除 estimated background/camera motion；
2. 计算速度、加速度和 jerk；
3. 使用置信度和可见性 mask；
4. 对轨迹做 robust smoothing；
5. projector 只修正高置信局部区域。

报告：

* valid track count；
* median track length；
* survival rate；
* acceleration；
* jerk；
* correction coverage；
* projector 前后能量。

## 8.5 分支边界

如果 generated track provider 未通过：

* object replay branch 标为 `blocked`；
* 不使用 GT future track 替代；
* 仅运行 static/self-consistency 分支；
* 正式结果不得报告 object improvement。

## 8.6 2026-07-13 实现状态（待 Base panel）

已接入 `RAFTChainGeneratedTrackProvider`：它按 background / dynamic-residual /
foreground-candidate 三层确定性选取 query，以 RAFT 相邻流链式传播，并在每一步执行
forward-backward、一致性、越界和低置信度筛除。所有点轨迹以 16px 可配置局部框进入
projector；轨迹和 projector 诊断会报告轨迹数、长度中位数、survival、去背景速度/加速度/jerk、
局部 correction coverage 与前后能量。generated 模式现在同时隔离 source future boxes，
因此 GT box 不能再影响 static mask 或 track。`cotracker3` 仅保留为需要显式注入 predictor
的独立 evaluator，缺依赖时 fail-closed，不会静默退回 RAFT。

工程测试覆盖：无 GT 轨迹、分层 query、链式传播、F/B 拒绝、source-box 隔离、
自估背景 projector 接线、配置和 CoTracker3 fail-closed。首个 clean Base panel
`p2-v2-gen04-panel1-s20260713-3cb8445` 已完成自动检查：72 个 query 中有 62 条有效
轨迹、长度中位数 5 帧、survival 44.44%、局部 correction coverage 6.01%，且
`uses_future_gt_track=false`。证据中的 track overlay 已供人工复核；单 case 只证明链路
可运行，不构成轨迹质量晋级，未通过后续人工 review 前 object replay 继续 blocked。

GEN-04 的轨迹人工门禁独立于已经被拒绝的 static V1：固定 8 个 clean Base rollout，
`uses_future_gt_track=false`、每例至少 1 条有效轨迹且轨迹长度中位数至少 3 帧；人工只评
panel 第二栏的点是否贴合可见局部并跨帧连续。8 例都必须填写 `yes/no/uncertain`，对 decisive
例的 `yes` 比例必须不低于 70%。static correction 栏仅为上下文，不能以其已知失败结果替代或
否决 point-track verdict；未达到该门槛时 object component 仍为 `blocked`。

---

# 9. Cache Schema V5

新增 replay cache schema，禁止复用 V1 fingerprint。

每条样本至少保存：

```text
sample_id
source = replay_v2
parent_kind = base
base_model_fingerprint
adapter_loaded = false
condition_id
condition_frame
generation_seed
generation_sampler
generation_steps
generation_settings
base_rgb
projected_rgb
base_latent
projected_latent
latent_residual
static_mask
object_mask
static_confidence
object_confidence
first_frame_frozen = true
auditor_version
projector_version
geometry_mode
uses_future_gt_ego = false
uses_future_gt_track = false
energy_before_by_component
energy_after_by_component
projector_diagnostics
VAE_fingerprint
cache_fingerprint
```

## 9.1 必须执行的校验

cache writer 在正式 V2 模式下必须拒绝：

* `parent_kind != base`；
* `adapter_loaded=true`；
* `uses_future_gt_ego=true`；
* `uses_future_gt_track=true`；
* frame 0 mask 非零；
* frame 0 projected RGB 与 Base 不一致；
* NaN/Inf；
* mask 为空且仍标记有效；
* latent/RGB frame count 不一致；
* Base/projected latent 使用不同 VAE fingerprint；
* component energy 在高置信区域明确上升；
* stale schema 或旧 fingerprint。

## 9.2 Soft confidence

不再使用单一 clip-level 70% eligibility 丢弃整条样本。

分别保存：

```text
static_valid_fraction
object_valid_fraction
static_confidence_mean
object_confidence_mean
```

只有以下情况硬拒绝：

* non-finite；
* 所有 component 均为空；
* projector 后高置信区域能量明确上升；
* frame/condition/cache 对齐失败；
* 灾难性视觉损坏；
* GT 泄漏；
* Base parent provenance 不正确。

---

# 10. 里程碑

| ID               | 状态       | 目标                              | 主要产物                      | 后续条件                       |
| ---------------- | -------- | ------------------------------- | ------------------------- | -------------------------- |
| P0-GEOMETRY-01   | done     | synthetic 100-case 几何验收         | 95/100 改善                 | 只作 projector unit evidence |
| P0-RUNTIME-02    | done     | resume 与连续训练逐位一致                | checkpoint/runtime 证据     | 保持单卡确定性                    |
| P1-PROJECTION-01 | done     | synthetic target 人工检查           | 20/20 reasonable          | 不外推 rollout                |
| P2-V1-TUNE-01    | rejected | V1 16×100、4×300 无正增益            | V1 归档                     | 不再续训                       |
| P2-V2-ARCHIVE-00 | done     | 归档 V1 和修复 watchdog 完成态          | `docs/EXPERIMENTS.md`、`tests/test_watchdog_terminal_state.py` | 开始 V2 条件门槛             |
| P2-V2-COND-00    | done     | 确认 SVD future ego 条件不匹配；判定 static V1 | condition validity report + 16-case review | GT static rejected；point-track 解锁 |
| P2-V2-API-01     | done     | raw-v、代数变换和 temporal-only API   | 单元测试                      | 解锁新 loss                   |
| P2-V2-GRAD-02    | done     | 当前 V1 与 V2 梯度审计                 | gradient JSONL/report     | 保留 residual-v + trust region |
| P2-V2-PILOT-03   | blocked  | 8-pair 单步容量测试                   | A/B/C/D curves            | 等待有效 Base replay pair       |
| P2-V2-GEN-04     | running  | generated point-track provider；static V1 blocked | provider tests / Base-panel review | 仅通过 review 后解锁 object component |
| P2-V2-REPLAY-05  | pending  | 64–128 Base replay cache        | schema V5、manual review   | 解锁训练                       |
| P2-V2-CAUSAL-06  | pending  | 因果配方对照                          | paired 25-step report     | 选择唯一主配方                    |
| P2-V2-SCALE-07   | pending  | low/mid/mixed sigma 对照          | scale report              | 判断 scale alignment         |
| P2-V2-REPRO-08   | pending  | 第二训练 seed 复现                    | two-seed report           | 晋级判断                       |
| P2-V2-PROMOTE-09 | pending  | 通过 32-clip 晋级门槛                 | promotion report          | 恢复 P2 主实验                  |
| P3-CAMERA-01     | blocked  | 五相机零样本泛化                        | —                         | 等待 P2-V2                   |
| P4-OPENDWM-01    | blocked  | 可控 backbone 迁移                  | —                         | 等待 P2 主结果                  |
| P5-PAPER-01      | pending  | 三 seed、完整统计与消融                  | 主表/附录                     | 10 月冻结                     |

---

# 11. P2-V2-ARCHIVE-00

状态：`done`（2026-07-12；实现与测试证据见本里程碑归档 commit；下一步仅执行 `P2-V2-COND-00`）

## 11.1 工作内容

1. 将 V1 全部事实写入 `docs/EXPERIMENTS.md`。
2. 将旧计划中的“Optuna 进行中”修改为 rejected。
3. 保存：

   * 16×100 trial 表；
   * 4×300 trial 表；
   * Base 指标；
   * 搜索空间；
   * 停止原因；
   * 未启动 800-step 的原因。
4. 修复 watchdog：

   * worker COMPLETE 后不再报告 heartbeat stale；
   * watchdog 识别 `COMPLETE / FAILED / REJECTED`；
   * 增加完成态测试。

## 11.2 验收

```bash
pytest -q tests/test_watchdog_terminal_state.py
```

并确认：

* 文档与 run summary 一致；
* 不把 t10 写成有效超参；
* 不修改历史 run；
* 新 commit worktree clean。

---

# 12. P2-V2-API-01：参数化与模块选择

状态：`done`（2026-07-12；commit `5bd7a18`；证据：`tests/test_svd_parameterization.py`、`tests/test_lora_scope.py`、全量 `105 passed`；下一步 `P2-V2-GRAD-02`）

## 12.1 需要修改的文件

优先检查并修改：

```text
motion_proj/backbones/base.py
motion_proj/backbones/svd_backbone.py
motion_proj/losses/
motion_proj/train/trainer.py
configs/model/svd.yaml
tests/
```

## 12.2 新 API

实现：

```python
predict_model_output
anchor_predict_model_output
x0_from_model_output
model_output_from_x0
```

`predict_x0` 保留兼容，但内部必须复用统一转换。

## 12.3 代数测试

新增：

```text
tests/test_svd_parameterization.py
```

测试 sigma：

```text
0.02, 0.05, 0.1, 0.5, 1.0, 5.0
```

验证：

[
x_0
\rightarrow v
\rightarrow x_0
]

和：

[
v
\rightarrow x_0
\rightarrow v
]

均满足：

```text
float32 max_abs_error < 1e-5
bf16 relative_error 在预注册容差内
```

同时测试：

* sigma floor；
* no NaN/Inf；
* target detach；
* adapters disabled 时 Base output 可复现；
* adapters 被恢复为原启用状态。

## 12.4 Temporal-only LoRA 测试

新增：

```text
tests/test_lora_scope.py
```

验收：

* temporal-only 选中模块数大于 0；
* temporal-only 不包含 spatial module；
* spatial-only 不包含 temporal module；
* manifest 中 module list 可复现；
* 训练参数数量与保存 adapter tensor 数一致。

## 12.5 验收结论

* 六个预注册 sigma 上，float32 的 `x0 → v → x0` 与 `v → x0 → v` 最大绝对误差均低于 `1e-5`；
* bf16 latent + float32 sigma 的 roundtrip 通过预注册 `2e-3` 相对误差容差；
* `sigma_floor=1e-3`，零 sigma 与 floor 行为一致，NaN/Inf fail closed；
* Base anchor 输出为 detached，adapter 在原先启用和原先禁用两种状态下均准确恢复；
* 完整 SVD-XT 结构 smoke 选中 `128` 个 temporal、`0` 个 spatial 模块；
* rank 16 下实际为 `256` 个 adapter tensor、`3,319,808` 个可训练参数，保存 tensor 数一致；
* selected module names 固化到 `selected_modules.txt` 与 run manifest。

该门槛只确认代数、参数隔离和工程 provenance，不构成 rollout 改善证据。

---

# 13. P2-V2-GRAD-02：梯度审计

状态：`done`（2026-07-13；实现 commit `ce52feb` / `63d9bd0`；证据：`/root/autodl-tmp/runs/p2-v2-gradient-audit/p2-v2-grad-s20260713-ce52feb-legacyv4`、`/root/autodl-tmp/runs/p2-v2-gradient-audit/p2-v2-grad-v1-s20260713-63d9bd0-legacyv4`；下一步：`P2-V2-PILOT-03` 因无有效 Base replay pair 而 blocked，先解锁 generated point-track provider）

## 13.1 工具

新增：

```text
motion_proj/diagnostics/gradient_audit.py
configs/diagnostics/p2_v2_grad.yaml
```

命令形式：

```bash
python -m motion_proj.diagnostics.gradient_audit \
  --config configs/diagnostics/p2_v2_grad.yaml
```

## 13.2 审计对象

在完全相同 batch、condition、sigma 和 noise 下分别计算：

[
g_{\text{real}},
\quad
g_{\text{x0-proj}},
\quad
g_{\text{direct-v}},
\quad
g_{\text{anchor}},
\quad
g_{\text{preserve}}.
]

记录：

* L2 gradient norm；
* 按参数数量归一化的 RMS norm；
* temporal LoRA norm；
* spatial LoRA norm；
* 各 loss 之间的 cosine similarity；
* sigma；
* mask coverage；
* static/object component；
* correction RMS；
* trust-region clipping fraction。

参数 RMS：

[
\operatorname{GradRMS}(g)
=========================

\sqrt{
\frac{\sum_i|g_i|_2^2}
{\sum_i\operatorname{numel}(g_i)}
}.
]

## 13.3 mask 内外审计

不能直接把参数梯度称为“mask 内梯度”。

必须分别构造：

```text
loss_corr_static
loss_corr_object
loss_preserve_outside
loss_boundary
```

再分别求参数梯度。

## 13.4 预注册判定

以下阈值只作工程诊断，不直接作论文结论：

1. 若：

[
\operatorname{median}
\frac{
|\lambda g_{\text{x0-proj}}|
}{
|g_{\text{real}}|+\varepsilon
}
<0.1
]

而 direct-v 显著提高该比率，则认定参数化缩放是 V1 的主要风险之一。

2. 若超过半数 batch：

[
\cos(g_{\text{proj}},g_{\text{anchor}})<-0.3
]

且：

[
\frac{|g_{\text{anchor}}|}
{|g_{\text{proj}}|+\varepsilon}
\ge 0.5,
]

则 full-image anchor 判为冲突设计。

3. 若 mixed LoRA 中 spatial GradRMS 超过 temporal 的 2 倍，且 LPIPS 改善方向与 spatial gradient 增长一致，则 temporal-only 成为强制主基线。

4. 若 direct-v 在最低 sigma bin 出现 target 或 gradient 爆炸，则提高 sigma floor 或收紧 trust region；禁止直接扩大 gradient clipping 掩盖问题。

## 13.5 产物

```text
gradient_audit.jsonl
gradient_audit_summary.json
gradient_norm_by_sigma.csv
gradient_cosine_matrix.csv
selected_modules.txt
```

## 13.6 验收结论

* V2 loss 单元测试与既有参数化/LoRA 测试均通过；完整测试为 `108 passed`。
* temporal-only 零初始化 LoRA 的 12 个固定 `(sample, sigma)` 行均为 finite；legacy synthetic V4 仅用于参数化工程诊断，不能作为 Base replay、target 合理性或 rollout 收益证据。
* 在已归档的 V1 `t10-300` all-attention adapter 上，direct-v 与 real 的 L2 比中位数为 `62.7214`，absolute x0 与 real 为 `2.7559`；最低 sigma 下 direct-v 没有 target/gradient NaN 或 Inf，trust-region 生效，故保留 teacher-relative residual-v + continuous trust-region 进入 pilot。
* direct-v 与 full-image anchor 的 cosine 在 12 行中均不低于 `0.3759`，未触发预注册的负冲突阈值；但这不推翻 full-image anchor 的设计替换，pilot 仍只使用 mask 外 preserve。
* V1 all-attention 的 direct-v spatial GradRMS 在 11/12 行超过 temporal 的 2 倍；由于本阶段未同时观测 LPIPS 方向，不把它单独升级为因果结论，temporal-only 仍作为已通过隔离门槛的 pilot 主基线。

`P2-V2-PILOT-03` 不能开跑：它要求 8 个人工合理、无 future-GT 泄漏的 Base replay pair，而 static V1 已被条件有效性门槛拒绝、generated point-track provider 尚未实现。因此先执行 `P2-V2-GEN-04`，不得用 legacy synthetic pair 替代该前置条件。

---

# 14. P2-V2-PILOT-03：单步容量与参数化测试

## 14.1 Pilot 数据

冻结 Base，25 inference steps，构建 8 个有效 replay pair。

要求：

* `parent_kind=base`；
* 无 adapter；
* 无 future GT ego/track 泄漏；
* frame 0 不修改；
* target 人工合理；
* 4 个用于 capacity overfit；
* 4 个只作可视化和实现 sanity check。

注意：

> 4 个 held-out pair 不足以检验泛化，不能把其结果写成 generalization conclusion。

## 14.2 固定 noise bank

为每个 pair 预生成固定：

```text
sigma
epsilon
z_sigma
```

所有版本使用完全相同 noise bank。

## 14.3 参数化对照

| 版本 | 目标                                  |
| -- | ----------------------------------- |
| A  | 当前 unweighted absolute x0 MSE       |
| B  | clipped weighted x0 MSE             |
| C  | absolute direct-v target            |
| D  | teacher-relative residual-v target  |
| E  | D + continuous trust-region scaling |

共同配置：

```text
temporal-only LoRA
rank 16
no real loss
no full-image anchor
no random data augmentation
fixed sampler order
deterministic optimizer
```

## 14.4 Capacity 验收

在最多 200 optimizer updates 内：

* train masked target error 至少下降 80%；
* outside-mask teacher drift 不超过 teacher output RMS 的 2%；
* frame 0 drift 接近数值零；
* gradient finite 且非零；
* 无 NaN；
* target-v roundtrip 正确；
* correction 方向与 latent target 一致。

如果 C/D/E 全部失败：

* 不进入 rollout 训练；
* 优先排查：

  * raw-v API；
  * sigma 解释；
  * VAE scaling；
  * Base/projected latent 对齐；
  * mask resize；
  * frame indexing；
  * condition cache；
  * LoRA 模块选择。

## 14.5 小规模泛化检查

capacity test 通过后，另构建：

```text
16 train pair
8 validation pair
```

只选择 D/E 中最稳定版本。

要求：

* validation one-step masked error 相比初始化有一致下降趋势；
* 不要求小样本下达到显著性；
* 不允许根据 8 个 validation pair 连续调参。

---

# 15. P2-V2-REPLAY-05：正式 Base rollout replay cache

## 15.1 数量

目标：

```text
64 conditions × 2 generation seeds = 128 candidate rollouts
```

最终有效数不得事先保证，由 component coverage 和质量门决定。

## 15.2 分层采样

条件必须覆盖：

* near-static ego；
* straight ego motion；
* turning ego motion；
* low/high foreground motion；
* sparse/dense traffic；
* day/night 或不同成像条件；
* easy/hard Base motion error。

不得只挖最高 drift 样本，否则会导致 replay 分布过窄。

建议分层：

```text
25% low-to-medium error
50% medium error
25% high error
```

灾难性生成失败单独归档，不直接作为 projector training pair。

## 15.3 人工复核

随机抽取 20 个有效 target，人工查看：

```text
Base
Projected
Mask
Static/track overlay
Difference
Independent evaluator
```

通过标准：

* 至少 70% 判为局部修正合理；
* 条件不确定样本单独计数；
* 不允许把 warping 伪影本身作为主要否决理由；
* 重点检查轨迹、背景运动和物体支撑关系；
* 如果 static 与 object component 判断不同，分开记录。

如果合理率低于 70%，禁止训练，返回 projector/auditor 修复。

---

# 16. 训练暴露控制

不再仅按 optimizer step 控制训练长度。

训练器必须实际记录每个 sample ID 的访问次数：

```text
sample_seen_count.json
```

分别统计：

```text
replay exposure
real exposure
component exposure
```

报告：

* mean；
* median；
* p90；
* p95；
* max。

诊断阶段：

```text
replay p95 exposure <= 8
```

不得仅用平均值掩盖少数样本被反复抽取。

当使用 `micro_batch=1`、`grad_accum=8` 时，训练预算必须由有效 replay 数量反推。

---

# 17. P2-V2-CAUSAL-06：因果配方对照

禁止 Optuna。固定：

* replay cache；
* Base checkpoint；
* LoRA scope；
* rank；
* optimizer；
* learning rate；
* sigma range；
* generation seeds；
* evaluation clips。

比较以下实验：

| ID | 配方                                    | 目的                         |
| -- | ------------------------------------- | -------------------------- |
| E0 | Base，不训练                              | 基准                         |
| E1 | real-only                             | 检验普通 SFT 的 motion collapse |
| E2 | V1 synthetic control                  | 复现已知负趋势                    |
| E3 | replay residual-v correction only     | 检验 replay 信号本身             |
| E4 | replay + real，independent sigma/noise | 噪声不对齐对照                    |
| E5 | replay + real，shared sigma/noise      | 检验 noise alignment         |
| E6 | E5 + outside-mask preserve            | 完整 V2 配方                   |

每组：

* 初始 50 optimizer steps；
* 仅在训练稳定、exposure 合法、one-step 诊断正常时延长到 100 step；
* 同一训练 seed；
* 同一 paired generation seed；
* 禁止根据中间结果修改阈值后重跑同名实验。

晋级到下一阶段的候选最多 1 个。

---

# 18. P2-V2-SCALE-07：噪声尺度诊断

仅对 `P2-V2-CAUSAL-06` 的唯一最佳配方运行。

比较：

```text
low
mid
mixed mid-low
mixed + sigma-conditioned eta
```

初始定义：

### Low

当前 V1 低噪 quantile 范围。

### Mid

[
q_\sigma\in[0.15,0.55].
]

### Mixed

按固定概率混合 low 与 mid，不做连续搜索。

### Sigma-conditioned correction

允许预注册形式：

[
\eta(\sigma)
============

\eta_0
\cdot
\operatorname{clip}
\left(
\frac{\sigma}{\sigma_{\text{ref}}},
\eta_{\min},
\eta_{\max}
\right).
]

所有参数必须在运行前写入 config。

如果只有 mid/mixed noise 改善完整 rollout，则方法表述调整为：

> Scale-Aligned Projected Replay Distillation

不再强调 low-noise 是核心贡献。

---

# 19. 快速评估协议

## 19.1 数据规模

### 诊断筛选

```text
16 validation clips
2 generation seeds per clip
```

### 晋级评估

```text
32 validation clips
2 generation seeds per clip
2 training seeds
```

### 论文级主结果

```text
完整或大规模 validation split
至少 3 training seeds
```

## 19.2 生成设置

* Base 与 tuned model 使用相同：

  * conditioning；
  * initial noise；
  * sampler；
  * inference steps；
  * generation seed；
  * motion bucket；
  * fps；
* 主结果使用 25 denoising steps；
* 8 steps 只作为低步数鲁棒性附加结果；
* 当前项目仍可保持 8-frame clip；
* 不得混淆“25 denoising steps”和“SVD-XT 原生 25 frames”。

## 19.3 Primary dynamics metrics

分别报告有效 coverage。

### Static/background

* training auditor static residual；
* independent background-motion residual；
* background temporal flicker；
* correction coverage。

### Object/point track

* camera-motion-removed acceleration；
* jerk；
* track survival；
* track length；
* disappearance/identity consistency；
* valid track count。

无有效 component 时：

```text
metric = invalid
```

不得填 0。

## 19.4 Visual preservation metrics

快速阶段使用：

* outside-mask LPIPS：tuned 对 Base，同 seed；
* outside-mask DINO feature distance：tuned 对 Base；
* subject/background consistency；
* temporal flicker；
* dynamic degree；
* GT LPIPS 作为次要 reconstruction 指标。

注意：

> 单一 GT future 不是 image-conditioned generation 的唯一合法未来，不能只用 tuned-to-GT LPIPS 判断视觉质量。

## 19.5 FVD

FVD 不进入 16/32-clip 配方筛选。

仅在：

```text
至少 256 clips，优先完整 732-clip val
```

上报告 FVD。

## 19.6 独立 evaluator

训练和验证不得完全同源。

建议：

```text
train flow: RAFT
eval flow: SEA-RAFT 或其他独立实现

train tracks: RAFT-chain
eval tracks: CoTracker3
```

并进行小规模人工 pairwise review。

## 19.7 统计单位

生成 seed 不是独立 clip。

统计时：

1. 先在同一 clip 内聚合两个 generation seeds；
2. 再以 clip 为 bootstrap unit；
3. 使用 paired hierarchical bootstrap；
4. 不把 32 clip × 2 seed 当作 64 个独立样本。

报告：

* paired delta；
* mean；
* median；
* win rate；
* bootstrap 95% CI；
* worst 10%；
* valid coverage；
* per-component coverage。

---

# 20. 晋级标准

配置只有同时满足以下条件，才能扩大 cache 或训练至 300 step：

1. 25-step rollout 中 dynamics win rate 不低于 60%；
2. 至少一个 primary dynamics metric 的 paired bootstrap 95% CI 上界小于 0；
3. 另一个有效 primary metric 的均值退化不超过 5%；
4. 独立 evaluator 的改善方向一致；
5. outside-mask visual metric 相对 Base 退化不超过 5%；
6. component valid coverage 下降不超过 5 个百分点；
7. 两个 generation seed 的方向一致；
8. 第二个 training seed 复现同方向改善；
9. 改善存在于完整 rollout，而不是只有 one-step error；
10. 人工 pairwise review 不显示明显的画质、身份或运动投机。

单 clip dynamics win：

* 至少一个有效 primary dynamics component 改善；
* 其他有效 component 退化不超过预注册容差；
* 无有效 component 的 clip 不进入该 component 的 win-rate 分母。

---

# 21. 明确停损线

## 情况 A：条件有效性失败

现象：

* SVD generated camera motion 与 GT ego trajectory 明显不一致；
* GT-ego projector 人工合理率低；
* self-estimated static projector 也无法稳定工作。

结论：

> SVD 不适合验证 control-consistent static geometry projection。

处理：

* SVD 只保留 object/point-track self-consistency；
* GT ego static projector 仅作 synthetic unit test；
* static control-consistent 主结论推迟到 OpenDWM；
* 不得继续使用 GT future ego 训练 SVD。

## 情况 B：generated track provider 失败

现象：

* 有效轨迹覆盖率过低；
* tracker 在生成伪影上不稳定；
* projector target 人工合理率不足。

处理：

* object replay branch 标为 blocked；
* 不以 GT track 替代；
* 只推进已通过的 static branch；
* 不报告 object improvement。

## 情况 C：单步无法拟合

现象：

* direct-v 和 teacher-relative residual-v 无法降低 fixed-noise target error。

结论：

> 优先视为实现或参数化问题，而非完整方法失败。

排查：

* raw-v；
* sigma；
* VAE；
* mask；
* frame；
* condition；
* LoRA scope。

问题未关闭前不得跑 rollout。

## 情况 D：单步成功，完整 rollout 不改善

结论：

> endpoint-level one-step correction 无法有效改变完整生成轨迹。

停止：

* 不扩大 replay；
* 不增加训练步数；
* 不继续调 lambda。

下一候选方向按顺序为：

1. temporal feature alignment；
2. motion-centric hidden-state alignment；
3. short-chain truncated rollout training。

不得同时实现三个 fallback。

## 情况 E：只有训练 auditor 改善

现象：

* training auditor 显著改善；
* independent evaluator 与人工评审不改善或反向。

结论：

> 存在 metric hacking 或 evaluator 不可靠。

处理：

* 暂停性能宣称；
* 重新验证 tracker、flow、coverage 和人工一致性；
* 禁止以训练 auditor 作为唯一晋级依据。

## 情况 F：完整 V2 仍为负

在以下全部完成后：

* condition validity；
* Base replay；
* direct-v；
* temporal-only LoRA；
* shared sigma/noise；
* outside-mask preserve；
* independent evaluation；

若仍无正 rollout 增益：

```text
P2-FRONT-01 = rejected
```

停止：

* P3 方法迁移；
* P4 当前方法迁移；
* 当前 endpoint projection distillation 路线。

保留：

* projector；
* auditor；
* cache；
* runtime；
* evaluation；
* 负结果与 failure analysis。

---

# 22. 工程任务清单

Coding Agent 按顺序执行：

## Phase A：事实归档与阻塞审计

* [x] 更新 `docs/EXPERIMENTS.md`
* [x] 更新本计划状态
* [x] 修复 watchdog terminal state
* [x] 增加 `generated_geometry_mode`
* [x] 完成 16-case condition validity report
* [x] 完成人工 condition review
* [x] 决定 SVD static branch 是否解锁（`blocked`）

## Phase B：参数化与 LoRA

* [x] 增加 raw-v backbone API
* [x] 增加 x0/v 双向变换
* [x] 增加 sigma floor
* [x] 增加 temporal/spatial LoRA scope
* [x] 保存 selected module list
* [x] 完成代数与模块测试

## Phase C：V2 loss

* [x] 新增 teacher-relative residual-v target
* [x] 新增 continuous trust-region scaling
* [x] 新增 static/object component loss
* [x] 新增 outside-mask preserve
* [x] 修改 real loss 接受外部 sigma/noise
* [x] 增加 shared-noise 模式
* [x] 增加所有 loss 单元测试

## Phase D：梯度与单步实验

* [x] 实现 gradient audit CLI
* [x] 完成 V1/V2 gradient report
* [ ] 构建 8-pair Base pilot（blocked：等待 `P2-V2-GEN-04`）
* [ ] 完成 A/B/C/D/E 对照
* [ ] 完成 16/8 one-step generalization sanity check

## Phase E：Generated replay

* [ ] replay miner 支持 `parent_kind=base`
* [ ] 正式 V2 禁止加载 adapter
* [ ] 实现 generated background provider
* [ ] 实现 generated track provider
* [ ] 增加 GT leakage guard
* [ ] 新增 cache schema V5
* [ ] 构建 64×2 candidate replay
* [ ] 完成 20-case 人工复核

## Phase F：因果与尺度

* [ ] 完成 E0–E6 因果对照
* [ ] 选出最多一个候选
* [ ] 完成 low/mid/mixed scale 对照
* [ ] 完成第二 training seed
* [ ] 完成 32-clip promotion report
* [ ] 作出 promote 或 reject 决策

---

# 23. 推荐测试文件

至少新增：

```text
tests/test_watchdog_terminal_state.py
tests/test_svd_parameterization.py
tests/test_lora_scope.py
tests/test_projected_replay_target.py
tests/test_trust_region_scaling.py
tests/test_outside_mask_preserve.py
tests/test_shared_noise.py
tests/test_first_frame_freeze.py
tests/test_replay_parent_base.py
tests/test_replay_no_gt_leakage.py
tests/test_cache_schema_v5.py
tests/test_component_confidence.py
tests/test_hierarchical_bootstrap.py
```

正式 V2 cache 前必须执行：

```bash
pytest -q
```

不得仅运行新增测试而忽略历史回归。

---

# 24. 建议执行节奏

| 时间窗口        | 工作                                         |
| ----------- | ------------------------------------------ |
| 7 月 12–13 日 | V1 归档、watchdog、条件有效性审计                     |
| 7 月 13–15 日 | raw-v、LoRA scope、gradient audit            |
| 7 月 15–17 日 | 8-pair 单步 A/B/C/D/E                        |
| 7 月 17–20 日 | generated provider、schema V5、64–128 replay |
| 7 月 20–24 日 | E0–E6 因果对照                                 |
| 7 月 24–27 日 | scale 消融、第二训练 seed、晋级或止损                   |
| 通过晋级后       | 扩大 replay 和 P2 正式主实验                       |
| P2 主结果通过后   | 解锁 P3/P4                                   |

日期为研发排序，不得作为降低验收标准的理由。

---

# 25. 外部研究依据与边界

## 25.1 SHIFT

SHIFT 在 SVD 上：

* 使用 temporal-attention-only LoRA；
* 发现普通 SFT 会改善 appearance、损害 motion；
* 让 SFT 与 motion alignment branch 共享相同 timestep 和 noise；
* 使用独立的瞬时与长时运动监督；
* 强调防止 reward hacking。

本项目只借鉴：

```text
temporal-only diagnosis
shared sigma/noise
motion/appearance decoupling
independent long-term evaluator
```

不实现其 AWR、reward model 或 adversarial training。

## 25.2 MotionDirector，ECCV 2024

MotionDirector 使用 spatial/temporal dual-path LoRA 和 appearance-debiased temporal loss，说明视频微调中直接混合空间外观与时间运动参数容易造成纠缠。

本项目只借鉴 temporal/spatial 参数隔离，不采用 motion customization 任务设定。

## 25.3 Track4Gen，CVPR 2025

Track4Gen 通过点跟踪监督 diffusion feature，改善 appearance drift 和时序一致性。

本项目当前仍优先检验 endpoint projected replay；只有在“单步 endpoint 可学、完整 rollout 无效”后，才允许将 feature-level tracking alignment 作为下一条路线。

## 25.4 DenseDPO

DenseDPO 使用结构对齐的视频 pair 和局部时间段监督，避免独立生成 pair 带来的粗粒度比较和低运动偏置。

本项目不采用 DPO，但借鉴：

```text
同源结构对齐 pair
局部时空监督
不把整条 clip 压成单一 scalar label
```

## 25.5 SIFT

SIFT 是 2026 年近期预印本，其核心观察是从模型自身生成结果学习，并使用 progressive hard-case replay，避免只做真实视频 reconstruction。

本项目只借鉴 Base-generated replay 和 hard-case 分层采样，不采用其 discriminative supervision。

## 25.6 ShortFT，ICCV 2025

ShortFT 使用较短去噪链进行 reward alignment，以降低完整 chain backpropagation 的计算和梯度风险。

本项目当前不展开去噪链。只有 endpoint correction 被明确证伪后，才把短链训练作为 fallback，而不是并行扩张当前范围。

## 25.7 MoAlign

MoAlign 使用与光流相关的 motion-centric representation alignment。

本项目将其视为 feature-level fallback 参考，而非当前 V2 的直接 baseline。

---

# 26. 变更日志

| 日期         | 基线 commit             | 变更                                                          | 原因                                                                  |
| ---------- | --------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------- |
| 2026-07-11 | `f11645b` 至 `ae826a1` | 建立 projector、cache、训练、恢复和调参链路                               | 保证实验可追溯                                                             |
| 2026-07-12 | `ae826a1`             | V1 rejected                                                 | 16×100 全部负动力学增益，300-step 未反转                                        |
| 2026-07-12 | `ae826a1`             | 增加 P2-V2 条件有效性门槛                                            | SVD 未接收 future ego control，但旧 generated audit 复用 GT future ego pose |
| 2026-07-12 | `ae826a1`             | 将 V2 改为 Base-parent、teacher-relative residual-v、区域解耦 replay | 修复 parent distribution、参数化、anchor 和噪声尺度风险                           |
| 2026-07-12 | `ae826a1`             | generated object supervision 改为无 GT point tracks            | 旧 replay 清空 boxes，无法形成 object dynamics 闭环                           |
| 2026-07-12 | `ae826a1`             | FVD 移出 16/32-clip 筛选                                        | 避免用小样本分布指标作不稳定决策                                                    |
| 2026-07-12 | `9a1f536`             | 完成 `P2-V2-ARCHIVE-00` 并修复 watchdog terminal state            | 正式归档 V1 negative result；终态 run 不再误报 heartbeat stale                    |
| 2026-07-12 | `fff5ccb`             | 完成 16-case condition validity 与人工复核                           | H0 确认；GT static rejected；self-estimated static V1 以 66.67% 未过门槛          |
| 2026-07-12 | `5bd7a18`             | 完成 SVD raw-v 参数化与 temporal-only LoRA 隔离                         | 关闭代数、sigma floor、adapter 恢复与完整路径模块选择风险；解锁 V2 loss/梯度审计             |
| 2026-07-13 | `ce52feb` / `63d9bd0` | 完成 V2 residual-v loss 与 V1/V2 fixed-noise 梯度审计                    | direct-v/trust-region 梯度有限；V1 all-attention spatial GradRMS 多数行高于 temporal；无有效 Base replay pair，pilot blocked |
| 2026-07-13 | `pending` | 开始 `P2-V2-GEN-04` RAFT point-track 工程接线 | generated mode 彻底隔离 source future boxes；真实 Base panel 尚未运行 |
| 2026-07-13 | `3cb8445` | 生成首个无 GT point-track Base panel | 自动检查通过、62 条有效轨迹；等待人工 review，不构建 replay cache |

---

# 27. Coding Agent 的当前唯一下一步

立即执行且只执行：

```text
P2-V2-GEN-04
```

在 `P2-V2-GEN-04` 完成前：

* 不构建 8-pair pilot 或 64–128 replay cache；
* 不启动训练；
* 不使用 future GT ego pose 生成正式 SVD target；
* 不恢复已被人工门槛拒绝的 SVD self-estimated static V1。

先实现并验证 generated point-track provider；只在 provider 的无 GT future 泄漏、轨迹有效性和人工检查均通过后，才构建 Base replay pair。

完成后输出：

```text
1. 修改文件列表
2. 测试结果
3. gradient audit 证据路径与 summary
4. raw-v / direct-v / preserve 梯度有限性与尺度结论
5. LoRA scope 与 selected module 统计
6. 下一 milestone 建议或阻塞原因
7. Git diff summary
8. 当前 worktree 状态
```
