# WorldSim V4：EviDelta-GS 顶会技术报告 / Paper 计划
## Evidence-Calibrated Temporal Delta Gaussian Assets for Editable Driving World Simulation

- **版本**：V4.0
- **日期**：2026-08-11
- **项目根目录**：`/root/autodl-tmp/motion_proj`
- **V4 目标**：从 V3.3 的“可信研究原型 + 强工程 artifact”升级为一份具有 CVPR / ICCV / ECCV 主会实验完整度的技术报告 / paper
- **当前代码事实**：
  - 当前干净代码：`main@144ed1900af1cdd40c5c0f22f0c566aba2be3fb2`
  - V3.3 收口提交：`e6663e1`
  - V3.3 分支：`research/worldsim-v3.3-object-maintenance`
  - V3.3 canonical R0：`20260810T222701Z__r0-integration-canonical-s0-r7`
  - V3.3 overall：`v33_supported`
- **建议 V4 分支**：`research/worldsim-v4-evidelta`
- **建议 V4 run namespace**：`/root/autodl-tmp/runs/worldsim_v4/`
- **默认硬件**：单卡 NVIDIA GeForce RTX 3090 24 GiB
- **扩卡原则**：
  - 所有方法、dataset adapter、evaluator、paper main table 必须先在**单卡闭环**
  - 单卡闭环之前不得用多卡掩盖显存 / 算法 / 数据问题
  - 单卡闭环以后，如仅剩场景规模与 wall-time 瓶颈，可增加多卡做**scene-level 横向并行**
  - 默认仍保持“一个 scene / 一个正式 run = 一张 GPU”，不做为了堆资源而改算法的 DDP
- **数据主线**：
  1. 扩充 nuScenes
  2. 冻结 V4 方法
  3. 加入 KITTI 做跨数据集 confirmation
- **KITTI 公共盘**：
  - `/root/autodl-pub/KITTI`
  - 官方来源：<http://www.cvlibs.net/datasets/kitti/>
  - **禁止重新下载 KITTI**
- **本文件性质**：V4 唯一当前研究执行计划；V3/V3.1/V3.2/V3.3 作为历史事实和 baseline，不再恢复为 active task

---

# 0. 一页执行结论

V4 不再继续“接一个模块、再接一个模块”。

V3.3 已经有：

```text
StreetGS / DriveStudio base
→ object-aware dual-opacity field
→ RoadPatch-Lite
→ Asset Harvester
→ spatial delta
→ semantic-safe renderer
→ mixed / chunk / content-addressed release
```

V4 将这些收束成一个可以写进论文 Method 的中心方法：

> **EviDelta-GS：Evidence-Calibrated Temporal Delta Gaussian Assets**

核心只保留三块：

```text
Immutable Dynamic Gaussian Base
            ↓
1. Evidence-Calibrated Gaussian Field
   - object ownership
   - geometry / LiDAR support
   - provenance authenticity
   - uncertainty
   - temporal memory
            ↓
2. Evidence-Prioritized Delta Compiler
   - ERASE
   - OBSERVED REPAIR
   - REAL DONOR PATCH
   - GENERATED INSERT
   - ABSTAIN
            ↓
3. Temporal Delta Renderer
   - SE(3) B-spline trajectory
   - time-consistent erase / repair / insert
   - exact rollback
   - semantic-safe visual layer
```

论文不是“我们用了 SAM、RoadPatch、Asset Harvester、Harmonizer”。

论文应该是：

> **我们把 driving neural asset 的对象归属、修复证据、生成可信度、时序编辑状态统一成一个带校准不确定性的 Gaussian evidence field，并使用风险最小化 delta compiler 决定“删什么、用什么修、何时生成、何时拒绝”，最后用连续时间 delta 表示实现可逆、时序一致的反事实编辑。**

---

# 1. 科研诚信与“数学包装”边界

V4 可以使用经典、成熟、理论味道强的数学工具组织方法，但**不允许为了“显得深”写没有对应实现或实验的公式**。

允许使用：

- 贝叶斯共轭更新（Beta-Bernoulli evidence）
- 概率校准（temperature / beta calibration）
- 贝叶斯决策理论（Bayes risk minimization）
- 鲁棒统计（Huber / trimmed statistics）
- 李群 `SE(3)` / `so(3)`
- 三次 B 样条（cubic B-spline）
- 时序总变差（temporal total variation）
- 有限状态 / Markov memory
- signed delta / reversible state operator
- Pareto / constrained optimization

禁止：

- 写 theorem 但无 proof / 无验证
- 把工程规则换成复杂符号后称“新理论”
- 使用不存在的“物理守恒定律”
- 把生成模型输出写成 ground truth
- 为了指标好看事后换 test、scene、actor、阈值
- 用像素数量做统计显著性，忽略 scene-level dependency

**原则：每一个 Method 公式必须能对应到：**
1. 一个 config；
2. 一个代码函数；
3. 一个 ablation；
4. 一个可计算指标。

---

# 2. V3.3 冻结事实：V4 的出发点

## 2.1 V3.3 已完成

P0、S1–S5、R0 均已收口：

```text
overall = v33_supported
```

当前主要结果：

### S1 Object-aware Field

O0 heuristic → O1 dual-opacity：

```text
Boundary F1: 0.068960 → 0.336158
IoU:         0.063253 → 0.330727
NBD:         0.144958 → 0.105280
FP mass:     0.900308 → 0.623276

FN mass:     0.061278 → 0.109356
```

结论：

```text
precision / boundary 明显提高
但 recall / FN 有 trade-off
```

### S2 RoadPatch

相对冻结 baseline：

```text
static PSNR:       +0.002865 dB
static LiDAR MAE:  0.895636 → 0.890384 m

global PSNR:       -0.084031 dB
SSIM:              -0.000908
LPIPS:             +0.001861
```

结论：

```text
通过非劣门
但没有形成 matched global visual advantage
```

### S3 Asset Harvester high actor

manual 2-view → auto 4-view：

```text
IoU:         +0.023490
Boundary F1: +0.059889

PSNR:        -0.015760 dB
LPIPS:       +0.008527
```

boundary actor generated override：

```text
rejected / abstain
```

### S4 Spatial Delta

```text
base immutable
20 / 20 rollback render SHA exact
deterministic replay exact
full checkpoint copy = 0
```

### S5

semantic-gated enhancer：

```text
development 局部改善
heldout 失败
production = raw 3D fallback
```

### R0

```text
主范围：
scene-0230_primary_frozen_views_single_RTX3090
```

因此：

> V3.3 是强研究原型，不是大规模 SOTA 证据。

---

# 3. V4 Paper 定位

## 3.1 建议正式标题

### 首选

**EviDelta-GS: Evidence-Calibrated Temporal Delta Assets for Editable Driving World Simulation**

中文：

**EviDelta-GS：面向可编辑驾驶 WorldSim 的证据校准时序 Gaussian Delta 资产**

### 备选 1

**Maintainable Gaussian Worlds: Evidence-Calibrated and Reversible Editing for Driving Simulation**

### 备选 2

**From Reconstruction to Maintenance: Bayesian Evidence and Temporal Delta Operators for Editable Driving Gaussians**

---

# 4. Paper 核心贡献设计

最终 paper 最多写 **4 个 contribution**。

## Contribution 1：Evidence-Calibrated Gaussian Field

将：

- instance ownership
- LiDAR / geometry support
- multi-view observation
- generated provenance
- temporal recurrence

统一成带后验概率和不确定性的 Gaussian evidence state。

不是：

```text
SAM mask → hard label
```

而是：

```text
multi-source evidence
→ posterior
→ calibration
→ uncertainty
```

## Contribution 2：Evidence-Prioritized Delta Compiler

把所有 edit repair 统一成：

```text
observe
donor
generate
abstain
```

的风险最小化决策。

核心不是“RoadPatch 更强”，而是：

> **系统能够根据证据质量选择最可信的修复来源，并在高风险时拒绝生成。**

## Contribution 3：Temporal Reversible Delta Operator

把每帧独立 edit：

```text
frame 0 edit
frame 1 edit
...
```

改成连续时间：

```text
trajectory / erase / repair / insert
→ temporal delta
```

使用：

- SE(3) B-spline
- temporal smoothness
- exact reversible authoring

## Contribution 4：跨场景 / 跨数据集 / 工业级验证

在：

```text
nuScenes
+
KITTI
```

上验证：

- PSNR
- SSIM
- LPIPS
- editing
- temporal
- geometry
- engineering production metrics

并把：

```text
wall / VRAM / asset size / cold load / FPS / production throughput / failure recovery
```

纳入 paper 主表或 supplement。

---

# 5. 核心数学方法：EviDelta-GS

---

# 5.1 Base Gaussian Asset

训练完成的场景：

\[
\mathcal G_0 = \{g_i\}_{i=1}^{N}
\]

其中：

\[
g_i =
(\boldsymbol\mu_i,
 \boldsymbol\Sigma_i,
 \alpha_i,
 \mathbf c_i,
 z_i)
\]

分别为：

- center
- covariance
- appearance opacity
- SH / color
- provenance / instance metadata

**V4 不要求重新发明 base reconstruction。**

主 baseline 继续基于 V3.3 / StreetGS asset。

---

# 5.2 Gaussian Evidence State

对 Gaussian \(i\)、actor/object \(a\)、时间 \(t\)：

定义：

\[
E_{i,a,t} =
(\alpha^{E}_{i,a,t},
 \beta^{E}_{i,a,t})
\]

表示 Beta posterior：

\[
p_{i,a,t}
=
\mathbb E[\theta_{i,a,t}]
=
\frac{\alpha^{E}_{i,a,t}}
     {\alpha^{E}_{i,a,t}+\beta^{E}_{i,a,t}}
\]

后验方差：

\[
u_{i,a,t}
=
\frac{
\alpha^{E}_{i,a,t}\beta^{E}_{i,a,t}
}{
(\alpha^{E}_{i,a,t}+\beta^{E}_{i,a,t})^2
(\alpha^{E}_{i,a,t}+\beta^{E}_{i,a,t}+1)
}
\]

解释：

```text
p = object ownership / support probability
u = uncertainty
```

---

# 5.3 Multi-View Evidence Update

每个训练观察 \(v\)：

构造：

\[
e_{i,a,v}\in[0,1]
\]

来源：

```text
SAM / dual-opacity object mask
× rendering visibility
× depth consistency
× LiDAR support
× camera confidence
```

权重：

\[
w_{i,v}
=
w^{vis}_{i,v}
w^{depth}_{i,v}
w^{lidar}_{i,v}
w^{mask}_{i,v}
\]

更新：

\[
\alpha^{E}_{i,a}
=
\alpha_0
+
\sum_v
w_{i,v}e_{i,a,v}
\]

\[
\beta^{E}_{i,a}
=
\beta_0
+
\sum_v
w_{i,v}(1-e_{i,a,v})
\]

这将替代 V3.3 中单纯 heuristic posterior threshold。

---

# 5.4 Temporal Evidence Memory

使用带遗忘因子的递归证据：

\[
\alpha_{t}
=
\rho\alpha_{t-1}
+
(1-\rho)\alpha^{obs}_{t}
\]

\[
\beta_{t}
=
\rho\beta_{t-1}
+
(1-\rho)\beta^{obs}_{t}
\]

其中：

\[
\rho \in [0,1]
\]

只在 development scenes 冻结。

目的：

- 防单帧 SAM 抖动
- 防短遮挡 identity drift
- 保留旧证据但允许新证据覆盖

---

# 5.5 Calibration

V4 必须做 probability calibration，而不是把 raw score 当概率。

候选：

```text
C0 raw posterior
C1 temperature scaling
C2 beta calibration
```

只在 `nuScenes development` 拟合。

正式测试：

- ECE
- Brier score
- reliability diagram

若：

```text
C1/C2 不改善
```

保留 raw posterior，不强制复杂化。

---

# 5.6 Authenticity / Support State

除了 object probability：

定义：

\[
q_i \in [0,1]
\]

表示 Gaussian 的真实证据可靠度。

来源先验：

```text
OBSERVED / LiDAR / real donor
    high prior

cross-view reconstructed
    medium-high

generative insert
    lower prior

unknown
    high uncertainty
```

V4 中：

```text
q_i
```

用于：

- repair routing
- generated Gaussian protection / suppression
- evaluation分层
- 不作为 GT 标签

---

# 5.7 Bayes Repair Decision

对编辑后 hole \(H\)：

候选 action：

\[
a
\in
\{
A_{\mathrm{obs}},
A_{\mathrm{donor}},
A_{\mathrm{gen}},
A_{\mathrm{abstain}}
\}
\]

定义 risk：

\[
R(a|E)
=
\lambda_p R_{photo}
+
\lambda_g R_{geom}
+
\lambda_t R_{temp}
+
\lambda_u U
+
\lambda_c C
\]

其中：

- \(R_{photo}\)：预测 appearance mismatch
- \(R_{geom}\)：depth / geometry conflict
- \(R_{temp}\)：时序不一致风险
- \(U\)：uncertainty
- \(C\)：工程成本

Bayes decision：

\[
a^*
=
\arg\min_a R(a|E)
\]

如果：

\[
\min_a R(a|E) > \tau_{\mathrm{risk}}
\]

则：

```text
ABSTAIN
```

这成为 V4 M2 的中心算法。

---

# 5.8 Evidence Priority

初始先验顺序：

```text
真实 cross-view observation
>
同场景 native Gaussian donor
>
生成式 repair
>
ABSTAIN
```

不是 hard rule；

最终由 risk 决定。

必须做消融：

```text
hard priority
vs
learned/calibrated risk
```

---

# 5.9 Temporal Delta Transform

动态 actor / edit trajectory 使用连续时间 SE(3)。

平移 / 旋转 Lie algebra：

\[
\boldsymbol\xi(t)
=
\sum_{k}
B_{k,3}(t)\mathbf c_k
\]

其中：

- \(B_{k,3}\) 为 cubic B-spline basis
- \(\mathbf c_k\in\mathfrak{se}(3)\)

transform：

\[
T(t)
=
T_0\exp(\hat{\boldsymbol\xi}(t))
\]

这样：

- lateral
- acceleration
- brake
- stop/restart
- insert trajectory

不再每帧独立定义。

这部分可参考 AD-GS “局部 B-spline motion” 的建模思想，但 V4 只使用与我们的 edit contract 对应的 B-spline，不为了包装强行加入无意义周期函数。

---

# 5.10 Temporal Smoothness

\[
\mathcal L_{\mathrm{acc}}
=
\int
\left\|
\frac{d^2\boldsymbol\xi(t)}
     {dt^2}
\right\|^2 dt
\]

delta image consistency：

\[
\mathcal L_{\mathrm{warp}}
=
\sum_t
\left\|
D_t -
\mathcal W_{t-1\rightarrow t}(D_{t-1})
\right\|_1
\]

object probability：

\[
\mathcal L_{\mathrm{evidence-temp}}
=
\sum_{i,t}
|p_{i,t}-p_{i,t-1}|
\]

只对共享可见 Gaussian 生效。

---

# 5.11 Reversible Delta Asset

base：

\[
\mathcal G_0
\]

编辑 delta：

\[
\Delta_t =
(
\mathcal E_t,
\mathcal G_t^{+}
)
\]

其中：

- \(\mathcal E_t\)：base Gaussian erase selector
- \(\mathcal G_t^{+}\)：repair / actor insert Gaussian

最终 state：

\[
\mathcal G(t)
=
\mathcal C(
\mathcal G_0,
\mathcal E_t,
\mathcal G_t^{+}
)
\]

强调：

> Gaussian renderer 是非线性 alpha compositing，因此 V4 **不声称图像空间线性可加**。

我们只声称：

```text
asset state 可组合
base immutable
delta 可撤销
```

rollback：

\[
\Delta_t=\varnothing
\Rightarrow
\mathcal G(t)=\mathcal G_0
\]

工程验证仍使用：

```text
render SHA exact
```

---

# 5.12 Unified V4 Objective

M1/M2/M3 总体可写成：

\[
\mathcal L =
\lambda_{obj}\mathcal L_{obj}
+
\lambda_{cal}\mathcal L_{cal}
+
\lambda_{geom}\mathcal L_{geom}
+
\lambda_{photo}\mathcal L_{photo}
+
\lambda_{temp}\mathcal L_{temp}
+
\lambda_{protect}\mathcal L_{protect}
\]

其中：

```text
L_obj:
object / instance mask

L_cal:
probability calibration / evidence consistency

L_geom:
LiDAR / depth / donor geometry

L_photo:
PSNR-oriented L1/SSIM training loss，非正式评测 PSNR

L_temp:
B-spline / warp / flicker consistency

L_protect:
non-target / immutable base protection
```

正式指标依然：

```text
PSNR / SSIM / LPIPS
```

不能把 training loss 当评测指标。

---

# 6. 数据策略

---

# 6.1 nuScenes：先从 3 scene 扩到 30 scene

V4 paper target：

```text
30 scenes
```

拆分：

```text
development:  6
validation:   6
test:        18
```

任何模型结果出现前冻结。

## 6.1.1 Scene sampling

根据输入 metadata 做分层，不看模型结果：

- 日间 / 夜间
- 晴 / 雨
- 直路
- 弯道
- 路口
- high actor support
- heavy occlusion
- road donor support strong / medium / weak
- stationary / low-speed / normal-speed
- actor 类别
- 近场 / 中场

## 6.1.2 每 scene 最少冻结

```text
1 high-support actor
1 difficult/boundary actor or explicit ABSTAIN
1 remove
1 lateral
1 insert
1 2–4 s continuous clip
```

如果该 scene 没有满足条件 actor：

```text
ABSTAIN_NO_ACTOR
```

保留在 denominator。

---

# 6.2 nuScenes 单卡 staged scale

为了防止 30 scene 一上来把项目拖死：

### D0-S

```text
2 scene smoke
```

只验证 adapter / manifest / training。

### D0-M

```text
6 development scenes
```

完成：

- V3.3 replay
- M1 development
- M2 development
- M3 development

### D0-V

```text
6 validation scenes
```

冻结方法后只读验证。

### D0-T

```text
18 test scenes
```

**test 只跑一次。**

只有：

```text
development + validation
```

单卡完整闭环后才授权 test。

---

# 6.3 KITTI：第二数据集

公共路径：

```text
/root/autodl-pub/KITTI
```

## 6.3.1 禁止下载

V4 coding agent：

```text
只审计本地目录
```

不得执行：

```text
wget / curl / aria2 / kaggle / gdown
```

获取 KITTI。

## 6.3.2 首选协议：KITTI Tracking Training

优先寻找：

```text
training/
├── image_02
├── image_03
├── velodyne
├── label_02
└── calib
```

因为 tracking training 有：

- 连续视频
- object ID
- 3D boxes
- LiDAR
- stereo color cameras

非常适合：

- dynamic actor
- editing
- temporal identity

如果公共盘实际是 Raw：

寻找：

```text
date_drive_sync
image_02 / image_03
velodyne_points
oxts
tracklets
calib
```

P0/D0 必须根据实际目录生成 manifest，不允许凭记忆硬编码。

## 6.3.3 KITTI camera contract

正式优先：

```text
image_02
image_03
```

即两路彩色相机。

不为了对齐 nuScenes 三相机协议伪造第三相机。

## 6.3.4 KITTI split

### adapter smoke

```text
2 sequences
```

只做：

- 坐标
- calib
- pose
- LiDAR
- actor ID
- mask / box
- render

不用于调方法阈值。

### cross-domain formal

目标：

```text
10 sequences
```

若资源允许：

```text
扩到所有满足合同的 tracking training sequences
```

V4 参数保持 nuScenes frozen。

KITTI：

```text
禁止重新搜：
evidence threshold
risk threshold
B-spline smoothness
repair router weight
```

否则不叫 cross-dataset generalization。

---

# 6.4 KITTI 正式适配门

必须通过：

1. meter / axis / handedness；
2. `T_velo_cam`；
3. rectification；
4. camera intrinsics；
5. frame timestamp order；
6. actor track ID；
7. 3D box → camera projection；
8. LiDAR → image；
9. object local → world → camera；
10. image_02 / image_03 stereo association；
11. heldout leak；
12. deterministic manifest hash。

任意失败：

```text
blocked_dataset_adapter
```

不输出质量表。

---

# 7. 实验任务注册表

状态只允许：

```text
pending
running
blocked
done
rejected
```

| Task ID | 初始状态 | 内容 |
|---|---|---|
| `WS-V4-P0-SCOPE-PAPER-FREEZE-01` | pending | 冻结论文命题、HEAD、baseline、source、数学定义 |
| `WS-V4-D0-NUSCENES-COHORT-01` | pending | 30-scene scene-disjoint nuScenes cohort |
| `WS-V4-D1-KITTI-ADAPTER-01` | pending | `/root/autodl-pub/KITTI` 本地 adapter |
| `WS-V4-B0-MATCHED-BASELINES-01` | pending | V3.3 / StreetGS / AD-GS / executable baseline |
| `WS-V4-M1-EVIDENCE-FIELD-01` | pending | Bayesian/calibrated temporal evidence |
| `WS-V4-M2-REPAIR-ROUTER-01` | pending | Bayes-risk repair compiler |
| `WS-V4-M3-TEMPORAL-DELTA-01` | pending | SE(3) B-spline temporal delta |
| `WS-V4-E0-NUSCENES-SCALE-01` | pending | 6 dev + 6 val + 18 test |
| `WS-V4-E1-KITTI-CROSSDATA-01` | pending | 2 smoke + 10 formal |
| `WS-V4-E2-ENGINEERING-BENCH-01` | pending | 工业生产指标 |
| `WS-V4-E3-DOWNSTREAM-GAP-01` | pending | perception real-to-sim gap |
| `WS-V4-H0-HUMAN-STUDY-01` | conditional | 人评 |
| `WS-V4-R0-RELEASE-01` | pending | paper release |
| `WS-V4-W0-PAPER-01` | pending | paper 技术报告 |

---

# 8. Baseline 体系

V4 必须避免“我们的每个模块对自己的旧版本”。

---

# 8.1 Tier A：必须可执行

## B0：V3.3 frozen

当前最重要 baseline。

```text
V3.3 O1
+ RoadPatch
+ auto view Asset Harvester
+ spatial delta
+ raw-3D fail-safe renderer
```

## B1：Native StreetGS / DriveStudio

固定同 scene / same split。

目的：

- original NVS
- dynamic object reconstruction
- engineering performance

## B2：AD-GS

项目已有 AD-GS 复现资产则：

```text
优先复用
```

新场景若需要训练：

- 固定官方实现
- same scene
- same train/heldout
- same resolution
- 不改论文模型

AD-GS 主要作为：

```text
dynamic motion / reconstruction baseline
```

不是 delta editing baseline。

---

# 8.2 Tier B：强 baseline，能跑则进主表

## SplatAD

如果官方 code 在单卡 + 数据 adapter 可执行：

报告：

- camera PSNR/SSIM/LPIPS
- LiDAR metrics
- FPS

否则：

```text
blocked_baseline_adapter
```

不伪造。

## IDSplat

官方 code 已发布，但其原生数据支持与当前数据可能不同。

只在：

```text
adapter 成本合理
+
单卡可运行
```

时加入。

## Inpaint360GS

如果 V3.3 被 blocked 的依赖现在仍不满足：

不继续消耗主线资源。

---

# 8.3 Paper-only comparison

下列如果官方 runnable implementation 不存在：

只进入：

```text
Related Work / qualitative boundary
```

不填数值：

- HorizonForge
- RecEdit-Drive
- GOR-IS
- 3D-GIMP
- 其他未公开实现

---

# 9. Main Ablation：只围绕 EviDelta-GS

不要再做 20 个模块表。

主消融：

```text
A0  V3.3 frozen

A1  + Bayesian evidence
A2  + calibration
A3  + risk-based repair routing
A4  + temporal evidence memory
A5  + SE(3) B-spline temporal delta

FULL = A5
```

---

# 9.1 Evidence Ablation

```text
E0 hard heuristic
E1 Beta evidence
E2 E1 + temperature calibration
E3 E1 + beta calibration
E4 E* + temporal memory
```

---

# 9.2 Repair Ablation

必须 matched：

```text
R0 ABSTAIN / no repair
R1 cross-view observation
R2 Telea
R3 RoadPatch
R4 generated repair（若可执行）
R5 V4 risk router
```

要求：

- same base
- same hole
- same view
- same resolution
- same heldout
- same metric
- same resource recording

禁止再出现：

```text
not_directly_ranked
```

作为主表核心比较。

---

# 9.3 Temporal Ablation

```text
T0 frame-independent V3.3
T1 linear interpolation
T2 cubic B-spline
T3 T2 + temporal evidence
T4 T3 + warp / flicker regularization
```

---

# 10. 正式图像指标

用户指定主指标：

```text
PSNR
SSIM
LPIPS
```

全部保留。

---

# 10.1 区域维度

每个方法必须至少报告：

### Global

- PSNR
- SSIM
- LPIPS

### Static

- PSNR
- SSIM
- LPIPS

### Actor

- PSNR
- SSIM
- LPIPS

### Boundary

- PSNR
- SSIM
- LPIPS

### Edited Hole / Insert ROI

- PSNR
- SSIM
- LPIPS

没有 GT 时：

```text
GT metric = undefined
```

不能用生成图自己当 GT。

---

# 10.2 LPIPS

固定：

```text
LPIPS-Alex
```

不要中途换 VGG。

若 supplement 想加：

```text
LPIPS-VGG
```

必须作为 secondary，不替换 Alex。

---

# 11. Editing Metrics

## 11.1 Remove

- operation success
- deleted semantic reintroduction rate
- hole coverage
- static-region preservation
- object mask residual
- background alignment

## 11.2 Lateral / trajectory

- target translation error
- target yaw error
- trajectory RMSE
- silhouette IoU
- Boundary F1
- fragmentation
- contact consistency

## 11.3 Insert

- object identity preservation
- actor IoU
- Boundary F1
- ground-contact error
- occlusion ordering
- shadow/contact consistency

---

# 12. Temporal Metrics

正式连续 clip：

```text
2–4 seconds
```

至少报告：

- temporal LPIPS
- optical-flow warp L1
- optical-flow warp LPIPS
- frame-to-frame flicker
- mask IoU jitter
- Boundary F1 jitter
- identity switch
- centroid jitter
- contact region jitter

如果样本量与依赖足够：

- FVD

FVD 不能因为只有几个 clip 就硬报。

---

# 13. Geometry / LiDAR Metrics

nuScenes 和 KITTI 都有 LiDAR。

最低：

- depth MAE
- depth median error
- relative depth error
- Chamfer distance

若有 sensor renderer：

- ray-drop accuracy
- intensity RMSE

如果没有可靠 intensity：

```text
not_supported
```

不构造。

---

# 14. 概率 / Evidence 指标

V4 特有：

- ECE
- Brier Score
- NLL
- reliability diagram
- uncertainty-error correlation
- selective risk curve
- coverage vs error

最关键：

```text
Selective Editing Curve
```

横轴：

```text
coverage
```

纵轴：

```text
edit error
```

说明：

> 当 uncertainty 高时选择 ABSTAIN，是否真的减少错误。

这会非常适合 paper。

---

# 15. 工业 / 工程指标

参考真实仿真产线需求，V4 工程主表不能只报 FPS。

---

# 15.1 End-to-End Production Time

每 scene：

```text
T_prepare
T_train
T_semantic
T_asset
T_repair
T_compile
T_render
T_eval
T_total
```

单位：

```text
seconds / minutes
```

同时报告比例：

\[
r_k = T_k / T_{total}
\]

这样可以定位真正瓶颈。

---

# 15.2 GPU / Memory

- peak NVIDIA VRAM
- peak torch allocated
- peak torch reserved
- peak cgroup RAM
- OOM count
- oom_kill count

---

# 15.3 Asset Size

- base checkpoint bytes
- semantic evidence bytes
- delta bytes
- actor asset bytes
- package bytes
- bytes / scene meter
- bytes / actor

---

# 15.4 Runtime

- process-cold load
- warm load
- compose wall
- P50 frame time
- P95 frame time
- FPS

固定：

- resolution
- camera count
- CUDA synchronize
- warmup
- number of frames

---

# 15.5 Production Yield

定义清楚，避免复用内部未定义“膨胀比”。

## Pipeline Success Rate

\[
SR =
\frac{
N_{\mathrm{completed}}
}{
N_{\mathrm{attempted}}
}
\]

## Valid Edit Yield

\[
Y_{\mathrm{edit}}
=
\frac{
N_{\mathrm{quality\_accepted}}
}{
N_{\mathrm{requested}}
}
\]

## Counterfactual Expansion Ratio

明确新定义：

\[
ER =
\frac{
N_{\mathrm{valid\ edited\ clips}}
}{
N_{\mathrm{source\ clips}}
}
\]

不能和历史周报中的内部“膨胀比”混为一谈。

---

# 15.6 Throughput

## Scene throughput

```text
scenes / GPU-day
```

## Edit throughput

```text
accepted counterfactual clips / GPU-hour
```

## GPU cost

```text
GPU-hours / accepted clip
```

---

# 15.7 Recovery Metrics

## Retry Amplification

\[
A_{\mathrm{retry}}
=
\frac{
T_{\mathrm{actual}}
}{
T_{\mathrm{ideal\ single\ pass}}
}
\]

## Resume Efficiency

\[
E_{\mathrm{resume}}
=
1-
\frac{
T_{\mathrm{rerun}}
}{
T_{\mathrm{full}}
}
\]

## Minimum Rerun Unit

例如：

```text
prepare
train
semantic
actor
repair
compile
render
```

---

# 15.8 Reproducibility / Reliability

- deterministic replay rate
- source hash preservation
- rollback exact rate
- package verifier pass
- missing artifact rate
- stale terminal rate

---

# 16. Downstream Utility

这是“工业需求 + 顶会审美”的强加分项。

---

# 16.1 2D detector consistency

冻结一个公开 detector。

输入：

```text
real heldout
base render
edited render
rollback render
```

报告：

- detection mAP
- recall
- prediction consistency
- real-to-sim gap

---

# 16.2 BEV / Map

如果当前已有可运行 BEV / online map model：

报告：

- drivable area IoU
- lane IoU
- road boundary IoU

如果环境成本过高：

只做一个冻结 detector 也可以。

不要为了 paper 开另一条大模型训练线。

---

# 16.3 Rollback

要求：

```text
base prediction
vs
rollback prediction
```

应该：

```text
exact / near-exact
```

这能把 spatial delta 的工程价值转成下游证据。

---

# 17. 统计协议

主会质量必须补。

## 17.1 统计单位

```text
scene
```

不是 pixel。

---

# 17.2 报告

每个主指标：

- mean
- median
- std
- IQR
- 95% scene-bootstrap CI

---

# 17.3 Pairwise Test

主比较：

```text
paired scene-level
```

优先：

- paired bootstrap
- permutation test
- Wilcoxon signed-rank

不依赖正态性。

---

# 17.4 Seed

确定性模块：

```text
byte-exact replay
```

即可。

随机模块：

至少：

```text
3 seeds
```

但只在最终候选做，不要所有 arm 都 3 seed 浪费单卡预算。

---

# 17.5 Denominator

所有：

- failed
- blocked
- abstain

必须保留 denominator。

例如：

```text
18/18 test scenes
15 success
2 abstain
1 blocked
```

不能只平均 15 个成功。

---

# 18. 预注册成功门

以下阈值在 test 前由 dev+validation 冻结。

---

# 18.1 M1 Evidence Field

相对 V3.3 O1：

```text
Boundary F1 scene mean >= +0.03
```

同时：

```text
FN semantic mass <= +0.01
```

概率：

```text
ECE 或 Brier 至少一项改善
另一项不关键退化
```

base RGB：

```text
exact
```

如果失败：

```text
M1 rejected
```

不继续加复杂 evidence feature。

---

# 18.2 M2 Repair Router

相对 matched best non-router baseline：

```text
global PSNR delta >= -0.10 dB
SSIM delta >= -0.002
LPIPS delta <= +0.01
```

同时 target/hole：

```text
PSNR / LPIPS / geometry 至少一个主端点改善
```

static LiDAR：

```text
MAE degradation <= 0.02 m
```

selective risk：

必须：

```text
abstain group error > accepted group error
```

证明 uncertainty 有意义。

---

# 18.3 M3 Temporal Delta

相对 frame-independent：

```text
operation success 不退化
```

并：

```text
tLPIPS 或 warp error >= 10% relative improvement
```

identity switch：

```text
>=25% reduction
或原本为0继续为0
```

deleted semantic reintroduction：

```text
0
```

rollback：

```text
100% exact
```

---

# 19. 单卡 RTX3090 执行原则

---

# 19.1 第一阶段绝对单卡

以下全部必须单卡完成：

- P0
- D0 2-scene smoke
- 6 dev baseline
- M1
- M2
- M3
- 6 validation
- KITTI 2-sequence adapter smoke

这是“single-card closure”。

---

# 19.2 Single-card Closure Gate

满足：

```text
6 nuScenes dev
+
6 nuScenes val
+
KITTI 2 adapter smoke
```

并且：

- no OOM
- no silent downscale
- no camera reduction
- evaluator complete
- paper tables can be generated

才允许考虑多卡。

---

# 19.3 扩卡用途

如果之后增加 2 / 4 / N GPU：

只允许：

```text
scene-level parallel
```

例如：

```text
GPU0 scene A
GPU1 scene B
```

不改变：

- batch
- model
- resolution
- method
- threshold

因此多卡结果和单卡算法严格等价。

---

# 19.4 禁止

单卡没闭环前：

- DDP
- FSDP
- tensor parallel
- model parallel
- 更大 diffusion
- 更高 resolution “换指标”
- 多卡重搜参数

---

# 20. Resource Gate

每个正式 GPU stage 前：

```bash
nvidia-smi
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
```

停止：

- CUDA OOM
- oom_kill delta > 0
- cgroup >=90% 连续两次
- free disk <20 GiB
- 必须降低正式 resolution
- 必须减少 camera
- source asset hash mismatch

---

# 21. P0：V4 Scope / Paper Freeze

Task：

```text
WS-V4-P0-SCOPE-PAPER-FREEZE-01
```

只做：

1. 核对真实 HEAD / status；
2. 创建 V4 plan；
3. 新分支；
4. 冻结数学定义；
5. 冻结 baseline；
6. audit 最新 SOTA；
7. audit `/root/autodl-pub/KITTI`；
8. 冻结 dataset split protocol；
9. 不训练。

输出：

```text
configs/worldsim_v4/p0_scope_v1.yaml
docs/WS_V4_P0_SCOPE.md
docs/WS_V4_LITERATURE_MATRIX.md
```

---

# 22. D0：nuScenes Cohort

新增：

```text
configs/worldsim_v4/nuscenes_cohort_v1.yaml
```

每 scene：

```yaml
scene:
role:
weather:
time_of_day:
road_geometry:
actors:
edits:
continuous_clip:
camera_set:
lidar:
train_frames:
development_frames:
heldout_frames:
donor_support:
```

manifest SHA 必须冻结。

---

# 23. D1：KITTI Adapter

新增：

```text
motion_proj/worldsim_v4/datasets/kitti.py
scripts/audit_worldsim_v4_kitti.py
scripts/build_worldsim_v4_kitti_manifest.py
```

必须由 public disk 自动发现。

不得假定：

```text
KITTI tracking 一定存在
```

必须：

```text
detect available layout
```

然后生成：

```text
KITTI_LAYOUT_AUDIT.md
```

---

# 24. B0：Matched Baseline Replay

正式方法前：

对 6 nuScenes dev scene：

跑：

```text
StreetGS
V3.3
AD-GS
```

至少。

目标不是赢；

目标：

```text
验证大规模 evaluator 和 split
```

检查：

- PSNR
- SSIM
- LPIPS
- actor
- boundary
- static
- wall
- VRAM
- bytes
- FPS

只有 baseline 能稳定复现才启动 M1。

---

# 25. M1：Evidence-Calibrated Gaussian Field

代码：

```text
motion_proj/worldsim_v4/
├── evidence_state.py
├── beta_fusion.py
├── evidence_calibration.py
├── evidence_temporal.py
├── evidence_renderer.py
└── evidence_metrics.py
```

配置：

```text
configs/worldsim_v4/m1_evidence_v1.yaml
```

执行顺序：

```text
2 scene smoke
→ 6 scene development
→ freeze
→ 6 validation
```

---

# 26. M2：Evidence-Prioritized Delta Compiler

代码：

```text
repair_candidates.py
repair_risk.py
repair_router.py
repair_compiler.py
selective_metrics.py
```

统一 candidate interface：

```python
RepairCandidate(
    method,
    gaussians,
    photo_risk,
    geometry_risk,
    temporal_risk,
    uncertainty,
    compute_cost,
    provenance,
)
```

输出：

```text
OBSERVED
DONOR
GENERATED
ABSTAIN
```

不得把 route 写死在 if/else 后无法 ablate。

risk 各项必须落盘。

---

# 27. M3：Temporal Delta

代码：

```text
se3_bspline.py
temporal_delta.py
temporal_compiler.py
temporal_metrics.py
```

连续 clip：

```text
2–4 sec
```

最低：

```text
remove
lateral
insert
```

其中 lateral / insert trajectory：

B-spline。

delete：

erase mask / repair delta 也需要 temporal support consistency。

---

# 28. E0：nuScenes 30-scene Scale

顺序：

```text
6 dev       已完成
6 val       已完成
18 test     只读一次
```

test 前生成：

```text
V4_TEST_FREEZE.json
```

包含：

- source commit
- config SHA
- split SHA
- method selection
- thresholds
- baseline list
- metrics list

然后：

```text
git commit
```

再跑 test。

---

# 29. E1：KITTI Cross-Dataset

先：

```text
2 adapter smoke
```

之后 frozen V4：

```text
10 formal sequences
```

如果场景长度太大：

冻结：

```text
固定 clip extraction rule
```

不是人为挑好看的片段。

不得在 KITTI 调：

- calibration
- risk
- repair
- B-spline
- thresholds

---

# 30. E2：Industry Engineering Benchmark

单独一张 paper table。

建议标题：

**System-Level Production Efficiency**

行：

```text
StreetGS
V3.3
EviDelta-GS
```

列：

- prepare min
- train min
- semantic min
- edit compile sec
- render FPS
- cold load sec
- peak VRAM GB
- scene asset MB
- delta MB
- scene / GPU-day
- edit clips / GPU-hour
- valid edit yield %
- pipeline success %
- retry amplification
- resume efficiency

这是与纯科研工作拉开差异的重要表。

---

# 31. E3：Downstream Gap

只冻结一个感知模型即可。

如果当前仓库已有 Grounding DINO：

可先做：

```text
object detection consistency
```

但论文主表最好使用一个标准 detector。

四臂：

```text
real
base render
edited render
rollback
```

目标：

证明：

```text
编辑后的视觉质量不只是 PSNR 好看
```

---

# 32. Human Study

可选但 HorizonForge / 生成编辑类 paper 很常见。

如果做：

至少：

```text
30–50 clips
```

三问题：

1. realism
2. edit correctness
3. temporal stability

pairwise blind：

```text
V3.3 vs V4
```

不要让 reviewer 看到方法名。

---

# 33. Paper Table 设计

## Table 1：nuScenes NVS / Preservation

```text
PSNR ↑
SSIM ↑
LPIPS ↓
```

Global / Static / Actor / Boundary。

## Table 2：Editing

remove / lateral / insert。

## Table 3：Temporal

tLPIPS / warp / flicker / ID switch。

## Table 4：Geometry

Depth MAE / median / Chamfer。

## Table 5：KITTI Cross-Dataset

PSNR / SSIM / LPIPS + edit / temporal。

## Table 6：Engineering

industry metrics。

## Table 7：Ablation

A0–A5。

---

# 34. Paper Figure 设计

## Figure 1

V3.3 problem → EviDelta-GS。

## Figure 2

Bayesian Gaussian Evidence Field：

```text
view evidence
→ Beta posterior
→ calibration
→ uncertainty
```

## Figure 3

Repair router：

```text
observed
donor
generated
abstain
```

## Figure 4

Temporal delta + B-spline。

## Figure 5

nuScenes qualitative。

## Figure 6

KITTI zero-adaptation qualitative。

## Figure 7

Reliability diagram + selective risk curve。

## Figure 8

Industry cost Pareto：

```text
quality
vs
wall
vs
VRAM
```

---

# 35. Paper Method 章节建议

```text
3. Method
3.1 Maintainable Dynamic Gaussian Asset
3.2 Evidence-Calibrated Gaussian Field
3.3 Bayesian Repair Routing
3.4 Temporal SE(3) Delta Representation
3.5 Reversible Delta Compilation
3.6 Training / Optimization
```

这样看起来是一个中心方法，不是 S1/S2/S3/S4 拼装。

---

# 36. Paper Related Work

四类：

## Driving Neural Rendering

- UniSim
- Street Gaussians
- DrivingGaussian
- NeuRAD
- HUGS
- AD-GS
- SplatAD
- IDSplat

## Driving Scene Editing

- HorizonForge
- RecEdit-Drive
- relevant reconstruct-edit pipelines

## 3DGS Object / Maintenance

- OP2GS
- GaME
- FocusGS
- object removal / inpainting

## Sensor-Realistic Simulation

- NeuRAD
- SplatAD
- LiDAR-EVS

---

# 37. Paper Claim Boundary

允许：

> 在 scene-disjoint nuScenes 与 frozen KITTI cross-domain 协议上，EviDelta-GS 改善了可编辑驾驶 Gaussian asset 的对象边界、风险路由与时序一致性，同时保持可逆 delta authoring 和单卡可部署性。

不允许：

- 完整世界模型
- 安全闭环 SOTA
- 所有天气泛化
- 所有城市泛化
- 生成区域真实 correctness
- KITTI test-server SOTA（除非真的提交）
- 多卡性能（单卡闭环前）
- 未运行 baseline 数值

---

# 38. 工程目录建议

```text
configs/worldsim_v4/
├── p0_scope_v1.yaml
├── nuscenes_cohort_v1.yaml
├── kitti_adapter_v1.yaml
├── baseline_matrix_v1.yaml
├── m1_evidence_v1.yaml
├── m2_router_v1.yaml
├── m3_temporal_v1.yaml
├── metrics_v1.yaml
├── engineering_bench_v1.yaml
└── paper_freeze_v1.yaml

motion_proj/worldsim_v4/
├── __init__.py
├── datasets/
│   ├── nuscenes.py
│   └── kitti.py
├── evidence_state.py
├── beta_fusion.py
├── evidence_calibration.py
├── evidence_temporal.py
├── repair_candidates.py
├── repair_risk.py
├── repair_router.py
├── se3_bspline.py
├── temporal_delta.py
├── delta_compiler.py
├── evaluator.py
├── temporal_metrics.py
├── geometry_metrics.py
├── engineering_metrics.py
├── downstream_metrics.py
├── statistics.py
└── release.py
```

---

# 39. Script 建议

```text
scripts/
├── audit_worldsim_v4_start.py
├── build_worldsim_v4_nuscenes_cohort.py
├── audit_worldsim_v4_kitti.py
├── build_worldsim_v4_kitti_cohort.py
├── run_worldsim_v4_baselines.py
├── run_worldsim_v4_m1.py
├── run_worldsim_v4_m2.py
├── run_worldsim_v4_m3.py
├── run_worldsim_v4_nuscenes_scale.py
├── run_worldsim_v4_kitti.py
├── run_worldsim_v4_engineering_bench.py
├── run_worldsim_v4_downstream.py
├── finalize_worldsim_v4_statistics.py
├── build_worldsim_v4_paper_tables.py
└── finalize_worldsim_v4_release.py
```

---

# 40. Test 建议

```text
tests/
├── test_worldsim_v4_beta_fusion.py
├── test_worldsim_v4_calibration.py
├── test_worldsim_v4_evidence_memory.py
├── test_worldsim_v4_repair_router.py
├── test_worldsim_v4_risk_abstain.py
├── test_worldsim_v4_se3_bspline.py
├── test_worldsim_v4_delta_rollback.py
├── test_worldsim_v4_nuscenes_split.py
├── test_worldsim_v4_kitti_projection.py
├── test_worldsim_v4_kitti_track_id.py
├── test_worldsim_v4_temporal_metrics.py
├── test_worldsim_v4_statistics.py
├── test_worldsim_v4_engineering_metrics.py
└── test_worldsim_v4_paper_freeze.py
```

---

# 41. Commit 规划

```text
docs(worldsim): 冻结 V4 paper-first 研究路线
data(worldsim): 冻结 nuScenes 30-scene cohort
data(kitti): 接入本地 KITTI tracking/raw adapter
eval(worldsim): 建立 V4 统一图像时序几何工程评测

research(evidence): 实现 Bayesian Gaussian evidence
research(evidence): 实现 evidence calibration 与 temporal memory
research(repair): 实现 Bayes-risk repair router
research(temporal): 实现 SE3 B-spline temporal delta

research(eval): 完成 nuScenes development / validation
research(eval): 冻结并执行 nuScenes heldout test
research(eval): 完成 KITTI cross-dataset confirmation

perf(worldsim): 完成工业生产效率 benchmark
research(downstream): 完成 real-to-sim perception gap
docs(paper): 生成 paper tables figures 和技术报告
research(worldsim): 完成 V4 release
```

---

# 42. 执行时间 / 算力控制建议

不把时间写成硬 deadline，但使用 stage budget。

## P0 / D0

```text
CPU-heavy
不启动大训练
```

## 6-scene development

先完成。

如果一个完整 scene：

```text
> 2 h GPU
```

需要先 profile 原因，不直接扩 30。

## validation

全部单卡串行。

## test

可单卡 overnight / 多天运行。

如果用户之后授权多卡：

可以：

```text
每卡不同 scene
```

缩短 wall clock。

---

# 43. 早停规则

## 如果 M1 失败

6 validation 多数 scene：

```text
Boundary / calibration 无方向性收益
```

则：

```text
M1 rejected
```

不要再加 feature / transformer。

Paper 可以退化为：

```text
Evidence-routed Delta Compiler
```

## 如果 M2 失败

matched repair：

```text
不优于 RoadPatch / Telea
```

则：

不要继续上更大的 diffusion。

分析：

```text
ABSTAIN / evidence selection
```

是否本身带来更高 valid yield。

## 如果 M3 失败

时序不改善：

不宣称 temporal。

保留：

```text
static reversible delta paper
```

## KITTI 失败

如果 adapter exact 通过但指标退化：

这是正式 cross-domain negative。

如果 adapter exact 不通过：

```text
blocked_dataset_adapter
```

不能当泛化失败。

---

# 44. Paper 最低发表门

如果最终仅：

```text
scene-0230
```

不写主会 paper。

最低：

```text
nuScenes >= 12 scene
+
KITTI >= 2 smoke
+
matched baseline
+
CI
```

可以形成 strong technical report / workshop。

主会目标：

```text
nuScenes 30
+
KITTI 10
+
M1/M2/M3 ablation
+
PSNR / SSIM / LPIPS
+
temporal
+
geometry
+
engineering
+
统计
```

---

# 45. V4 Paper 成功标准

### 方法成功

至少两个成立：

1. M1 evidence 在 6-val + 18-test 保持方向；
2. M2 router 明显优于 best matched repair baseline；
3. M3 temporal 连续 clip 改善。

### 数据成功

```text
nuScenes 30
KITTI 10
```

或有清晰 blocked reason。

### 图像成功

主表至少：

```text
PSNR
SSIM
LPIPS
```

有可解释 improvement / non-degradation。

### 工业成功

EviDelta-GS 至少在以下之一形成 Pareto 优势：

- accepted edit / GPU-hour
- valid edit yield
- delta bytes
- cold load
- resume efficiency
- FPS / VRAM

### 科学成功

负结果完整保留，统计严谨。

---

# 46. Coding Agent：V4 首轮提示词

```text
执行 docs/WORLDSIM_V4_PLAN.md。

当前用户目标：
把已经完成的 V3.3 从单场景工程原型升级为顶会质量技术报告 / paper。

硬约束：
1. 默认单卡 RTX 3090 24 GiB。
2. 单卡 dev+validation+KITTI adapter smoke 未闭环前禁止多卡。
3. 先扩 nuScenes，再接 KITTI。
4. KITTI 已在公共盘：
   /root/autodl-pub/KITTI
   禁止下载。
5. 正式图像主指标必须包括：
   PSNR / SSIM / LPIPS。
6. 工程指标必须保留：
   wall / VRAM / RAM / bytes / cold load / FPS / throughput /
   valid yield / pipeline success / resume。
7. V3.3 所有 canonical asset / terminal 只读。
8. V2/M5 dirty files 不纳入 V4。

当前只授权：
WS-V4-P0-SCOPE-PAPER-FREEZE-01

开始：

A. 读取：
- AGENTS.md
- README.md
- docs/RESEARCH_STATUS.md
- docs/RESEARCH_FAILURES.md
- docs/EXPERIMENTS.md
- docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md
- docs/WS_V33_R0_INTEGRATION.md
- docs/WORLDSIM_V4_PLAN.md
- 用户提供 WORLDSIM_V4_ITERATION_DRAFT.md（若仓库已同步）

B. 核对：
- git branch
- HEAD
- git status
- V3.3 e6663e1 是否在 main history
- main@144ed19 事实是否仍成立
- active GPU process
- cgroup
- disk

C. 不训练。
先修正 status / branch / HEAD provenance。

D. 审计：
- /root/autodl-pub/KITTI
- 只 ls / find / metadata
- 禁止网络下载
- 判断 tracking / raw / odometry / object 格式实际存在情况

E. 创建 V4：
- research/worldsim-v4-evidelta
- configs/worldsim_v4/
- motion_proj/worldsim_v4/

F. 冻结：
1. EviDelta-GS mathematical schema
2. nuScenes cohort construction protocol
3. KITTI adapter contract
4. baseline matrix
5. metrics schema
6. engineering metrics schema
7. test-freeze protocol

G. source audit：
只使用官方 paper / repo。
重点：
- Street Gaussians
- AD-GS
- NeuRAD
- SplatAD
- IDSplat
- HUGS
- DrivingGaussian
- HorizonForge
- GaME
- RecEdit-Drive
- OP2GS
- LiDAR-EVS
不可执行方法只写 paper-only。

H. P0 输出：
- configs/worldsim_v4/p0_scope_v1.yaml
- docs/WS_V4_P0_SCOPE.md
- docs/WS_V4_LITERATURE_MATRIX.md
- docs/WS_V4_KITTI_AUDIT.md
- updated RESEARCH_STATUS / EXPERIMENTS / FAILURES / README
- tests

I. P0 不允许：
- 模型训练
- 大权重下载
- KITTI 下载
- nuScenes 新 scene 训练
- KITTI quality run
- test scene 选择基于结果

J. 提交：
docs(worldsim): 冻结 V4 paper-first 研究路线

P0 done 后唯一允许解锁：
WS-V4-D0-NUSCENES-COHORT-01
```

---

# 47. D0 Coding Agent 提示词

```text
只执行 WS-V4-D0-NUSCENES-COHORT-01。

目标：
冻结 30 个 scene-disjoint nuScenes scenes。

不得训练模型。

流程：
1. 枚举所有候选 scene metadata。
2. 只使用结果前可观测属性分层。
3. 冻结：
   6 development
   6 validation
   18 test
4. 每个 scene 自动找：
   high actor
   difficult actor / ABSTAIN
   remove
   lateral
   insert
   2–4 sec continuous clip
5. 固定 train/development/heldout frame。
6. test scene 配置在方法训练前 hash。
7. 输出：
   configs/worldsim_v4/nuscenes_cohort_v1.yaml
   artifacts/nuscenes_cohort.json
   split SHA
8. 先只对 2 个 dev scene 做数据/preprocess smoke。
9. smoke 不运行 M1/M2/M3。
10. 不读取 test quality。
```

---

# 48. M1 Coding Agent 提示词

```text
执行 WS-V4-M1-EVIDENCE-FIELD-01。

只在 development scenes 调。

1. 从 V3.3 O1 初始化 instance evidence。
2. 实现 Beta-Bernoulli multi-view evidence：
   mask / visibility / depth / LiDAR。
3. 保存 alpha / beta / posterior / uncertainty。
4. base RGB Gaussian immutable。
5. smoke：
   2 scenes
6. 比：
   raw
   temperature calibration
   beta calibration
7. 指标：
   IoU
   Boundary F1
   FP/FN mass
   ECE
   Brier
8. 实现 temporal memory。
9. 6 dev 冻结 M1*。
10. 6 val 只读确认。
11. 如果 majority val 不改善：
    M1 rejected。
12. 不读取 18 test。
```

---

# 49. M2 Coding Agent 提示词

```text
执行 WS-V4-M2-REPAIR-ROUTER-01。

先冻结 matched repair table。

candidate：
R0 abstain
R1 observed
R2 Telea
R3 RoadPatch
R4 generated if executable
R5 router

全部：
same base / hole / view / resolution / heldout。

实现：
RepairCandidate
RiskTerms
BayesRouter

risk：
photo
geometry
temporal
uncertainty
compute

在 dev 冻结 lambda 和 tau。
validation 不调。

输出 selective risk curve：
coverage vs edit error。

必须证明：
高 uncertainty 样本确实更容易错。
否则 uncertainty contribution 无效。
```

---

# 50. M3 Coding Agent 提示词

```text
执行 WS-V4-M3-TEMPORAL-DELTA-01。

连续 2–4s clip。

对照：
frame independent
linear
cubic B-spline
B-spline + evidence memory
full temporal

实现：
SE(3) Lie algebra B-spline。

不改变 V3.3 base。

指标：
PSNR/SSIM/LPIPS
tLPIPS
warp L1/LPIPS
flicker
IoU jitter
Boundary F1 jitter
identity switch
trajectory RMSE
semantic reintroduction
rollback exact

只有 development 调控制点间隔 / smoothness。
validation 后冻结。
```

---

# 51. KITTI Coding Agent 提示词

```text
执行 WS-V4-D1-KITTI-ADAPTER-01。

禁止下载。

根目录：
/root/autodl-pub/KITTI

1. 审计实际结构。
2. 优先 tracking training。
3. 若 tracking 不存在，审计 raw。
4. 自动建立：
   camera
   LiDAR
   calibration
   pose
   actor ID / box
   timestamp
5. 两 sequence 仅做 adapter smoke。
6. 所有 V4 method threshold 来自 nuScenes。
7. KITTI 不允许调参。
8. adapter exact 后才解锁 10-sequence formal。
```

---

# 52. Paper 写作执行提示词

实验收口后：

```text
WS-V4-W0-PAPER-01
```

输出：

```text
paper/
├── abstract.md
├── introduction.md
├── related_work.md
├── method.md
├── experiments.md
├── limitations.md
├── conclusion.md
├── tables/
├── figures/
├── appendix/
└── reproducibility.md
```

要求：

- 所有主张自动链接 canonical JSON
- 表格数字从 machine-readable CSV 生成
- 不人工复制指标
- 每张图记录 source run
- failure / abstain 进入 appendix
- theory 不超出代码实现

---

# 53. Paper Abstract 骨架

不要提前填数字，最后由 R0 自动填：

> Neural driving simulators can reconstruct photorealistic scenes, yet maintaining such assets under counterfactual edits remains brittle: object ownership is uncertain, occluded regions admit multiple repair sources, and frame-wise edits introduce temporal inconsistency. We present EviDelta-GS, an evidence-calibrated temporal delta representation for maintainable driving Gaussian assets. EviDelta-GS aggregates multi-view semantic, geometric, LiDAR, provenance, and temporal observations into calibrated Gaussian-level evidence, routes each repair through a Bayesian risk criterion among observed support, real-scene donors, generation, or abstention, and compiles edits into reversible continuous-time Gaussian deltas parameterized by SE(3) B-splines. The base reconstruction remains immutable. We evaluate on scene-disjoint nuScenes sequences and a frozen cross-dataset KITTI protocol using image fidelity, temporal consistency, geometry, edit correctness, and production-system metrics. ...

最后数字必须来自 canonical test。

---

# 54. 最终路线

```text
V3.3:
Strong single-scene maintainable WorldSim prototype

        ↓

V4 / EviDelta-GS:

30-scene nuScenes
        ↓
Evidence-Calibrated Gaussian Field
        ↓
Bayes-Risk Repair Router
        ↓
Temporal SE(3) B-Spline Delta
        ↓
18-scene heldout test
        ↓
KITTI frozen cross-domain
        ↓
PSNR / SSIM / LPIPS
+ temporal
+ geometry
+ engineering
+ statistics
        ↓
Top-conference-quality technical report / paper
```

---

# 55. 参考工作（Coding Agent 必须优先核对一手资料）

## Driving reconstruction / simulation

- Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting — <https://arxiv.org/abs/2401.01339>
- DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes — <https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_DrivingGaussian_Composite_Gaussian_Splatting_for_Surrounding_Dynamic_Autonomous_Driving_Scenes_CVPR_2024_paper.html>
- NeuRAD: Neural Rendering for Autonomous Driving — <https://openaccess.thecvf.com/content/CVPR2024/html/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.html>
- HUGS: Holistic Urban 3D Scene Understanding via Gaussian Splatting — <https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_HUGS_Holistic_Urban_3D_Scene_Understanding_via_Gaussian_Splatting_CVPR_2024_paper.html>
- AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving — <https://arxiv.org/abs/2507.12137>
- SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving — <https://openaccess.thecvf.com/content/CVPR2025/html/Hess_SplatAD_Real-Time_Lidar_and_Camera_Rendering_with_3D_Gaussian_Splatting_CVPR_2025_paper.html>
- IDSplat: Instance-Decomposed 3D Gaussian Splatting for Driving Scenes — <https://github.com/zenseact/idsplat>

## Closed-loop / editing

- UniSim: A Neural Closed-Loop Sensor Simulator — <https://openaccess.thecvf.com/content/CVPR2023/html/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.html>
- HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles — <https://openaccess.thecvf.com/content/CVPR2026/html/Wang_HorizonForge_Driving_Scene_Editing_with_Any_Trajectories_and_Any_Vehicles_CVPR_2026_paper.html>
- RecEdit-Drive: 3D Reconstruction-Guided Spatiotemporal Video Editing for Autonomous Driving Scenes — <https://openaccess.thecvf.com/content/CVPR2026/html/Wu_RecEdit-Drive_3D_Reconstruction-Guided_Spatiotemporal_Video_Editing_for_Autonomous_Driving_Scenes_CVPR_2026_paper.html>

## Maintainable / object-aware Gaussian assets

- OP2GS: Object-Aware 3D Gaussian Splatting with Dual-Opacity Primitives — <https://arxiv.org/abs/2605.20044>
- GaME: Gaussian Mapping for Evolving Scenes — <https://github.com/VladimirYugay/GaME>
- GOR-IS: 3D Gaussian Object Removal In the Intrinsic Space — <https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_GOR-IS_3D_Gaussian_Object_Removal_In_the_Intrinsic_Space_CVPR_2026_paper.html>
- FocusGS: Spatial Delta Layers for Local Repair and Deterministic Editing of Trained 3D Gaussian Assets — <https://arxiv.org/abs/2607.28834>

## Sensor robustness

- LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision — <https://arxiv.org/abs/2603.14763>

## Dataset

- nuScenes — <https://www.nuscenes.org/>
- KITTI — <http://www.cvlibs.net/datasets/kitti/>

---

# 56. V4 最后一条执行原则

> **优先把“一个 scene 的复杂系统”变成“30+10 scene 上一个清晰方法”。**

从现在开始：

```text
新增 scene
>
新增严谨 baseline
>
新增统计
>
新增 temporal
>
新增 geometry
>
新增工程指标
>
最后才是新增模型模块
```

除非 M1/M2/M3 的正式验证显示存在明确机制缺口，否则不再无边界扩展 SOTA 模块。
