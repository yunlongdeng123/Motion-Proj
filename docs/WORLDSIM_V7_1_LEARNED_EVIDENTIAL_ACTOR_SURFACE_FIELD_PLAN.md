# WorldSim V7.1 研究计划

## HARP-Field：面向危险保真的可学习 Actor 物理表面场

**版本定位：** V7.1 不是继续增加 selector、threshold 或 authority head，而是专门解决 V7 的 C1 剩余核心问题：`COMPLETE` 生成的未观测表面会造成 literal first-return ghost；现有模型只能保留、删除或转 UNKNOWN，不能把错误表面移动到正确位置。

**工作名：** **HARP-Field — Learned Evidential Actor Surface Field for Hazard-Preserving 3D World Compilation**

**论文关系：** V7.1 直接更新现有 CVPR/ICCV 论文，不另起一篇完全不同的故事。C2「有效性—危险性结构解耦」和 C3「连续任务代价密度」冻结为继承模块；V7.1 的主要研究预算全部用于把 C1 从“物理审计 + 局部规则修复”升级成“可训练的连续几何修复”。

---

# 1. 当前事实与 V7.1 的研究问题

V7 已经确认：

1. Actor canonical 多帧融合能够显著提升 held-out 表面覆盖；
2. matched-ray `PROJECT` 可以消除已观测 FREE 区中的矛盾；
3. literal first-return 指标比 target-nearest proxy 更严格，后者会系统性低估 early-return ghost；
4. fresh AV2 中，新增 literal early returns 几乎全部来自 `COMPLETE`；
5. P16 的 FREE/OCCUPIED/UNKNOWN 点级分类失败；
6. P17/P17R 的射线级可微训练能够降低 early return，但所有 keep/delete 策略都会损害 hit retention 或 Chamfer；
7. 纯删除只对 early-return 方向有单调保证，不对 Chamfer、表面完整性或碰撞正确性提供保证。

因此，V7.1 的核心研究问题不是：

> 哪些 COMPLETE 点应该被删掉？

而是：

> **错误或无充分证据的 completed surface 应该位于哪里；能否通过可学习的 Actor-local 几何场，将其移动、重投影或重新生成到物理自洽的位置？**

形式上，V7 的动作空间主要是：

\[
\alpha_j\in\{0,1\},
\]

而 V7.1 要引入：

\[
\Delta x_j\in\mathbb R^3
\quad\text{或}\quad
s_\theta(x)\in\mathbb R^K,
\]

使模型能够改变表面位置，而不是只改变点是否存在。

---

# 2. 文献机制迁移，而不是完整复现

V7.1 组合四类已经被公开工作验证过的机制，但不伪称完整复现任何单篇论文。

## 2.1 EvOcc：证据目标，而不是普通置信度

迁移：

- LiDAR endpoint 作为 OCCUPIED evidence；
- ray transmission 作为 FREE evidence；
- 未经过观测的区域保留 UNKNOWN；
- 冲突与未观测不能被多数票压成确定标签；
- 训练和评价都绑定射线/首返回物理语义。

不迁移：

- 不直接复现其完整 image-based occupancy backbone；
- 不把 V7.1 包装成 EvOcc faithful reproduction。

## 2.2 Object-Centric Occupancy Completion：Actor canonical tracklet

迁移：

- 将多帧 Actor 点和射线对齐到 Actor-local canonical coordinates；
- 使用长时序 tracklet 形成高分辨率 object-centric geometry target；
- 用 implicit decoder 处理不同尺寸 Actor；
- 将 free 与 unobserved 分开标注。

## 2.3 SelfOcc：连续场与可微射线监督

迁移：

- 把三维表示视为连续 SDF/geometry field；
- 通过可微 ray rendering 监督 first-return/depth；
- 让几何表征与最终可观测量处于同一个训练闭环。

## 2.4 SparseOcc++：completion 是几何回归，不是体素分类

迁移：

- scene/object completion 优先建模为 signed-distance / scene-completion-field regression；
- 将几何完成与语义/危险标签分离；
- 对道路 Actor 的各向异性形状，允许平面与垂直方向采用不同几何分量。

V7.1 的方法概括为：

```text
EvOcc-style evidence target
+
Object-centric temporal canonicalization
+
SDF/SCF geometry regression
+
V7 literal first-return renderer
+
HARP Actor/hazard immutability
```

---

# 3. 版本北极星与边界

## 3.1 唯一主目标

在保持 Actor identity、trajectory、extent 和 hazard state 不变的前提下，实现：

\[
\text{LiteralFirstReturnRisk}\downarrow
\quad\land\quad
\text{TargetSurfaceChamfer}\not\uparrow.
\]

这就是 V7.1 的 Pareto 目标。

## 3.2 不作为 V7.1 主目标

- 不重新训练 C2 validity/hazard selector；
- 不重新搜索 C3 density、copula、quantile 或 task authority；
- 不深入 RL；
- 不刷 Occupancy mIoU；
- 不进行 full IR-WM / GS backbone unfreeze；
- 不继续做 COMPLETE selector、threshold、router 或 top-k veto；
- 不用 AV2 target quality 调参；
- 不声称真实道路安全保证。

## 3.3 C2/C3 的冻结方式

```text
C2：保留结构性 non-interference 和 hazard preservation；仅重算新表面产生的物理指标。
C3：保留 V6.7/V7 已冻结的 actor residual、continuous cost density 和 joint-H reliability。
```

V7.1 的成功不依赖 C2/C3 再次提升。

---

# 4. 数学接口

## 4.1 Actor canonical evidence

对 Actor \(i\)，将多帧传感器点、射线和 pose 变换到 canonical frame：

\[
x^{\mathrm{actor}}_{i,t}
=
T^{-1}_{\mathrm{city}\leftarrow\mathrm{actor},t}
T_{\mathrm{city}\leftarrow\mathrm{ego},t}
T_{\mathrm{ego}\leftarrow\mathrm{sensor},t}
x^{\mathrm{sensor}}_t.
\]

构造证据集合：

\[
\mathcal E_i=
\{x,\,m_O,\,m_F,\,m_U,\,n_{time},\,n_{view},\,r,\,v\},
\]

其中：

- \(m_O\)：endpoint/reflection evidence；
- \(m_F\)：ray-transmission evidence；
- \(m_U\)：unobserved/contradictory evidence；
- \(n_{time}\)：时序支持；
- \(n_{view}\)：视角多样性；
- \(r,v\)：量程与射线方向。

所有计数优先表示为 per-opportunity ratio，避免 V7 已发现的 raw frame/hit-count 跨传感器 shortcut。

## 4.2 通用表面场接口

V7.1 固定语义接口，但不在计划阶段预先钦定最终参数化：

\[
F_\theta(\xi\mid\mathcal E_i,D_i)
=
\left(
\mathbf s_\theta(\xi),
\mathbf m_\theta(\xi)
\right),
\]

其中：

- \(\xi\)：Actor canonical coordinate；
- \(D_i\)：Actor 尺寸；
- \(\mathbf s_\theta\)：surface displacement / scalar SDF / anisotropic SCF；
- \(\mathbf m_\theta=(m_F,m_O,m_U)\)：evidential physical state。

最终物理表面：

\[
S_i^\theta
=
\{\xi:\mathbf s_\theta(\xi)=0,\;m_O\ge m_F,\;m_O\ge m_U\}.
\]

UNKNOWN 仍是一级状态，不自动变为 FREE。

## 4.3 Literal first-return renderer

沿 held-out ray \(r=(o,v)\) 采样 \(x_k=o+d_kv\)，由 field 转成 occupancy/opacity \(\alpha_k\)：

\[
T_k=\prod_{j<k}(1-\alpha_j),
\qquad
w_k=T_k\alpha_k,
\]

\[
\hat d(r)=\sum_k w_kd_k+T_{K+1}d_{fallback}.
\]

该 renderer 必须与 V7 P20/P22/P23 的 literal minimum-positive-depth evaluator 使用相同 beam-tube 和 depth tolerance 语义。

---

# 5. 数据与角色

## 5.1 Source：nuScenes

第一正式模型只训练刚性道路车辆：

```text
car / regular vehicle
bus / articulated bus / school bus
truck / box truck / large vehicle
trailer
```

摩托车、自行车、行人和可变形 Actor 暂作外推审计，不作为第一轮 C1 的主训练对象。

推荐 metadata-only 划分：

| 角色 | 场景 | 目标规模 |
|---|---:|---:|
| Train | 120 | ≥1000 rigid tracklets |
| Selection | 20 | ≥150 tracklets |
| Source final | 20 | ≥150 tracklets |

若固定场景数下 tracklet 不足，只允许在读取质量前扩充 Train 场景；不得通过放宽结果标准、删除难 Actor 或移动 selection/test 场景来补数量。

## 5.2 帧角色

对每条 tracklet 采用固定 modulo-3 划分：

```text
rank mod 3 in {0,1} → model input/build evidence
rank mod 3 == 2     → held-out target rays/surface
```

Train 中 held-out 部分只用于监督；Selection/Final 中 held-out 部分只用于评价。

S1 的 per-Actor oracle 在 Train tracklet 内再把 held-out frames 交替划分为：

```text
oracle-fit rays
oracle-check rays
```

从而避免“在同一批目标射线上优化并汇报”的平凡上界。

## 5.3 External：Argoverse 2

从尚未被 V7 三套 cohort 消费的 AV2 Sensor-val logs 中，metadata-only 冻结 20 个新日志。

AV2 只允许：

- 坐标和单位转换；
- 类别映射；
- timestamp interpolation；
- Actor-local canonicalization；
- sensor-opportunity normalization。

禁止：

- AV2 fine-tuning；
- AV2 calibration；
- AV2 threshold selection；
- 根据结果替换日志；
- 用 AV2 选择 M0/M1。

Waymo 仅在官方访问权限和磁盘条件允许时作为可选第二外测，不阻塞 V7.1。

---

# 6. Baseline staircase

| ID | 方法 | 作用 |
|---|---|---|
| B0 | Single-frame query | 不做 completion |
| B1 | V7 always-COMPLETE | 表面完整性强，但 literal ghost 多 |
| B2 | V7 ray-certified four-action | 当前论文主基线 |
| B3 | V7 P17R ray+Chamfer keep/delete | 最强 learned selector 基线 |
| B4 | Actor-local TSDF/evidential fusion | 外部非学习几何基线 |
| B5 | Implicit occupancy classification | 与 M1 同 encoder/decoder，但 completion 仍按 voxel classification |
| M0 | Learned ray/surface displacement | 第一正式 learned geometry model |
| M1 | Actor-local evidential SDF/SCF | 条件解锁的完整 field model |

B4 必须实现，避免论文继续只有“ours vs ours-minus-X”。

若 Object-Centric Occupancy Completion 官方代码能以合理成本接入，可作为附加 baseline；其复现不阻塞主线。

---

# 7. 研究阶段

## S0：分支、代码骨架和论文冻结

从 V7 terminal HEAD 建立：

```text
research/worldsim-v7.1-learned-evidential-surface
```

新增：

```text
motion_proj/worldsim_v71/
  evidence_volume.py
  surface_oracle.py
  ray_displacement.py
  actor_surface_field.py
  first_return_renderer.py
  dataset_nuscenes.py
  dataset_av2.py
  evaluate_surface.py

configs/worldsim_v71/
scripts/run_worldsim_v71_*.py
```

论文继续使用现有 `paper/`：

- 先复制一份 V7 final PDF 作为 immutable baseline；
- 修正现有 C3 Eq.11 与真实 P182 输入的表述；
- C2/C3 数值暂不改变；
- 在 `results_macros.tex` 新增 V7.1 C1 占位宏。

只做必要的可运行检查，不增加仓库级回归矩阵、哈希或指纹。

---

## S1：连续几何动作空间 Oracle

这一阶段不训练可泛化网络，只回答：

> 当前 evidence 是否支持一个同时改善 first-return 和 Chamfer 的连续几何解？

### S1-A：Candidate displacement oracle

固定 KEEP、matched PROJECT、Actor state 和 target operator，仅令每个 COMPLETE candidate 学习：

\[
\Delta d_j^{ray},\quad\Delta d_j^{normal}
\]

并限制在 Actor cuboid/局部表面邻域内。

使用 oracle-fit rays 优化：

\[
L_{oracle}
=
L_{first-return}
+
L_{surface}
+
L_{anchor}
+
L_{smooth}.
\]

在 oracle-check rays 上报告。

### S1-B：Sparse field oracle（仅当 S1-A 不可行）

如果固定 completion candidates 即使可移动也无法形成 Pareto，则在 Actor-local sparse grid 上直接优化小型 SDF/SCF 参数，允许生成原 candidate set 中不存在的表面。

### S1 唯一决策

```text
A 可行 → V7.1 首选 M0 ray/surface displacement
A 不可行、B 可行 → 跳过 M0，直接 M1 implicit field
A/B 均不可行 → 证据/target ceiling；停止 learned C1，不回到 selector
```

“可行”指 oracle-check 上至少存在：

\[
\text{hazard literal first-return 相对下降}\ge5\%
\]

且：

\[
\Delta\text{Chamfer}\le0.5\text{ mm},
\]

Actor/hazard retention 必须为 100%。

---

## S2：Actor evidential completion corpus

以 Train tracklets 构建 object-centric training corpus。

### 输入

- build-only multi-frame surfels；
- LiDAR origins、directions、endpoint depths；
- FREE transmission samples；
- temporal/view support；
- Actor size、canonical coordinate；
- query surface 与 V7 deterministic KEEP/PROJECT anchors。

### 监督

- target-only endpoints；
- target-only FREE intervals；
- FREE/OCCUPIED/UNKNOWN soft evidence mass；
- target object-centric occupancy；
- target boundary distance / SCF；
- literal first-return depth。

### 跨传感器增强

从训练第一天就加入：

- beam/ray thinning；
- frame dropout；
- view dropout；
- range-dependent dropout；
- hit-count/opportunity normalization；
- actor-size normalized coordinate。

不使用 raw observation-frame count 作为未经归一化的模型输入。

### 数据缓存

按 Actor 保存 compact NPZ/Parquet：

```text
actor identity + size
build evidence
anchor surface
target rays
target surface
hazard label（只用于分层评测，不进入模型）
```

I/O producer 与 GPU training consumer 并行；不等待全部 corpus 完成才启动训练。

---

## S3：M0 — Learned Ray/Surface Displacement

M0 是最低容量、最容易归因的 representation-changing model。

### 输入

在 V7 现有 11D completion features 基础上增加：

- Actor-normalized canonical coordinate；
- evidence masses；
- local normal；
- matched/support ray direction statistics；
- local surface neighborhood summary。

不输入 hazard/TTC/cut-in 标签。

### 输出

\[
(\Delta d_{ray},\Delta d_{normal},u)
\]

其中 \(u\) 是 UNKNOWN/abstain logit。

表面更新：

\[
x_j'=x_j+\Delta d_{ray}v_j+\Delta d_{normal}n_j.
\]

有 direct matched hit 的 PROJECT 仍使用确定性 matched termination，不由网络覆盖。

### 架构

首版固定为低容量 local MLP/PointNet residual：

```text
per-point evidence
+ local neighborhood pooled feature
→ 128 → 128 → displacement + unknown
```

一个 seed，不做 hidden-size sweep。

### 训练顺序

1. distill S1 displacement oracle；
2. 加入 differentiable literal first-return fine-tuning；
3. 加入 target surface 和 temporal consistency；
4. observed anchors 始终冻结。

不通过任意 loss-weight sweep救结果。各项先按 always-COMPLETE reference scale 归一化，再固定等权；若梯度诊断显示明确冲突，只允许一次 PCGrad 式恢复。

### M0 决策

在 source Selection 上必须同时：

- hazard literal first-return 相对 B1 至少下降 5%；
- symmetric Chamfer 不劣于 B1 超过 0.5 mm；
- target hit recall 不低于 B1 超过 1 pp；
- Actor/hazard state 100% 保留。

通过后立即冻结 M0，并跳过 M1；不因“模型更简单”继续追求更复杂 field。

---

## S4：M1 — Actor-Local Evidential Surface Field（条件解锁）

只有以下情况才执行 M1：

- S1 field oracle 可行；
- M0 无法逼近 oracle，或 candidate support 明显限制 surface placement；
- 失败不是 target/evaluator/工程问题。

### 表示

首版采用：

```text
multi-frame evidence point encoder
→ Actor latent + sparse local features
→ implicit coordinate decoder
→ anisotropic SCF + evidential masses
```

建议容量：

```text
PointNet/sparse encoder: 128D latent
implicit decoder: 4 × 128 MLP
SCF: planar distance + vertical distance
state: FREE/OCCUPIED/UNKNOWN masses
```

不同尺寸 Actor 通过真实 `size_lwh` 和 normalized canonical coordinates 条件化，不使用 hazard features。

### Surface extraction

- sparse anchors 位于 observed surface、FREE rays 和 candidate hole 周边；
- 通过 SCF zero crossing / geometry-guided propagation 生成表面；
- evidence argmax 为 UNKNOWN 时不进入 collision surface；
- known KEEP 和 matched PROJECT 是硬 anchor。

### 分阶段训练

#### Stage A：Evidence/geometry pretraining

\[
L_A=L_{evidence}+L_{SCF}+L_{anchor}.
\]

#### Stage B：Physical fine-tuning

\[
L_B=L_{first-return}+L_{surface}+L_{temporal}.
\]

先让模型学会表面位置，再加入 downstream ray objective，避免从随机场直接优化 first return。

### 条件 LoRA

V7.1 默认不改 IR-WM/visual backbone。

只有：

1. S1 oracle 明确可行；
2. M1 在 train 也无法逼近 oracle；
3. target、field parameterization 和 optimizer 均无异常；

才允许对 Actor temporal encoder/geometry decoder 最后模块做一次 `rank=16, alpha=32` 的 targeted LoRA。不得 full unfreeze，也不得 rank sweep。

---

## S5：Source final 与 AV2 zero-shot

### Source final

在方法、checkpoint 和 surface extraction 全部冻结后，读取 20 个 source final scenes 一次。

主要比较：

```text
B0/B1/B2/B3/B4/B5/M0 or M1
```

主指标：

- literal first-return new-early：all / hazard / clear；
- symmetric Chamfer；
- target LiDAR depth error；
- matched-hit recall；
- surface completeness；
- temporal surface jitter；
- observed-FREE violation；
- Actor/ID/trajectory/TTC/hazard retention。

### AV2 zero-shot

Source final 通过后，冻结运行 20 个全新 AV2 logs。

V7.1 的外域主结论只允许两种：

```text
Zero-shot Pareto transfer supported
或
Source-only geometry improvement; cross-sensor transfer rejected
```

不允许在 AV2 上重新标定 unknown threshold、field scale、SDF zero level、ray tolerance 或模型参数。

### External primary criterion

\[
\text{hazard literal first-return relative reduction}\ge5\%
\]

且：

\[
\Delta\text{Chamfer}\le1\text{ mm}
\]

同时 Actor/hazard retention 为 100%。

若 source pass、AV2 fail，V7.1 不继续做 target adaptation；跨传感器 completion 作为下一版本问题。

---

## S6：Appearance bridge、C2/C3 接口与论文收敛

### S6-A：最小 appearance/GS bridge

为避免 CVPR reviewer 将工作理解成纯 LiDAR post-processing，至少选择一个已有重建场景，建立：

```text
appearance Gaussians / rendered asset
        ↕ Actor identity ownership
V7.1 physical surface
```

要求展示：

- RGB appearance 与 Actor trajectory 不变；
- collision/physical surface 改变；
- literal first-return ghost 减少；
- 危险 Actor 不消失。

该 bridge 可以是 inference-only，不需要把 GS 重新作为 C1 训练 backbone。

### S6-B：C2

- validity/hazard input separation 和 Actor/hazard immutability 不变；
- 不重新训练新的 selector；
- 用新 C1 表面重算 clean/hazard 分层物理指标；
- 增加 shared/factorized coverage-matched AURC，修复 V7 表格中覆盖率不可直接比较的问题。

### S6-C：C3

- 冻结 inherited P182/P199；
- 修正 paper 中 density head 的真实输入为：

\[
p_\theta(y\mid s_{actor},H,d_{clearance}),
\]

而不是暗示其直接消费全部 \(z^{val},z^{haz}\)；
- 仅做新 physical surface 与 frozen task reliability 的接口一致性审计，不重新训练 density。

---

# 8. 损失与物理约束

## 8.1 Evidential mass loss

训练 target 是 soft evidence mass，而非把未观测位置强标为 FREE：

\[
L_{evidence}
=
-\sum_{c\in\{F,O,U\}}m_c^*\log m_c.
\]

## 8.2 Geometry field loss

如果使用 SDF/SCF：

\[
L_{SCF}
=
\operatorname{Huber}(\mathbf s_\theta,\mathbf s^*).
\]

## 8.3 Literal first-return loss

\[
L_{first-return}
=
\operatorname{SmoothL1}(\hat d(r),d^*(r)).
\]

必须使用 literal ray operator，而不是 target-nearest proxy。

## 8.4 Surface loss

\[
L_{surface}
=
\frac12
\left[
CD(S_\theta,S^*)+CD(S^*,S_\theta)
\right].
\]

## 8.5 Anchor and temporal loss

\[
L_{anchor}=\|S_\theta^{observed}-S^{observed}\|,
\]

\[
L_{temporal}
=CD(T^{-1}_{actor,t}S_t,T^{-1}_{actor,t'}S_{t'}).
\]

## 8.6 Hazard preservation

不通过 hazard loss 来“奖励保留危险 Actor”；Actor state 本身在数据结构和编译器中保持 immutable：

\[
(ID,X_{0:H},D,h)_{before}
=
(ID,X_{0:H},D,h)_{after}.
\]

这样 hazard preservation 是接口合同，不与 geometry loss 交换。

---

# 9. 只保留三个研究决策点

V7.1 不再建立长链 gate。

## D1：动作空间是否可行？

由 S1 oracle 决定 displacement 还是 implicit field。

## D2：source 是否形成真实 Pareto？

要求 first-return 明显下降，同时 Chamfer 非劣，Actor/hazard 保留。

## D3：AV2 zero-shot 是否迁移？

方法冻结后一次读取，不做 target adaptation。

除此之外，普通工程错误直接修复；没有必要为每个脚本增加 smoke、hash、fingerprint 或大规模回归测试。

---

# 10. 代码交付

最终 paper-facing 代码必须收敛为：

```text
motion_proj/worldsim_v71/
  evidence_volume.py          # FREE/OCC/UNKNOWN evidence
  actor_canonical.py          # tracklet canonicalization
  surface_oracle.py           # displacement/field capability
  ray_displacement.py         # M0
  actor_surface_field.py      # M1
  first_return_renderer.py    # differentiable + literal eval
  surface_extract.py
  evaluate_surface.py
  av2_adapter.py
```

正式训练入口不应 import P16/P17/P18 等历史 experiment script；历史结果只作为 baselines。

最小测试只覆盖：

- coordinate transform；
- observed anchors 不移动；
- UNKNOWN 不进入 collision surface；
- literal first-return synthetic ray；
- Actor state immutability；
- SDF/SCF zero-crossing。

---

# 11. 论文交付

V7.1 必须持续产出可编译 CVPR/ICCV 论文，而不是研究结束后再重写。

## Draft 0

- 新增 HARP-Field 方法占位；
- 修正 C3 Eq.11；
- 更新 Related Work：EvOcc、Object-Centric Occupancy Completion、SelfOcc、SparseOcc++；
- 保留 V7 final PDF 作为 baseline。

## Draft 1：S1/S2 完成

- Figure：selection-only action space 与 continuous field 的区别；
- 数据生成和 evidential target；
- oracle upper bound。

## Draft 2：M0/M1 source result

- 主表加入 learned geometry model；
- 可视化 literal first-return、surface relocation 和 target-only geometry；
- 把 P17/P17R 作为关键 negative baseline。

## Draft 3：AV2 zero-shot

- nuScenes→AV2 主表；
- frozen qualitative panels；
- sensor-normalized evidence ablation；
- failure cases 原样进入 supplement。

## Draft 4：paper-ready

主文 8 页：

```text
1. Introduction
2. Related Work
3. Problem Formulation
4. Method
   4.1 Evidential Actor Surface Target
   4.2 Continuous Geometry Action Space
   4.3 Learned Actor Surface Field
   4.4 Literal First-Return Compilation
5. Experiments
   5.1 Source Physical Pareto
   5.2 AV2 Zero-Shot
   5.3 Validity–Hazard Preservation
   5.4 Continuous Task Reliability Interface
6. Limitations
7. Conclusion
```

主图优先展示：

1. same hazardous Actor：before ghost → repositioned surface → hazard retained；
2. ray depth profile；
3. Actor canonical field；
4. nuScenes→AV2 zero-shot；
5. actual GS/appearance bridge。

---

# 12. 资源合同

默认：

```text
1 × RTX 3090 24 GB
```

首版建议：

- max 4096 input surfels / Actor；
- 2048–8192 query/ray samples / Actor；
- Actor batch 4–8；
- AMP；
- query chunking；
- gradient accumulation。

如果 faithful model 超 24 GB，先使用 query/ray chunking 和 checkpointing；不得降低 actor-local voxel resolution、缩短 tracklet 或删除远距/危险 Actor。只有仍无法放入时才申请多卡。

数据和训练流水：

```text
CPU Actor corpus producer
→ compact cache
→ GPU training consumer
→ held-out evaluator
```

AV2 下载与 source GPU 训练并行。

---

# 13. 明确关闭的路线

V7.1 不得再次启动：

- candidate FREE/OCC/UNKNOWN point classifier；
- P17 式 alpha keep/delete 的容量升级；
- visibility head v2；
- threshold、radius、temperature 或 top-k sweep；
- another expert router；
- no-COMPLETE provenance gate；
- P4/P6-C selector recovery；
- C3 density/copula/quantile 继续搜索；
- full IR-WM / GS unfreeze；
- 使用 AV2 target 调参；
- all-UNKNOWN 冒充物理安全。

---

# 14. V7.1 完成定义

## 最优完成态

- learned displacement 或 field model 在 source final 同时降低 literal first-return 和 Chamfer；
- 在新 AV2 logs 零样本保持同方向；
- Actor/hazard state 100% 保留；
- 有真实 appearance/GS bridge；
- CVPR main/supplement 更新完成。

此时 C1 可以从：

> physically grounded audit and partial repair

升级为：

> **learned evidential Actor-surface reconstruction that repairs completion-induced first-return artifacts without sacrificing surface fidelity or hazardous actors.**

## 中间完成态

- source Pareto 成立，AV2 失败。

则论文主张限制为 source-domain learned physical completion，并把 cross-sensor transfer 明确列为未解问题；不在 AV2 上微调救结果。

## 失败完成态

- 连续 displacement oracle 与 sparse field oracle 均无法在 held-out rays 形成 Pareto。

则结论是：当前公开日志证据不足以同时恢复 unseen Actor surface 和 first-return validity。V7.1 停止 learned C1，不回到 selector；V7 论文改为 physical evaluation/boundary paper。

---

# 15. 推荐执行顺序

```text
S0 branch + paper consistency fix
        ↓
S1 continuous geometry oracle
        ↓
S2 evidential object-centric corpus
        ↓
M0 ray/surface displacement
        │
        ├─ source Pareto PASS → freeze
        │
        └─ representation ceiling + field oracle PASS
                              ↓
                    M1 evidential SDF/SCF
                              ↓
                    source final exact-once
                              ↓
                    AV2 zero-shot exact-once
                              ↓
appearance bridge + C2/C3 frozen integration + CVPR paper
```

**V7.1 的核心不是“再训练一个分类头”，而是第一次训练一个能够改变三维物理表面位置的 Actor-local geometry model。**
