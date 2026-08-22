# WorldSim V6.1 — Occupancy-Verified Generative World Compiler

> 中文名：**占据验证的生成式世界编译器**
>
> 推荐方法名：**OccCompiler**
>
> 推荐论文题目：
>
> **OccCompiler: Occupancy-Grounded and Task-Verifiable Compilation of Generative 4D Driving Worlds**
>
> 中文表述：
>
> **OccCompiler：面向生成式四维驾驶世界的占据约束与任务可验证编译**
>
> 默认资源：单卡 RTX 3090 24GB
>
> 目标形态：硬核 arXiv 技术报告，可继续演化为 CVPR / ICCV / ECCV 主会投稿
>
> 下游任务：
>
> 1. GS + LogSim：case 精确复现、回归拦截；
> 2. GS + WorldSim：新路线、新 Actor、新交互下的功能泛化测试；
> 3. GS + NWM 强化学习：生成多 Actor 安全反事实 episode，降低碰撞 False-safe。

---

# 0. 结论先行

V6 已经证明：

- SceneIR、Provenance、Factorized Validity、Typed Edit、Bake、Deterministic Runtime 能够形成真实闭环；
- 纯 2D 生成提案在几何上失败；
- V6 R10 最终的 `3/28 = 10.7%` 安全通过率主要来自跨前端同刻证据，不是 2D Diffusion 本身；
- 多 Actor 编译、生命周期、传感器回放、选择性下游执行均已具备可复用基础设施。

V6.1 不应继续研究：

```text
2D inpainting
→ 再加 depth/semantic verifier
→ 继续降低阈值
```

V6.1 正式北极星应改为：

> **将 Occupancy / SDF / Collision Volume 提升为世界中的权威物理状态，使高斯只负责外观与传感器渲染；任何生成器只负责提出候选，不拥有几何真值。**

核心表示：

```text
World State
├── Appearance State
│   └── 3D/4D Gaussians
├── Collision State
│   └── Occupancy / SDF / mesh collision body
├── Dynamic State
│   └── canonical actor + trajectory + lifecycle + flow
├── Evidence State
│   └── observed / predicted / generated / unknown
└── Task Validity
    └── photo / geometry / semantic / dynamics / interaction
```

V6.1 的主要科学问题不是：

> Occupancy 能否让图像更漂亮？

而是：

> **Occupancy 能否在不增加 False-safe 的前提下，让更多生成式提案安全进入显式四维世界，并在多 Actor 与 NWM rollout 中保持可验证碰撞语义？**

---

# 1. 必须澄清的基线事实

V6 的两个结果不能混写：

## 1.1 纯 2D generator

Big-LaMa / SD-v1.5 在既有 28-case 上：

```text
photo / geometry verifier 大面积拒绝
纯 2D proposal 的 geometry 支持接近 0
```

这证明：

```text
2D 图像补洞
≠
3D 世界补全
```

## 1.2 R10 最终基线

R10 的最终：

```text
3 ACCEPT / 28
= 10.7%
```

来自：

```text
same-time cross-frontend proposal
+
P1 photo
+
P2 geometry
+
AND fusion
```

因此 V6.1 正式比较必须同时保留：

```text
B0 = 2D generator baseline
B1 = V6 R10 cross-frontend verified baseline
```

不能把 B1 的 10.7% 写成 2D Diffusion 的成果。

---

# 2. V6.1 核心方法：Appearance–Collision Dual World

## 2.1 双状态世界

定义编译世界：

\[
\mathcal W_t =
\left(
\mathcal G_t^{app},
\mathcal O_t^{col},
\mathcal X_t^{dyn},
\mathcal P_t,
\mathbf q_t
\right)
\]

其中：

- \(\mathcal G_t^{app}\)：外观高斯场；
- \(\mathcal O_t^{col}\)：占据 / SDF / collision body；
- \(\mathcal X_t^{dyn}\)：Actor canonical state、trajectory、lifecycle；
- \(\mathcal P_t\)：provenance；
- \(\mathbf q_t\)：任务条件有效性。

关键原则：

```text
Gaussian opacity
≠ Occupancy probability
```

高斯可以：

- 近透明；
- 漂浮；
- 拉长；
- 为了渲染而覆盖表面附近空间。

碰撞体必须：

- 封闭或有明确体积；
- 具有单位与坐标合同；
- 能进行 swept-volume collision；
- 不随外观优化静默漂移。

---

## 2.2 四种 Occupancy 身份

必须区分：

### `O_observed`

由独立真实证据直接支持：

- LiDAR hit；
- LiDAR ray free-space；
- map drivable / boundary；
- tracked 3D box；
- held-out multi-view depth；
- 真实 actor geometry。

### `O_predicted`

由 GaussianWorld / OccWorld / Drive-OccWorld 等模型预测。

它是：

```text
teacher / proposal / supporting evidence
```

不是 GT。

### `O_asset`

生成或重建后固化的 collision body：

- mesh；
- voxel occupancy；
- SDF；
- canonical actor volume。

这是仿真定义真值，但不是真实世界历史 GT。

### `O_runtime(t)`

由：

\[
O_{a,t}^{runtime}
=
T_a(t) O_a^{canonical}
\]

组成的运行时占据。

RL、collision、reward、label writer 必须消费同一个 `O_runtime(t)`。

---

# 3. Occupancy 不能重新压成一个手工 scalar

V6 已经证明 factorized validity 的价值。

V6.1 的 `q_geometry` 应是一个命名因子集合，而不是：

```text
q_geometry = 0.2 * depth + 0.3 * occ + 0.5 * lidar
```

推荐：

```text
q_geometry:
  free_space_conflict
  occupied_surface_support
  unknown_volume_fraction
  collision_body_closure
  depth_consistency
  ground_contact
  temporal_occupancy_consistency
  actor_static_penetration
  actor_actor_swept_collision
```

最终决策：

```text
REJECT:
  与 certified free-space 冲突
  或 collision body 不合法
  或发生 actor/static、actor/actor 穿透

UNKNOWN:
  独立证据不足
  或预测 Occupancy 与真实证据冲突但无法判定

ACCEPT:
  所有 required geometry factor 通过
  且 photo / semantic / dynamics 的任务要求也通过
```

---

# 4. Occupancy 证书公式

对候选体积 \(V_p\)：

## 4.1 Free-space violation

\[
r_{\mathrm{free}}
=
\frac{
|V_p \cap F_{\mathrm{obs}}|
}{
|V_p| + \epsilon
}
\]

其中 \(F_{\mathrm{obs}}\) 只能由真实 ray carving / map 证据获得。

若：

\[
r_{\mathrm{free}} > \tau_{\mathrm{free}}
\]

直接 REJECT。

---

## 4.2 Occupied support

\[
r_{\mathrm{support}}
=
\frac{
|\partial V_p \cap \operatorname{Dilate}(O_{\mathrm{obs}} \cup O_{\mathrm{pred}})|
}{
|\partial V_p| + \epsilon
}
\]

必须分别报告：

```text
observed support
predicted-only support
```

不能合并后把预测写成真实观测。

---

## 4.3 Unknown volume

\[
r_{\mathrm{unknown}}
=
\frac{
|V_p \cap U_{\mathrm{obs}}|
}{
|V_p| + \epsilon
}
\]

高 UNKNOWN 不等于 background，也不等于安全。

---

## 4.4 Actor canonical occupancy

\[
O_{a,t}
=
T_a(t) O_a^{canonical}
\]

V6.1 不在每帧重新生成 Actor 几何。

生成一次 canonical collision body，随后只变换。

---

## 4.5 Swept collision

Actor \(a,b\) 的时序碰撞：

\[
C_{ab}
=
\max_t
\frac{
\operatorname{Vol}
\left(
O_{a,t} \cap O_{b,t}
\right)
}{
\min
\left(
\operatorname{Vol}(O_a),
\operatorname{Vol}(O_b)
\right)
+\epsilon
}
\]

Broad phase：

```text
OBB / AABB
```

Narrow phase：

```text
Occupancy / SDF / mesh
```

不能再用 coarse AABB 直接作为最终物理真值。

---

# 5. 生成器路线取舍

## 5.1 第一优先级：Hunyuan3D-Omni

用途：

```text
Actor shape proposal
```

输入：

- source actor image；
- 3D bbox；
- observed actor point cloud；
- canonical occupancy voxel。

输出：

- 3D shape / mesh proposal。

优势：

- bbox / point / voxel 控制；
- 约 10GB VRAM 推理；
- 单卡 3090 可行；
- mesh 更容易构建 collision volume。

限制：

- 它不是驾驶场景 4D world model；
- 不负责 trajectory；
- 不保证车辆尺度、底盘、轮胎、朝向物理正确；
- 仍需 occupancy verifier；
- 许可证必须独立审计，不能默认工业可商用。

V6.1 用法：

```text
Hunyuan3D-Omni
→ canonical actor mesh
→ voxel / SDF
→ collision body
→ surface Gaussian appearance
→ SceneIR actor bundle
```

---

## 5.2 DiffSplat

用途：

```text
快速 appearance Gaussian proposal
```

不建议承担：

```text
collision volume
```

原因：

- 3DGS 是表面外观表示；
- Gaussian 本身不提供封闭实体；
- 高质量 rendering 不等于稳定碰撞几何。

建议只作为消融：

```text
Hunyuan mesh geometry + DiffSplat appearance
```

---

## 5.3 Diff4Splat

用途：

```text
后续 4D dynamic appearance proposal
```

不作为第一阶段原因：

- 单图到 4D 的 domain gap 大；
- deformable Gaussian 不天然等于碰撞体；
- 先把 canonical actor + typed trajectory 做稳更可归因。

---

## 5.4 GaussianWorld

推荐作为第一批 learned occupancy teacher：

- 与 Gaussian / Occupancy 表示接近；
- 有官方代码与 pretrained weight；
- 支持 streaming occupancy；
- 适合给 `O_predicted` 与 `q_geometry` 提供 sidecar。

不承担：

```text
new actor generation
```

它主要是预测与补全当前/未来 Occupancy。

---

## 5.5 OccWorld

适合作为：

```text
24GB 级 autoregressive occupancy forecasting baseline
```

优点：

- 官方 pretrained model；
- 官方说明 RTX 4090 24GB 可训练/评估；
- 单卡 3090 值得做 inference smoke。

限制：

- 官方明确指出不能可靠预测输入中不存在的新进入车辆；
- 因此不能单独承担 actor insertion。

---

## 5.6 Drive-OccWorld / IR-WM

适合作为第二阶段：

```text
action-conditioned future occupancy
+
planning coupling
```

它们更贴近 NWM RL，但环境和数据链更重。

V6.1 Minimum Experiment 不应一开始就被其训练链绑定。

---

## 5.7 OccSora

只进入 related work / external-resource track。

官方训练与评估都要求 A100 80GB，训练 OccSora 还使用 8 卡。

当前单卡 3090 不应：

- 重训；
- 反复 OOM；
- 降规格后仍声称 faithful。

---

## 5.8 UniScene / WorldSplat / GenieDrive / OG-Gaussian

它们是必须面对的 novelty baseline：

- UniScene：Occupancy → video + LiDAR；
- GenieDrive：4D Occupancy 作为 physics foundation 指导视频；
- WorldSplat：feed-forward 4D Gaussian generation；
- OG-Gaussian：Occupancy 初始化 static/dynamic Gaussian。

因此 V6.1 不能把下列内容单独写成贡献：

```text
Occupancy guidance
Occupancy initialization
Occupancy → video
4D Gaussian generation
```

V6.1 的独立贡献必须是：

```text
task-verifiable compilation
+
authoritative collision state
+
independent evidence
+
false-safe controlled bake
+
deterministic downstream runtime
```

---

# 6. SceneIR-O 设计

在现有 SceneIR 上新增：

```text
SceneIR-O
├── Appearance
│   ├── static Gaussian chunks
│   └── actor Gaussian assets
├── OccupancyEvidence
│   ├── observed_occupied
│   ├── observed_free
│   ├── predicted_occupied
│   └── unknown
├── CollisionAssets
│   ├── static SDF / occupancy
│   └── actor canonical SDF / occupancy
├── DynamicState
│   ├── actor transforms
│   ├── lifecycle
│   ├── velocity / flow
│   └── swept volume
├── Validity
└── Provenance
```

每个 collision asset 至少记录：

```text
frame
units
resolution
extent
axis convention
semantic class
source type
source hash
generator hash
voxelizer / SDF hash
watertight status
volume
surface area
support statistics
```

---

# 7. 禁止复用旧 OccGS 实现作为正式方法

旧 Occupancy 路线只能复用：

- UNKNOWN / FREE / OCCUPIED 三态思想；
- ray-carving 基础组件；
- WorldState / typed label 接口；
- failure ledger。

不能复用为正式结论的内容：

```text
固定绝对路径
0.4m 全局粗网格
rotated box → corner AABB fill
frame 语义不清
动态 Actor 扁平覆盖
learned occupancy 自评
高 UNKNOWN 通过降阈值解决
```

V6.1 必须：

1. 使用 `T_dst_src`；
2. oriented-box / mesh / SDF voxelization；
3. Actor 层独立；
4. source removal 后恢复 UNKNOWN，不恢复 FREE；
5. observed / predicted / generated 分层；
6. learned occupancy 不作为唯一 evaluator。

---

# 8. Minimum Experiment 总览

## 目标

在既有 28-case development denominator 上回答：

> **Occupancy-conditioned 3D proposal 是否能在 0 False-safe 下，把安全通过率从 R10 的 3/28=10.7% 提升到至少 5/28=17.9%？**

为什么是 `5/28`：

- 不是仅多通过一个偶然 case；
- 至少增加两个独立 case；
- 能形成比 10.7% 明确更强的最小信号；
- 单卡成本可控。

---

# 9. ME-0：SceneIR-O / Occupancy Truth Tier

Task：

```text
WS-V61-ME0-OCCIR-01
```

不运行生成器。

## 输入

- 现有 R10 28-case；
- logged LiDAR；
- multi-sweep LiDAR；
- 3D boxes / actor tracks；
- camera calibration；
- map（若合法可用）；
- held-out target evidence。

## 输出

对每个 case：

```text
O_method:
  方法允许读取的 occupancy evidence

O_eval:
  只用于最终 evaluator 的独立 occupancy evidence
```

方法与 evaluator 严格分离。

## 必须通过

- coordinate round-trip；
- oriented volume；
- actor identity；
- lifecycle；
- free / occupied / unknown；
- method/eval evidence hash disjoint；
- fresh process exact；
- 旧 AABB inflation anti-regression；
- no hard-coded path。

---

# 10. ME-1：Oracle Occupancy Upper Bound

Task：

```text
WS-V61-ME1-ORACLE-OCC-PROPOSAL-01
```

## Arms

### `B0-2D`

冻结的 2D generator baseline。

### `B1-R10`

冻结 R10：

```text
cross-frontend + P1 + P2
```

### `O1-GATE`

```text
B1 proposal
+
Occupancy q_geometry
```

用途：

> 只验证 gate，理论上不能提高 coverage。

### `O2-OCC-GEOMETRY`

```text
R10 appearance
+
oracle occupancy surface / volume
```

静态 hole：

```text
occupancy boundary
→ deterministic surface samples
→ Gaussian appearance attachment
```

Actor hole：

```text
oracle actor canonical occupancy
→ mesh / SDF
→ surface Gaussian appearance
```

### `O3-OCC-4D`

```text
O2
+
trajectory
+
lifecycle
+
swept collision
```

## Primary gate

必须同时满足：

```text
accepted_cases >= 5/28
false_safe = 0
R10 原 3 个 ACCEPT 全部保留
至少新增 1 个 actor case
至少新增 1 个 static/disocclusion case
accepted mask-area yield >= 12%
```

## Stop rule

若 oracle occupancy 都不能达到：

```text
5/28
```

则停止：

- Hunyuan3D；
- GaussianWorld；
- Drive-OccWorld；

先修 compiler representation / evaluator。

---

# 11. ME-2：Hunyuan3D-Omni Actor Proposal

Task：

```text
WS-V61-ME2-HY3D-OCC-ACTOR-01
```

只使用 6 个 actor stratum。

## Matched arms

### `A0-image`

image-only actor generation。

### `A1-bbox`

image + bbox control。

### `A2-point`

image + observed point control。

### `A3-voxel`

image + occupancy voxel control。

一次只比较控制方式，不同时改变 texture / trajectory / verifier。

## 转换

```text
generated mesh
→ canonical scale alignment
→ watertight / connected-component audit
→ canonical occupancy / SDF
→ surface Gaussian sampling
→ SceneIR actor asset
```

## Actor-specific gate

```text
>=2/6 actor cases ACCEPT
false-safe = 0
free-space conflict = 0
swept collision = 0
ground contact PASS or UNKNOWN
photo / semantic 不灾难退化
```

若只有外观好、几何失败：

```text
Hunyuan arm rejected
```

---

# 12. ME-3：Predicted Occupancy 替换 Oracle

Task：

```text
WS-V61-ME3-PREDICTED-OCC-01
```

优先顺序：

1. GaussianWorld pretrained；
2. OccWorld pretrained；
3. Drive-OccWorld inference-only；
4. IR-WM；
5. OccSora 仅外部资源。

## Baseline

```text
oracle occupancy upper bound
```

## 目标

预测 Occupancy 替代 oracle 后：

```text
valid yield 保留率 >= 80%
false-safe = 0
UNKNOWN 诚实保留
```

例如 oracle 为 `5/28`：

```text
predicted occupancy 至少 4/28
```

但最终 paper candidate 仍要求安全超过 V6 的 `3/28`。

---

# 13. ME-4：多 Actor False-safe Stress

Task：

```text
WS-V61-ME4-MULTIACTOR-COLLISION-01
```

## 数据

至少：

```text
3 scenes
2 / 3 / 4 actor
每场 50 proposals
总计 >=150 proposals
```

Proposal 包含：

- translation；
- yaw；
- size；
- trajectory perturbation；
- new actor insertion。

## Baselines

### `C0-AABB`

当前 broad-phase AABB。

### `C1-OBB`

oriented box。

### `C2-OCC`

canonical occupancy + per-frame transform。

### `C3-SWEPT`

occupancy / SDF swept collision。

## 独立 evaluator

不能使用同一个 learned occupancy 输出作为 evaluator。

Evaluator 使用：

- exact mesh / voxel intersection；
- held-out LiDAR / map；
- simulation-defined collision body；
- native actor boxes；
- temporal continuous interpolation。

## Primary

\[
\mathrm{FalseSafe}
=
\frac{
\#(\mathrm{ACCEPT}\land \mathrm{collision})
}{
\#\mathrm{ACCEPT}
}
\]

Gate：

```text
0 false-safe / >=150 proposals
Clopper–Pearson 95% upper bound 报告
accepted yield 比 C0 提高 >=10%
无 actor-static 穿透
无 lifecycle 外 phantom collision
```

---

# 14. 三种下游任务的统一方式

## 14.1 GS + LogSim

Occupancy 的作用：

- 校验原 case；
- collision/event sidecar；
- 保持原日志因果；
- 检测重建或编辑是否篡改关键几何。

验收：

```text
same log
→ same world state
→ same occupancy
→ same labels
→ same collision events
→ sensor deterministic replay
```

不允许生成内容改变原 case。

---

## 14.2 GS + WorldSim

Occupancy 的作用：

- 新路线支持域；
- Actor placement；
- ground contact；
- actor-static；
- actor-actor；
- disocclusion geometry；
- generated chunk bake。

主要指标：

```text
Safe Valid Yield
False-safe
Verified route length
Verified world area
Actor insertion yield
Multi-actor interaction yield
UNKNOWN / abstain
```

---

## 14.3 GS + NWM 强化学习

推荐架构：

```text
NWM:
  负责 reactive multi-agent dynamics

Occupancy runtime:
  负责 authoritative physical state / collision / reward

GS runtime:
  负责 sensor observation

V6.1 compiler:
  负责三者一致性与资产资格
```

即：

```text
NWM proposal
→ actor trajectories
→ O_runtime(t)
→ collision / reward
→ GS sensor render
→ planner action
→ next rollout
```

这可以避免：

```text
视频世界模型把障碍物生成没了
→ RL 误以为动作安全
```

---

# 15. NWM RL 最小验收

V6.1 第一版不要求训练大型 NWM。

先做：

## RL-0：Collision Critic Dataset

三臂：

```text
Real-only
Real + naive generated
Real + OccCompiler verified
```

相同：

- planner；
- NWM；
- training budget；
- seed；
- episode 数；
- action set。

Primary：

```text
collision false-safe
collision critic recall
unsafe proposal rejection precision
```

Secondary：

```text
route progress
stuck
comfort
overall reward
```

不得只报告 Reward。

## RL-1：轻量 model-based RL

若 RL-0 通过，再用：

- REINFORCE；
- GRPO；
- preference optimization；
- correction policy；

中的一个。

Gate：

```text
False-safe 显著低于 Real-only 与 naive
collision 降低
progress 不灾难下降
无全刹车 reward hacking
```

---

# 16. 防止 Reward / Evaluator 循环论证

必须有两套物理链：

## Method chain

```text
O_predicted
+
proposal collision body
+
compiler q_geometry
```

## Evaluation chain

```text
O_eval
+
held-out LiDAR/map
+
exact mesh/voxel collision
+
native box trajectory
```

若 method 和 evaluator 使用同一 occupancy tensor：

```text
结果无效
```

RL reward 可以使用 simulation-defined Occupancy，但论文评估必须用独立 collision audit。

---

# 17. 论文贡献建议

## Contribution 1

**Appearance–Collision Dual World Representation**

将高斯外观和物理 collision state 解耦，但在 SceneIR 中绑定。

## Contribution 2

**Occupancy-Verified Proposal Compilation**

任何 2D/3D/4D generator 都只产生 proposal，通过因子化 Occupancy 证书后才 bake。

## Contribution 3

**4D Swept Occupancy for Multi-Actor Interaction**

canonical actor occupancy + trajectory + lifecycle，解决多 Actor False-safe。

## Contribution 4

**Task-Conditioned Runtime**

同一 compiled world 支持：

- LogSim；
- WorldSim；
- NWM RL。

## Contribution 5

**Safety-Centric Evaluation**

用：

```text
Safe Valid Yield
False-safe
abstention
task utility
```

替代单纯 PSNR / FVD / Reward。

---

# 18. 与已有工作的差异

| 工作 | 已有能力 | V6.1 不应重复 | V6.1 差异 |
|---|---|---|---|
| Hunyuan3D-Omni | controllable 3D asset | 3D shape generation | 驾驶 Actor 的 Occupancy 验证、4D trajectory 与 safe bake |
| DiffSplat | 快速 3DGS 生成 | 3D Gaussian proposal | collision body 与 task validity |
| Diff4Splat | 单图 4D scene | 4D GS generation | authoritative collision state |
| OccWorld | 未来 Occupancy | occupancy forecasting | persistent GS/Occ world compilation |
| Drive-OccWorld | action-conditioned Occ + planning | occupancy planner | proposal qualification、deterministic bake、多下游 runtime |
| OccSora | trajectory-conditioned 4D Occ | 4D Occupancy diffusion | 单卡可执行 compiler 与安全验证 |
| UniScene | Occ → video/LiDAR | occupancy-centric generation | false-safe-controlled world acceptance |
| GenieDrive | Occ-guided video | physics-guided visual generation | explicit collision assets and deterministic runtime |
| WorldSplat | feed-forward 4D Gaussians | generic 4D GS generation | verifier/compiler/provenance/false-safe |
| OG-Gaussian | Occ initialized GS | Occupancy densification | multi-Actor collision and task-valid bake |

---

# 19. ArXiv 报告实验表

## Table 1：28-case proposal

```text
2D generator
R10 cross-frontend
Occ gate
Oracle Occ proposal
Predicted Occ proposal
```

报告：

- ACCEPT；
- ABSTAIN；
- REJECT；
- false-safe；
- mask area；
- photo / geometry / semantic。

## Table 2：Actor generator

```text
image
bbox
point
voxel
```

报告：

- shape support；
- free violation；
- collision closure；
- ground contact；
- actor rendering；
- final valid yield。

## Table 3：Multi-Actor

```text
AABB
OBB
Occupancy
Swept Occupancy
```

报告：

- false-safe；
- false reject；
- coverage；
- penetration；
- runtime。

## Table 4：三下游

```text
LogSim
WorldSim
NWM RL
```

报告任务级指标，不混成总分。

## Table 5：资源

- 3090 VRAM；
- compile time；
- bake size；
- runtime FPS；
- occupancy memory；
- collision query latency。

---

# 20. 第一阶段正式执行顺序

```text
P0  V6 closeout / V6.1 scope freeze
↓
P1  SceneIR-O + truth tiers
↓
P2  ME-0 exact occupancy contract
↓
P3  ME-1 oracle occupancy upper bound
↓
     FAIL → 停止模型接入，修 compiler
     PASS
↓
P4  Hunyuan3D-Omni 3090 smoke
↓
P5  ME-2 controlled actor generation
↓
P6  GaussianWorld / OccWorld smoke
↓
P7  ME-3 predicted occupancy
↓
P8  ME-4 multi-actor false-safe stress
↓
P9  LogSim / WorldSim
↓
P10 NWM collision critic
↓
P11 optional RL post-training
```

---

# 21. Stop Rules

## 停止 Hunyuan3D 路线

若：

```text
voxel-controlled actor
仍不能达到 2/6 actor safe acceptance
```

停止调 prompt / seed / texture。

## 停止 learned Occupancy

若：

```text
predicted occupancy false-safe > 0
且 calibration 无法在未读 confirmation 前修复
```

保持 UNKNOWN / abstain，不降阈值。

## 停止 RL

若：

```text
Real-only 与 V6 一样好
```

不削弱 Real-only 制造增益。

若：

```text
碰撞降低来自全刹车
```

以 progress/completion gate 拒绝。

---

# 22. 最终推荐

V6.1 最优定位不是：

> 3DGS + Occupancy 模块。

而是：

> **一个 Occupancy-authoritative、Gaussian-rendered、task-verifiable 的四维驾驶世界编译器。**

最小实验的唯一关键问题：

> **在同一 28-case denominator 上，Occupancy-controlled 3D proposal 能否以 0 False-safe 将 valid yield 从 3/28 提升到至少 5/28？**

只要该实验通过，V6.1 就获得一个非常清晰、可继续扩展的硬核技术报告故事：

```text
2D proposal geometry failure
→ 3D/4D Occupancy authority
→ verified world compilation
→ multi-actor collision safety
→ LogSim / WorldSim / NWM RL
```

若该实验失败，V6.1 也会得到一个高价值结论：

> 当前瓶颈不是 verifier 不够强，而是没有可用的三维生成提案；此时应转向更强 4D scene generator 或外部算力，而不是继续包装 2D Diffusion。
