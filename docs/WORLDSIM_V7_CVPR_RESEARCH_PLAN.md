# WorldSim V7 研究计划

## HARP-3D：危险保真、伪影修复的三维物理世界编译器

> **Hazard-Preserving, Artifact-Repairing Physical 3D World Compiler**
>
> 面向 CVPR / ICCV 主会投稿；先发布 arXiv，后提交匿名主会版本。

---

## 0. 计划定位

本文档是 **V7 的 Markdown 研究执行计划**，不是论文正文。

V7 的核心研究成果之一，是在研究推进过程中同步产出并持续更新一套 **CVPR/ICCV 风格的 LaTeX 论文初稿与可编译 PDF**：

```text
paper/
├── main.tex
├── supplement.tex
├── sections/
├── figures/
├── tables/
├── bibliography.bib
└── results/results_macros.tex
```

计划本身继续使用 Markdown，方便 Codex / AutoResearch 直接读取、修改和执行；论文初稿是 V7 的正式交付物，而不是计划格式。

---

# 1. 项目北极星

V7 不再继续扩展 V6.7 后期的多条件 authority head，也不以强化学习（Reinforcement Learning，RL）为主论文贡献。

主问题收敛为：

> **高质量驾驶世界不应通过删除危险 Actor 获得“安全”，而应修复违反传感器证据、三维几何与时空连续性的伪影，同时保留合法但危险的交通参与者。**

论文中心命题：

\[
\boxed{\text{World validity} \neq \text{World difficulty}}
\]

进一步分解为：

\[
\boxed{\text{Artifactness} \neq \text{Hazardness}}
\]

\[
\boxed{\text{Appearance} \neq \text{Physical collision state}}
\]

\[
\boxed{\text{Physical validity} \neq \text{Task-conditioned reliability}}
\]

V7 的目标不是继续做一个更强的 Occupancy 小头，而是形成以下显式世界表示：

\[
\mathcal W_t =
\left(
\mathcal G_t^{app},
\mathcal S_t^{phys},
\mathcal A_t,
\mathcal P_t,
\mathbf q_t^{val}
\right)
\]

其中：

- \(\mathcal G^{app}\)：外观高斯或渲染表征；
- \(\mathcal S^{phys}\)：显式可碰撞物理表面、符号距离场或占据表面；
- \(\mathcal A\)：带身份、轨迹、生命周期和危险属性的动态 Actor；
- \(\mathcal P\)：来源、射线、多帧支持和传感器证据；
- \(\mathbf q^{val}\)：局部有效性、未知性和物理证书。

强制保留：

```text
Gaussian opacity != occupancy probability
Appearance Gaussian != collision volume
Unknown != free
Hazardous Actor != artifact
```

---

# 2. 面向论文的三项主贡献

V7 将 70% 左右的主文篇幅、图表和实验预算集中在 C1–C3。

## C1：危险保真的三维物理几何编译

### 核心问题

V6.7 已证明运动补偿后的 Actor 内向射线修复可以降低局部冲突并保留 Actor 身份、轨迹和危险语义，但当前操作仍主要是：

```text
KEEP / UNKNOWN
```

容易被视觉审稿人视为几何过滤或后处理规则。

### V7 升级目标

将 Actor-local 物理编译升级为：

```text
KEEP
PROJECT
COMPLETE
UNKNOWN
```

并建立显式 Actor canonical physical surface：

\[
S_i^{phys}
=
\operatorname{Fuse}
\left(
\{T_{actor,t}^{-1}T_{ego,t}T_{sensor}x_t\}_{t}
\right)
\]

目标是证明：

1. 被删除或修复的是证据不一致的伪影；
2. 合法危险 Actor 的存在、身份、轨迹、尺寸、TTC 和危险行为保持；
3. 修复后的表面在三维空间、射线可见性和时间维度上更自洽；
4. 外观层和物理层明确分离。

---

## C2：有效性与危险性的条件解耦

### 核心问题

不能将重建困难、风险较高或与 Ego 接近的 Actor 自动判为无效。

V7 不追求全局统计独立，而追求 **配对条件不变性（paired conditional invariance）**。

定义：

\[
z_i^{val}
=
f_{val}(E_i,S_i^{phys},P_i)
\]

\[
z_i^{haz}
=
f_{haz}(X_{i,0:H},\tau,\mathcal M)
\]

对应：

\[
a_i=P(\text{artifact}\mid z_i^{val})
\]

\[
h_i=P(\text{hazard}\mid z_i^{haz})
\]

配对约束：

- 仅把同一 Actor 从普通驾驶改成合法加塞、急刹或近碰撞时：

\[
a_i(A_i^{safe})\approx a_i(A_i^{hazard})
\]

- 仅在同一 Actor 上注入 duplicate shell、ghost、flicker 或 teleport 时：

\[
h_i(A_i^{clean})\approx h_i(A_i^{artifact})
\]

V7 需要通过四象限数据、配对反事实、交叉探针和表征可视化证明：

```text
合法且普通
合法且危险
伪影且普通
伪影且危险
```

可以被正确区分，而不是把“危险”清洗掉。

---

## C3：连续任务代价密度与多时域可靠性

V6.7 已获得较强证据：直接建模连续任务代价密度，比为多个阈值分别训练二元分类器更稳定。

定义候选 Ego 轨迹 \(\tau\)、Actor 残差 \(\epsilon_{i,t}\)、轨迹边界法向 \(n_{\tau,t}\) 和净空 \(d_{i,t}(\tau)\)：

\[
C(\tau,H)
=
\max_{i,t\le H}
\frac{
|n_{\tau,t}^{\top}\epsilon_{i,t}|
}{
\max(|d_{i,t}(\tau)|,\varepsilon)
}
\]

学习连续条件密度：

\[
p_\theta\left(\log(1+C)\mid z^{val},z^{haz},\tau,H\right)
\]

查询可靠性：

\[
R(\tau,H,b)
=
P_\theta(C(\tau,H)\le b)
\]

必须满足：

\[
\frac{\partial R}{\partial b}\ge0
\]

\[
\frac{\partial R}{\partial H}\le0
\]

V7 不重新搜索大量 density family，而是整合并简化 V6.7 已成立的链路：

```text
Actor residual distribution
→ analytic trajectory boundary query
→ continuous cost density
→ input-conditioned multi-horizon dependence
→ monotone runtime reliability surface
```

重点转为：

- 跨数据集零样本迁移；
- 与物理几何质量的因果联系；
- 解释性；
- 统一 paper implementation；
- 最终一次 exact-once test。

---

# 3. 论文定位与 Claim Boundary

## 3.1 推荐论文标题

首选：

> **Validity Is Not Difficulty: Hazard-Preserving Physical 3D World Compilation for Autonomous Driving**

备选：

> **HARP-3D: Hazard-Preserving and Artifact-Repairing Physical World Compilation for Autonomous Driving**

## 3.2 一句话摘要

> A valid driving-world compiler should repair evidence-inconsistent 3D artifacts while preserving legitimate hazardous actors, and should expose task-conditioned reliability through a continuous multi-horizon cost distribution rather than a single confidence score.

## 3.3 可以主张

- Actor-preserving、ray-consistent 三维物理编译；
- validity 与 hazard 的配对条件解耦；
- learned Actor uncertainty 与 analytic task geometry 的因子化；
- 连续代价密度和 input-conditioned multi-horizon reliability；
- nuScenes 训练、Argoverse 2 零样本测试；
- 显式、可查询、确定性的 runtime SceneIR；
- 下游候选轨迹选择可以消费该可靠性接口。

## 3.4 不能主张

- 形式化现实道路安全保证；
- 所有物理伪影都能被修复；
- Gaussian opacity 是 collision occupancy；
- 所有 Surface/CVaR/UQ 方法均失败或均成立；
- 完整 RL-ready simulator 已经成立；
- 仅凭 proxy metric 即证明 closed-loop policy 提升；
- Argoverse 2 上做过 fine-tuning 时仍称零样本。

---

# 4. V7 总体架构

```text
Real Logs / Reconstructed World
              │
              ▼
┌──────────────────────────────────────┐
│ Actor-Preserving Physical Compiler   │
│                                      │
│ multi-frame Actor alignment          │
│ ray FREE/OCC evidence                │
│ canonical physical surface / SDF     │
│ KEEP / PROJECT / COMPLETE / UNKNOWN  │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ Validity–Hazard Factorization        │
│                                      │
│ validity: evidence / geometry        │
│ hazard: dynamics / TTC / route       │
│ paired invariance / leakage probes   │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ Task-Conditioned Reliability         │
│                                      │
│ Actor residual distribution          │
│ analytic trajectory-boundary query   │
│ continuous task-cost density         │
│ joint multi-horizon dependence       │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ Monotone Runtime Reliability Surface │
│ R(trajectory, horizon, budget)       │
└────────────────┬─────────────────────┘
                 │
                 ▼
       Planner / Action Selector / RL demo
```

---

# 5. 执行阶段

## V7-P0：Paper-first 分支与最小代码收敛

### P0 执行记录（2026-09-01）

- 基线：V6.7 `d97c3f2`；分支：`research/worldsim-v7-harp3d-cvpr`。
- 论文：固定官方 `cvpr-org/author-kit@2917585`（当前官方 CVPR 2026 kit；2027 kit 尚未发布），建立匿名
  `main.tex`、`supplement.tex`、章节、结果宏、架构图和 contribution map；主稿与补充材料均已编译。
- 代码：已从历史链抽出 paper-facing `worldsim_v7` 六模块，固定
  `KEEP/PROJECT/COMPLETE/UNKNOWN`、validity--hazard 输入隔离、Actor residual、连续 cost density、单调 runtime
  surface 与 AV2 SceneIR 适配器；不回改历史 Pxxx。
- 数据：AV2 Sensor `val` 的 150 个 UUID 仅按 metadata 排序，每隔 5 个冻结 30 logs；前 20 个 quantitative、后
  10 个 qualitative。冻结发生在任何方法输出或质量 read 之前，且禁止 AV2 fine-tune、校准、阈值选择和失败场景删除。
- 传输：远端小文件下载实测约 `0.58 MiB/s`，本地 D 盘约 `2.86 MiB/s`，故按预案改为本地串行下载后手工上传；
  远端只保留 `s5cmd`，测速残片删除。
- 验证：P0 定向单元测试 `5 passed`；完整首个 AV2 log 的 metadata-only SceneIR smoke 得到
  `2671` ego poses、`11` sensors、`59` Actors、`5337` states；本轮不训练、不做 repo-wide regression。
- 失败账本继承：`V66-F01`（Actor existence 与 local geometry 分权）、`V66-F02`（triage 不等于 physical
  repair）、`V67-F232`--`V67-F236`（后期可靠性目标错配）；本轮资源/协议事件登记为 `V7-F01`。

### 目标

从实际远端 V6.7 最新 HEAD 新建：

```text
research/worldsim-v7-harp3d-cvpr
```

建立 paper-facing 目录：

```text
paper/
├── main.tex
├── supplement.tex
├── sections/
│   ├── 01_intro.tex
│   ├── 02_related.tex
│   ├── 03_problem.tex
│   ├── 04_method.tex
│   ├── 05_experiments.tex
│   ├── 06_limitations.tex
│   └── 07_conclusion.tex
├── figures/
├── tables/
├── results/results_macros.tex
└── bibliography.bib
```

重构 paper-facing 实现：

```text
motion_proj/worldsim_v7/
├── physical_compiler.py
├── validity_hazard.py
├── actor_reliability.py
├── boundary_cost_density.py
├── runtime_surface.py
└── sceneir_adapter.py
```

### 原则

- 不做 repo-wide regression；
- 不加哈希、指纹、checksum；
- 不重构所有历史 Pxxx；
- 仅将论文最终依赖链从历史实验脚本中抽出来；
- 每个 V7 里程碑同步更新 LaTeX 结果宏、表格和图占位。

### 交付物

- `WORLDSIM_V7_CVPR_RESEARCH_PLAN.md`；
- 可编译的 CVPR/ICCV 风格匿名论文骨架；
- 第一版 `main.pdf`，允许结果为 TBD；
- 一张总架构图草稿；
- 一页 paper contribution map。

---

## V7-P1：四象限物理—危险证据图谱

### P1 执行冻结（2026-09-02）

- Formal task：`WS-V7-P1-AV2-FACTORIAL-ATLAS-01`；冻结 AV2 quantitative 20 logs，zero-shot only。
- 使用官方 AV2 ego-frame LiDAR/cuboid 合同，在 GPU 上完成 Actor-local 入盒、坐标变换、`0.12m` surfel fusion
  和 held-out surface evaluation；固定 2/3 build、1/3 held-out，不按质量换帧或 Actor。
- 迁移 NeuRAD/SplatAD 的 Actor-local point seeding 边界，但禁止其 mirrored seed augmentation 进入物理层。
- 每个真实 surfel 产生 clean 与 paired artifact probes，Actor ID、轨迹、尺寸和 hazard 不变；Validity/Hazard
  输入严格隔离。配置与 gates 在 formal quality read 前提交并推送。
- 首次 verdict 后按冻结停止条件决定：surface 指标支持则进入 P2 四动作真实几何编译，否则直接进入 P2 canonical
  surface 表示修复，不在 P1 扫 voxel/threshold/cohort。

### P1 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P1-AV2-FACTORIAL-ATLAS-01/20260902T104500Z__av2-factorial-s0-r1`；10/10 gates，
1,046 Actors（241 hazard）、1.075M surfels、134,914 paired probes。真实 held-out LiDAR target distance
`0.864→0.131m`，recall `26.16%→85.72%`；Actor/hazard retention=100%。paired synthetic corruption 的 action/
artifact score=100%、clean-hazard false-artifact=0，只登记为接口合同证据。P1 参数族关闭，进入 P2。

### 数据构造

建立四象限：

| 物理有效性 | 普通行为 | 危险行为 |
|---|---|---|
| 合法 | valid-safe | valid-hazard |
| 含伪影 | artifact-safe | artifact-hazard |

危险行为至少包括：

- cut-in；
- hard braking；
- close merge；
- pedestrian/cyclist crossing；
- near miss；
- physically feasible collision。

伪影至少包括：

- observed-FREE ghost；
- duplicate shell；
- floating surface；
- temporal pop/flicker；
- teleport；
- identity/surface mismatch；
- ray-inconsistent geometry。

### 评测

不先训练复杂网络，只回答：

1. V6.7 inward-ray repair 能消除哪些伪影？
2. clean-hazard 是否被误判为 artifact？
3. 修复是否改变 Actor ID、trajectory、size、TTC？
4. artifact-hazard 是否能够修伪影而保留 hazard？

### 主要指标

- artifact recall / precision；
- clean-hazard false-artifact rate；
- Actor identity retention；
- trajectory/TTC shift；
- free-space violation；
- temporal flicker；
- primitive/surface retention。

### 停止条件

如果现有 inward-ray 只能改善内部 conflict proxy，却无法改善任何三维或时序物理指标，则 C1 不允许直接包装成 geometry repair，必须进入 P2 canonical surface。

---

## V7-P2：Actor canonical physical surface

### P2 formal freeze（2026-09-02）

- 分支从 `research/worldsim-v6.7-anisotropic-surface@d97c3f2` 直接拉出 V7。
- Formal task=`WS-V7-P2-AV2-FOUR-ACTION-COMPILE-01`；冻结 30 AV2 logs 一次完成 quantitative 与 qualitative
  confirmation，不在两阶段之间改 candidate。
- build/query/target 按固定 frame index 分离；所有 action 仅读取 build canonical 与 query evidence，target-only 几何
  只用于最终 before/after 评价。
- `KEEP/PROJECT/COMPLETE/UNKNOWN` 均产生真实坐标 surface；completion 必须有 build-side `>=3` temporal 与 `>=2`
  view support。13-gate AND rule 在 formal read 前提交。
- r1 仅在首 log metadata 后因漏配复用的冻结 hazard 段退出，未进入 point/surface/action/metric；r2 只恢复 P1
  同一 hazard 段并保持上述合同不变，不把工程入口失败计作科学 trial。

### P2 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P2-AV2-FOUR-ACTION-COMPILE-01/20260902T125000Z__four-action-s0-r2`；
30/30 logs，quantitative/qualitative 各 13/13 gates。Quantitative 433 Actors（147 hazard），真实 target-only
recall=`.5336→.6260`、precision=`.7959→.9767`、Chamfer=`.2544→.1681m`；Actor/hazard retention=`1/1`。
paired action rates 只作 deterministic contract 证据。P2 参数族关闭，不回扫，进入 P3 硬证据与预冻结视觉结果。

### P2-A：多帧 Actor 对齐

将每个 Actor 的多帧 LiDAR / point / primitive 转换到 canonical Actor frame：

\[
x_{i,t}^{actor}
=
T_{actor,t}^{-1}
T_{ego,t}
T_{sensor}
 x_t^{sensor}
\]

构造：

- point/surfel surface；
- FREE/OCC ray evidence；
- temporal support count；
- local normal；
- Actor box/shell prior；
- provenance。

### P2-B：物理表面表示

第一版优先比较：

1. motion-compensated surfel fusion；
2. truncated signed distance field（TSDF）或 signed distance field（SDF）。

不同时上大型神经隐式场。

### P2-C：四动作编译

- `KEEP`：已有稳定表面支持；
- `PROJECT`：靠近可信零水平集但违反局部射线一致性；
- `COMPLETE`：多帧、多方向支持下的局部孔洞补全；
- `UNKNOWN`：无足够证据。

### P2-D：外观—物理解耦

外观高斯可以通过：

- center-to-surface distance；
- covariance normal alignment；
- depth consistency；
- visibility consistency；

附着到物理表面。

但下游碰撞和占据查询只读取 `S_phys`。

### 学习模块解锁条件

只有确定性表面在以下方面出现系统性缺口时，才允许学习 residual completion：

- 远侧不可见面；
- 小型 Actor；
- 长遮挡；
- 稀疏 LiDAR；
- 局部多帧支持不足。

第一学习候选必须是低容量 Actor-local residual SDF；最多一个主方案和一个结构 fallback。

---

## V7 reviewer-facing 补强主线（2026-09-02）

1. **视觉与几何硬证据**：V7 必须从 feature filtering 推进到 ray/depth/surface/collision 的真实三维自洽；失败时优先
   修物理 provenance 或表示，而不是增加后处理规则。
2. **外部泛化降维打击**：模型训练/选择限定 nuScenes，AV2 为冻结 zero-shot 主外域；资源允许时再加入 Waymo 第二外域，
   但不以扩大数据面替代 AV2 深证据。
3. **理论边界与工程极致**：将 sensor opportunity、`UNKNOWN`、immutable Actor/hazard state、可修复/不可修复条件和
   runtime/resource 明确暴露，提供可解释边界；禁止包装成 road-safety formal guarantee。

后续 auto-research 聚焦这三条，不横向堆叠与论文主张无关的模块。

---

## V7-P3：三维硬证据与视觉结果

### P3-A formal freeze（2026-09-02）

- Formal task=`WS-V7-P3-AV2-HARD-EVIDENCE-01`，冻结 30 logs 与 P2 canonical config；新 overlay 只定义硬指标、
  8-gate non-regression rule 和 renderer，不复制 Actor/hazard 参数。
- target-ray depth/termination 使用各 target LiDAR point 原始 Actor-local sensor origin，target 仍不参与 action；
  temporal jitter 是 build-frame-to-canonical residual 的跨帧标准差。paired free-space/ghost component 明确为合成合同。
- qualitative 10 logs 不删；每 log 词法序前三 eligible Actor。main 固定前 8 logs 首 Actor，supplement 固定全部 30；
  首阶段产出 point/surface/ray panel，P3-B 再接 frozen camera RGB/depth/video，不按结果换案例。
- r1 30/30 完成但两 role 均 6/8；任意 compiled-surface Euclidean residual 破坏逐 primitive ray provenance，
  free-space/ghost 两门无效且总 verdict rejected（`V7-F07`）。r2 仅迁移 NeuRAD/SplatAD/LiDAR-RT 的最小
  `ray_o/ray_d + hit + aligned output` 边界；不改案例、动作、阈值、gates 或其余六项指标。
- r2 的正确 ray metric 仍为 6/8：nearest-canonical 输出沿 matched beam 平均早于 hit 约 9cm，确认 PROJECT
  operator 的真实物理失败（`V7-F08`）。r3 冻结为 ray-certified PROJECT：有 direct observed hit provenance 才投回
  hit，无 matched hit 则 UNKNOWN；不调门/案例/target isolation，不把 zero-level 最近点冒充 line-of-sight 证据。

### P3-A 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T143000Z__ray-certified-s0-r3`；
两 role 各 8/8、视觉 3/3。Quantitative free-space=`1→0`、ghost components=`10983→0`、target depth=
`.2067→.1541m`、ray termination=`.5694→.6438`、zero-level=`.1400→.0583m`、Chamfer=`.2544→.1753m`，
Actor/hazard state shift=0。相较 rejected nearest-surface r2 的 `.1681m` Chamfer，r3 明确选择约 7.2mm 的拟合代价
换 matched-ray free-space 安全边界。P3-A 关闭，进入 P3-B RGB/camera-depth/video，不扫 tolerance。

### P3-B formal freeze（2026-09-02）

- Formal task=`WS-V7-P3B-AV2-CAMERA-EVIDENCE-01`；逐项复用 P3-A 8 main + 30 supplement cases，case identity 与
  main/supplement 身份不可变。compiler 只新增 query timestamp 与 Actor ego rigid transform diagnostics，不改 action/metric。
- 相机候选固定为 AV2 官方 7 ring cameras。选择只计算 query Actor LiDAR points 经官方 motion-compensated projection
  后的 in-frame count；取最大者，平局按 config 顺序，完成选择后才 decode RGB。crop 只由 query projection 决定。
- 每 case 固定输出 RGB+observed returns、paired artifact overlay、four-action output、sparse camera-depth 四联图与动态
  before/after MP4。全部 30 cases 写出，不按可见结果换图，也不扫 camera/crop/palette。
- paired ghost/duplicate/flicker 与视频闪烁是 synthetic contract evidence；RGB panel 是 calibrated evidence overlay，
  不是 photorealistic reconstruction。真实几何结论继续由 P3-A target-only LiDAR depth/ray/surface 承担。

### P3-B 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P3B-AV2-CAMERA-EVIDENCE-01/20260902T150000Z__camera-evidence-s0-r1`；
10/10 logs 与 30/30 frozen cases 完成，8 main + 30 supplement panels、30 MP4。query visibility points min/median=
`14/82`，minimum visibility fraction=`.7278`，zero-visible=0；8 main 人工检查均可读且未换图。wall=`43.38s`、
GPU=`.0663GiB`、RSS=`1.339GiB`、run=`46MiB`。

不挑图同时暴露 `V7-F09`：全 634 Actors 的 aggregate Chamfer 改善下仍有 `109/634=17.19%` individual worsening，
depth/ray 方向变差分别为 `271/253` Actors。故 P3 只支持 aggregate zero-shot geometry，不支持 per-Actor universal
repair certificate。P4 必须把 validity head 改成 nuScenes-only selective repairability/risk--coverage，并保持 hazard 输入
独立；AV2 未知 shift 下只做 zero-shot evaluation，不声称 conformal exchangeability guarantee。

### 必须补齐的几何指标

- free-space violation rate；
- ray termination consistency；
- surface precision / recall；
- Chamfer distance 或 point-to-surface error；
- SDF consistency；
- LiDAR depth consistency；
- temporal surface jitter；
- ghost connected components；
- Actor surface completeness；
- collision-shell consistency。

### 必须补齐的危险保真指标

- Actor retention；
- ID/lifecycle retention；
- trajectory displacement；
- speed/acceleration shift；
- TTC shift；
- cut-in / hard-brake / near-miss label retention；
- hazard event count change。

### 视觉证据

至少制作：

- 6–8 个主文 qualitative cases；
- 20+ supplement cases；
- RGB / depth / point / surface / ray evidence 并列图；
- 动态视频：修复前后 ghost、flicker、危险 Actor 保留。

禁止按最终质量挑图。Qualitative scene 在看方法结果前按 metadata 和预注册 failure strata 冻结。

---

## V7-P4：有效性—危险性条件解耦

### P4 执行冻结（2026-09-02）

- Formal task=`WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01`；nuScenes `11/14/38` scenes 分别用于
  train/calibration/test，角色完全 scene-disjoint；30 个 AV2 frozen logs 仅作外域 zero-shot。
- P3 `before` 带 synthetic corruption，P4 不再以此定义“修复成功”。逐 Actor label 固定为
  `Chamfer(compiled,target) <= Chamfer(clean query,target)`；abstain 返回 clean query，target 不进入 action/features。
- validity 仅用 13 个 runtime surface/ray/provenance/support 特征，hazard 仅用 TTC/clearance/closing/brake/crossing；
  比较唯一 shared-input two-head baseline 与 structurally factorized two-head candidate。
- 单 seed、固定 32 hidden/80 epochs，无 architecture/seed/threshold sweep。标准化=train only；false-repair threshold
  只在 nuScenes calibration 以 `.05` monotone adjusted risk 冻结，随后 AV2 不调任何参数。
- formal guarantee 只限 exchangeable nuScenes calibration 边界；未知 nuScenes→AV2 shift 只报告 risk--coverage/geometry，
  不声称 conformal、collision、planning、closed-loop 或 road-safety guarantee。Waymo 在本地无冻结 sensor corpus 时暂缓。

### P4 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01/20260902T161000Z__selective-factor-s70401-r2`；
7/7 gates，nuScenes Actors=`29/56/228`，AV2=30 logs/634 Actors。factorized nuScenes test repair/hazard AUROC=
`.6491/.9811`，shared=`.6426/.9166`；factorized paired swap shift=`0/0`，shared=`.2014/.2886`。

相同 nuScenes-only threshold 在 AV2 覆盖 `75.55%`，false repair `16.56%→8.99%`（always→selective）；
clean-query/always/selective Chamfer=`.2577/.1770/.1821m`，hazard coverage=`92.17%`。结果支持 empirical
nuScenes→AV2 selective transfer 与结构解耦，但 calibration→AV2 coverage=`8.93%→75.55%` 明确否定跨域 formal
exchangeability guarantee；selective 也不在平均 Chamfer 上支配 always repair。train 仅 29 Actors，冻结为限制。

### Baselines

- shared encoder；
- two-head shared trunk；
- independent encoders；
- paired-invariance factorized encoders（V7）。

### 输入

Validity encoder：

```text
ray support
FREE/OCC contradiction
surface residual
multi-frame consistency
source provenance
visibility
local geometry
```

Hazard encoder：

```text
Actor trajectory
relative pose/velocity
TTC
route relation
map context
interaction state
```

### 损失

\[
\mathcal L
=
\mathcal L_{artifact}
+
\mathcal L_{hazard}
+
\lambda_{pair}\mathcal L_{pair}
+
\lambda_{leak}\mathcal L_{leak}
\]

其中重点是 paired invariance，而非追求全局零相关。

### 评测

- artifact AUROC/AUPRC；
- hazard AUROC/AUPRC；
- clean-hazard false artifact；
- artifact-hazard hazard retention；
- hazard-from-validity probe；
- artifact-from-hazard probe；
- paired latent distance；
- paired prediction consistency。

### 成功标准

V7 模型必须相对 shared encoder：

- artifact 与 hazard 主任务均不退化；
- clean-hazard 误删率明显降低；
- 配对不变性改善；
- cross-probe leakage 下降；
- 不能通过 all-UNKNOWN 或降低危险样本分母获得改善。

---

## V7-P5：C3 论文整合与轻量重训

### P5-A alignment audit freeze（2026-09-02）

P4 `scene_name/instance_token` 与 V6.7 `scene_index/numeric_actor_id` 只允许通过 nuScenes official scene ordering 和
DriveStudio `instances_info[id]` exact join。Direct joint fit 需 P4 train 对齐 `>=3 scenes/20 Actors`，否则无法建立非平凡
scene-heldout split，禁止用 calibration/test Actors 回流补样本；仅执行 frozen descriptive interface audit。

Alignment canonical=`run://worldsim_v7/WS-V7-P5-PHYSICAL-RELIABILITY-ALIGNMENT-AUDIT-01/20260902T180000Z__physical-reliability-alignment-s0-r1`。
P4 train/cal/test 对齐=`2/5/14 scenes`、`5/25/88 Actors`、`1,320/7,645/39,285 rows`；train 未达
`3/20`，因此 `V7-F13` 拒绝 direct joint fit。P5 后续只允许冻结 P4/P346 的 descriptive multi-horizon interface audit。

P5-B 固定使用 calibration/test `25/88` exact-match Actors，比较 frozen P4 selected/abstained、geometric
helpful/harmful、selected-and-harmful 在 retained V6.7 `.8/1.5/2.5/3.0s` cost/state-error/decision-flip/
false-safe 的分层。无 fit、threshold、gate 或 P346 execution；只给共现安全边界，不做因果改善声明。

P5-B canonical=`run://worldsim_v7/WS-V7-P5B-FROZEN-PHYSICAL-RELIABILITY-INTERFACE-01/20260902T183000Z__frozen-physical-reliability-interface-s0-r1`。
test `88 Actors/39,285 rows` 中 selected=`5/1,781`，false-safe/flip=`0/0`，23 个 geometric-harm Actors 全部
abstained；但 selected q90 state error=`1.577m` vs abstained `.693m`，3.0s=`4.246m`。因此 paper graph 固定为
physical repair gate 与 downstream multi-horizon reliability authority 分离；零事件小样本不升级为 safety guarantee。

CVPR draft 已把该边界写入 composed-authority equation、P5 table、abstract/limitations/conclusion；official template full
compile=`6 pages`，pages 2--6 visual check 通过。后续 P6-C fresh external 结果出来前不改现有 P5/P7 claim。

### 保留模块

- V6.7 已支持的 Actor residual distribution；
- analytic trajectory boundary query；
- conditional continuous cost density；
- input-conditioned multi-horizon dependence；
- monotone runtime reliability surface。

### 关闭模块

不复活：

- fixed Student-t 重尾分布；
- mixture component sweep；
- CRPS/Energy-loss family；
- DCT temporal family；
- end-to-end occupancy-flip classifier；
- learned selective authority head；
- P333 后继续扩展的条件校准树。

### V7 新增重点

1. 让 C3 输入显式使用 V7 validity representation；
2. 做 `with/without physical repair` 对照；
3. 检查 C1 是否改善 density 的 NLL/Brier/calibration；
4. 检查危险 Actor 保留是否维持或增加困难样本覆盖；
5. 简化 checkpoint 依赖为一套 paper-facing end-to-end inference graph。

---

## V7-P6：nuScenes → Argoverse 2 零样本迁移

### P6-B sensor-opportunity recovery freeze（2026-09-02）

P7 的 `V7-F11` 将失败层级定位为 `sensor opportunity shift`。唯一恢复候选删除 raw observation-frame count，并将
canonical surfels、temporal/view support 变为 per-observation dimensionless ratios；只用 nuScenes train/calibration 训练与
定阈值，hazard head 和 structural factorization 不变。模型固定为 hidden32/seed70601/80 epochs，无 feature/model sweep。

External recovery cohort 不复用已消费 30 logs：对 150 个 AV2 val UUID 排序，排除 v1 indices `0,5,...,145`，再从
120-log complement 每隔 6 个取一个，共 20 fresh logs。元数据选择在任何 recovery score/quality 前冻结。通过条件只包括
nuScenes non-inferiority、机会变换不变性、fresh coverage/false-repair/Chamfer，以及相对 frozen P4 的 fresh score-shift
改善；不得在 fresh read 后调 threshold。通过也不产生 formal cross-domain risk guarantee。

### P6-B ratio result / P6-C sparsity-consistency direction（2026-09-02）

Ratio-only canonical=`run://worldsim_v7/WS-V7-P6-OPPORTUNITY-INVARIANT-SELECTOR-01/20260902T170000Z__opportunity-invariant-s70601-r1`。
fixed opportunity shift=`0`，但 nuScenes-test repair AUROC=`.60728`，低于 P4 `.64908` 与 floor `.62908`；因此
`V7-F12` 在 external read 前关闭 ratio-normalization family，fresh AV2 quality 仍未读。

顶会迁移方向改为 CVPR 2023 DGLSS 的 source-only sparsity augmentation + invariant consistency：保留 raw evidence amount，
训练时构造固定下采样机会视图并约束 repair score 一致。P6-C 必须在读取 fresh AV2 前独立冻结 model/seed/loss/gates；
不得根据 ratio candidate 或未来 fresh 结果扫描 consistency weight。

P6-C 已冻结为原始 13D validity inputs + `.5x/.75x` joint opportunity subsampling，BCE(original+augmented)+weight-1
probability consistency，hidden32/seed70602/80 epochs。fresh 前两门固定为 nuScenes AUROC non-inferiority `.02` 与
intervention-score shift 相对 P4 `<=.70x`；通过后外域仍沿用 coverage/false-repair/Chamfer/Wasserstein 四门，不调阈值。

P6-C fit canonical=`run://worldsim_v7/WS-V7-P6C-SPARSITY-CONSISTENT-SELECTOR-01/20260902T173000Z__sparsity-consistent-s70602-r1`；
nuScenes AUROC=`.63239>=.62908`，candidate/P4 intervention shift=`.019526/.183026`、ratio=`.10668`，2/2 通过。
model/standardizer/threshold 已冻结；fresh AV2 仍未编译/打分，external 四门保持未读。

### 主域

- 训练、模型选择和校准：仅 nuScenes；
- 测试：Argoverse 2 Sensor Dataset；
- Waymo Open Dataset 为资源允许时的第二外测，不阻断 V7。

### AV2 adapter 允许做

- 坐标系转换；
- 单位统一；
- 类别映射；
- timestamp 对齐与 pose 插值；
- Actor-local frame；
- sensor opportunity normalization；
- SceneIR 字段适配。

### 禁止

- AV2 fine-tuning；
- AV2 post-hoc calibration；
- 看 AV2 quality 后改 threshold；
- 删除失败场景；
- 用 AV2 validation 选择模型；
- 因类别差异重新定义 target。

### 推荐规模

```text
AV2 main quantitative: 20 logs
AV2 qualitative:       10 logs
Total:                  30 logs
```

场景在结果读取前按 metadata 冻结。

### 零样本主表

报告：

- C1 physical metrics；
- C2 artifact/hazard disentanglement；
- C3 density、Brier、calibration、multi-horizon；
- runtime；
- relative degradation from nuScenes to AV2。

### 如果零样本失败

只允许判断失败层级：

```text
sensor opportunity shift
coordinate/interface mismatch
representation shift
density calibration shift
Actor dynamics shift
```

V7 主实验不允许通过 AV2 fine-tuning 救结果；可以在 supplement 增加少量-shot adaptation，但必须与零样本结果分开。

---

## V7-P7：理论边界与可解释性

### P7-A 执行冻结（2026-09-02）

- 先解释 P4 frozen selector：21-point risk--coverage/geometry/hazard envelope、固定 threshold 标记、跨域 score shift，
  以及 factorized validity head 的 Integrated Gradients；不重训、不重校准、不从 AV2 曲线重选 operating point。
- exact proposition：factorized repair/hazard heads 的输入计算图不连通，故 cross-input derivative=`0`；IG 仅解释各自
  合法输入内的模型敏感度，不主张 causality。
- P5 C3 integration 暂缓到 physical Actor 与 V6.7 trajectory-density row 建立充分、scene-disjoint 对齐之后；当前少量
  overlap 不足以把轻量重训解释成可靠性改善。

### P7-A 结果与边界（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P7-INTERPRETABLE-SAFETY-ENVELOPE-01/20260902T163000Z__safety-envelope-s0-r1`。
在不训练/重校准/适配的条件下，P7 保留 P4 empirical zero-shot operating point 和 factorized exact-zero cross-input
derivative，但暴露 validity head 的 sensor-opportunity shortcut：calibration→AV2 score Wasserstein=`.2170`、KS=`.7041`，
AV2 median score=`.999992`；`observation_frame_count` 的 IG attribution 从 nuScenes test `18.25%` 增到 AV2 `49.53%`。

因此 `V7-F11` 把主张边界收窄为“nuScenes-trained empirical AV2 repair-or-abstain transfer”，不得称 domain-invariant
selection 或外域 formal risk guarantee。恢复只允许在 nuScenes development 中把 raw observation opportunity 替换为
dimensionless density/support 或施加 sparsity invariance；已消费 30 AV2 logs 不允许参与后验 feature/threshold 选择。
任何恢复后的 external claim 必须使用 metadata-frozen 的新 AV2 cohort 或 Waymo。

### 命题 1：危险保真

若 physical compiler 不改变 Actor identity、trajectory 和尺寸：

\[
\mathcal R(A_i)
=
(ID_i,X_{i,0:H},D_i,\widetilde S_i)
\]

则任何只依赖 \(ID_i,X_{i,0:H},D_i,\tau\) 的危险函数保持：

\[
h(\mathcal R(A_i),\tau)=h(A_i,\tau)
\]

### 命题 2：确定性编译幂等性

\[
\mathcal R(\mathcal R(W))=\mathcal R(W)
\]

允许固定数值容差。

### 命题 3：可靠性单调性

\[
b_1\le b_2
\Rightarrow
R(\tau,H,b_1)\le R(\tau,H,b_2)
\]

\[
H_1\le H_2
\Rightarrow
R(\tau,H_1,b)\ge R(\tau,H_2,b)
\]

### 命题 4：几何误差与任务代价边界

若任务代价关于 surface 和 residual distribution 为 Lipschitz：

\[
|\mathbb E_{\hat P}C-\mathbb E_{P^*}C|
\le
L_eW_1(\hat P,P^*)
+
L_sd(\hat S,S^*)
\]

理论部分只解释接口和误差传播，不声称现实道路 safety guarantee。

### 可解释性图

- ray evidence → surface action；
- validity latent vs hazard latent；
- trajectory normal projection；
- conditional cost density；
- reliability surface；
-跨数据集失效分解。

### P7-B geometry-to-cost sensitivity freeze（2026-09-02）

对 `C(d)=max_j |n_j^Te_j|/max(d_j,epsilon)` 使用 max 1-Lipschitz 与 clipped reciprocal 推导
`|Delta C| <= max_j a_j|delta|/(m_jm'_j) <= a_max|delta|/epsilon^2`。固定在 575,596 retained P109 rows 与
P5 test strata 上评估 `±.05/±.10/±.20m`、`epsilon=.05m` 的 shift/bound/tightness/clearance crossing；无训练、
校准、阈值或新传感器读取。只解释误差传播，不能称扰动概率或 safety certificate。

P7-B r1 的 FP32 maximum `shift-bound=3.8147e-6` 超过 frozen `1e-6`，登记 `V7-F14`；P5 strata 为 0 violation。
只把 algebra compute 改 FP64，deltas/floor/tolerance/groups/rows 均不变后 r2，不放宽判据。

P7-B r2 canonical=`run://worldsim_v7/WS-V7-P7B-GEOMETRY-COST-SENSITIVITY-01/20260902T191500Z__geometry-cost-sensitivity-s0-r2`：
575,596 rows×6 shifts 为 0 violation，max overage=`1.42e-14`。`-.20m` full-source mean/q99 cost shift=
`.02922/.14501`，仅 `.8409%` sign crossing；P5 selected mean shift=`.000847` vs abstained `.004560` 且 0 crossing。
结合 P5-B selected motion error 更大，结论固定为 geometry sensitivity、repairability、motion uncertainty 三轴分离。

CVPR draft 已同步 P7-B theorem/result/limitation，official template=`7 pages`；page 7 仅 references continuation，无 float/
clipping。P6-C fresh external 前不根据该解析 audit 改 selector 或 threshold。

---

## V7-P8：Final exact-once evaluation

### P8-A fresh nuScenes freeze（2026-09-02）

- 排除 P4 train/calibration/test 后剩余 106 个本地可用 trainval scenes；只按 official scene index 与文件可用性排序，
  用 `round(linspace(0,105,20))` 冻结 20 scenes，冻结前 quality unread。
- frozen candidate=P6-C source sparsity-consistent selector；baseline/hazard=P4 factorized model；standardizer、threshold、
  compiler 与 Actor policy 均不变。
- one formal read；同 rows 报告 repair/hazard、coverage、false-repair、selective Chamfer 与 score shift；不换 scene、
  不 refit/recalibrate/调阈值。
- core gates：candidate repair AUROC 不低于 P4 `-.02`、coverage `>=.10`、false-repair 低于 always-repair failure、
  selective Chamfer 不差于 clean query。失败即保留 negative result。
- AV2 fresh 20-log 部分继续由 P6-C frozen external runner等待单实例下载完成；P8-A 结果不得改变 AV2 candidate/protocol。

### P8-A exact-once result（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P8A-FRESH-NUSCENES-EXACT-ONCE-01/20260902T200000Z__fresh-nuscenes-final-s0-r1`。
20/20 scenes、123 Actors、one read、0 replacement/update。P6-C/P4 repair AUROC=`.747253/.782280`，退化
`.035027 > .02`，因此 3/4 gates、verdict rejected（`V7-F15`）。P6-C coverage=`.26829`、false-repair=`.02439`、
selective/query Chamfer=`.196891/.234408m` 的 operating-point support 保留，但 descriptive AURC=`.126429` vs P4
`.105633` 也确认 global ranking 较弱。P6-C 不晋升；P4 保持 paper primary selector。禁止 recovery fit/threshold/scene
replacement；fresh AV2 formal read 仍按既定 frozen protocol 执行并独立报告。

CVPR main 已同步 P8-A two-row table、negative recovery、AURC boundary 与 P4 retention；official-template=`7 pages`，
pages 4/6/7 visually valid。该 paper update 不新增 metric gate/model/cohort 或第二次 test read。

所有 architecture、threshold、surface operation、density model、calibration 和 adapter 冻结后，建立：

```text
nuScenes final test: >=20 fresh scenes
AV2 zero-shot test: >=20 frozen logs
```

要求：

- distinct logs/session；
- metadata-only selection；
- quality unread；
- exact-once；
- 不换场景救结果；
- resource failure 与 scientific failure 分离。

Final test 前不再设计新模型。

---

## V7-P9：最小下游应用

### P9 composed-authority freeze（2026-09-02）

为避免把物理 repair 冒充 planner benefit，四臂改写为可识别的 `2×2` factorial：

```text
B0 query surface + no task authority
B1 HARP-3D surface + no task authority
B2 query surface + frozen P346 task authority
B3 HARP-3D surface + frozen P346 task authority
```

固定 P5 test exact identities 的 retained P109 rows、6-query/4-horizon lattice、P346 heldout 3.0s、authority set size 2、
middle frozen cost ceiling、requested reliability .90 与 heldout task conditions。无训练/calibration/threshold/budget sweep、
无新 sensor read、无 critic/RL/closed-loop。核心只检验 physical CD、authorized visited-cost/risk、coverage 与 denominator/
Actor/hazard retention；physical branches 的 action result 相同是预期 non-interference，不是无效实验。

RL 不再是主线，也不阻断论文。

下游只回答：

> V7 编译后的物理世界和 reliability surface，是否能帮助固定候选轨迹选择器避开不可靠访问状态，同时保留危险但合法 Actor？

### 四臂

```text
B0 Naive reconstruction
B1 q0-filtered world
B2 V6.7 inward-ray world
B3 V7 HARP-3D + task reliability
```

### 固定条件

- 同一 Actor IDs；
- 同一 Actor trajectories；
- 同一 hazard distribution；
- 同一 action lattice；
- 同一 planner / scoring budget；
- 同一 case denominator。

### 指标

- actual visited-state cost；
- collision/TTC proxy；
- progress；
- stuck；
- hazard Actor retention；
- abstention；
- artifact exploitation proxy。

### 篇幅

主文仅 0.4–0.6 页和一张小表。

如果可低成本完成小型 RL demo，可放 supplement；不为 RL 新开大规模模型搜索。

---

# 6. 论文交付计划

## 6.1 V7 必须持续产出 CVPR/ICCV 格式论文初稿

从 P0 开始保持 `paper/main.tex` 可编译。

每个里程碑完成后：

1. 将数值写入 `results/results_macros.tex`；
2. 更新对应主表；
3. 更新一张图或 qualitative panel；
4. 更新 Method/Experiments 对应段落；
5. 编译 anonymous PDF；
6. 不要求每次润色全文。

## 6.2 论文阶段性交付

### Draft 0：结构骨架

- 完整章节；
- 标题、摘要、贡献；
- Figure/Table 占位；
- V7 数值允许 TBD。

### Draft 1：C1 完整

- C1 method；
- 物理指标；
- qualitative；
- Figure 1/3；
- Table 1。

### Draft 2：C2+C3 完整

- paired disentanglement；
- density / joint-H；
- Table 2/3；
- zero-shot 初步结果。

### Draft 3：实验完整

- final exact-once；
- AV2；
- minimal downstream；
- limitations；
- supplement failure ledger。

### arXiv v1

- 非匿名；
- 完整 8 页风格主文；
- supplement；
- 视频/项目页可后补。

### Conference submission

- 匿名；
- 使用届时正式 CVPR/ICCV author kit；
- 保持与 arXiv 的方法和主结果一致；
- 不在审稿期继续通过新 test 调主方法。

---

# 7. 主图与主表

## Figure 1：Teaser

四象限并列：

```text
valid-safe
valid-hazard
artifact-safe
artifact-hazard
```

展示：

- RGB；
- 3D surface；
- ray evidence；
- 修复动作；
- 修复后结果；
- 危险 Actor 保留。

## Figure 2：总架构

```text
Physical Compiler
→ Validity/Hazard Factorization
→ Actor Reliability
→ Analytic Task Query
→ Cost Density
→ Reliability Surface
```

## Figure 3：物理几何机制

展示：

- Actor canonical frame；
- motion compensation；
- FREE ray；
- hit endpoint；
- inward support；
- zero-level surface；
- KEEP / PROJECT / COMPLETE / UNKNOWN。

## Figure 4：特征解耦

同一 Actor 的两组配对：

- safe → cut-in；
- clean → ghost。

展示：

- artifact score；
- hazard score；
- latent distance；
- cross-probe。

## Figure 5：连续密度和跨数据集

展示：

- conditional cost density；
- CDF；
- joint-H reliability；
- nuScenes→AV2 zero-shot degradation。

## Table 1：物理与危险保真

- free-space violation；
- surface error；
- depth consistency；
- temporal jitter；
- ghost count；
- Actor retention；
- TTC / trajectory shift。

## Table 2：Validity–Hazard 解耦

- artifact AUROC/AUPRC；
- hazard AUROC/AUPRC；
- paired invariance；
- clean-hazard false artifact；
- cross-probe leakage。

## Table 3：连续可靠性与零样本

- Spearman；
- NLL/Brier；
- calibration；
- joint-H Brier；
- runtime；
- nuScenes；
- AV2 zero-shot。

## Table 4：最小下游

- actual visited-state cost；
- collision/TTC proxy；
- progress；
- hazard retention；
- abstention。

---

# 8. CVPR/ICCV 主文篇幅分配

8 页正文建议：

| 模块 | 比例 |
|---|---:|
| C1 物理几何编译 | 32% |
| C2 有效性—危险性解耦 | 18% |
| C3 连续代价密度与多时域可靠性 | 20% |
| 跨数据集实验 | 22% |
| 下游、限制、结论 | 8% |

\[
\boxed{C1+C2+C3\approx70\%}
\]

---

# 9. 研究纪律与资源合同

## 9.1 简化原则

每个阶段最多：

- 一个主方法；
- 一个结构性 fallback；
- 2–4 个真正决定结论的指标。

## 9.2 禁止

```text
hash / checksum / fingerprint
repo-wide regression matrix
大量 smoke tests
seed sweep
threshold sweep
mixture-count sweep
hidden-size sweep
为了过门更换 final cohort
复活已关闭的方法家族
用 hazard score 删除 Actor
把 opacity 当 occupancy
无限追加 P 编号
```

## 9.3 允许直接修复

- 路径和运行入口；
- archive locator；
- scene-ready pipeline；
- CPU/GPU overlap；
- chunked surface fusion；
- normal numerical stabilization；
- LaTeX 编译和表格宏更新。

## 9.4 资源

默认：

```text
1 × RTX 3090 24GB
```

优先：

- Actor/scene chunking；
- mixed precision；
- gradient accumulation；
- activation checkpointing。

禁止为了显存：

- 降空间分辨率；
- 缩 ROI；
- 缩时域；
- 丢危险 Actor；
- 只保留容易场景。

只有 faithful C1 配置经 chunking 后仍超过 24GB，才判定 `blocked_resource` 并申请多卡。

---

# 10. 六周执行节奏

| 周次 | Research | Paper |
|---|---|---|
| Week 1 | P0 + 四象限图谱 | Draft 0、Introduction、teaser storyboard |
| Week 2 | Actor canonical surface、ray-SDF | C1 Method、Figure 3、Table 1 初版 |
| Week 3 | PROJECT/COMPLETE、C2 paired disentanglement | Figure 1/4、Table 2 |
| Week 4 | C3 整合、AV2 adapter、zero-shot | Figure 5、Table 3 |
| Week 5 | 理论、final exact-once、最小下游 | Table 4、Limitations、Supplement |
| Week 6 | 收敛实验、8 页压缩、arXiv | Draft 3、arXiv PDF、匿名版本 |

---

# 11. 版本停止条件

V7 不以所有实验都成功为结束条件。

当以下四项完成后，停止方法扩展：

1. C1 有明确三维物理硬证据和 qualitative；
2. C2 有 paired counterfactual 解耦证据；
3. C3 有 nuScenes→AV2 零样本结果；
4. final exact-once 和最小下游完成。

之后仅允许：

- 论文写作；
- 图表重绘；
- 代码收敛；
- supplement；
- 合法的 bug 修复。

禁止因某个表格不够漂亮继续增加方法分支。

---

# 12. V7 最终交付物

## 研究资产

- HARP-3D physical SceneIR；
- Actor canonical physical surface；
- validity/hazard factorized representation；
- continuous task-cost density；
- joint multi-horizon reliability surface；
- nuScenes→AV2 zero-shot benchmark；
- exact-once final cohort；
- minimal downstream demo。

## 论文资产

- `paper/main.tex`；
- `paper/main.pdf`；
- `paper/supplement.tex`；
- `paper/supplement.pdf`；
- 主图 1–5；
- 主表 1–4；
- `results_macros.tex`；
- arXiv 非匿名版本；
- CVPR/ICCV 匿名版本；
- 失败账本精简版 appendix；
- 项目页与视频素材清单。

---

# 13. 最终决策

V7 的重点不再是：

```text
继续扩展 authority 的 H / q / k / task 条件轴
```

而是：

```text
把 V6.7 已经验证的可靠性链，建立在真正自洽、危险保真的 3D 物理世界上；
通过 paired disentanglement 和跨数据集零样本，形成视觉顶会可以直接理解的硬证据。
```

最终论文因果链应当是：

\[
\boxed{
\text{Physical Artifact Repair}
\rightarrow
\text{Hazard Preservation}
\rightarrow
\text{Task-Conditioned Reliability}
\rightarrow
\text{Minimal Downstream Utility}
}
\]

而不是一个由数百个实验编号组成的 autoresearch diary。
