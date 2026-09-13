# WorldSim V8.1 研究计划

## Sparse-View × Low-Texture Failure Discovery for Driving Reconstruction

**中文题目：自动驾驶稀疏视角 × 低纹理三维/四维重建失效发现与证据稀缺图谱**\
**版本：V8.1**\
**日期：2026-09-13**\
**阶段定位：科学发现 / badcase mining / failure atlas；不是方法开发阶段**

---

# 0. V8.1 的战略定位

V7.4 已以 `CLOSED_WITHOUT_VALIDATED_MAIN_METHOD` 收尾：没有形成经验证的论文主方法；旧 ordered-chain 实现、ownership-stable 立项依据以及围绕法向漂移继续造新机制的路线均已关闭。V8.1 **不重开 V7.4 已关闭 family**，也不把 V7.4 的 first-return surface dynamics 换名继续做。

V8.1 调整研究问题，但仍留在 WorldSim 的上游核心环节：

```text
Sparse Driving Observations
        ↓
3D / 4D Reconstruction
        ↓
Reusable Scene Geometry / Gaussian Scene
        ↓
Simulation / Novel View / Physical Query
```

本轮不再预设“应该发明哪种新 representation / operator”，而严格遵循最新 Auto Research Scaling Law：

> **failure discovery → 简单强控制 → 找到真正技术缺口 → V8.2 再立项方法。**

V8.1 的主要目标不是把某个 SOTA 打低，而是回答：

> **当视觉纹理、跨视角几何证据、外部几何锚点和 foundation prior 同时变弱时，2026 年自动驾驶三维/四维重建方法究竟在哪里、以什么形式失效？**

当前论文 readiness **不因 failure analysis 自动上升**，在没有 V8.2 正方法前仍按约 **3/6 Weak Reject** 看待。

---

# 1. 中心研究问题

不再笼统写成：

> Sparse-view + Low-texture 很难。

而定义一个更严格的 **证据稀缺区（Evidence Scarcity Regime）**：

\[
\boxed{
\text{弱视觉纹理}
\cap
\text{低跨视角重叠 / 低视差}
\cap
\text{外部几何锚点缺失或偏置}
\cap
\text{几何 foundation prior 不可靠}
}
\]

V8.1 首先研究共同的前两个轴：

1. **纹理证据（Texture Evidence）**
2. **多视角几何证据（Cross-view Geometric Evidence）**

再对不同方法研究它们各自依赖的额外证据：

3. **LiDAR 几何锚点（LiDAR Anchor）**：重点对应 DriveMVS；
4. **Foundation Geometry Prior**：重点对应 VGGD / VGGT 系方法；
5. **Ambiguity Localization / Structural Completion**：重点对应 FocusGS；
6. **Temporal / Dynamic Evidence**：作为 DGGT、PointForward、LGS 的扩展轴，不作为 V8.1 第一主轴。

---

# 2. 截至 2026-09-13 的 SOTA 审计边界

必须严格区分：

- **作者明确写出的 failure / problem；**
- **我们提出、尚待验证的 attack hypothesis。**

不能把后者提前写成论文事实。

## 2.1 第一优先级四个方法

| 方法 | 状态 | 作者明确解决/暴露的内容 | V8.1 待验证攻击假说 | 优先级 |
|---|---|---|---|---:|
| **DVGT** | CVPR 2026 | 无位姿多相机/多时序到度量稠密几何；作者在**伪真值生产**中明确暴露低纹理大平面、特殊平面、曝光、高速模糊、LiDAR 稀疏/集中导致错误 | 这些伪真值 failure signature 是否也会转移到 **DVGT inference geometry**；低纹理 × 低重叠是否出现局部平面弯折、深度漂移、跨视图不一致 | ★★★★★ |
| **DriveMVS** | CVPR 2026 | 用 sparse LiDAR 同时作为 cost-volume 硬几何锚点和软特征引导，结合时序 MVS 消除歧义 | **视觉最弱的位置恰好没有 LiDAR prompt** 时，是否出现显著 geometry collapse；是局部 hole，还是 prior 可跨区域补全 | ★★★★★ |
| **FocusGS** | ECCV 2026 | 从全局 densification 转向 geometric ambiguity manifold + targeted structure completion | 极弱视觉证据下，**ambiguity localization 本身是否漏检/错检**；以及正确定位后 completion 是否会在证据不足时 confident hallucination | ★★★★★ |
| **VGGD** | 2026-08 arXiv | 用 VGGT geometry prior 解决 single-frame surround-view 极低 overlap 的几何不稳定；改善 weakly observed region | 当 **raw VGGT prior 自身在目标区域错误** 时，VGGD 能否纠错，还是把 prior bias 传播/放大到 Gaussian geometry | ★★★★★ |

### 一个非常重要的 claim 修正

DVGT 论文明确展示的是**训练伪标签生成流程**的 failure patterns，不等价于“DVGT 模型推理在这些场景一定失败”。V8.1 必须专门检验：

\[
\text{Pseudo-label failure signature}
\stackrel{?}{\Longrightarrow}
\text{Trained model inference failure}
\]

不能把二者混写。

## 2.2 第二优先级方法

| 方法 | 状态 | V8.1 角色 |
|---|---|---|
| **DGGT** | CVPR 2026 | 开源 4D feed-forward 对照；重点观察 sparse input 下 ghosting / disocclusion / diffusion refinement 是否掩盖几何错误 |
| **PointForward** | 2026-05 arXiv | 检查 world-space 3D query 在极低 overlap 下是否真的比 pixel-aligned representation 更能稳定聚合弱证据 |
| **LGS** | 2026-08 arXiv | 检查“应该 prune / add primitive”的结构策略在 evidence 极弱或偏置时是否误增生、误裁剪 |
| **ReconDrive** | 2026-03 arXiv | foundation-based feed-forward 4DGS 对照；更偏 photometric / dynamic fidelity，非主战场 |
| **P2GS** | CVPR 2026 | 主要解决曝光/HDR一致性；作为 photometric confound control，而不是低纹理 geometry 主对象 |

---

# 3. 可执行性分级：不要为了凑 SOTA 自己重实现论文

V8.1 的结论必须区分**可复现实验**和**文献观察**。

截至 2026-09-13 的公开状态：

## Tier A：可直接进入统一实验

### DVGT

- 官方代码、推理、训练/评测流程和预训练权重已公开；
- 支持任意数量/顺序视角和最长固定帧输入；
- 作为 V8.1 **第一主模型**。

### DGGT

- 官方代码、训练代码、推理代码及 Waymo / nuScenes / AV2 支持已公开；
- 作为 4D / dynamic 扩展模型；
- 不要求第一阶段把 diffusion refinement 当 geometry GT。

### VGGT（机制控制）

不是 V8.1 的 2026 目标 SOTA，但必须作为 VGGD 的 foundation-prior probe：

> raw VGGT 自己在 badcase ROI 上到底错没错？

## Tier B：代码发布通过 Gate 后才进入主量化表

### DriveMVS

论文页面声明 code，但当前官方 GitHub 检索到的仓库仍只有极少 README 内容。执行时重新确认；若完整代码/权重仍不可用：

- 不自行重实现然后称“DriveMVS failure”；
- 只保留论文级 hypothesis；
- V8.1 主表用可执行 MVS / LiDAR-prompt proxy 做**机制控制**，明确不是 DriveMVS 本体。

### FocusGS

项目页标注 Code，但当前 Code 链接仍指向 placeholder/404。若执行时仍未正式开放：

- 不把作者项目图当成我们发现的 failure；
- 不自己复刻 ambiguity manifold 后宣称攻击 FocusGS；
- 保留其 failure hypothesis，等待代码后补正式审计。

### VGGD

当前为 2026-08 arXiv；若没有官方代码/权重：

- raw VGGT prior audit 可以先做；
- “VGGD 会继承 prior 错误”只能作为 hypothesis；
- 不能用另一个 decoder 代替 VGGD 后做方法级结论。

### PointForward / LGS / ReconDrive

只有官方可运行 checkpoint + eval contract 满足时才进入主表；否则放到 Secondary / Literature Boundary。

---

# 4. V8.1 的主科学假设

本轮不预注册某个方法一定失败，而预注册**可证伪的 failure hypotheses**。

## H1：Texture × Overlap 非加性退化

单独低纹理或单独低 overlap 未必致命，但二者联合时 geometry error 可能显著高于简单叠加：

\[
I_{T\times O}
=
(E_{11}-E_{10})-(E_{01}-E_{00}).
\]

其中：

- \(T=1\)：低纹理；
- \(O=1\)：低 overlap / 低 parallax；
- \(E\)：几何误差。

V8.1 要检验 \(I_{T\times O}\) 是否稳定为正，而不是只挑几张失败图。

## H2：LiDAR Prompt Hole

对于依赖 sparse LiDAR anchor 的方法：

> 当视觉歧义最大的区域恰好没有 prompt，而 held-out LiDAR 证明那里确有结构时，性能是否出现局部突变？

重点不是简单“LiDAR 越少越差”，而是比较：

```text
相同 LiDAR 数量
但点落在 ambiguity ROI 内
vs
点全部落在 ambiguity ROI 外
```

检验**空间覆盖**而不是只检验点数。

## H3：Ambiguity Localization Failure

对于 FocusGS：

> 真正 geometry error region 是否一定落在它的 geometric ambiguity manifold 内？

需要区分两类 failure：

1. **Localization miss**：错的地方根本没被标为 ambiguous；
2. **Completion miss**：找对区域，但补错结构。

如果代码可用，必须分别评价，不可只看最终 PSNR。

## H4：Foundation Prior Lock-in

对于 VGGD / foundation-based 方法：

> 当 raw geometry foundation prior 在弱证据 ROI 上已经错误时，下游 reconstruction 是纠正、忽略，还是继承这个错误？

V8.1 不把“foundation prior 错”与“VGGD 错”混为一谈，而做三段式：

```text
raw image evidence
      ↓
raw foundation geometry
      ↓
final reconstruction geometry
```

## H5：Geometry Wrong but Rendering Looks Good

低纹理平面天然缺少外观细节，可能出现：

\[
\text{PSNR/LPIPS 看起来正常}
\quad\text{但}\quad
\text{metric geometry 已经显著错误}.
\]

因此 V8.1 必须同时输出 rendering 和 metric geometry，不允许靠单纯 novel-view quality 判断 badcase。

---

# 5. 数据设计：主数据域和独立确认域

## 5.1 Primary：nuScenes

优先原因：

- VGGD 明确以 nuScenes single-frame surround-view 为主要实验；
- FocusGS 项目页展示 nuScenes ego-centric reconstruction；
- DVGT 支持 nuScenes；
- 有多相机 + LiDAR + ego pose，适合构造 texture / overlap / LiDAR coverage 三轴；
- 可以进行静态世界多帧 LiDAR 累积，形成相对独立的 geometry audit reference。

## 5.2 Secondary confirmation：Waymo 或 Argoverse 2

在 V8.1 找到确定 failure pattern 后再确认：

- DVGT / DGGT 均覆盖多数据域；
- 避免把 nuScenes 特定相机 overlap / LiDAR pattern 当成一般规律。

**开发集与最终确认集严格分开。**

Badcase mining、阈值、图例挑选所用 logs 全部标为 `DISCOVERY`；最终确认使用未参与任何选择的新 logs。

---

# 6. Reference Geometry：避免“低纹理 badcase”其实是 GT 错

这是 V8.1 成败的关键。

## 6.1 第一阶段优先静态 ROI

先选：

- 建筑大平面；
- 声屏障 / 墙面；
- 路侧大板；
- 停放大型车辆侧面（若可确认静止）；
- billboard / sign back；
- 路面局部平面（需单独标注地面特殊性）。

第一主表暂不让动态 actor、运动补偿和形变混进低纹理结论。

## 6.2 多帧 LiDAR 累积作为 audit reference

对静态世界：

```text
多帧 LiDAR
  + ego pose
  + deskew / motion compensation
  + occlusion-aware filtering
        ↓
高密度 held-out geometry reference
```

reference 不是绝对真值，仍需记录：

- LiDAR range / incidence；
- 累积帧数；
- 支撑点数；
- 点云空间分布；
- 是否靠近 depth discontinuity；
- 是否可能有动态污染。

## 6.3 DriveMVS 必须做 LiDAR 数据隔离

如果被测方法使用 LiDAR prompt：

\[
L_{input}\cap L_{GT}=\varnothing.
\]

建议按**时间 + 空间**双重隔离：

- `INPUT_PROMPT`：当前帧或明确子采样 LiDAR；
- `HELDOUT_GT`：邻近时间累积后投影到目标时刻，但从未馈入模型；
- 在目标 ROI 中可主动制造 `prompt hole`，但 GT 仍来自 held-out points。

禁止“同一 LiDAR 点既做 prompt 又做 GT”。

---

# 7. Evidence Scarcity Grid：V8.1 的核心 cohort

不要一开始做 4 维 16-cell 大网格。先用最干净的 2×2 主实验：

| Cohort | 纹理 | overlap / parallax | 用途 |
|---|---|---|---|
| C00 | 高 | 高 | easy control |
| C10 | **低** | 高 | isolated low-texture |
| C01 | 高 | **低** | isolated sparse-view |
| C11 | **低** | **低** | 核心 joint badcase |

在 C11 内再分：

- **C11-L+**：LiDAR anchor 覆盖较好；
- **C11-L−**：LiDAR anchor 缺失/集中；
- **C11-P+**：foundation prior 与 independent geometry 一致；
- **C11-P−**：foundation prior 本身错误。

最终最值得攻击的是：

\[
\boxed{
C11\cap L^-\cap P^-
}
\]

但只有数据里真实存在足够样本时才成立，不强行制造论文故事。

---

# 8. 四个轴如何量化

## 8.1 纹理强度（Texture Strength）

不使用单一 Laplacian variance 就宣布“低纹理”。每个 ROI 保存：

1. 局部梯度能量；
2. 局部灰度 / 颜色熵；
3. 可重复局部特征匹配密度；
4. DINO/VGGT feature variation（仅作辅助，不作为定义真值）。

低纹理标签由**模型运行前**的统计分位数确定，例如 bottom quantile，并人工 spot-check。

自然低纹理为主；合成 texture attenuation 只做因果 sanity check，不进入主真实 badcase 表。

## 8.2 Cross-view overlap / parallax

即便被测模型声称 unposed，**审计可以使用 dataset calibration** 衡量输入信息量：

- ROI 在相邻 camera 的可见重叠比例；
- feature-match coverage；
- triangulation angle / expected parallax；
- 相邻 temporal frame 的视角变化。

不要只用“输入 2 views / 4 views”替代真实几何 overlap。

## 8.3 LiDAR anchor quality

至少保存：

- ROI 内 projected point density；
- 最近 prompt 点距离；
- 2D/3D spatial variance；
- points 是否只集中在一角；
- depth distribution span。

DVGT 论文已经指出**LiDAR 极稀疏或空间集中会让其 pseudo-label alignment ill-conditioned**，所以 spatial dispersion 必须单独量化。

## 8.4 Foundation prior reliability

用 raw VGGT / 对应 foundation 输出，在**独立 held-out LiDAR reference**上评价：

- depth / point error；
- scale drift；
- local plane residual；
- normal deviation。

定义 `P−` 时必须说明：

> 这是 raw foundation geometry 在该 ROI 上与独立 reference 不一致，**不是在声称整个 foundation 模型失败**。

---

# 9. 统一输出空间：不要被各论文 representation 绑架

不同方法输出：

- point map；
- depth；
- 3D Gaussian；
- 4D Gaussian；
- implicit / renderable scene。

V8.1 主评测统一投到三个公共空间。

## 9.1 Ray / Depth Space

对 held-out LiDAR ray：

- Absolute Depth Error；
- AbsRel；
- RMSE；
- \(\delta<1.25\)；
- 可复用 V7 的 `HIT / EARLY / LATE / MISS` 诊断，但仅作为 geometry audit metric，不把 V8.1 story 拉回 V7.4 first-return 方法线。

## 9.2 Surface / Point Space

静态 ROI：

- point-to-plane residual；
- point-to-point / Chamfer（只作辅助）；
- local normal angle error；
- plane bending / curvature proxy；
- completeness / support coverage。

低纹理大平面尤其要观察：

> **平面是否被模型“弯曲、起伏、分层、推远/拉近”，而不仅是有没有点。**

## 9.3 Image / Render Space

- PSNR；
- SSIM；
- LPIPS；
- error crop。

但明确：

\[
\text{Rendering metric}
\neq
\text{Geometry correctness}.
\]

---

# 10. Failure taxonomy：先定义候选，不预设成立

V8.1 在输出中允许出现以下 failure code，但必须由实际证据触发：

| Code | 候选 failure | 可视化特征 |
|---|---|---|
| F-TEX-PLANE | 低纹理平面深度弯折 / 起伏 | depth heatmap + plane fit residual |
| F-OVL-DRIFT | 低 overlap 下 local scale/depth drift | cross-view depth disagreement |
| F-LAYER | 多层 / duplicated geometry / floaters | side-view point cloud + ray depth |
| F-HOLE | evidence hole 内 coverage collapse | GT support exists but prediction absent |
| F-HALLU | weak evidence 区 hallucinated structure | prediction exists where held-out free-space/reference rejects |
| F-AMBI-MISS | ambiguity detector 漏掉真正错误区域 | error mask vs ambiguity manifold |
| F-AMBI-OVER | ambiguity detector 把确定区域错误标成 ambiguous | ambiguity false-positive visualization |
| F-PRIOR-LOCK | raw foundation prior 错，final geometry 继续错 | prior → final error correspondence |
| F-PRIOR-RECOVER | raw prior 错，但 final model 成功修正 | **goodcase**，帮助划清攻击边界 |
| F-TEMP-GHOST | sparse temporal input 下 ghost/disocclusion | multi-time overlay |
| F-DENSIFY-WRONG | primitive prune/add 在弱证据区误增生/误裁剪 | structure intervention map |

Failure code 只是索引，不是论文贡献。

---

# 11. V8.1 实验阶段

## P0：冻结方法、数据与可执行性

输出：`V81_METHOD_AVAILABILITY.md`

做：

1. 锁定各论文版本、官方 checkpoint、代码 commit；
2. 记录每个方法输入要求；
3. 统一 nuScenes frame / camera / crop / resolution contract；
4. 将方法分成 Tier A / Tier B；
5. 不因为某个方法代码暂未发布阻塞整个 V8.1。

**Stop rule**：没有官方可运行实现的方法，不进入“我们复现失败”的主 claim。

---

## P1：构造 Evidence Scarcity Atlas

输出：

- `v81_roi_registry.jsonl`
- `v81_scene_registry.jsonl`
- `texture_overlap_lidar_prior_stats.parquet`

流程：

```text
nuScenes candidate logs
       ↓
Static ROI mining
       ↓
Texture metrics
       ↓
Overlap / parallax metrics
       ↓
LiDAR coverage / dispersion
       ↓
Held-out reference quality check
       ↓
C00 / C10 / C01 / C11 cohort
```

### 采样原则

- 先按 scene/log 采样，再选 ROI；
- 一个 log 内大量 ROI 不能冒充大量 independent samples；
- semantic category、range、ROI size 尽量匹配；
- 不看模型结果后再定义 low-texture 阈值；
- 单独保存“自然 badcase”和“人为干预 case”。

---

## P2：Tier-A Baseline Inference

第一轮至少跑：

1. **DVGT**；
2. **VGGT**（foundation prior control）；
3. **DGGT**（4D / Gaussian extension，资源允许时）；
4. 一个成熟 driving reconstruction control（只作为 sanity，不要求成为论文目标）。

这一阶段**不训练新方法、不 fine-tune、不修 baseline**。

目的：

> 先看 joint evidence scarcity 是否真的产生稳定、可视、可量化的 failure。

---

## P3：Controlled Interventions

自然 badcase 只能说明相关性；P3 用控制干预判断因果方向。

### P3-A：View sparsification

同一 scene 固定 reference：

- 保留全部视角；
- remove adjacent view；
- temporal stride ↑；
- 保持 image content 不变，只改变几何 evidence。

观察 degradation curve，而不是只看一个 sparse setting。

### P3-B：Texture attenuation（仅诊断）

对高纹理静态平面 ROI 做局部高频衰减 / edge-preserving texture flatten：

- 保留主要轮廓；
- 不修改 camera / pose / GT；
- 不把这种合成图当真实分布 claim。

目的：确定模型错误是否对 texture evidence 有 causal sensitivity。

### P3-C：LiDAR prompt hole（DriveMVS code 可用时）

至少比较：

```text
相同点数 + ROI内均匀覆盖
相同点数 + ROI外集中
ROI内全部 dropout
随机 dropout
```

如果“同点数、不同空间覆盖”差异明显，说明关键变量是 **anchor placement** 而不是纯 LiDAR density。

### P3-D：Foundation prior probe

对 raw VGGT：

- 先定义 prior-good / prior-bad ROI；
- 再观察 foundation-based final model 的修复/继承关系。

禁止先看 VGGD 结果再挑 “VGGT 恰好错” 的例子。

---

## P4：Method-specific Forensics

只有某方法在 P2/P3 出现稳定 failure 后才启动。

### DVGT

问：

1. 大低纹理平面是局部 depth bias，还是整体 metric scale drift？
2. 低 overlap 时空间 attention 是否得到互相矛盾的几何？
3. failure 是否与论文伪标签 failure 类型同源，还是完全不同？

不能仅因作者伪标签过滤了某类样本，就推断 trained model 一定不会/一定会失败。

### DriveMVS

问：

1. cost-volume hard anchor 缺失时，soft cue 是否能补；
2. spatially concentrated prompt 是否造成局部错误传播；
3. temporal decoder 能否从邻帧补回当前 frame prompt hole。

### FocusGS

如果官方 code 可用，必须把最终 failure 分解：

```text
actual geometry error
       ↓ compare
ambiguity manifold
       ↓
localization correct ?
       ↓ yes
completion correct ?
```

输出 ambiguity localization 的 precision/recall 或 error-region AUROC，而不是只报 final PSNR。

### VGGD

如果官方 code 可用：

```text
raw VGGT error
      ↓
Dual-Path / decoder
      ↓
final VGGD error
```

重点找：

- prior error 被修正；
- prior error 被保留；
- prior error 被放大；
- raw prior 正确但 downstream decoder 破坏。

---

# 12. 最重要的可视化产物

V8.1 的价值很大程度上体现在**一眼能看懂的 failure evidence**。所有主图必须使用冻结的 case selection 规则，禁止论文写作阶段只挑最好看的图。

## Figure A：Evidence Scarcity Grid

推荐布局：

```text
                    High overlap              Low overlap
              ┌──────────────────┬──────────────────┐
High texture  │ C00              │ C01              │
              │ input/output/err │ input/output/err │
              ├──────────────────┼──────────────────┤
Low texture   │ C10              │ C11 ★            │
              │ input/output/err │ input/output/err │
              └──────────────────┴──────────────────┘
```

每格固定展示：

1. RGB ROI；
2. held-out geometry reference；
3. predicted depth / point geometry；
4. geometry error map；
5. side-view 3D crop。

这张图用于回答：

> joint failure 到底是否比单因素更明显？

## Figure B：Same-scene Factor Escalation

同一 scene/ROI：

```text
Full evidence
→ sparse views
→ low texture
→ sparse + low texture
→ + LiDAR prompt hole
```

避免不同 scene 之间的语义混淆。

## Figure C：Geometry Wrong, Rendering Fine

同一 case 并排：

- RGB render；
- PSNR/LPIPS；
- predicted depth；
- LiDAR/reference error；
- side-view geometry。

如果 render 近似正确而 geometry 错，这是很强的视觉论据。

## Figure D：LiDAR Prompt Coverage Curve

DriveMVS 可运行时：

横轴不是只用 “LiDAR %”，而同时画：

- density；
- spatial dispersion；
- ROI prompt distance。

展示同点数不同空间布置的差异。

## Figure E：Ambiguity Localization Audit

FocusGS 可运行时：

```text
RGB
| predicted ambiguity manifold
| actual high-error region
| false negative
| completion output
```

## Figure F：Foundation Prior Lock-in / Recovery

VGGD 可运行时：

```text
RGB
→ raw VGGT geometry
→ VGGT error
→ final VGGD geometry
→ final error
```

必须同时展示 **prior-lock badcase** 和 **prior-recovery goodcase**，否则会形成确认偏误。

## Figure G：Badcase Cards

每个正式 badcase 生成一张统一卡片：

- scene/log/frame/camera；
- semantic ROI；
- texture metrics；
- overlap/parallax；
- LiDAR coverage；
- raw prior error；
- 每个模型输出；
- geometry/reference；
- failure code；
- 是否在 independent confirmation 中复现。

最终形成可浏览的 **Failure Atlas**。

---

# 13. 统计与科学判定

V8.1 不是通过百万 pixel / ray 把 p-value 做小。

独立单位优先：

- log；
- scene；
- 或严格定义的独立 ROI cluster。

## 13.1 主比较

### Joint effect

\[
I_{T\times O}
=
(E_{11}-E_{10})-(E_{01}-E_{00}).
\]

按 scene/log 做 paired bootstrap / interval。

### Conditional LiDAR effect

\[
\Delta_L
=
E(C11,L^-)-E(C11,L^+).
\]

### Prior reliability effect

\[
\Delta_P
=
E(C11,P^-)-E(C11,P^+).
\]

这些是分析量，不要求所有模型都用同一输入 modality。

## 13.2 Promotion gate：什么 failure 值得进入 V8.2？

一个 candidate badcase 至少满足：

1. **真实**：不是 preprocessing / calibration / GT artifact；
2. **重复**：多个独立 scene/log 同方向出现；
3. **可控**：通过 view / texture / anchor intervention 至少有一条 causal evidence；
4. **显著**：不是只差 1–2 个 pixel 的视觉 cherry-pick；
5. **机制边界清楚**：知道现有方法依赖的 evidence 在哪里断掉；
6. **有 headroom**：oracle / stronger evidence / easy control 能明显修复，说明不是数据本身不可辨识到完全无解；
7. **可攻击**：V8.2 能提出一个自然解，而不是只能“加更多数据/更大模型”。

建议筛查阈值只作为工程 gate，而不是科学定律：

- matched cohort median geometry degradation 有实质量级，例如 ≥15%；
- ≥70% independent scenes 同方向；
- scene-level interval 不跨 0；
- 至少一组可视化可明确看出 geometry failure；
- simple strong control 未完全解释全部问题。

如果一个 failure 只满足“图很好看”，不进入 V8.2。

---

# 14. Headroom / Oracle 设计

V8.1 发现 badcase 后，优先当天做以下简单控制：

## Oracle 1：增加视角

同场景更多 camera / temporal views 能否修复？

如果能：

> failure 是 evidence scarcity，而不是 reference 错。

## Oracle 2：增加 ROI 内 LiDAR anchor

在不改变网络的前提下，prompt 恢复到 badcase ROI 后是否明显修复？

如果能：

> 证明 “anchor location” 有 headroom。

## Oracle 3：可靠 foundation geometry

用 held-out geometry / high-confidence prior 替代错误 prior 做诊断，不作为部署方法。

如果 final model 立刻恢复：

> prior propagation 是可能机制。

## Oracle 4：per-scene optimization / strong reconstruction

在同一 badcase 上用强优化式 reconstruction 做上界参照：

- 如果优化式方法也失败，说明 observation 本身可能不可辨识；
- 如果它能恢复，说明 feed-forward / prior mechanism 仍有方法空间。

---

# 15. V8.1 不做什么

明确禁止：

1. 不在发现 failure 前设计 V8.2 fancy module；
2. 不调 DVGT / DGGT loss 让 baseline 变好或变差；
3. 不把 synthetic low-texture 当真实-world claim；
4. 不用 rendering PSNR 代替 geometry；
5. 不把作者论文 qualitative failure 当成我们的新证据；
6. 不因为某个方法没开源就自己随意重实现后宣布它失败；
7. 不同时把低纹理、运动模糊、曝光、动态物体、遮挡全部混成一个 hard subset；
8. 不把 low-texture ROI 中 LiDAR GT 自己也稀疏的 case 当确定 failure；
9. 不因为某模型在 C11 差，就立即宣称 foundation prior / Gaussian / MVS 范式无效；
10. 不在 V8.1 开始大训练。

---

# 16. Stop Rules

## Stop A：Sparse × Low-texture 本身没有明显 joint failure

如果强 SOTA 在 C11 并未比 matched controls 明显退化：

> 关闭“泛化 sparse-view × low-texture”作为 V8.2 主问题。

再检查更具体的 L− / P− / ambiguity localization，而不是强行写 badcase。

## Stop B：所有 failure 都能被 trivial control 完全解释

例如只是输入 resize bug、LiDAR alignment、曝光预处理或简单增加 view 即完全解决且无剩余方法空间：

> 不立项新方法。

## Stop C：Reference 不可靠

若 badcase 区域没有可靠 independent geometry reference：

> 只能进入 qualitative candidate pool，不能进入主 scientific claim。

## Stop D：某论文不可运行

代码/权重不可用：

> 降级为 hypothesis / literature boundary；不阻塞其他模型。

## Stop E：找到一个强、稳定、可控的 failure

一旦某 failure 通过 Promotion Gate：

> **V8.1 即可收敛，不需要再凑 10 种 failure。**

冻结 cohort、证据和强控制，进入 V8.2 方法设计。

---

# 17. V8.2 的入口条件，而不是 V8.1 的方法预案

V8.1 结束时只允许产生以下形式的结论：

### 例 A

> 在低纹理 × 低 overlap ROI 中，只要 LiDAR prompt spatially misses 目标区域，LiDAR-prompted MVS 出现稳定局部 geometry collapse；同点数但 ROI 内覆盖可恢复。

那么 V8.2 才研究：

> 如何在缺失 anchor 的 ambiguous region 生成/传播可靠几何。

### 例 B

> FocusGS 的主要问题不是 completion，而是 geometric ambiguity manifold 漏掉某类弱纹理平面。

那么 V8.2 才研究：

> ambiguity localization。

### 例 C

> raw foundation geometry 错误与 final reconstruction 错误高度耦合，而额外视角/LiDAR 能明显纠正。

那么 V8.2 才研究：

> prior 与 observation evidence 的冲突解决机制。

### 例 D

> 多类模型都在 C11 失败，但 strong optimization oracle 能恢复。

这时才有资格研究更通用的：

> evidence-scarcity-aware generative reconstruction。

**不要提前选 A/B/C/D。**

---

# 18. V8.1 最终交付物

## 18.1 文档

1. `WORLDSIM_V8_1_PLAN.md` —— 本计划；
2. `WORLDSIM_V8_1_METHOD_AUDIT.md` —— 论文/代码/权重/输入输出审计；
3. `WORLDSIM_V8_1_FAILURE_ATLAS.md` —— 所有可视化 badcase/goodcase；
4. `WORLDSIM_V8_1_SCIENTIFIC_REPORT.md` —— 统计、强控制、结论边界；
5. `WORLDSIM_V8_1_TO_V8_2_DECISION.md` —— 只允许 `GO_<failure>` / `NO_GO`。

## 18.2 数据索引

```text
scene_registry
roi_registry
method_outputs
reference_geometry
factor_metrics
failure_codes
case_selection
independent_confirmation
```

## 18.3 图

至少产出：

1. Evidence Scarcity Grid；
2. Same-scene Factor Escalation；
3. Geometry-vs-Rendering mismatch；
4. Method-specific mechanism figure（LiDAR / ambiguity / prior 中至少一个）；
5. independent confirmation badcase figure；
6. goodcase boundary figure。

**必须有 goodcase。** 只有 badcase 没有 boundary，很容易变成 cherry-picking。

---

# 19. 当前方法执行优先级

## 第一批：无需等待

1. **DVGT** —— 主 geometry target；
2. **VGGT** —— foundation prior probe；
3. **DGGT** —— 4D / sparse input 扩展；
4. nuScenes Evidence Scarcity Atlas；
5. multi-frame LiDAR audit reference。

## 第二批：代码 Gate 通过后

1. **DriveMVS**；
2. **FocusGS**；
3. **VGGD**。

## 第三批：只在主 failure 已经明确时扩展

1. PointForward；
2. LGS；
3. ReconDrive；
4. P2GS photometric confound。

不要九个方法一起开跑。

---

# 20. V8.1 最希望得到的“一句新知识”形式

V8.1 最终必须得到一句**由实验证据决定**的新知识，而不是预填答案。

理想形式例如：

> **Sparse views are not uniformly hard: modern driving reconstruction fails most severely when weak texture removes photometric correspondence and the remaining geometric anchor misses the same region.**

中文：

> **稀疏视角本身并不会均匀导致重建失败；真正严重的退化集中在视觉对应关系因低纹理变弱、且剩余几何锚点又恰好缺失于同一区域时。**

或者，如果证据指向 foundation prior：

> **Foundation priors reduce sparse-view ambiguity only when the prior itself remains geometrically reliable; under joint evidence scarcity, prior error can become the dominant reconstruction bias.**

但在 V8.1 结果出来前，以上都只是**目标句型，不是当前结论**。

---

# 21. Skeptical Reviewer Gate

在 V8.1 收尾时，用下面六个问题决定是否值得进入 V8.2：

1. **这个 badcase 是真实场景中自然存在，还是我们人为造出来的？**
2. **reference 足够可靠，还是 GT 自己在低纹理/稀疏区域也有问题？**
3. **它在多个独立 scene/log 复现了吗？**
4. **简单增加信息或成熟强控制能否证明存在 headroom？**
5. **它是 2026 SOTA 的共同缺口，还是某一实现 bug？**
6. **即使 V8.2 把它解决，是否足以形成 CVPR/ICCV main-track 的明确方法增量？**

只有这些问题大体通过，才允许从 failure atlas 转向 method development。

---

# 22. 参考工作与当前公开状态

- **DVGT: Driving Visual Geometry Transformer**, CVPR 2026\
  Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Zuo_DVGT_Driving_Visual_Geometry_Transformer_CVPR_2026_paper.html\
  Code: https://github.com/wzzheng/DVGT

- **DriveMVS: LiDAR Prompted Spatio-Temporal Multi-View Stereo for Autonomous Driving**, CVPR 2026\
  Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Sun_LiDAR_Prompted_Spatio-Temporal_Multi-View_Stereo_for_Autonomous_Driving_CVPR_2026_paper.html\
  Repo: https://github.com/Akina2001/DriveMVS

- **FocusGS: Targeted Structure Completion for Sparse-View 3D Reconstruction in Autonomous Driving**, ECCV 2026\
  Project: https://focusgs.github.io/\
  arXiv: https://arxiv.org/abs/2607.04661

- **VGGD: Visual Geometry Foundation-Aware Gaussians for Single-Frame Surround-View Driving Reconstruction**, 2026-08 arXiv\
  arXiv: https://arxiv.org/abs/2608.10682

- **PointForward: Feedforward Driving Reconstruction through Point-Aligned Representations**, 2026-05 arXiv\
  arXiv: https://arxiv.org/abs/2605.11594

- **Learning Gaussian Structure: Intervention-Guided Density Control for Feed-Forward Driving Reconstruction**, 2026-08 arXiv\
  arXiv: https://arxiv.org/abs/2608.11077

- **DGGT: Feedforward 4D Reconstruction of Dynamic Driving Scenes using Unposed Images**, CVPR 2026\
  Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Chen_DGGT_Feedforward_4D_Reconstruction_of_Dynamic_Driving_Scenes_using_Unposed_CVPR_2026_paper.html\
  Code: https://github.com/xiaomi-research/dggt

- **ReconDrive: Fast Feed-Forward 4D Gaussian Splatting for Autonomous Driving Scene Reconstruction**, 2026-03 arXiv\
  arXiv: https://arxiv.org/abs/2603.07552\
  Repo: https://github.com/TuojingAI/ReconDrive

- **P2GS: Physical Prior-guided Gaussian Splatting for Photometrically Consistent Urban Reconstruction**, CVPR 2026\
  Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Shimomura_P2GS_Physical_Prior-guided_Gaussian_Splatting_for_Photometrically_Consistent_Urban_Reconstruction_CVPR_2026_paper.html

---

# 23. 最终执行原则

V8.1 的目标不是证明我们已经知道答案，而是构造一条可信的证据链：

```text
2026 SOTA
   ↓
统一真实数据合同
   ↓
Evidence Scarcity Grid
   ↓
自然 badcase mining
   ↓
controlled intervention
   ↓
strong simple controls / oracle
   ↓
independent confirmation
   ↓
一个值得解决、边界清楚的 failure
   ↓
V8.2 method design
```

如果最后得到的是：

> “DVGT、DriveMVS、FocusGS、VGGD 在我们定义的 joint evidence scarcity 下其实都很稳健。”

这也必须诚实接受，**关闭这条方向，而不是继续制造更极端 perturbation 直到它们失败。**

V8.1 的最高价值，是让 V8.2 的方法不是从想象中长出来，而是从 2026 SOTA 的真实 failure boundary 中被逼出来。
