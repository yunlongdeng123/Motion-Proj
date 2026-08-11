# 面向 WorldSim 的对象感知 3DGS、道路原生修复与可维护编辑资产计划 V3.3

- **版本**：V3.3
- **日期**：2026-08-11
- **项目根目录**：`/root/autodl-tmp/motion_proj`
- **当前硬件**：单卡 NVIDIA GeForce RTX 3090 24 GiB
- **当前历史分支**：`research/worldsim-v3.2-semantic-repair`
- **建议新分支**：`research/worldsim-v3.3-object-maintenance`
- **V3.2 scoped closeout commit**：`f44decacca812fe1b476253b6bdc8aac1869f873`
- **V3.2 canonical R0**：
  - run=`20260810T134658Z__r0-final-integration-s0-r1`
  - 8/8 gates passed
  - regression=`36 passed`
- **V3.3 当前唯一授权任务**：`WS-V33-P0-ROUTE-SOTA-AUDIT-01`
- **本计划替代关系**：
  - V3.2 保留为已完成、不可改写的历史事实；
  - V3.3 只在新的分支与新的 run namespace 中执行；
  - 不删除、不覆盖、不重新解释 V3.2 canonical terminal / checkpoint / registry / package。

---

# 0. V3.3 修订目的

V3.2 已经证明一条单卡 3090 可执行的 WorldSim 资产链可以完整闭环：

```text
V3.1 D2 StreetGS
→ SAM2.1 temporal semantic masks
→ heuristic multi-view Gaussian semantic sidecars
→ depth-guided cross-view generated background
→ NVIDIA Asset Harvester generated actor
→ mixed-precision checkpoint
→ static / actor exact chunk package
```

V3.2 解决了工程闭环，但还没有把以下三个问题做到方法层：

1. **对象归属仍主要依赖 heuristic lifting**
   - 现有 S1 使用 SAM2.1 mask、visibility、depth consistency 和 posterior threshold；
   - RGB Gaussian 的“视觉存在”与“实例归属”没有真正解耦；
   - Gaussian 为了 RGB reconstruction 必须存在，不代表它就应该属于 actor；
   - 这是典型的 label contamination 问题。

2. **背景修复仍不是完整的当前 SOTA**
   - 当前 S2 采用 3DGIC 的 depth-guided cross-view 原则；
   - 未观测区域的二维补全仍为确定性 OpenCV Telea；
   - 当前生成背景可以安全落盘，但并不能代表高质量、多视角一致的 3D inpainting；
   - V3.2 r2 已经证明“把所有二维 unseen completion 写入 3D”会明显伤害 held-out 质量。

3. **编辑结果仍以修改后的完整 checkpoint 为主要状态**
   - 当前已有 exact chunk / actor package；
   - 但每次 remove / background repair / actor replacement 仍缺少“base + delta”的可维护编辑语义；
   - 工业 WorldSim 更需要可回滚、可审计、可组合的编辑层，而不是复制整场景资产。

因此 V3.3 不再继续扩大 V3.2 的工程矩阵，研究问题收缩为：

### Q1：对象感知 Gaussian

> 能否将 RGB opacity 与 instance occupancy 解耦，用 SAM3.1 / SAM2.1 的训练视图 mask 学习一个轻量 object-occupancy Gaussian field，从根本上降低 heuristic 2D→3D label contamination？

### Q2：驾驶场景原生背景修复

> 删除/位移 actor 后，能否优先利用同一驾驶场景中已有的 3D Gaussian road/background patch 完成修复，只在真实三维支持不足时调用生成式 inpainting？

### Q3：可维护编辑资产

> 能否把 remove / background repair / actor replacement 表达为 immutable base 上的 spatial delta layer，实现 erase + insert + generated provenance，而不是产生新的完整 base checkpoint？

### Q4：完整 actor 的输入视角选择

> Asset Harvester 已经跑通后，能否用自动 view selection 选择更互补、更少遮挡的 1/2/4 个真实观测，提高完整 actor 的多视角一致性和边界质量？

---

# 1. V3.2 必须冻结的现状

## 1.1 当前生产 candidate

V3.2 R0 生产链固定为：

```text
S1 extended semantic sidecars
→ S2 generated-background mixed scene
→ S3 generated-actor override
→ mixed checkpoint
→ exact chunk package
```

当前已知事实：

```text
mixed checkpoint bytes = 432,347,490
vs FP32 reduction      = 25.363333%

static assets           = 133
actor assets            = 24

Background rows         = 1,207,060
RigidNodes rows         = 104,704

chunk missing rows      = 0
chunk duplicated rows   = 0

fixed-view source→mixed PSNR
= 68.2993 / 67.2399 / 68.4322 dB
```

**重要解释边界：**

上述约 `67–68 dB` 是：

```text
V3.2 source representation
vs
mixed / exact-reassembled representation
```

的工程保真度，不是最终 WorldSim 对真实 GT 的重建 PSNR，不得在 V3.3 中改写。

## 1.2 当前关键资产

V3.1 D2 FP32 source：

```text
SHA-256
1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c
```

V3.2 S2 generated-background checkpoint：

```text
SHA-256
3d6e13d47291f5b5949ff3adf5598b6e0cffb930c4cbff2200c6e708d82e6e0f
```

V3.2 S3 high-support 2-view actor asset：

```text
SHA-256
b0c1f413e1a462292a1e3396ad45b8a8fc10f87f647e4bc3e1b98a4c8913caf0
```

V3.2 R0 mixed candidate：

```text
SHA-256
6d4e4c489f53bf4e7de3f5c405ec37dc63d3f79155aad5237fe175ce0fcd7e5d
```

V3.2 R0 chunk manifest：

```text
SHA-256
af7b402e0b171b11f8c22e4123002f4f844db746ea72f53b77c3de878bf0947d
```

所有上述资产在 V3.3 中默认：

```text
immutable / read-only
```

## 1.3 当前 V3.2 已知局限

### S1

- SAM2.1 Hiera Large；
- 398 个 train-only masks：
  - accepted=334；
  - rejected=64；
- heuristic Gaussian semantic posterior；
- module sidecar，不修改 source checkpoint。

### S2

- depth-guided cross-view；
- OpenCV Telea 负责未观测二维区域；
- r3 追加 `1,896` 个 `GENERATED_BACKGROUND` Gaussian；
- held-out 平均：
  - PSNR delta=`-0.022958 dB`
  - SSIM delta=`-0.000528`
  - LPIPS delta=`+0.000301`
- 通过冻结 safeguard，但不主张 unseen completion accuracy。

### S3

- Asset Harvester 已真实跑通；
- 单卡峰值约 20.1 GiB；
- 2-view 优于当前 1-view 综合结果；
- 目前只对 scene-0230 high-support actor 形成生产 override。

### S4

- NVIDIA Harmonizer 非时序 arm 跑通；
- delete scene 中重新生成了被删除车辆语义；
- 因语义保持门失败，只保留 diagnostic；
- temporal arm 受 gated base 权重授权阻塞。

---

# 2. V3.3 SOTA 事实源与使用方式

V3.3 只允许 coding agent 使用以下三种类型的能力：

```text
A. 官方代码 + 合法权重 → 可运行
B. 论文 + 无可执行官方实现 → inspired implementation，禁止称 reproduction
C. 外部许可/权重阻塞 → audit_only / blocked
```

## 2.1 SAM 3.1

官方：

- Meta `facebookresearch/sam3`
- SAM 3 unified detection / segmentation / tracking
- 2026-03 发布 SAM 3.1 Object Multiplex
- 支持：
  - text prompt
  - exemplar prompt
  - box / mask / point
  - joint multi-object tracking
- 官方当前 prerequisites：
  - Python >= 3.12
  - PyTorch >= 2.7
  - CUDA >= 12.6
- checkpoint 需要 Hugging Face access

V3.3 用法：

```text
首选：SAM3.1 exemplar + box prompt
fallback：现有 SAM2.1 canonical mask
```

SAM3.1 权重如果无授权，不阻塞主计划。

官方：

- https://github.com/facebookresearch/sam3
- https://ai.meta.com/research/sam3/

## 2.2 OP2GS

论文：

**OP2GS: Object-Aware 3D Gaussian Splatting with Dual-Opacity Primitives**
arXiv:2605.20044

关键思想：

```text
visual opacity      σ
instance opacity    σ*
```

RGB branch 继续使用原 opacity；
object-mask branch 使用独立 instance opacity。

V3.3：

> 实现 `OP2GS-inspired instance opacity sidecar`。

除非 P0 审计发现作者公开且可执行官方代码，否则不得称为 OP2GS reproduction。

论文：

- https://arxiv.org/abs/2605.20044

## 2.3 Inpaint360GS

WACV 2026，官方代码已公开，Apache-2.0：

- object-aware Gaussian training；
- 2D segmentation distillation；
- multi-object removal；
- virtual camera context；
- 2D color/depth inpainting；
- 3D inpainting。

官方：

- https://github.com/dfki-av/Inpaint360GS
- https://openaccess.thecvf.com/content/WACV2026/html/Wang_Inpaint360GS_Efficient_Object-Aware_3D_Inpainting_via_Gaussian_Splatting_for_360deg_WACV_2026_paper.html

V3.3：

```text
strong external baseline
```

不允许为了适配当前 StreetGS 偷换相机、split 或 heldout。

## 2.4 GS-RoadPatching

SIGGRAPH Asia 2025：

**GS-RoadPatching: Inpainting Gaussians via 3D Searching and Placing for Driving Scenes**

关键思想：

```text
driving scene 中存在大量结构重复
→ feature-embedded Gaussian patches
→ BEV / 3D patch search
→ substitution
→ fusion
```

当前公开 GitHub 主体为 project-page 资源；P0 必须重新检查是否已经出现可执行源码。

若仍无 runnable official implementation：

```text
实现 RoadPatch-Lite
```

并明确称：

```text
GS-RoadPatching-inspired
```

官方/论文：

- https://arxiv.org/abs/2509.19937
- https://github.com/Shanzhaguoo/GS-RoadPatching

## 2.5 3D-GIMP

2026-07：

**3D-GIMP: When 3D Gaussian Inpainting Meets PatchMatch**

核心：

```text
single reference-view generative inpaint
→ 3D-aware PatchMatch
→ texture propagation
→ 3D Gaussian reconstruction
```

论文：

- https://arxiv.org/abs/2607.20789

P0 若仍无官方 runnable code：

```text
audit_only
```

不得从论文手写完整系统作为 V3.3 主线。

## 2.6 FocusGS

2026-07：

**FocusGS: Spatial Delta Layers for Local Repair and Deterministic Editing of Trained 3D Gaussian Assets**

核心：

```text
immutable base
+
spatial delta

repair = additive delta

editing =
erase old carrier
+
insert new content
```

V3.3 不需要复现全部 FocusGS；
只吸收：

```text
spatial delta layer
erase-insert factorization
```

作为工程/模型资产接口。

论文：

- https://arxiv.org/abs/2607.28834

## 2.7 Asset Harvester

官方：

- https://github.com/NVIDIA/asset-harvester

已在 V3.2 跑通。

V3.3 不换模型，改：

```text
manual 1/2 view
→ automatic 1/2/4-view selection
```

## 2.8 R3D2

官方代码：

- Zenseact
- CVPR Workshop on Autonomous Driving 2026
- Apache-2.0
- one-step diffusion
- 解决 complete 3D asset 插入后的 shadow / lighting integration

官方：

- https://github.com/zenseact/R3D2

V3.3：

```text
P0 审计是否提供可合法获取的 exported pretrained model
```

若没有：

```text
blocked_pretrained_model_unavailable
```

禁止在单张 3090 上从零训练 R3D2 主模型。

## 2.9 GOR-IS

CVPR 2026 Highlight：

**GOR-IS: 3D Gaussian Object Removal in the Intrinsic Space**

解决：

- material/light decomposition；
- object removal；
- lighting consistency；
- non-Lambertian appearance。

V3.3 只作：

```text
audit / optional comparison
```

不阻塞 RoadPatch 主线。

论文：

- https://arxiv.org/abs/2605.00498

## 2.10 LiDAR-EVS

2026：

**LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision**

V3.3 只登记未来接口：

```text
conditional future task
```

主计划先把 RGB/object/background 资产做扎实。

论文：

- https://arxiv.org/abs/2603.14763

---

# 3. V3.3 总体路线

```text
                    V3.2 immutable production baseline
                               │
                               ▼
                  ┌────────────────────────┐
                  │ S1 Object-aware field  │
                  │ SAM3.1 / SAM2 fallback │
                  │ + dual instance opacity│
                  └───────────┬────────────┘
                              │
                  object-aware Gaussian masks
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
       S2 Road/background repair     S3 complete actor asset
       RoadPatch-Lite                Asset Harvester
       + Inpaint360GS baseline       auto 1/2/4 views
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    S4 Spatial Delta Layer
                      erase + insert
                              │
                              ▼
                   S5 semantic-gated render
                   / R3D2 conditional
                              │
                              ▼
                   mixed storage + exact chunk
                              │
                              ▼
                        V3.3 R0
```

---

# 4. 任务注册表

| Task ID | 初始状态 | 目标 | 完成门禁 |
|---|---|---|---|
| `WS-V33-P0-ROUTE-SOTA-AUDIT-01` | pending | 冻结 V3.2、建立新分支、审计 SAM3.1/OP2GS/Inpaint360GS/GS-RoadPatching/3D-GIMP/FocusGS/R3D2 | source/license/weights/hardware 全记录 |
| `WS-V33-S1-OBJECT-AWARE-GS-01` | pending | SAM3.1/SAM2 + instance-opacity Gaussian field | base RGB bitwise 不变；mask field 可复现；优于 heuristic 或透明记录 no-gain |
| `WS-V33-S2-ROADPATCH-INPAINT-01` | pending | RoadPatch-Lite + Inpaint360GS strong baseline | 至少一个真实 delete background 方法过门 |
| `WS-V33-S3-ASSET-VIEWSELECT-01` | pending | Asset Harvester automatic 1/2/4-view selection | 高支持 actor 必做；通过后扩 boundary actor |
| `WS-V33-S4-SPATIAL-DELTA-01` | pending | immutable base + erase/insert delta | exact compose / rollback / package |
| `WS-V33-S5-SEMANTIC-RENDER-01` | pending | semantic-gated Harmonizer；R3D2 conditional | 不允许删除语义重引入 |
| `WS-V33-R0-INTEGRATION-01` | pending | object-aware + background repair + actor + delta + packaging | 全部 canonical hash / gates / single-GPU 资源通过 |
| `WS-V33-F0-LIDAR-EVS-AUDIT-01` | conditional | LiDAR extrapolated view future audit | 不阻塞 R0 |

---

# 5. 场景和实验角色

继续沿用：

```text
development:
scene-0230

confirmation:
scene-0242
scene-0255
```

V3.3 不能重新把三场景当盲测。

规则：

- scene-0230 允许调 threshold / patch size / view selector；
- 0242/0255 只跑冻结后的 S1*/S2*/S3*；
- boundary actor 缺失时继续 `ABSTAIN`；
- 不为补齐表格更换 actor；
- 所有 heldout frame 禁止进入：
  - SAM prompt selection；
  - instance-opacity optimization；
  - patch search candidate selection；
  - Asset Harvester view selection；
  - color fitting；
  - candidate selection。

---

# 6. P0：路线切换与 SOTA 审计

## 6.1 新分支

```text
research/worldsim-v3.3-object-maintenance
```

第一个提交：

```text
docs(worldsim): 启动 V3.3 对象感知与可维护资产路线
```

## 6.2 run namespace

```text
/root/autodl-tmp/runs/worldsim_v33/
```

不得复用：

```text
worldsim_v32 run ID
```

## 6.3 第三方目录

```text
/root/autodl-tmp/third_party/worldsim_v33/
├── sam3/
├── inpaint360gs/
├── gs-roadpatching/
├── 3d-gimp/
├── op2gs/
├── r3d2/
└── optional/
```

## 6.4 审计输出

新增：

```text
configs/worldsim_v33/p0_sources_v1.yaml
docs/WS_V33_P0_SOTA_AUDIT.md
```

每个 source：

```yaml
name:
official_url:
paper_url:
commit:
tree_sha:
license:
license_sha256:
weights:
weights_revision:
weights_sha256:
python:
torch:
cuda:
single_3090:
input_schema:
output_schema:
execution_state:
  executable | audit_only | license_blocked | weights_blocked | source_not_released
```

## 6.5 P0 禁止

- 不下载 >5 GB 权重；
- 不启动训练；
- 不修改 DriveStudio；
- 不修改 V3.2 canonical；
- 不把论文中的 code-available 声明等同于当前仓库实际可执行；
- 不绕过 HF gated license。

---

# 7. S1：对象感知 Gaussian Field

## 7.1 目标

把 V3.2：

```text
2D mask
→ heuristic posterior threshold
→ CORE / SEMANTIC / AMBIGUOUS / NEGATIVE
```

升级为：

```text
2D temporal / concept masks
→ Gaussian instance identity prior
→ learnable instance opacity σ*
→ physically rendered object occupancy mask
```

同时：

```text
RGB Gaussian 参数完全冻结
```

## 7.2 SAM3.1 arm

优先：

```text
SAM3.1
```

prompt：

```text
instance_token
→ best train-view crop exemplar
+
frozen projected 3D box
```

不要只用文本 `"car"`，避免同类车辆混淆。

开发 arm：

```text
P0 = SAM2.1 canonical masks
P1 = SAM3.1 box
P2 = SAM3.1 exemplar + box
P3 = SAM3.1 Object Multiplex (多 actor，条件式)
```

如果 SAM3.1 checkpoint 无合法权限：

```text
P1/P2/P3 = weights_blocked
```

S1 主方法继续用 V3.2 SAM2.1 mask，不能因此阻塞 dual-opacity。

## 7.3 OP2GS-inspired representation

新增 sidecar：

```text
instance_field.npz
```

最小字段：

```text
gaussian_id
base_model
base_index
hard_instance_id
instance_opacity_logit
instance_opacity
source_semantic_score
num_positive_views
num_negative_views
visibility_mass
trainable
provenance
```

RGB 原始：

```text
opacity σ
```

完全不改。

新 object branch：

```text
instance opacity σ*
```

只用于：

```text
render_instance_mask(actor_id)
```

## 7.4 mask rendering

使用原 Gaussian：

- means；
- scales；
- quats；
- camera；
- projection；
- depth ordering。

但 alpha：

```text
alpha_instance_i =
sigmoid(instance_opacity_logit_i)
× projected_gaussian_i
```

对于指定 actor：

```text
hard_instance_id == requested_actor
```

的 Gaussian 进入主 instance branch。

对 V3.2 `SEMANTIC_POSITIVE` 的背景 Gaussian：

允许在一个冻结候选集内：

```text
background → actor candidate
```

但不修改 base model，只修改 semantic sidecar 的 `hard_instance_id_candidate`。

## 7.5 训练范围

第一阶段只做：

```text
scene-0230
high-support
boundary-support
```

trainable Gaussian：

```text
现 actor RigidNodes
+
S1 semantic-positive Background
+
3px/5px mask boundary 投影邻域
```

其他全部冻结。

## 7.6 loss

主 loss：

```text
L_mask =
BCE(rendered_instance_mask, target_mask)

L_dice =
1 - Dice(...)

L_sparse =
mean(instance_opacity for negative candidates)

L_prior =
deviation from V3.2 semantic prior

L_temporal =
same Gaussian's occupancy stability across train views
```

总：

```text
L =
L_mask
+ λdice L_dice
+ λsparse L_sparse
+ λprior L_prior
+ λtemporal L_temporal
```

所有 λ：

先用 100–300 step smoke；
只在 scene-0230 development 冻结一次。

不允许用 heldout 选 λ。

## 7.7 random-object training

若扩到 24 actor：

参考 OP2GS 的 object-level sampling 思路：

```text
每 step 随机选一个可用 actor
→ render 该 actor instance mask
→ update only relevant instance opacity
```

禁止 24 actor × 全视图全量同时渲染造成无意义显存增长。

## 7.8 S1 对照

```text
O0：V3.2 heuristic sidecar
O1：O0 + learnable instance opacity
O2：SAM3.1/SAM2 mask + instance opacity
O3：O2 + background-to-actor candidate reassignment
```

## 7.9 S1 主要指标

只对 heldout：

- mask IoU；
- boundary F1；
- normalized boundary distance；
- temporal identity stability；
- false-positive semantic mass；
- false-negative semantic mass；
- RGB before/after SHA exact；
- checkpoint SHA exact；
- sidecar bytes；
- wall；
- peak VRAM。

RGB source 必须：

```text
bitwise unchanged
```

## 7.10 S1 成功定义

候选必须至少满足：

```text
boundary F1 ↑
或
boundary distance ↓
```

并且：

```text
IoU 不关键退化
false-positive 不关键增加
RGB bitwise exact
```

若无提升：

```text
S1*=V3.2 heuristic
```

仍可继续 S2。

---

# 8. S2：驾驶场景原生 Background Repair

S2 是 V3.3 的核心模型阶段。

目标：

> 优先从同一场景已有真实 3D Gaussian 中找可复用背景，而不是先让 2D 模型凭空 hallucinate。

---

# 9. S2-A：RoadPatch-Lite

## 9.1 定位

明确命名：

```text
GS-RoadPatching-inspired RoadPatch-Lite
```

除非 P0 得到官方 runnable source 并直接执行，否则不得写：

```text
reproduced GS-RoadPatching
```

## 9.2 使用 V3.2 已有能力

直接复用：

- Background Gaussian；
- 133 static chunk inventory；
- camera；
- depth；
- SAM semantic sidecar；
- actor footprint；
- generated provenance；
- original/delete render；
- heldout split。

## 9.3 target hole 3D anchor

由：

```text
S1 object-aware delete mask
+
target-view first-hit depth
+
cross-view observed support
```

得到：

```text
hole footprint in 3D / BEV
```

如果完全没有合法 depth：

```text
ABSTAIN
```

不得使用 Telea depth 或 hallucinated depth 作为 search anchor truth。

## 9.4 static patch index

只使用：

```text
V3.1 / V3.2 native Background
```

作为 patch donor。

默认禁止：

```text
GENERATED_BACKGROUND
```

作为 donor，防止生成误差自我复制。

每个 patch anchor 建特征：

### geometry

```text
mean_z
std_z
local_plane_normal
local_plane_residual
gaussian_density
mean_scale
max_scale
opacity_stats
depth_range
```

### appearance

```text
SH-DC mean/std
RGB render mean/std
gradient energy
```

### semantics

优先：

```text
SAM3.1 road / sidewalk / lane marking / curb concept
```

若 SAM3.1 无权重：

使用当前可用：

```text
V3.2 masks + image statistics
```

不增加新的大型 feature backbone 作为主门禁。

### support

```text
train-view observation count
multi-camera count
visibility mass
```

## 9.5 patch size

scene-0230 development：

```text
1 m
2 m
4 m
```

只允许这三个。

根据 hole 的 BEV bbox 选能完全覆盖的最小 patch size。

结果前不增加新 patch size。

## 9.6 donor 排除

donor patch：

- 不能和 target hole 重叠；
- 不能含动态 actor 高语义质量区域；
- 不能来自 heldout-only observation；
- 不能包含 V3.2 generated background；
- visibility support 不足则拒绝；
- 局部 geometry outlier 则拒绝。

## 9.7 candidate search

top-K：

```text
K = 5
```

距离：

```text
D =
wg D_geometry
+ wa D_appearance
+ ws D_semantic
+ wv D_visibility
```

权重只在 scene-0230 train/development 冻结。

## 9.8 placement

V1 只允许：

```text
BEV translation
+
optional yaw alignment
+
z-plane offset
```

不允许非刚性 warp。

yaw 来源：

优先：

```text
local road tangent
```

无可靠道路 tangent：

```text
translation-only
```

## 9.9 Gaussian copy

donor Gaussian copy 为：

```text
GENERATED_BY_PATCH_REUSE
```

不是：

```text
OBSERVED_AT_TARGET
```

保留：

```text
donor_chunk_id
donor_flat_indices
transform
source_gaussian_hash
```

## 9.10 seam fusion

只对新增 patch：

- opacity feather；
- SH-DC / RGB small affine；
- scale clamp；
- duplicate suppression。

base Background：

```text
完全不修改
```

禁止 whole-scene retrain。

## 9.11 candidate selection

只能在 train/development view 选择 donor。

指标：

- hole coverage；
- seam L1；
- local depth discontinuity；
- adjacent-view consistency；
- non-target drift。

heldout 只做最终 confirmation。

---

# 10. S2-B：Inpaint360GS 强 baseline

## 10.1 目标

真正执行官方 Inpaint360GS 代码，作为：

```text
generative / retrained 3D inpainting strong baseline
```

## 10.2 adapter 原则

优先适配数据，而不是修改论文核心算法。

允许写：

```text
StreetGS → Inpaint360GS scene adapter
```

包含：

- camera intrinsics；
- camera extrinsics；
- images；
- object masks；
- train/test split；
- initial point cloud / Gaussian export；
- COLMAP-like camera metadata。

不允许：

- 使用 heldout 图参与 retraining；
- 为了让 repo 跑通改变 scene split；
- 把 nuScenes metric pose 当 COLMAP optimization 输出；
- 静默降低 resolution。

## 10.3 单卡约束

先：

```text
1 object
1 scene
minimum official settings
```

如果官方 pipeline 在 24 GiB：

- OOM；
- 需要多卡；
- 必须大规模重新训练；

则：

```text
blocked_single_3090
```

RoadPatch-Lite 仍继续。

## 10.4 对照

```text
B0：V3.2 Telea-generated Background
B1：RoadPatch-Lite
B2：Inpaint360GS
```

条件式：

```text
B3：3D-GIMP
B4：GOR-IS
```

只有官方 code/weights 可执行才加入。

---

# 11. S2 最终裁决

主指标：

### observed / verifiable regions

- heldout PSNR / SSIM / LPIPS；
- local crop LPIPS；
- depth consistency；
- multi-view warp consistency；
- seam artifact；
- non-target drift。

### unsupported generated region

不报：

```text
GT accuracy
```

只报：

- coverage；
- cross-view consistency；
- provenance；
- donor real-support ratio；
- generated fraction。

最终优先级：

```text
真实 3D donor patch
>
cross-view observed reconstruction
>
generative inpainting
>
ABSTAIN
```

---

# 12. S3：Asset Harvester 自动 View Selection

V3.2 已证明 Asset Harvester 可运行，不重做模型本身。

## 12.1 当前问题

当前 1-view / 2-view 来源：

```text
frame 91
frame 51 + 91
```

仍偏人工冻结。

V3.3 做：

> automatic observation selection

## 12.2 候选 view pool

只从 train frames：

- actor visible；
- SAM mask accepted；
- bbox 面积足够；
- 非 heldout。

## 12.3 单 view quality score

```text
Q_view =
w_area × projected_area
+ w_mask × mask_confidence
+ w_sharp × sharpness
+ w_vis × visible_fraction
- w_occ × occlusion_score
- w_edge × truncation_score
```

sharpness：

```text
Laplacian variance
```

truncation：

```text
mask touching image boundary
```

occlusion：

基于：

- projected 3D box；
- mask ratio；
- D2 effect mask；
- nearby actor overlap。

## 12.4 multi-view diversity

选择 2/4 view 时：

```text
Q_set =
Σ Q_view
+
λyaw × yaw diversity
+
λtime × temporal diversity
+
λcam × camera diversity
```

禁止只选同一侧几乎重复的 view。

## 12.5 view count

开发：

```text
1
2
4
```

先 high-support actor。

选出策略后：

```text
boundary-support actor
```

再条件式确认：

```text
scene-0242 high actor
scene-0255 high actor
```

## 12.6 output

`actor_asset_manifest.json` 新增：

```text
selection_policy
candidate_views
selected_views
view_score
set_score
yaw_distribution
mask_quality
occlusion
sharpness
```

## 12.7 S3 比较

```text
A0：V3.2 manual 2-view
A1：auto best 1-view
A2：auto best 2-view
A4：auto best 4-view
```

指标：

- observed-view IoU；
- boundary F1；
- LPIPS；
- PSNR；
- cross-view consistency；
- lateral +1m artifact；
- asset bytes；
- inference wall；
- peak VRAM。

背面仍：

```text
generated completeness only
```

不声明 GT correctness。

---

# 13. S4：Spatial Delta Layer

这是 V3.3 的核心工程/方法接口。

## 13.1 目标

不再把每次 edit 表达为：

```text
完整新 checkpoint
```

而是：

```text
immutable base
+
delta
```

## 13.2 delta 类型

```text
ERASE
INSERT_BACKGROUND
INSERT_ACTOR
REPAIR
RENDER_ONLY
```

## 13.3 目录

```text
worldsim_asset/
├── base/
│   ├── checkpoint
│   └── registry
├── deltas/
│   ├── delete_actor_13/
│   │   ├── erase.json
│   │   ├── background_patch.npz
│   │   └── manifest.json
│   ├── lateral_actor_13/
│   │   ├── actor_override.npz
│   │   └── manifest.json
│   └── ...
└── package_manifest.json
```

## 13.4 erase

不要删除 base tensor。

保存：

```text
model
source_flat_indices
gaussian_ids
instance_token
reason
mask_hash
```

runtime：

```text
effective_opacity = 0
```

仅在 composite renderer 内生效。

## 13.5 insert

background：

```text
RoadPatch-Lite / Inpaint360GS generated Gaussian
```

actor：

```text
Asset Harvester Gaussian
```

以独立 asset 加载。

## 13.6 composition order

固定：

```text
base
→ erase delta
→ background repair delta
→ actor override / insertion delta
→ render-only postprocess
```

## 13.7 exact rollback

必须支持：

```text
compose(edit)
→ render
→ remove_delta(edit)
→ source render SHA exact
```

## 13.8 delta 不变量

- base checkpoint SHA 不变；
- source registry SHA 不变；
- unaffected actor index 不变；
- delta 不得复制完整 source checkpoint；
- 每个 inserted Gaussian 必须有 provenance；
- erase / insert 可单独开关；
- two edits composition 顺序明确；
- duplicate Gaussian index=0。

---

# 14. S5：语义约束的视觉融合

## 14.1 delete 不再走 generic Harmonizer 主链

V3.2 已证明：

```text
delete image
→ generic Harmonizer
→ deleted car semantics may reappear
```

因此 V3.3：

```text
delete:
不启用 unconstrained Harmonizer
```

## 14.2 insertion / lateral 才使用 enhancement

目标：

- actor-ground contact；
- shadow；
- local lighting；
- seam。

## 14.3 R3D2 conditional

P0 审计：

- official code；
- license；
- pretrained/exported model availability。

若合法 pretrained model 可用：

```text
R3D2-fast first
```

单卡运行。

如果只有训练代码，没有官方可用 checkpoint：

```text
S5-R3D2 = blocked_pretrained_model_unavailable
```

禁止从零训练 R3D2。

## 14.4 Semantic-Gated Harmonizer

无 R3D2 权重时，实现现有 Harmonizer 的安全 wrapper：

允许改变区域：

```text
actor boundary ring
ground-contact region
shadow support region
small seam ring
```

删除 region：

```text
hard preserve generated background
```

far non-target：

```text
small residual only
```

blend：

```text
output =
input
+ gate × clamp(harmonizer(input)-input, ±R)
```

其中：

```text
gate ∈ [0,1]
```

来自：

- object-aware S1；
- actor footprint；
- ground plane；
- seam distance。

## 14.5 semantic reintroduction detector

对 delete：

render enhancer 后用：

```text
SAM3.1 exemplar
或 SAM2.1 existing actor prompt
```

检查 deleted actor 是否重新出现。

如果：

```text
inside-delete semantic mass > threshold
```

则：

```text
reject enhanced frame
fallback raw 3D render
```

---

# 15. Single RTX 3090 资源合同

## 15.1 硬上限

每个 GPU stage：

```text
NVIDIA sampled memory < 24,000 MiB
```

停止：

- CUDA OOM；
- cgroup OOM/kill 增加；
- cgroup 连续两次 >=90%；
- 磁盘 <20 GiB；
- 必须静默降低正式分辨率；
- 必须减少正式相机；
- 必须使用第二 GPU 才能运行必做 arm。

## 15.2 不并行 GPU stage

单卡下禁止：

```text
SAM + DriveStudio 同时占 GPU
Asset Harvester + renderer 同时占 GPU
Inpaint360GS + eval 同时占 GPU
```

统一：

```text
stage serial execution
```

## 15.3 GPU release gate

每个 stage 后：

```text
nvidia-smi
```

要求不存在本 task 的遗留 compute PID。

## 15.4 environment isolation

```text
envs/worldsim-v33-sam3
envs/worldsim-v33-inpaint360
envs/worldsim-v33-r3d2
```

模型通过文件交互，不把新依赖装入 DriveStudio env。

---

# 16. V3.3 fixed experiment matrix

## 16.1 S1

scene-0230：

```text
O0 heuristic
O1 dual-opacity
O2 SAM3.1 + dual-opacity (if weights)
O3 dual-opacity + reassignment
```

confirmation：

只跑：

```text
O0 + frozen O*
```

## 16.2 S2

scene-0230：

```text
B0 V3.2 Telea
B1 RoadPatch-Lite
B2 Inpaint360GS (if single-card executable)
B3 3D-GIMP (conditional)
B4 GOR-IS (conditional)
```

confirmation：

```text
B0 + frozen B*
```

## 16.3 S3

scene-0230 high actor：

```text
manual 2-view
auto 1-view
auto 2-view
auto 4-view
```

boundary actor：

只跑：

```text
manual/native
+
frozen view-selection policy
```

## 16.4 S4

```text
base only
base + erase
base + erase + background
base + actor override
base + erase + background + actor
```

全部支持：

```text
on/off exact rollback
```

## 16.5 S5

```text
H0 raw 3D render
H1 semantic-gated Harmonizer
H2 R3D2-fast (conditional)
```

---

# 17. 指标合同

V3.3 不建设大型新 benchmark。

## S1

- IoU；
- boundary F1；
- boundary distance；
- temporal identity stability；
- positive/ambiguous Gaussian 数；
- false-positive semantic mass；
- sidecar bytes；
- wall；
- VRAM。

## S2

- target-hole coverage；
- heldout PSNR/SSIM/LPIPS；
- local crop LPIPS；
- seam L1；
- cross-view consistency；
- depth discontinuity；
- donor-real-support fraction；
- generated fraction；
- wall；
- VRAM。

## S3

- observed-view IoU；
- boundary F1；
- PSNR/LPIPS；
- multi-view consistency；
- lateral artifact；
- asset bytes；
- wall；
- VRAM。

## S4

- delta bytes；
- base bytes；
- rollback SHA；
- compose wall；
- load wall；
- additional VRAM；
- duplicated rows；
- source immutability。

## S5

- deleted-semantic reintroduction；
- outside-mask drift；
- seam improvement；
- shadow/contact QA；
- temporal consistency（若视频）；
- wall；
- VRAM。

---

# 18. 代码结构

建议新增：

```text
motion_proj/worldsim_v33/
├── __init__.py
├── instance_field.py
├── instance_renderer.py
├── semantic_reassignment.py
├── road_patch_index.py
├── road_patch_features.py
├── road_patch_search.py
├── road_patch_fusion.py
├── inpaint360_adapter.py
├── asset_view_selector.py
├── spatial_delta.py
├── delta_renderer.py
├── semantic_gate.py
├── r3d2_adapter.py
├── provenance.py
└── integration.py
```

configs：

```text
configs/worldsim_v33/
├── p0_sources_v1.yaml
├── s1_instance_field_v1.yaml
├── s1_sam3_v1.yaml
├── s2_roadpatch_v1.yaml
├── s2_inpaint360_v1.yaml
├── s3_view_selection_v1.yaml
├── s4_spatial_delta_v1.yaml
├── s5_semantic_gate_v1.yaml
└── r0_integration_v1.yaml
```

scripts：

```text
scripts/audit_worldsim_v33_sources.py
scripts/run_worldsim_v33_s1_instance_field.py
scripts/run_worldsim_v33_s1_sam3.py
scripts/build_worldsim_v33_road_patch_index.py
scripts/run_worldsim_v33_s2_roadpatch.py
scripts/run_worldsim_v33_s2_inpaint360.py
scripts/run_worldsim_v33_s3_view_selection.py
scripts/run_worldsim_v33_s3_asset_harvester.py
scripts/materialize_worldsim_v33_delta.py
scripts/render_worldsim_v33_delta.py
scripts/run_worldsim_v33_s5_semantic_gate.py
scripts/audit_worldsim_v33_r3d2.py
scripts/finalize_worldsim_v33.py
```

tests：

```text
tests/test_worldsim_v33_instance_field.py
tests/test_worldsim_v33_instance_render.py
tests/test_worldsim_v33_no_rgb_mutation.py
tests/test_worldsim_v33_road_patch_index.py
tests/test_worldsim_v33_patch_exclusion.py
tests/test_worldsim_v33_patch_transform.py
tests/test_worldsim_v33_view_selector.py
tests/test_worldsim_v33_spatial_delta.py
tests/test_worldsim_v33_delta_rollback.py
tests/test_worldsim_v33_semantic_gate.py
tests/test_worldsim_v33_provenance.py
tests/test_worldsim_v33_integration.py
```

---

# 19. Provenance V3.3

保留 V3.2：

```text
OBSERVED_RGB
LIDAR_MEASURED
MULTIVIEW_GEOMETRIC
NATIVE_3DGS
SAM_SEMANTIC
GENERATED_BACKGROUND
GENERATED_ACTOR
HARMONIZED_2D
```

新增：

```text
INSTANCE_OPACITY_LEARNED
PATCH_REUSED_BACKGROUND
INPAINT360_GENERATED
DELTA_ERASE
DELTA_INSERT_BACKGROUND
DELTA_INSERT_ACTOR
SEMANTIC_GATED_2D
R3D2_2D
```

任何新增 Gaussian：

必须记录：

```text
generation method
source Gaussian IDs
source patch/chunk
source views
transform
mask hash
config hash
model revision
```

---

# 20. Commit 计划

```text
docs(worldsim): 启动 V3.3 对象感知与可维护资产路线

research(seg): 冻结 SAM3.1 与 object-aware Gaussian 协议
feat(seg): 实现 dual instance opacity Gaussian field
research(seg): 完成对象感知语义场消融

research(inpaint): 建立驾驶场景 Gaussian patch index
feat(inpaint): 实现 RoadPatch-Lite 搜索与 delta fusion
research(inpaint): 接入 Inpaint360GS 强基线
research(inpaint): 收口 V3.3 背景修复候选

feat(asset): 实现 Asset Harvester 自动多视角选择
research(asset): 完成 1/2/4-view actor asset 消融

feat(edit): 实现 spatial delta erase-insert 资产层
research(render): 实现 semantic-gated post-render
research(render): 审计 R3D2 单卡 inference

research(worldsim): 完成 V3.3 单卡集成
```

每个提交不得混入 V2/M5 dirty files。

---

# 21. Coding Agent 首轮提示词

```text
执行 docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md。

当前事实：
- V3.2 已 done；
- canonical R0=20260810T134658Z__r0-final-integration-s0-r1；
- V3.2 scoped commit=f44decacca812fe1b476253b6bdc8aac1869f873；
- 当前硬件只有单张 RTX 3090 24 GiB；
- V3.2 canonical 资产全部只读；
- 用户已有 V2/M5 dirty 工作树不得纳入 V3.3。

当前唯一授权任务：
WS-V33-P0-ROUTE-SOTA-AUDIT-01

开始前：

1. 读取：
   AGENTS.md
   docs/RESEARCH_STATUS.md
   docs/RESEARCH_FAILURES.md
   docs/EXPERIMENTS.md
   docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md
   docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md
   docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md

2. 核对：
   - 当前 branch；
   - HEAD；
   - dirty files；
   - GPU；
   - cgroup；
   - disk；
   - active tmux/controller/process。

3. 核对 V3.2 canonical：
   - R0 terminal done；
   - mixed checkpoint SHA；
   - chunk manifest SHA；
   - S2 checkpoint SHA；
   - S3 actor asset SHA；
   - 36 tests；
   - 输入 before/after SHA 不变。

4. 新建：
   research/worldsim-v3.3-object-maintenance

5. 新建：
   configs/worldsim_v33/
   motion_proj/worldsim_v33/

6. 不修改：
   motion_proj/worldsim_v32/ 的历史语义；
   V3.2 run；
   V3.2 checkpoint；
   V3.2 package。

7. 只读审计并固定：
   - Meta SAM3
   - OP2GS
   - Inpaint360GS
   - GS-RoadPatching
   - 3D-GIMP
   - FocusGS
   - R3D2
   - GOR-IS
   - LiDAR-EVS

8. 对每个项目核对：
   official source
   commit/tree
   license
   checkpoint availability
   checkpoint license
   Python/Torch/CUDA
   single 3090 feasibility
   input/output schema

9. 重点：
   - SAM3.1 checkpoint 是 gated 时，不绕权限；
   - OP2GS 无官方可执行源码时，只登记 paper_inspired；
   - GS-RoadPatching GitHub 若仍无算法源码，只登记 paper_inspired；
   - 3D-GIMP 若无官方代码，只 audit；
   - R3D2 若无官方 pretrained exported model，不从零训练。

10. S0 不能：
    - 下载大型 gated weights；
    - 训练；
    - 启动 Inpaint360GS；
    - 修改 DriveStudio；
    - 启动 Asset Harvester。

11. 生成：
    configs/worldsim_v33/p0_sources_v1.yaml
    docs/WS_V33_P0_SOTA_AUDIT.md

12. 同步：
    docs/RESEARCH_STATUS.md
    docs/EXPERIMENTS.md
    docs/RESEARCH_FAILURES.md
    README.md
    V3.3 plan 更新日志

13. 运行：
    WorldSim V3.3 新测试
    现有 V3.2 回归
    git diff --check

14. 提交：
    docs(worldsim): 启动 V3.3 对象感知与可维护资产路线

S0 done 后，
只允许解锁：
WS-V33-S1-OBJECT-AWARE-GS-01
```

---

# 22. S1 Coding Agent 提示词

P0 done 后：

```text
只执行 WS-V33-S1-OBJECT-AWARE-GS-01。

核心原则：
V3.2 RGB 3DGS immutable。
只学习 instance-opacity sidecar。

1. 固定 scene-0230。
2. 固定 high-support / boundary-support actor。
3. heldout 禁止进入 optimization。
4. SAM3.1 可用则：
   exemplar + projected-box；
   不可用则 exact fallback V3.2 SAM2.1 masks。
5. base means/scales/quats/SH/RGB opacity 全部 requires_grad=false。
6. instance field 独立保存。
7. 先 synthetic render correctness。
8. 再真实 100-step smoke。
9. smoke 后冻结：
   LR / loss weights / trainable candidate radius / steps。
10. 再 development formal。
11. RGB source checkpoint before/after SHA 必须 exact。
12. 如果 dual-opacity 不改善 boundary/identity：
    S1*=V3.2 heuristic；
    负结果正常收口；
    不追加 post-hoc loss。
```

---

# 23. S2 Coding Agent 提示词

S1 收口后：

```text
只执行 WS-V33-S2-ROADPATCH-INPAINT-01。

先实现 RoadPatch-Lite；
Inpaint360GS 是强 baseline，不得抢先扩大环境。

RoadPatch-Lite：
1. 使用 immutable native Background 作为 donor。
2. GENERATED_BACKGROUND 禁止作为 donor。
3. target hole 用 object-aware delete mask + 合法 depth 定位。
4. 建 1/2/4m static patch index。
5. 冻结 geometry/appearance/semantic/support feature schema。
6. donor exclusion fail closed。
7. top-K=5。
8. 只允许 translation / optional yaw / z offset。
9. 新 Gaussian 写 delta，不写 source checkpoint。
10. train/dev 选 donor，heldout 只 confirmation。
11. 保存 donor Gaussian ID、chunk、transform、hash。
12. 对比 V3.2 Telea。

RoadPatch-Lite 完成后才审计 Inpaint360GS adapter：
如果单卡 24 GiB 官方最小配置可运行，则跑一个 scene-0230 delete；
否则 blocked_single_3090，不改协议救火。

禁止：
- whole-scene retrain 冒充 local patch；
- heldout patch donor；
- Telea-generated donor；
- 事后新增 patch size；
- unknown-depth hole 强行 patch。
```

---

# 24. S3 Coding Agent 提示词

```text
只执行 WS-V33-S3-ASSET-VIEWSELECT-01。

不改 Asset Harvester 网络。
只做 automatic view selection。

1. 枚举 train-only actor observations。
2. 保存：
   projected area
   SAM quality
   blur
   boundary truncation
   occlusion
   camera
   timestamp
   estimated actor yaw
3. 冻结单 view score。
4. 冻结 2/4-view diversity score。
5. high-support：
   auto 1 / 2 / 4 views。
6. 和 V3.2 manual 2-view exact baseline 比。
7. 选出固定 policy 后才跑 boundary actor。
8. 如果 boundary actor 输入质量不足：
   ABSTAIN。
9. 不使用 heldout 选 view。
10. 不把生成背面当 GT。
```

---

# 25. S4 Coding Agent 提示词

```text
只执行 WS-V33-S4-SPATIAL-DELTA-01。

目标：
把 V3.3 edit 变成 immutable base + delta。

1. base checkpoint/registry read-only。
2. 实现 ERASE：
   不删除 base tensor；
   runtime opacity mask。
3. 实现 INSERT_BACKGROUND。
4. 实现 INSERT_ACTOR。
5. 每个 insert 都保存 provenance。
6. composition order 固定。
7. delta 单独 hash。
8. compose→render→rollback。
9. rollback RGB 必须与 source SHA exact。
10. 两个 independent delta 组合必须确定性。
11. 不输出新的完整 checkpoint 作为唯一编辑状态。
12. 最终仍允许为部署生成 materialized mixed/chunk package，
    但 canonical authoring state 是 base + delta。
```

---

# 26. R0 最终目标

V3.3 合法 final chain 示例：

```text
V3.2 D2 immutable base
→ S1 object-aware instance field
→ S2 RoadPatch-Lite background delta
→ S3 auto-selected Asset Harvester actor
→ S4 erase/insert delta composition
→ S5 semantic-gated render
→ mixed persistent storage
→ exact static/actor/delta package
```

R0 必须报告：

```text
1. object field 是否改善；
2. RoadPatch 是否优于 V3.2 Telea；
3. Inpaint360GS 是否能在单卡运行；
4. Asset Harvester view selection 是否改善；
5. delta package 是否 exact rollback；
6. semantic-gated renderer 是否避免 delete semantic reintroduction；
7. 每个生成区域的 provenance；
8. single RTX3090 wall / VRAM / disk；
9. 所有 rejected / blocked SOTA；
10. 不把 unavailable code/weights 包装成失败算法。
```

---

# 27. V3.3 成功标准

不要求每项 SOTA 都胜出。

满足以下 4 项即可视为 V3.3 明显优于 V3.2：

### 必须

1. **Object-aware Gaussian field**
   - dual instance opacity 跑通；
   - base RGB bitwise 不变；
   - 至少一个 actor 的 boundary/identity 比 heuristic 更好，或形成可信负结果。

2. **3D-native background repair**
   - RoadPatch-Lite 完成；
   - donor provenance 完整；
   - 不依赖 Telea 作为主方法；
   - heldout 不明显伤害。

3. **Automatic Asset Harvester view selection**
   - 1/2/4 view 完成；
   - 不使用 heldout；
   - 选出固定 policy 或合法 no-gain。

4. **Spatial Delta**
   - erase / background insert / actor insert；
   - exact rollback；
   - base immutable。

### 加分项

- SAM3.1 成功替代/补充 SAM2.1；
- Inpaint360GS single-GPU 跑通；
- R3D2 pretrained inference 跑通；
- semantic-gated Harmonizer 改善；
- 0242/0255 confirmation；
- future LiDAR-EVS audit。

---

# 28. 合法终态

```text
object_field_supported
object_field_no_gain
sam3_weights_blocked

roadpatch_supported
roadpatch_no_gain
inpaint360_supported
inpaint360_single_gpu_blocked
generative_inpaint_visual_only

asset_view_selection_supported
asset_view_selection_no_gain

spatial_delta_supported
spatial_delta_blocked

semantic_gate_supported
r3d2_pretrained_unavailable

v33_supported
v33_partial_supported
v33_engineering_blocked
```

任何负结果都保留，不统一写成项目失败。

---

# 29. 研究叙事

V3.3 最终不要写成：

> 我接了很多 SOTA。

而应写成：

> **V3.1–V3.2 建立了可编辑动态驾驶 3DGS 的模型与生产基础；V3.3 进一步发现工业 WorldSim 的核心问题不是无限增加生成模型，而是让显式场景资产具备对象归属、真实来源优先的局部维护和可回滚编辑语义。因此，本阶段将“对象感知 occupancy、驾驶场景 3D patch repair、完整 actor generation 和 spatial delta”统一进同一资产链。**

最终技术结构：

```text
Reconstruction
→ Object-aware representation
→ Evidence / patch prioritized repair
→ Generative asset completion
→ Deterministic erase-insert
→ Deployment packaging
```

这比“纯 3DGS 重建”更接近：

> **maintainable neural world asset for WorldSim**。

---

# 30. 更新日志

## 2026-08-11 — V3.3 计划创建

- 冻结 V3.2 canonical R0、S1/S2/S3/S4 与 mixed/chunk 事实；
- 单卡 RTX 3090 继续作为硬资源合同；
- 不再扩大 heuristic semantic lifting；
- 引入 OP2GS-inspired dual instance opacity；
- SAM3.1 作为 gated 条件增强，SAM2.1 保持可执行 fallback；
- 将 background 主研究方向从 Telea-based generated completion 切到 driving-specific Gaussian patch reuse；
- Inpaint360GS 升级为必须审计的强官方 baseline；
- 3D-GIMP / GOR-IS 只在官方代码可执行时进入；
- Asset Harvester 保留，但新增自动 1/2/4-view selection；
- 引入 FocusGS-inspired spatial delta / erase-insert authoring；
- generic Harmonizer 不再用于 delete 主链；
- R3D2 只在官方 pretrained/exported model 可合法获得时运行；
- LiDAR-EVS 保留为 R0 后条件式 WorldSim 传感器扩展。
