# 面向 WorldSim 的动态驾驶 3DGS 语义资产修复与生成式补全计划 V3.2

- **版本**：V3.2
- **日期**：2026-08-10
- **项目根目录**：`/root/autodl-tmp/motion_proj`
- **基线分支**：`research/worldsim-v3`
- **当前分支**：`research/worldsim-v3.2-semantic-repair`
- **执行环境**：
  - 默认：单卡 NVIDIA GeForce RTX 3090 24 GiB；
  - 可选：双卡 RTX 3090 24 GiB × 2；
  - cgroup memory 继续沿用现有资源合同；
- **历史权威计划**：`docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md`
- **V3.1 当前终态**：
  - A1=`done_off`；
  - A2=`done / tradeoff_non_dominated`，后续研究资产选 `D2-boundary-priority`；
  - A3=`done`，旧 `R1 local refine` 因资源门失败且诊断为 tradeoff，被正式 `rejected`；
  - A3*=R0-off，即 D2 checkpoint immutable exact alias；
  - A4=`done`，P2 mixed-precision + P3 exact chunk package 已完成；
  - F0=`done`，Instant NuRec 当前机器本地 inference 未解锁；
  - R0=`done`，V3.1 主计划已收口。
- **V3.2 当前任务**：`WS-V32-R0-INTEGRATION-01`（`done`；全部单卡可执行工作已收口，S5 保持 `blocked`）
- **V3.2 核心目标**：不再继续增加旧 A3 局部 opacity/scale 优化步数，而是围绕工业和 2025–2026 SOTA 的共同路线，建立：
  1. 高质量 2D/视频语义掩码；
  2. 多视角一致的 2D→3D Gaussian semantic lifting；
  3. 删除/位移后背景的 3DGS inpainting；
  4. 稀疏驾驶日志中的完整动态 actor 资产恢复；
  5. 在线生成式 artifact / lighting / shadow harmonization；
  6. 与现有 StreetGS actor registry、trajectory editor、P2/P3 资产链保持可追踪集成。

---

# 0. V3.2 为什么必须换路线

V3.1 A3 已经回答了一个重要问题：

> 只在已有 Gaussian 上做严格局部的 opacity / scale 重激活，并不能稳定解决编辑后 WorldSim 的真实缺口。

旧 R1 的问题有两层：

1. **工程层**
   - held-out 评估路径峰值显存约 14.24 GiB；
   - 超过旧协议冻结的 12 GiB ceiling；
   - 虽然没有 CUDA OOM，但正式资源门失败。

2. **方法层**
   - depth-order violation 有改善；
   - non-target / global RGB safeguard 出现严格退化；
   - exact comparator 为 `tradeoff_non_dominated`；
   - 继续加 step、放大 affected set、加卡，只能扩大同一个优化问题，不能补出真正不存在的背景和完整动态资产。

V3.1 还发现：

```text
S-A observed          真实其他时间/相机 RGB 支持
S-B geometric         LiDAR / 至少两视图几何支持
S-C unsupported       完全未观测
```

真实 A3 工程 smoke 中：

```text
S-A = 0
S-B mutable Gaussian = 51
S-C abstain Gaussian = 16,451
```

因此最大的缺口不是“优化得不够久”，而是：

> **大量受影响区域本来就没有足够真实三维/颜色证据。**

V3.2 不再试图让纯 3DGS 从不存在的数据中恢复真实世界，而改成工业上更实际的分层架构：

```text
已观测 / 几何支持
    → 3DGS / multi-view geometry 保持和修复

动态 actor 边界和身份
    → SAM 系列 + 3D Gaussian semantic lifting

完全未观测背景
    → multi-view 3D inpainting / diffusion completion

不完整动态 actor
    → generative asset harvesting

插入后的颜色 / 光照 / 阴影 / NVS artifact
    → online Harmonizer
```

---

# 1. V3.2 研究定位

## 1.1 项目定位

> **面向自动驾驶 WorldSim 的语义可编辑 3DGS 场景资产修复与生成式补全。**

本项目不重新发明：

- actor identity；
- scene graph；
- trajectory editing；
- basic remove / lateral API；
- 3DGS renderer；
- 大型 world foundation model。

本项目只回答三个问题：

### Q1：语义 Grounding

> 现有 token-first actor registry 已经知道“是哪辆车”，但能否利用 SAM 系列模型得到更准确的真实图像边界，并稳定 lift 到 Gaussian，从而改善删除、资产抽取和局部编辑的边界？

### Q2：背景补全

> actor 删除/位移后，能否利用多视角几何 + 生成式 inpainting 重建原来被遮挡、甚至完全未观测的背景，而不是继续对已有 Gaussian 做无证据优化？

### Q3：完整动态资产

> 对只在少数驾驶视角出现的车辆，能否采用最新 AV 资产生成模型恢复更完整的 3D Gaussian actor，使其在 lateral / insert / cross-scene manipulation 中比原始 per-scene RigidNodes 更稳定？

---

# 2. V3.1 必须冻结、不允许被 V3.2 改写的事实

## 2.1 最终基础资产

算法研究默认使用：

```text
A2-D2 FP32 source checkpoint
SHA-256:
1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c
```

理由：

- 它是当前 actor boundary-priority research asset；
- 保留最完整原始 Gaussian 参数；
- 外部 SOTA repo 通常默认普通 FP32 3DGS/gsplat 资产；
- 不直接在 P2 mixed/P3 package 上开发第三方算法，减少 schema 兼容干扰。

V3.1 P2 mixed checkpoint 与 P3 chunk package继续保持 immutable production artifact。

只有 V3.2 模型链被选中后，才重新执行：

```text
V3.2 selected scene asset
→ P2-style mixed storage converter
→ P3-style chunk / actor package
```

不得修改原 V3.1 P2/P3 canonical 文件。

## 2.2 继续复用

- `motion_proj/worldsim_v3/gaussian_ancestry.py`
- `motion_proj/worldsim_v3/actor_metrics.py`
- `motion_proj/worldsim_v3/boundary_residual.py`
- `motion_proj/worldsim_v3/chunk_package.py`
- `motion_proj/worldsim_v3/mixed_precision.py`
- `motion_proj/worldsim_v3/local_refinement.py`：只作为旧 R1 历史基线和类型定义，不继续扩旧 R1；
- `motion_proj/dynamic_editing_v2/drivestudio_registry.py`
- `motion_proj/dynamic_editing_v2/actor_projection.py`
- `motion_proj/resim/` 的 WorldState / trajectory editor / counterfactual render；
- scene-0230 / 0242 / 0255 的 frozen actor cohort；
- scene-0230 的 `original / lateral +1m / delete` 编辑合同。

## 2.3 禁止

- 不为了 V3.2 把旧 A3 R1 从 rejected 改回 running；
- 不通过提高旧 R1 显存门、增加 step、扩大 affected set 来“救活”历史结果；
- 不把 2×3090 当成 48 GiB 单卡共享显存；
- 不修改 V3.1 canonical checkpoint / summary / terminal；
- 不把生成式结果冒充真实观测；
- 不把 SAM mask 冒充 GT segmentation；
- 不把 2D Harmonizer 输出写回三维 Gaussian state；
- 不把第三方 SOTA demo 的网页能力写成本地已跑通能力。

---

# 3. 2026-08-10 SOTA 路线冻结

以下工作只采用论文、官方项目页和官方代码。

| 方向 | 方法 | 年份/会议 | V3.2 定位 | 是否主路径 |
|---|---|---|---|---|
| 视频语义分割 | SAM 2 | 2024, Meta | 时序 mask propagation，成熟官方代码 | 是 |
| Gaussian prompt segmentation | SAGA | AAAI 2025 | 3D Gaussian affinity / prompt segmentation 对照 | 参考 |
| SAM→3DGS | Gaussian Grouping | ECCV 2024 | 2D SAM knowledge lift 到 Gaussian identity | 参考 |
| 高质量 3DGS segmentation | Robust Prior-Guided Segmentation for Editable 3DGS | ICIP 2026 | SAM-HQ + prior-guided multi-view label reassignment | **主算法参考** |
| 多视角 SAM | MV-SAM | 2026 | pointmap-guided view-consistent segmentation | 条件式；代码可用才跑 |
| 驾驶动态资产补全 | NVIDIA Asset Harvester | 2026 | 稀疏 AV object views → multi-view diffusion → full 3D Gaussian asset | **主路径** |
| 深度引导 3D inpainting | 3DGIC | CVPR 2025 | SAM mask + depth-guided cross-view unseen mask + 3DGS refinement | **主 baseline** |
| 360° object-aware inpainting | Inpaint360GS | WACV 2026 | 2D mask distillation + virtual views + 3DGS inpainting | **强 baseline** |
| visibility uncertainty inpainting | VISTA | 2025 | visibility uncertainty + scene concept + diffusion | 强参考 |
| 最新 2D↔3D inpainting | CoIn | 2026-06 | diffusion → GS → diffusion → GS 双向一致性，论文报告 SOTA | **条件式 SOTA** |
| multi-view feed-forward edit | Omni-3DEdit | CVPR 2026 Highlight | 单参考编辑图 → multi-view consistent edit；约分钟级 | **强上界** |
| 百视角编辑 | 100Editor | CVPR 2026 | training-free 大批量多视角编辑 | audit-only |
| online simulation enhancement | NVIDIA Harmonizer | CVPR 2026 | 单步 temporal diffusion，artifact / lighting / shadow 修复 | **主后处理** |
| AV asset insertion realism | R3D2 | 2025/2026 AD | complete asset insertion 后的 shadow / lighting realism | 次级参考 |

## 3.1 方法选择原则

### 必须实际实现或跑通

1. SAM2-based temporal mask；
2. prior-guided multi-view Gaussian semantic lifting；
3. 3DGIC **或** Inpaint360GS 至少一个真实 3D inpainting baseline；
4. NVIDIA Asset Harvester image-to-3D smoke；
5. Harmonizer pretrained inference preflight，资源允许则正式跑。

### 条件式

- MV-SAM：只有官方代码和 checkpoint 已可公开获取时跑；
- CoIn：只有官方代码/模型可获得时跑，不从论文手搓完整模型；
- Omni-3DEdit：双 3090 可用时优先跑；单卡也可做小规模 smoke；
- 100Editor：不阻塞主链。

---

# 4. V3.2 总体模型链

```text
                V3.1 A2-D2 3DGS
                       │
                       ▼
          ┌─────────────────────────┐
          │ S1 语义 Grounding       │
          │ SAM2 / SAM-HQ / prior   │
          └────────────┬────────────┘
                       │
                       ▼
       per-view mask + per-Gaussian semantic posterior
                       │
        ┌──────────────┴───────────────┐
        │                              │
        ▼                              ▼
  删除/位移后的背景                动态 actor 完整性
        │                              │
        ▼                              ▼
  3DGIC / Inpaint360GS          NVIDIA Asset Harvester
  / CoIn conditional             complete 3D asset
        │                              │
        └──────────────┬───────────────┘
                       ▼
                 WorldState merge
                       │
                       ▼
        original / lateral / delete / insert render
                       │
                       ▼
             NVIDIA Harmonizer
         artifact / lighting / shadow
                       │
                       ▼
             P2/P3 asset packaging
```

---

# 5. 任务注册表

| Task ID | 状态 | 核心产物 | 解锁条件 |
|---|---|---|---|
| `WS-V32-S0-ROUTE-AND-SOTA-AUDIT-01` | done | V3.1 freeze、新分支、第三方 source/license/weight/hardware audit | 11 个官方 source HEAD 固定、SAM2.1 large exact weight、无历史资产 mutation |
| `WS-V32-S1-SEMANTIC-LIFT-01` | done | SAM2 mask、multi-view Gaussian semantic posterior、actor semantic registry | canonical r6 identity 合同、0 heldout leak、6 smoke 通过 |
| `WS-V32-S2-BACKGROUND-INPAINT-01` | done | 3DGIC/Inpaint360GS 背景补全，CoIn 条件式 | canonical r3 追加 1,896 generated rows；两目标与 held-out 门通过 |
| `WS-V32-S3-ASSET-HARVEST-01` | done | Asset Harvester 动态 actor 资产 + StreetGS adapter | canonical r3 完成 1/2-view，选定 2-view；4 个真实视角回注与资源门通过 |
| `WS-V32-S4-HARMONIZER-01` | done | pretrained Harmonizer 输出 | canonical r3 non-temporal 跑通但删除语义门失败，仅 diagnostic；temporal 受 gated Cosmos base 阻塞 |
| `WS-V32-S5-MULTIVIEW-UPPERBOUND-01` | blocked | Omni-3DEdit/100Editor 多视角编辑上界 | 许可证门未通过，2×3090 或单卡 smoke 均未授权 |
| `WS-V32-R0-INTEGRATION-01` | done | 最终语义资产修复 WorldSim chain | canonical r4 8/8 gates；extended semantics、mixed scene、S3 override 与 exact chunk selected |

---

# 6. S0：路线切换、第三方审计与环境隔离

## 6.1 Git

新建：

```text
research/worldsim-v3.2-semantic-repair
```

建议提交：

```text
docs(worldsim): 启动 V3.2 语义资产修复路线
```

不得改写 V3.1 历史 terminal。

## 6.2 第三方目录

统一：

```text
/root/autodl-tmp/third_party/worldsim_v32/
├── sam2/
├── saga/
├── gaussian-grouping/
├── segment-anything-in-3d/
├── 3dgic/
├── inpaint360gs/
├── vista/
├── asset-harvester/
├── harmonizer/
└── omni3dedit/
```

每个 repo 记录：

```json
{
  "official_url": "",
  "commit": "",
  "license": "",
  "weights_repo": "",
  "weights_revision": "",
  "weights_sha256": "",
  "python": "",
  "torch": "",
  "cuda": "",
  "minimum_vram": "",
  "input_schema": "",
  "output_schema": ""
}
```

## 6.3 独立环境

禁止把所有 SOTA 包装进现有 DriveStudio env。

建议：

```text
envs/worldsim-v32-sam
envs/worldsim-v32-inpaint
envs/worldsim-v32-asset-harvester
envs/worldsim-v32-harmonizer
envs/worldsim-v32-omni3dedit
```

所有输出通过文件 schema 交互，不跨环境 import 大型模型。

## 6.4 S0 必须核实

### Asset Harvester

官方当前说明：

- driver >= 570；
- CUDA 12.8 compatible；
- image-to-3D 约 16 GiB VRAM；
- 支持 CPU offload；
- checkpoints：
  - multiview diffusion；
  - TokenGS lifting；
  - camera estimator；
  - object segmentation；
- 单图 + mask 也可运行。

因此 1×3090 理论资源可满足其 image-to-3D VRAM 级别，但仍必须做本机 smoke，不提前宣称可用。

### Harmonizer

- pretrained temporal model；
- NVIDIA Cosmos 0.6B base；
- 单 GPU online enhancer；
- 只做 inference，不训练；
- 若本机依赖/显存/磁盘不满足则 `blocked`，不影响 S1–S3。

### Omni-3DEdit

- 官方代码支持 `nproc_per_node=1`；
- 官方 removal 示例使用 2 GPU；
- 双 3090 可优先用于该分支；
- 输出是 multi-view images，不自动保留现有 actor registry；
- 因此只做 strong upper-bound，不作为默认生产 state。

## 6.5 S0 收口（2026-08-10）

- 分支已切换到 `research/worldsim-v3.2-semantic-repair`；
- 事实配置：[`../configs/worldsim_v32/s0_sources_v1.yaml`](../configs/worldsim_v32/s0_sources_v1.yaml)；
- 审计报告：[`WS_V32_S0_SOTA_AUDIT.md`](WS_V32_S0_SOTA_AUDIT.md)；
- 11 个公开官方仓库的本地 commit 与审计时 upstream 默认分支 HEAD exact；MV-SAM 未找到可固定的官方代码/权重；
- SAM2.1 Hiera Large 已通过 `hf-mirror.com` 下载并通过 bytes/SHA-256 校验；
- 当前无卡实例 cgroup memory max=`2,147,483,648 bytes`，GPU 推理必须等待有卡重启后的 preflight；
- V3.1 D2 checkpoint、A3*=R0-off、P2/P3 assets 与所有 terminal 未修改；
- S0=`done`，只解锁 `WS-V32-S1-SEMANTIC-LIFT-01`；S2–S5 仍未授权。

---

# 7. S1：SAM 驱动的多视角 Gaussian Semantic Lifting

## 7.1 为什么不是直接“SAM 分割完就删”

单视图 SAM 可能出现：

- 相机间边界漂移；
- 遮挡后 ID 漂移；
- 轮胎/玻璃/阴影漏分；
- 一视图把相邻车辆并入目标；
- 视频传播在长期遮挡后错误恢复。

因此 S1 要解决的是：

```text
2D promptable segmentation
→ temporal consistency
→ multi-view consistency
→ Gaussian semantic posterior
```

## 7.2 新增模块

```text
motion_proj/worldsim_v32/
├── __init__.py
├── semantic_schema.py
├── sam2_temporal.py
├── sam_boundary_refine.py
├── gaussian_semantic_lift.py
├── multiview_label_reassign.py
├── generated_provenance.py
└── third_party_registry.py

scripts/
├── audit_worldsim_v32_s0_sources.py
├── build_worldsim_v32_sam_masks.py
├── lift_worldsim_v32_semantics.py
└── finalize_worldsim_v32_s1.py

configs/worldsim_v32/
├── s0_sources_v1.yaml
├── s1_sam2_v1.yaml
└── s1_gaussian_lift_v1.yaml
```

## 7.3 Prompt 来源

不人工逐帧点。

主 prompt：

```text
nuScenes instance_token
→ frozen 3D box
→ camera projection
→ 2D prompt box
```

辅助：

- 原有 token-first actor registry；
- original/delete effect mask 只作为模型内部 prior；
- 不作为 GT。

## 7.4 SAM2 mask production

每个相机独立处理时间序列：

```text
CAM_FRONT
CAM_FRONT_LEFT
CAM_FRONT_RIGHT
```

流程：

1. 在 actor 首个高质量可见帧投影 2D box；
2. SAM2 box prompt；
3. 向前/向后 temporal propagation；
4. 每隔固定 `K` 帧使用 3D box 重提示；
5. 遮挡或 IoU/centroid jump 异常时 fail closed；
6. mask 保存原始 logits、binary mask、prompt、timestamp。

建议开发网格只在 scene-0230：

```text
K ∈ {5, 10, 20}
```

只根据 train/development frames 冻结一次。

确认场景不再调 K。

## 7.5 boundary refinement

优先级：

1. 若 Robust Prior-Guided/SAM-HQ 代码可直接运行：
   - 用其 SAM-HQ 作为 key-frame boundary refinement；
2. 若只能稳定运行 SAM2：
   - SAM2 mask + 形态学小修正；
   - 不伪造 SAM-HQ 结果。

## 7.6 2D→3D Gaussian posterior

对 Gaussian `i`：

```text
semantic_score_i =
Σ_v [ contribution_iv
      × visibility_iv
      × depth_consistency_iv
      × mask_v(project(mu_i)) ]
/
Σ_v valid_weight_iv
```

其中：

- `contribution_iv`：渲染 alpha contribution；
- `visibility_iv`：该 Gaussian 在该训练视图可见；
- `depth_consistency_iv`：投影深度与 first-hit depth 一致；
- `mask_v`：SAM mask 概率。

必须保存：

```text
num_positive_views
num_negative_views
weighted_positive
weighted_total
semantic_score
depth_consistency_rate
boundary_score
```

## 7.7 existing actor registry 作为 prior，而不是被替换

对已有 RigidNodes actor：

```text
registry core
→ high-confidence positive prior
```

SAM 主要允许：

- 修正边界；
- 排除与 actor 混叠的背景 Gaussian；
- 找回遗漏的同 actor Gaussian；
- 生成高质量 image-space edit mask。

不允许仅因单帧 SAM 把另一辆相邻车重标成目标。

## 7.8 prior-guided multi-view reassignment

参考 2026 Robust Prior-Guided Editable 3DGS：

Gaussian label 的最终状态：

```text
CORE_POSITIVE
SEMANTIC_POSITIVE
AMBIGUOUS
NEGATIVE
```

规则在结果前冻结。

`AMBIGUOUS`：

- 不删除；
- 不作为 hole ground-truth；
- 可以进入 S2 unseen-mask uncertainty。

## 7.9 S1 对照

```text
M0：原 registry/effect-mask
M1：SAM2 temporal mask
M2：M1 + multi-view Gaussian lifting
M3：M2 + prior-guided label reassignment
M4：MV-SAM（条件式，仅官方代码可用）
```

S1 不训练新的 3DGS 主模型。

---

# 8. S2：删除/位移后的 Background 3D Inpainting

## 8.1 核心原则

V3.1 S-C：

```text
unsupported
```

V3.2 改为：

```text
unsupported_for_real_truth
but
allowed_for_generated_completion
```

生成内容必须显式 provenance：

```text
GENERATED_BACKGROUND
```

不得与：

```text
OBSERVED_RGB
LIDAR_GEOMETRIC
MULTIVIEW_GEOMETRIC
```

混写。

## 8.2 S2-A：3DGIC，主 baseline

选择原因：

- CVPR 2025；
- 官方代码公开；
- 直接使用 SAM object mask；
- 利用 render depth 和跨视图投影，把“其他相机已经看过的背景”从 inpaint mask 中去掉；
- 只对所有训练视图都看不到的区域进行 inpainting；
- 非常适合当前 delete / vacated-hole 问题。

### 接入

新增：

```text
motion_proj/worldsim_v32/inpainting_adapter.py
motion_proj/worldsim_v32/depth_guided_unseen_mask.py
scripts/run_worldsim_v32_s2_3dgic.py
configs/worldsim_v32/s2_3dgic_v1.yaml
```

### 输入

- D2 scene render；
- train-view SAM mask；
- camera intrinsics/extrinsics；
- 3DGS rendered depth；
- heldout frames 禁止进入 inpainting optimization。

### 输出

```text
background_completion/
├── edited_train_views/
├── unseen_masks/
├── depth/
├── gaussian_checkpoint/
├── provenance.json
└── source_hashes.json
```

## 8.3 S2-B：Inpaint360GS，强 baseline

WACV 2026、代码已发布。

适合原因：

- object-aware；
- multi-object；
- 360/unbounded scene；
- 2D segmentation→3D object labels；
- 利用 virtual camera views 补上下文；
- 比 front-facing 单物体 inpainting 更接近驾驶场景。

正式测试只在：

```text
scene-0230
high-support delete
boundary-support delete
```

先跑。

如果它依赖 COLMAP / 原始 3DGS schema 而改造成本过大，可以输出：

```text
adapter_blocked
```

但必须完成 source/schema audit。

## 8.4 S2-C：CoIn，2026 最新 SOTA 条件线

CoIn 2026-06 的论文设计：

```text
2D diffusion inpainting
→ Reference Adaptive GS
→ GS-guided feature warping
→ diffusion consistency
→ texture-enhanced 3D reconstruction
```

适合：

- removal；
- arbitrary mask；
- insertion；
- 2D↔3D 双向一致性。

但：

> 只有官方代码和权重公开且可审计时启动。

禁止为了“最新”从论文手写整套大型模型。

如果代码在执行时仍未公开：

```text
S2-C = audit_only / not_executable_publicly
```

## 8.5 S2 生成资产 provenance

每个新 Gaussian：

```json
{
  "provenance": "GENERATED_BACKGROUND",
  "generator": "3DGIC|Inpaint360GS|CoIn",
  "source_views": [],
  "mask_sha256": "",
  "model_revision": "",
  "confidence": 0.0
}
```

它永远不允许标成 LiDAR truth。

---

# 9. S3：NVIDIA Asset Harvester 动态 actor 完整资产

## 9.1 为什么它是 V3.2 最重要的工业 SOTA

当前 StreetGS RigidNodes 是：

```text
从真实轨迹中有限视角逐场景优化出的 actor Gaussian
```

它天然存在：

- 车背面缺失；
- 远端/遮挡面缺失；
- 反射和阴影烘焙；
- cross-scene reuse 差；
- 大 lateral edit 后新视角异常。

Asset Harvester 2026 的目标正是：

```text
one/few in-the-wild AV views
→ sparse-view-conditioned multiview diffusion
→ feed-forward Gaussian lifting
→ complete simulation-ready 3D Gaussian asset
```

官方当前 image-to-3D 路径标称约 16 GiB VRAM，单卡 3090 值得优先实跑。

## 9.2 输入构造

从当前资产：

```text
instance_token
→ raw camera observations
→ SAM2 masks
→ object crops
```

优先选：

- 高可见面积；
- 不同方位；
- 遮挡较少；
- 不使用 heldout evaluation frame。

只允许：

```text
1 / 2 / 4 observed views
```

做小型消融。

## 9.3 不强制走 NCore

如果官方 repo 的 NCore parser 与当前 nuScenes 不兼容：

优先走其官方 image-to-3D 输入：

```text
512×512 frame.jpeg
512×512 mask.png
```

若已知相机信息 adapter 容易接则使用；

否则使用官方 camera estimator，并显式记录：

```text
camera_source = estimated
```

不得把 estimated pose 写成 nuScenes GT。

## 9.4 StreetGS actor adapter

新增：

```text
motion_proj/worldsim_v32/asset_harvester_adapter.py
motion_proj/worldsim_v32/actor_asset_schema.py
scripts/run_worldsim_v32_s3_asset_harvester.py
scripts/import_worldsim_v32_actor_asset.py
configs/worldsim_v32/s3_asset_harvester_v1.yaml
```

输出统一：

```text
instance_token
→ canonical actor-local coordinates
→ means
→ scales
→ quats
→ SH / RGB
→ opacity
→ asset bounds
→ source views
→ generation provenance
```

## 9.5 关键坐标合同

必须明确：

```text
T_world_actor(t)
T_actor_asset
T_camera_world
```

生成 asset 不得直接使用任意坐标。

需要单元测试：

- actor local → world → camera → pixel；
- original track replay；
- lateral +1m；
- delete；
- reload exact；
- asset scale 单位；
- wlh 与 Gaussian bounds 一致性。

## 9.6 S3 实验

```text
A0：原 StreetGS actor
A1：Asset Harvester 1-view asset
A2：Asset Harvester 2-view asset
A4：Asset Harvester 4-view asset
```

只先跑：

```text
scene-0230 high-support actor
```

通过后再：

```text
boundary-support actor
```

## 9.7 不主张

Asset Harvester 背面是生成式补全：

```text
GENERATED_ACTOR
```

不是“恢复真实背面”。

主张应为：

> complete / reusable / simulation-ready generative actor asset

而不是 ground-truth reconstruction。

---

# 10. S4：NVIDIA Harmonizer 在线修复

## 10.1 定位

Harmonizer 不是 world-state 模型。

它只作用于：

```text
rendered RGB sequence
```

目标：

- novel-view artifact correction；
- lighting mismatch；
- inserted asset appearance；
- missing / unrealistic shadows；
- temporally consistent enhancement。

## 10.2 为什么比自己训练 diffusion 更合理

官方：

- CVPR 2026；
- 预训练 checkpoint 已公开；
- 建于 Cosmos 0.6B；
- 单步、temporal-conditioned；
- 设计目标就是 NuRec / robotics / AV simulation online enhancement。

因此 V3.2：

```text
pretrained inference only
```

不训练 Harmonizer。

## 10.3 输入

至少：

```text
G0 original render
G1 semantic-remove + inpaint render
G2 Asset Harvester lateral render
```

固定：

- frame order；
- camera identity；
- resize/crop rule；
- 800×450 → 官方要求尺寸的 pad/resize mapping；
- 输出再恢复原尺寸；
- 不使用 heldout target image 作为条件。

## 10.4 输出 provenance

```text
2D_HARMONIZED_RENDER
```

严禁写入 Gaussian checkpoint。

---

# 11. S5：Omni-3DEdit 多视角生成上界

## 11.1 定位

CVPR 2026 Highlight。

特点：

```text
multi-view source images
+ one edited reference view
→ one-pass multi-view consistent edited images
```

支持：

- remove；
- add；
- appearance/color change。

可选：

```text
edited multi-view
→ AnySplat
→ new 3D Gaussian
```

## 11.2 为什么不作为主生产线

它会重新生成 multi-view scene：

- actor registry 不天然保留；
- Gaussian identity 不天然保留；
- source actor token / trajectory 需要重新绑定；
- 更像强生成上界。

因此：

```text
S5 = upper bound / rethink
```

## 11.3 双 3090 使用

如果开 2×3090：

优先把双卡预算给 Omni-3DEdit inference，而不是给旧 StreetGS R1 DDP。

官方示例本身提供双 GPU torch.distributed inference。

流程：

```text
GPU0 + GPU1
→ Omni-3DEdit remove
→ 3/10/20 source views
→ multi-view outputs
```

如果单卡可以稳定运行：

也记录单卡 wall / VRAM。

---

# 12. V3.2 固定实验矩阵

## 12.1 开发场景

继续：

```text
scene-0230
```

## 12.2 确认场景

只有在开发场景选出候选后：

```text
scene-0242
scene-0255
```

不做所有 SOTA × 3 scene 全排列。

## 12.3 主链

```text
G0 = V3.1 D2 baseline / no A3 refine

G1 = G0 + SAM2 semantic mask
G2 = G1 + prior-guided Gaussian semantic lifting

G3a = G2 + 3DGIC background inpainting
G3b = G2 + Inpaint360GS background inpainting
G3c = G2 + CoIn (conditional)

G4 = G2 + Asset Harvester complete actor

G5 = selected G3 + G4

G6 = G5 + pretrained Harmonizer
```

生成上界：

```text
U0 = Omni-3DEdit
U1 = 100Editor (conditional)
```

## 12.4 不再跑

```text
old A3-R1/R2/R3/R4
```

除非未来有完全新的独立 hypothesis，不得在 V3.2 内复活。

---

# 13. 最小指标合同

V3.2 不是评测项目，只保留选择模型需要的最小指标。

## 13.1 S1 semantics

- per-view mask area；
- temporal mask stability；
- multi-view reprojection consistency；
- projected actor boundary distance；
- Gaussian positive / ambiguous / negative counts；
- manual QA panel。

## 13.2 S2 background

真实可验证区域：

- heldout observed PSNR / SSIM / LPIPS；
- depth consistency；
- cross-view consistency；
- temporal warp error。

完全未观测：

- 不报“准确率”；
- 报：
  - provenance；
  - view consistency；
  - artifact rate；
  - qualitative panel。

## 13.3 S3 actor asset

在真实曾观测视角：

- RGB LPIPS；
- silhouette IoU / boundary；
- scale / rigidity；
- camera-view consistency；
- lateral edit artifact。

生成背面：

- 只报 completeness / consistency；
- 不报 GT correctness。

## 13.4 S4 Harmonizer

- temporal consistency；
- non-target image drift；
- artifact/lighting/shadow QA；
- wall time；
- peak VRAM。

## 13.5 工程

- model/checkpoint bytes；
- per-stage wall；
- peak VRAM/RAM；
- cold load；
- render FPS；
- failure recovery；
- generated provenance coverage。

---

# 14. 生成来源必须进入 WorldState

新增 provenance enum：

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

每个 Gaussian / asset / rendered image 必须能追到：

```text
source method
source checkpoint
source views
generation model revision
mask hash
camera / pose source
```

这是 V3.2 与普通 demo 最大的工程区别。

---

# 15. 双 3090 资源策略

## 15.1 基本原则

```text
2×24 GiB != 48 GiB shared memory
```

不得因为有两张卡把单 GPU OOM 预算直接翻倍。

## 15.2 默认并行

### GPU0

```text
DriveStudio / StreetGS
render
Gaussian adapter
3D evaluation
```

### GPU1

```text
SAM2
Asset Harvester
3D inpainting diffusion
Harmonizer
```

通过文件 sidecar 交换，不共享进程。

这样能够：

- 减少环境冲突；
- 减少 GPU memory fragmentation；
- 同时生产 SAM mask / image-to-3D asset 与 3D render；
- 不需要改 StreetGS 为 DDP。

## 15.3 双卡专用任务

只有官方上游明确支持时使用两卡：

```text
Omni-3DEdit
部分 diffusion model inference
```

## 15.4 不建议

- 旧 A3 local refinement DDP；
- 跨 GPU 拆单个 Gaussian tensor；
- 自己写 model parallel 只为把 14 GiB heldout 路径拆卡；
- 为了利用两卡重训已冻结 A1/A2。

---

# 16. 资源和停机合同

每个新 third-party GPU task：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
```

## 停止

- 任意 GPU CUDA OOM；
- cgroup 连续两次 >= 90%；
- OOM/OOM-kill 增加；
- disk free < 20 GiB；
- 需要静默降低 camera 数 / resolution / view count；
- 第三方输出 schema 与论文/官方 README 不一致；
- 必须使用未知来源 checkpoint；
- 需要修改 V3.1 canonical asset 才能继续。

## 双卡

每个 run 必须记录：

```text
CUDA_VISIBLE_DEVICES
GPU0 peak
GPU1 peak
P2P/NCCL state
per-process PID/PGID
```

---

# 17. 第三方模型的决策优先级

如果资源有限，执行顺序严格为：

```text
1. SAM2 + semantic lift
2. Asset Harvester
3. 3DGIC
4. Inpaint360GS
5. Harmonizer
6. Omni-3DEdit
7. CoIn conditional
8. MV-SAM conditional
9. 100Editor conditional
```

原因：

- 前 5 个和当前 WorldSim 资产链直接兼容；
- Omni-3DEdit 很强，但更像重新生成 multi-view；
- CoIn 最新但若无官方代码，不值得本项目从零复现；
- MV-SAM 当前若官方代码不可用，只做方法审计。

---

# 18. S1 判定

S1 可以 `done`，即使没有最终质量提升，只要：

- SAM2 train-view mask 完整生成；
- token / prompt / camera / timestamp provenance 完整；
- multi-view semantic lift 跑通；
- Gaussian semantic posterior 可 round-trip；
- 原 `instance_token → RigidNodes` 映射不被破坏；
- delete/lateral 使用 semantic mask 后真实渲染非空；
- 没有 heldout leakage。

候选：

```text
S1* = M0 / M1 / M2 / M3
```

若 prior-guided lift 没有明显改善，允许回退 SAM2 image mask only。

## 18.1 S1 收口（2026-08-10）

- r5=`20260810T093248Z__s1-semantic-lift-s0-r5` 后续发现 high-support 的 dataset ID/token/rigid index
  三元 identity 错配，现只保留为 identity-invalid 历史运行；
- config SHA-256=`ecb6c2bc6f68376c9cd81e3e2a362a30506edfba5772226ad27125f0dcbad706`，prompt
  manifest SHA-256=`771817828acb689e8cab19c4f4c368d8ead24c0d1c154bd1d8bcc283a9b6c071`；
- SAM2 双向传播产生 263 个 train-only mask，212 个通过冻结 QC，51 个 fail-closed，heldout leaks=`0`；
- high-support：`CORE_POSITIVE=4,525`、`SEMANTIC_POSITIVE=3,927`、`AMBIGUOUS=6,275`；
- boundary-support：`CORE_POSITIVE=3,728`、`SEMANTIC_POSITIVE=21,043`、`AMBIGUOUS=8,202`；
- posterior sidecar 保存 `num_positive_views / num_negative_views / weighted_positive / weighted_total /
  semantic_score / depth_consistency_rate / boundary_score` 并通过 round-trip finalizer；
- D2 checkpoint 前后 SHA-256 均为
  `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c`；
- 6 个 `original / delete / lateral` smoke 均非空且每个 actor 三种输出 SHA 互异；
- SAM2 wall=`58.34s`、peak allocated=`1,895,410,176 bytes`；semantic lift wall=`770.73s`、peak
  allocated=`15,726,013,440 bytes`，单卡 RTX 3090 通过且无 OOM；
- 修正配置 v3 把 high-support dataset ID 从 `5` 改为与 token `af663…` 对应的 `13`，并新增
  fail-closed identity 合同；canonical r6=`20260810T101739Z__s1-semantic-lift-s0-r6` 已完成 identity-aware
  finalizer：398 masks=`334 accepted / 64 rejected`、heldout leaks=`0`，6 个真实 smoke 通过；S1=`done`，S3 已解锁。

---

# 19. S2 判定

至少一个：

```text
3DGIC
Inpaint360GS
```

必须完成真实 scene-0230 delete。

优先看：

- 原位置 hole 是否真正填充；
- 其他相机看到的已有背景是否被破坏；
- multi-view consistency；
- 是否需要全场景重训；
- wall / VRAM。

如果只有生成式 visual improvement：

仍可选为：

```text
GENERATED_BACKGROUND visual layer
```

但不得宣称 geometry-grounded world state。

## 19.1 canonical 判定（2026-08-10）

- 3DGIC 原生 Gaussian-Grouping/Inria checkpoint schema 与当前 StreetGS 不兼容；本项目完成的是官方
  depth-guided cross-view 原理、外部 2D inpainter 和 RGB-D unprojection 的显式 StreetGS 适配，不冒充
  untouched upstream run；Inpaint360GS 因要求独立 COLMAP、训练、交互 segmentation/track 与 LaMa 链保持
  `adapter_blocked`，但其 source/license 审计已完成；
- r1 因 boundary 原支持帧几何重叠不足而 `rejected`；只用 train-only exhaustive audit 冻结 frame `24..29`；
- r2 方法链完成，但持久化未观测 Telea 几何造成 held-out PSNR/SSIM 平均退化
  `0.495842 dB / 0.007160`，候选不选；
- canonical r3=`20260810T121829Z__s2-3dgic-adapted-s0-r3`：完整 2D unseen completion 及 provenance 保留，
  high-support checkpoint 只持久化 cross-view observed geometry，小 boundary 目标保留完整补全；
- Background 新增 `1,896` 行（`1,205,164 → 1,207,060`），权威 sidecar 标记
  `GENERATED_BACKGROUND`；候选 strict reload，旧行与 D2 source SHA exact；
- high/boundary effect=`9,928 / 176` pixels，outside L1=`0.042503 / 0.005122`；四路 held-out
  PSNR/SSIM/LPIPS delta=`-0.022958 dB / -0.000528 / +0.000301`，全部通过冻结门；
- selected checkpoint SHA-256=`3d6e13d47291f5b5949ff3adf5598b6e0cffb930c4cbff2200c6e708d82e6e0f`；
  unseen 仍只声明生成资产一致性/provenance，不声明 GT accuracy 或 geometry-grounded world state。S2=`done`。

---

# 20. S3 判定

Asset Harvester 至少完成：

```text
scene-0230 high-support
1-view / 2-view
```

并接回现有 trajectory。

必须真实执行：

```text
original
lateral +1m
delete
```

如果完整 asset：

- 在新视角明显更稳定；
- 不破坏 actor identity；
- adapter 坐标正确；
- 单 3090 资源可控；

则：

```text
S3*=AssetHarvester
```

否则保留原 StreetGS actor。

---

# 21. S4 判定

Harmonizer 只做 final render enhancer。

若：

- temporal artifact 降低；
- lighting/shadow 更自然；
- non-target drift 可接受；
- 单 GPU 延迟可接受；

则加入最终视觉输出链。

否则：

```text
Harmonizer = optional diagnostic
```

不影响三维资产终态。

---

# 22. 最终可能的 V3.2 生产链

理想：

```text
D2 3DGS
→ SAM2 + prior-guided 3D semantic lifting
→ 3DGIC/Inpaint360GS background completion
→ Asset Harvester complete dynamic actor
→ original/lateral/delete WorldState render
→ Harmonizer
→ P2 mixed storage
→ P3 static chunk + actor package
```

如果生成式背景不适合写回 3D：

```text
D2
→ semantic lift
→ Asset Harvester
→ 3D render
→ background/harmonizer 2D enhancement
```

也属于合法结果。

## 22.1 canonical 最终裁决（2026-08-10）

实际选定链为：

```text
S1 extended semantic sidecars
→ S2 generated-background scene
→ P2-style mixed storage
→ S3 high-support GENERATED_ACTOR registry override
→ original / lateral / delete WorldState render
→ P3-style exact static/actor chunk package
```

S4 non-temporal 输出因 remove 区域语义重引入而只保留 optional diagnostic，不串入 production chain；
S4 temporal 与 S5 因 gated 权重/许可证等外部门禁保持 `blocked`，不影响 R0 `done`。

R0 canonical run=
`/root/autodl-tmp/runs/worldsim_v32/WS-V32-R0-INTEGRATION-01/20260810T134658Z__r0-final-integration-s0-r1`。
混合精度 checkpoint SHA-256=`6d4e4c489f53bf4e7de3f5c405ec37dc63d3f79155aad5237fe175ce0fcd7e5d`；
chunk manifest SHA-256=`af7b402e0b171b11f8c22e4123002f4f844db746ea72f53b77c3de878bf0947d`；
summary SHA-256=`40624cbc79a004e9e07e57b00cebc535b900297a10f0d070fb4e9305a5f7937a`。

---

# 23. 新增文件建议

```text
configs/worldsim_v32/
├── s0_sources_v1.yaml
├── s1_sam2_v1.yaml
├── s1_gaussian_lift_v1.yaml
├── s2_3dgic_v1.yaml
├── s2_inpaint360gs_v1.yaml
├── s2_coin_audit_v1.yaml
├── s3_asset_harvester_v1.yaml
├── s4_harmonizer_v1.yaml
├── s5_omni3dedit_v1.yaml
└── r0_integration_v1.yaml

motion_proj/worldsim_v32/
├── __init__.py
├── semantic_schema.py
├── sam2_temporal.py
├── sam_boundary_refine.py
├── gaussian_semantic_lift.py
├── multiview_label_reassign.py
├── background_inpainting.py
├── asset_harvester_adapter.py
├── actor_asset_schema.py
├── harmonizer_adapter.py
├── multiview_editor_adapter.py
├── generated_provenance.py
└── integration.py

scripts/
├── audit_worldsim_v32_sources.py
├── build_worldsim_v32_sam_masks.py
├── lift_worldsim_v32_semantics.py
├── run_worldsim_v32_s2_3dgic.py
├── run_worldsim_v32_s2_inpaint360gs.py
├── audit_worldsim_v32_s2_coin.py
├── run_worldsim_v32_s3_asset_harvester.py
├── import_worldsim_v32_actor_asset.py
├── run_worldsim_v32_s4_harmonizer.py
├── run_worldsim_v32_s5_omni3dedit.py
└── finalize_worldsim_v32.py

tests/
├── test_worldsim_v32_semantic_lift.py
├── test_worldsim_v32_mask_provenance.py
├── test_worldsim_v32_inpainting_adapter.py
├── test_worldsim_v32_asset_harvester_adapter.py
├── test_worldsim_v32_actor_coordinates.py
├── test_worldsim_v32_generated_provenance.py
└── test_worldsim_v32_integration.py
```

---

# 24. Commit 计划

建议：

```text
docs(worldsim): 启动 V3.2 语义资产修复路线
research(seg): 接入 SAM2 时序掩码与语义 provenance
feat(seg): 实现多视角 Gaussian semantic lifting
research(inpaint): 接入 3DGIC 驾驶背景补全
research(inpaint): 审计并接入 Inpaint360GS
research(asset): 接入 NVIDIA Asset Harvester
feat(asset): 将完整生成 actor 绑定到 WorldState
research(render): 接入 NVIDIA Harmonizer 推理
research(edit): 加入 Omni-3DEdit 多视角上界
research(worldsim): 完成 V3.2 集成与模型链收口
```

每个 third-party 方法一个独立 commit，不混。

---

# 25. Codex Agent 首轮执行提示词

```text
执行 docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md。

这是 V3.1 完成后的新研究计划。V3.1 的 A1/A2/A3/A4/F0/R0 terminal 和 canonical assets 全部保持历史只读；
不得恢复旧 A3 R1，也不得通过增加 step、提高资源门或双卡 DDP 去挽救旧 local refinement。

V3.2 当前只执行：
WS-V32-S0-ROUTE-AND-SOTA-AUDIT-01

首轮动作：

1. 读取：
   - AGENTS.md
   - docs/RESEARCH_STATUS.md
   - docs/RESEARCH_FAILURES.md
   - docs/EXPERIMENTS.md
   - docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md
   - docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md

2. 核对 V3.1 canonical：
   - A2-D2 checkpoint SHA = 1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c
   - A3*=R0-off
   - P2/P3 production assets immutable
   - R0 none_plan_complete

3. 建立新分支：
   research/worldsim-v3.2-semantic-repair

4. 不修改旧 worldsim_v3 模块语义。
   新代码优先进入：
   motion_proj/worldsim_v32/
   configs/worldsim_v32/

5. 对以下官方 source 做只读审计并固定 commit/license/weights/hardware：
   - facebookresearch/sam2
   - Jumpat/SegAnyGAussians
   - Gaussian Grouping official repo
   - Robust Prior-Guided / SegmentAnythingin3D
   - peterjohnsonhuang/3dgic
   - dfki-av/Inpaint360GS
   - Aswhalefall/VISTA
   - NVIDIA/asset-harvester
   - NVIDIA/harmonizer
   - mt-cly/Omni3DEdit
   - CoIn 官方代码若存在
   - MV-SAM 官方代码若存在

6. 生成：
   configs/worldsim_v32/s0_sources_v1.yaml
   docs/WS_V32_S0_SOTA_AUDIT.md

7. 检查当前机器：
   - GPU 数量；
   - 每卡 VRAM；
   - driver/CUDA；
   - cgroup；
   - 磁盘；
   - Docker；
   - HF 访问；
   - GitHub 访问。

8. S0 不安装大依赖、不下载大权重、不启动模型推理。
   只允许 source/license/schema/hardware audit。

9. S0 完成后只解锁 S1：
   SAM2 temporal mask + Gaussian semantic lifting。

10. 不得直接跳 Asset Harvester / Harmonizer / Omni-3DEdit，
    不得一次启动所有第三方环境。

11. 每个状态更新同步：
    - V3.2 plan
    - RESEARCH_STATUS.md
    - EXPERIMENTS.md
    - RESEARCH_FAILURES.md
    - README

12. commit：
    docs(worldsim): 启动 V3.2 语义资产修复路线
```

---

# 26. S1 Coding Agent 提示词

S0=`done` 后：

```text
只执行 WS-V32-S1-SEMANTIC-LIFT-01。

目标：
用 SAM2 时序 mask + 多视角 Gaussian semantic lifting 替换旧 A3 的 model-self-difference affected mask，
但不修改 actor identity 与 D2 checkpoint。

步骤：
1. 固定 scene-0230、两个 actor role、train/heldout split。
2. 用 nuScenes 3D box projection 自动产生 SAM2 box prompt。
3. CAM_FRONT / FRONT_LEFT / FRONT_RIGHT 各自做 temporal propagation。
4. heldout frame 不进入 prompt 选择、mask 调参或 3D lift。
5. 保存 mask logits/binary/prompt/source hash。
6. 实现 alpha contribution + visibility + depth consistency 的 Gaussian semantic posterior。
7. actor registry 作为 hard identity prior；SAM 只细化边界/遗漏，不重写 instance_token。
8. 输出 CORE_POSITIVE / SEMANTIC_POSITIVE / AMBIGUOUS / NEGATIVE。
9. module-off 必须 bitwise 不修改 D2 checkpoint。
10. 只生成 sidecar semantic assets，不写入 source checkpoint。
11. 做 original/delete/lateral smoke。
12. freeze S1* 后才进入 S2/S3。
```

---

# 27. 推荐资源使用结论

## 只有单卡 3090

优先：

```text
SAM2
→ Gaussian semantic lift
→ Asset Harvester
→ 3DGIC / Inpaint360GS
→ Harmonizer preflight
```

Asset Harvester 官方 image-to-3D 约 16 GiB VRAM，优先级很高。

## 有双 3090

最佳使用方式：

```text
GPU0：StreetGS / render / 3D adapter
GPU1：SAM2 / Asset Harvester / inpainting / Harmonizer
```

需要 Omni-3DEdit 时：

```text
GPU0 + GPU1：按其官方 torch.distributed inference
```

不建议：

```text
GPU0 + GPU1：
旧 A3 R1 DDP
```

因为它只能解决资源限制，不能解决 S-C unsupported 和方法 tradeoff。

---

# 28. V3.2 最终成功标准

V3.2 不要求所有 SOTA 都赢。

满足以下即可认为项目升级成功：

1. SAM/semantic lifting 明确改善 actor selection / boundary 或至少形成稳定可用语义 sidecar；
2. 3DGIC/Inpaint360GS 中至少一个能真实修复 actor 删除后的背景；
3. Asset Harvester 至少完成一个驾驶 actor 的 1/2-view → complete Gaussian asset → lateral render；
4. 生成内容 provenance 与真实观测严格分离；
5. Harmonizer 或其他 2D enhancement 作为独立视觉层；
6. 旧 D2/P2/P3 资产链可继续使用或重新 package；
7. 单卡与双卡资源边界有明确结论；
8. 即使某 SOTA 在当前数据/硬件失败，也保留可复现负结果。

合法终态：

```text
semantic_lift_supported
semantic_lift_no_gain
background_inpaint_supported
background_inpaint_visual_only
asset_harvester_supported
asset_harvester_adapter_blocked
harmonizer_supported
harmonizer_resource_blocked
multiview_upper_bound_supported
v32_partial_supported
v32_engineering_blocked
v32_production_chain_selected
```

不得统一写成“项目失败”。

---

# 29. 一手参考

## Semantic / SAM

- SAM 2: Segment Anything in Images and Videos
  Meta AI / official code: `facebookresearch/sam2`
- Gaussian Grouping: Segment and Edit Anything in 3D Scenes
  ECCV 2024
- SAGA: Segment Any 3D Gaussians
  AAAI 2025, official: `Jumpat/SegAnyGAussians`
- Robust Prior-Guided Segmentation for Editable 3D Gaussian Splatting
  ICIP 2026, arXiv:2605.16065
- MV-SAM: Multi-view Promptable Segmentation using Pointmap Guidance
  2026, arXiv:2601.17866

## 3D Inpainting / Editing

- 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency (3DGIC)
  CVPR 2025, official code: `peterjohnsonhuang/3dgic`
- Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
  WACV 2026, official code: `dfki-av/Inpaint360GS`
- Visibility-Uncertainty-guided 3D Gaussian Inpainting via Scene Conceptual Learning (VISTA)
  2025, official code: `Aswhalefall/VISTA`
- CoIn: Comprehensive 2D-3D Inpainting with Gaussian Splatting Guidance
  arXiv:2606.27584
- Omni-3DEdit: Generalized Versatile 3D Editing in One-Pass
  CVPR 2026 Highlight, official code: `mt-cly/Omni3DEdit`
- 100Editor: 100+ Views per Batch and Minute-Scale View-Consistent 3D Editing
  CVPR 2026

## Autonomous Driving / WorldSim

- Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation
  NVIDIA 2026, official code: `NVIDIA/asset-harvester`
- Harmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer
  CVPR 2026, official code: `NVIDIA/harmonizer`
- R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation
  official code: `zenseact/R3D2`

---

# 30. 最终路线摘要

```text
V3.1：
A1 off preferred
→ A2 D2 boundary-priority
→ A3 local refine rejected
→ A4 packaging done

V3.2：
D2 source
→ SAM2 temporal mask
→ prior-guided multi-view Gaussian semantic lifting
→ {
     background: 3DGIC / Inpaint360GS / CoIn conditional
     actor: NVIDIA Asset Harvester
   }
→ WorldState original/lateral/delete/insert
→ NVIDIA Harmonizer
→ P2 mixed storage
→ P3 static/actor package

强生成上界：
Omni-3DEdit
```

**V3.2 的核心不再是“继续优化已有 Gaussian”，而是把工业 WorldSim 真正需要的语义、完整资产、未观测区域生成与在线视觉修复接入现有可编辑 3DGS 资产链。**
