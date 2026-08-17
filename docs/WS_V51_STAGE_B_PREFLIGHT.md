# WorldSim V5.1 Stage B 独立授权前预检

> **Task**：`WS-V51-M1-B-LUDVIG-UPLIFT-01`
> **预检日期**：2026-08-17
> **仓库基线**：`3d332623eed5189c4c68464dfbda5922f18f3598`
> **状态**：`pending/locked`；本文只记录 source、资产、资源、协议与实现边界，不构成 Stage B 执行授权。

## 1. 结论

Stage B 尚不能直接启动。Stage A 已无新 unary survivor，当前只保留 `U2/B3`；但 normative plan 同时存在两条
不兼容的解锁规则：

1. §10.8 规定“所有 Stage A arm 都失败”时保留 U1/U2 并进入 Stage B；
2. 附录“八、Stage A 后如何解锁”又规定只有 Stage A candidate 通过 S 才允许进入 Stage B。

当前事实恰好落在冲突分支：A1–A4 全部 rejected、fallback=`U2/B3`。因此不得由执行者静默选择其中一条规则。
独立授权必须明确选择：

- 允许 `U2/B3` 作为 fallback seed 进入 Stage B；或
- 按“candidate 必须通过 S”的严格规则关闭 M1，不执行 Stage B。

该协议问题登记为 `V51-F11`。P0 已按 SHA 冻结 normative plan，所以本文与短执行入口只记录冲突，不回写长计划；
该工程边界见 `V51-F13`。Stage B 解锁前还要补齐 DINOv2 资产与 24GB 单卡执行合同，见 `V51-F12`。

## 2. 官方实现冻结

预检只读取官方公开仓库，没有 clone、vendor 或执行上游代码：

- LUDVIG upstream：`https://github.com/naver/ludvig`
- 预检时 `main`：`4461fc515439bb498a75d71738a1e73cf7a452ed`
- 论文：`https://openaccess.thecvf.com/content/ICCV2025/html/Marrie_LUDVIG_Learning-Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_ICCV_2025_paper.html`
- 关键语义：`utils/solver.py::uplifting()` 用 renderer transpose 累积每 Gaussian feature 与 weight，最后执行
  `features_3d /= weights + 1e-8`；可选 pruning 不进入 V5.1，因为 base Gaussian 必须 immutable。
- 第一版 backbone 应冻结为官方 DINOv2 ViT-g/14 registers 路线；官方 demo config 使用 `40` 维 PCA。不得把
  任意 DINOv2 小模型、SAM probability 或 CLIP feature 直接称为 faithful port。
- 官方 DINO pipeline 包含 dataset-level standardization/PCA 与可选随机 subsampling。V5.1 必须在任何 feature
  metric 前冻结 fit population、view 顺序、seed、subsample 数量和 PCA 维数；scaler/PCA 只能在允许的 development
  数据上拟合，并作为 immutable sidecar 应用于后续场景。
- 上游 license 含非商业研究和再分发约束。第一版优先按论文/源码语义做最小 clean-room 风格实现并记录 provenance；
  若复制任何上游代码，必须先保留完整 license/NOTICE 并单独审计许可边界。

## 3. 本机资产与环境事实

2026-08-17 只读审计结果：

| 项目 | 事实 | 影响 |
|---|---|---|
| GPU | RTX 3090，`24576 MiB`；预检时 free=`24126 MiB` | official README 的测试平台为 A6000 48GB，24GB feasibility 未证明 |
| 容器资源 | cgroup memory=`96636764160 bytes`（90 GiB），CPU quota=`14 cores` | CPU PCA/sidecar 可分阶段执行，但必须遵守容器限额；`free` 的宿主机总量不是可用额度 |
| `motionproj` | Python 3.10.20；Torch 2.4.1+cu121；timm/transformers/sklearn/einops/xformers 可用 | 适合 DINO extraction 独立进程，不代表 checkpoint 已就绪 |
| `drivestudio` | Python 3.9.25；Torch 2.1.2+cu118；timm/transformers/sklearn 可用 | renderer 继续绑定该环境；不要在运行中临时升级依赖 |
| DINOv2 source | 仅有 Depth-Anything-V2 内部 DINOv2 模块 | 不是 LUDVIG 官方 ViT-g/14 registers 实现/权重冻结 |
| DINOv2 checkpoint | torch/HuggingFace cache 中未找到 ViT-g/14 registers checkpoint | 资产下载、SHA-256、license/model-card 冻结是启动前硬门 |

此前 Stage A 单个 unary materialization 的显存峰值曾达到约 `20–22 GiB`。因此禁止让 DINO ViT-g extraction 与
DriveStudio renderer 在同一 GPU 进程或同卡并发常驻；“依赖已安装”不能写成 Stage B resource gate 已通过。

## 4. 与现有代码的接口边界

可以复用但不能直接等同 LUDVIG 的现有能力：

- `motion_proj/worldsim_v5/renderer_intersections.py` 已按 alpha compositing 顺序恢复逐 pixel 的 Gaussian
  contribution weight；它可作为 `W` 的本地候选来源。
- `motion_proj/worldsim_v5/observation_aggregation.py` 当前把同一 view 的 intersection 收缩成每 Gaussian 一条
  observation，并将 visibility 变为 `1-exp(-mass)`。该饱和 visibility 是 Bayesian evidence 合同，不能替代
  B1 的线性 `W^T F_2D` 权重。
- B1 必须在 pixel feature 被 view-level aggregation 丢失之前计算 contribution-weighted feature numerator 和
  denominator；不得从现有 B3 posterior、SAM union probability 或 per-view centroid 反推 dense feature uplift。
- base checkpoint、Gaussian 行序、renderer state 和 contribution operator 必须 immutable；只新增 feature sidecar。

## 5. 获得授权后的最小落地序列

### B-F0：第二轮 freeze-only

- 明确选择冲突分支，并冻结 `U2/B3` fallback 身份；
- 用显式 supersession/migration 更新 P0 plan binding；不得直接改写已冻结 plan 后继续沿用旧 SHA；
- 冻结 H/S/C 的 Stage B 角色、禁止读取项、one-shot gate 和 `failure_ledger_refs`；
- 冻结上游 commit、DINO model ID/checkpoint SHA、license、preprocess、PCA fit population/seed/dimension；
- 不读取 DINO feature quality，不启动模型或 renderer。

### B-F1：operator-only parity

- 实现独立 `renderer_transpose` 核心，先用 tiny synthetic tensor 验证
  `sum(w*f)/(sum(w)+1e-8)`、重复 index、零 denominator、chunk-order invariance 和 dtype；
- 用一个已授权 H view 做 contribution inventory/denominator smoke，仍不读取 same-actor 或 background quality；
- checkpoint 前后 SHA exact，峰值显存与 sidecar schema 必须落盘。

### B-F2：离线 DINO feature sidecar

- DINO extraction 在独立进程中运行，写 image-feature sidecar 后完全释放 GPU；
- scaler/PCA 在冻结 H population 上 fit，一次持久化，S/C 只 transform；
- feature sidecar 记录 model/checkpoint、preprocess、view key、shape/dtype、PCA、source image SHA 和内容 SHA。

### B-F3：learning-free uplift 与 one-shot gate

- DriveStudio renderer 进程只消费冻结的 image-feature sidecar；不加载 DINO；
- 首先比较 B0 current intersection lift 与 B1 normalized transpose；
- 同表报告 repeatability、same-actor cosine、actor-background margin、coverage、walltime、sidecar bytes、peak VRAM；
- 若未形成预注册的稳定 `same-actor > actor-background` separation，则 Stage B rejected，Graph 保持锁定。

## 6. 未冻结项

以下都必须在第二轮 freeze-only commit 中精确化，不能在看到 feature quality 后补写：

- Stage B H/S/C 的 scene 与 view denominator；
- DINOv2 ViT-g/14 registers 的官方下载源、checkpoint SHA-256 与本地 immutable path；
- 输入 resize/crop、patch grid、last-intermediate-layer 选择、mixed precision 与输出 dtype；
- PCA fit population、最大 sample 数、seed、whitening 与 40-D 是否原样继承；
- B0 的精确定义，以及 B0/B1 的 matched coverage floor；
- same-actor pair、actor-background pair、abstain 与 scene-balanced aggregation 的精确定义；
- separation 的数值阈值、最小场景数和 S→C 解锁条件；
- 24GB 的 batch/chunk ceiling、OOM fail-closed 与 checkpoint/sidecar cleanup 合同。

在这些项被冻结并得到独立授权前，不创建 method run、不下载权重、不读 C/validation/test/KITTI quality，也不实现
Graph、SigLIP/CLIP/SAM backbone search 或参数搜索。
