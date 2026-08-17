# WorldSim V5.1 Stage B freeze-only 提案

> **Task**：`WS-V51-M1-B-LUDVIG-UPLIFT-01`
> **状态**：`pending / draft_freeze_only_not_authorized`
> **机器可读提案**：`configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`
> **硬边界**：本提案不是独立授权，不下载权重、不实现方法、不读取质量、不启动 run。

## 1. 提案目的

Stage A 已冻结 `U2/B3`，但 Stage B 的 fallback 解锁仍等待用户裁决。为了让授权后的第一步只剩“确认并冻结”，
本提案预先补齐能由公开源码、既有 immutable checkpoint 和只读 metadata 确定的内容：

- faithful DINOv2 模型、输入、PCA 与 checkpoint identity；
- H/S/C 的 scene、frame、camera、checkpoint 与完整 denominator；
- B0/B1 的 matched operator 公式；
- sidecar schema、24GB 单卡执行方式、指标和 result-blind gate；
- proxy label、许可、随机性和质量锁。

未获独立授权前，所有字段都是 proposal，不得被 runner 当作 executable config。

## 2. 官方机制与资产身份

### 2.1 LUDVIG 语义

- upstream=`https://github.com/naver/ludvig`，commit=`4461fc515439bb498a75d71738a1e73cf7a452ed`；
- `utils/solver.py::uplifting()` 对每个 view 用 renderer weight 累加 feature numerator/weight，最后除以
  `weights+1e-8`；V5.1 禁用 optional pruning，保持 base Gaussian immutable；
- demo 第一 backbone 是 DINOv2 ViT-g/14 registers，PCA=`40`；
- LUDVIG 总体 license 是 non-commercial，并要求保留 copyright/license。V5.1 不 vendor LUDVIG 源码，只按论文与
  公开 operator 语义实现本地稀疏 transpose，同时在文档和模块中保留 provenance。

### 2.2 DINOv2 身份

- official Meta repo=`https://github.com/facebookresearch/dinov2`，预检 main=
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`，hub entrypoint=`dinov2_vitg14_reg`；
- direct checkpoint URL=
  `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth`；
- HEAD metadata：bytes=`4,546,140,349`，Last-Modified=`2023-10-27T10:37:55Z`，S3 version=
  `T_6GA9ukHl7r7daSEhzWpsymxbGdAvFR`，ETag=`3d1b1c4501eac45d83af24b811e3bea9-542`；
- 该 ETag 是 542-part multipart ETag，不是 SHA-256。授权下载后必须计算完整 SHA-256，再形成 asset freeze commit；
  HEAD/size/ETag 只能证明 proposed remote object，不能替代内容哈希。

## 3. Feature preprocessing freeze proposal

LUDVIG 当前实现的关键语义：

1. 图像按 ImageNet mean/std=`[0.485,0.456,0.406]/[0.229,0.224,0.225]` 标准化；
2. 最大边不超过 1600，然后宽高向下取 patch-14 整数倍；
3. ViT-g/14 registers 在 fp16 autocast 下取最近 4 个 intermediate layers 的最后一个；
4. raw 1536-D feature 转 float32，做 dataset-level 标准化和 40-D PCA；
5. 40-D patch grid bilinear 上采样后进入 renderer transpose。

本数据的冻结输入全部是 `1600×900`，所以 model input=`1596×896`，patch grid=`114×64`，每 view=`7,296`
patch。H evidence 固定为 `3 scenes × 5 frames × 3 cameras=45 views`，PCA population=
`45×7,296=328,320`，低于上游 `500,000` subsample cap，因此不执行随机 subsample。

但上游 `PCA(n_components=40)` 没有固定 `random_state`；对该矩阵 sklearn 会走 randomized solver。上游 GPU 分支还用
PyTorch `std`（correction=1），CPU 分支用 NumPy `std`（correction=0），直接切换实现会改变 feature。提案固定：

- standardization std correction=`1`；
- PCA solver=`randomized`、random_state=`20260814`、whiten=`false`；
- fit 只用 H evidence，S/C/evaluation view 只 transform，禁止 refit；
- 持久化 mean/std/PCA mean/components/singular values 及整体 SHA。

这是 reproducibility hardening，不是结果驱动的 backbone/PCA 搜索，详见 `V51-F14`。

## 4. 数据与 denominator

所有 scene 沿用 `development_roles_v1.yaml`，只用三前向相机 `0/1/2`：

- uplift evidence frames=`0/40/80/120/160`，均属于 train remainder 0；
- heldout evaluation frames=`2/42/82/122/162`，属于 development remainder 2；
- remainder 4 继续禁止读取；
- 每 scene=`15 uplift + 15 evaluation=30` images。

只读 header audit 已确认 8/8 scene 都是 `30/30` 文件、统一 `1600×900`。checkpoint 均来自 V5 formal30k r027–r034，
8/8 status=`done` 且此前已逐 bytes/SHA 审计。角色和 Gaussian metadata 如下：

| Role | Scene | Index | Background | Rigid | 30 images |
|---|---|---:|---:|---:|---:|
| H | 0471 | 382 | 809,902 | 49,711 | exact |
| H | 1087 | 827 | 930,979 | 244 | exact |
| H | 0379 | 296 | 1,186,659 | 632 | exact |
| S | 0998 | 756 | 1,578,331 | 79,068 | exact |
| S | 0359 | 276 | 1,387,860 | 9,896 | exact |
| C | 0875 | 663 | 889,059 | 59,915 | exact |
| C | 0535 | 425 | 1,162,092 | 21,981 | exact |
| C | 0436 | 350 | 1,140,699 | 151,229 | exact |

1087/0379 的 Rigid denominator 很小，不能为了凑稳定性删场景或降低 eligibility 后重算。每场需保留
`EVALUABLE` 或 `ABSTAIN_NO_ELIGIBLE_ACTOR`；H 至少 2/3、S 必须 2/2、C 至少 2/3 evaluable。

授权后的 freeze-only commit 还需给这 240 个 image 文件逐 SHA-256；本轮没有读取像素或 DINO feature quality。

## 5. Matched B0/B1 operator

对 Gaussian `g`、view `v`、pixel `p`，现有 renderer 提供真实 alpha compositing contribution `w_gvp`。
共同 support 沿用 V5：intersection contribution>=`1e-4`、Gaussian-view mass>=`1e-3`。

先定义：

\[
N_{gv}=\sum_p w_{gvp}f_{vp},\qquad m_{gv}=\sum_p w_{gvp}.
\]

### B0：view-saturated current intersection lift

\[
f_{gv}=N_{gv}/(m_{gv}+10^{-8}),\quad
a_{gv}=1-e^{-m_{gv}},\quad
F_g^{B0}=\frac{\sum_v a_{gv}f_{gv}}{\sum_v a_{gv}+10^{-8}}.
\]

这保留 V5 的“同 view 不把 pixel area 当独立票、visibility 饱和后再跨 view 聚合”语义。

### B1：normalized renderer transpose

\[
F_g^{B1}=\frac{\sum_v N_{gv}}{\sum_v m_{gv}+10^{-8}}.
\]

B1 保留真实 contribution mass，不做 view saturation。两臂使用同一 DINO/PCA、image、checkpoint、intersection support
和 Gaussian row order；唯一差别是跨 view normalization。base checkpoint 前后 SHA 必须 exact。

## 6. Sidecar 与 24GB 执行合同

官方 extractor 把全部 raw feature 预分配在 GPU；本场景 H population 若为 float32 约 2.0GB，叠加 1.1B ViT-g 和
renderer 不适合 24GB 3090。提案使用等价的分阶段实现：

1. DINO 单进程逐 image 推理，H raw 1536-D 写 CPU memmap；
2. H 一次 fit PCA，生成每 view `40×64×114 float32` patch-grid sidecar；
3. S/C raw feature 逐 view transform 后立即释放，不积累 1536-D 全集；
4. 完全退出 DINO 进程；
5. DriveStudio renderer 单进程逐 scene/view 输出 sparse contribution；CPU 端分块完成 B0/B1 scatter-add；
6. Gaussian sidecar 固定 `N_gaussian×40 float32`，另存 weight/coverage/identity metadata。

不落盘 `40×900×1600` dense map；按 pixel center lazy bilinear sampling patch grid，但必须先用 tiny tensor 与 dense
`torch.interpolate(..., align_corners=False)` 做数值 parity。DINO smoke GPU peak 上限=`22,528 MiB`，container peak
上限=`80 GiB`。OOM 时 fail closed，不能换 ViT-L/B、降低图像分辨率或改 PCA 后继续称 faithful port。

## 7. Metrics 与 proxy 边界

预注册指标：

- same-Gaussian cross-view cosine repeatability；
- same-actor cosine：只在 `RigidNodes.points_ids[:,0]` 相同的 deterministic pairs 上计算；
- actor-background cosine：actor Gaussian 对 reference-frame nearest Background Gaussian；
- actor-background margin=`same_actor - actor_background`；
- heldout reprojection cosine：`W F_G` 与冻结 evaluation-view DINO feature；
- Background/Rigid coverage、walltime、sidecar bytes、peak VRAM/cgroup memory。

`RigidNodes/Background/points_ids` 是 frozen base-model membership proxy，不是真实 ownership GT，只允许作 evaluation
stratum，绝不能进入 DINO、PCA、B0/B1 weight 或 candidate selection feature。报告必须写
`model_membership_proxy_not_ground_truth`，并同时保留不消费该 proxy 的 heldout reprojection/repeatability；否则属于循环
评测，见 `V51-F15`。

每 actor 至少 32 个 covered Gaussian，最多 4,096 对，hash seed=`20260814`；无 eligible actor 的 scene 保留 abstain。

## 8. Draft gate 与停止条件

### H direction gate

- operator parity/resource/checkpoint exact 全通过；
- 至少 2/3 scene evaluable，至少 2/3 B1 margin>0，scene-balanced mean margin>0；
- mean Rigid coverage>=0.60；
- heldout reprojection cosine `B1-B0>=-0.01`。

### S one-shot replication

- 2/2 evaluable、2/2 B1 margin>0；
- 至少 1/2 B1 margin>=0.02，scene-balanced mean margin>0；
- mean Rigid coverage>=0.60；heldout `B1-B0>=-0.01`。

S 失败即 reject feature-graph path，不读 C、不改 pair、PCA、support 或阈值。S 通过后先 freeze B1 family，再一次性读 C。

### C confirmation

- 至少 2/3 evaluable、至少 2/3 B1 margin>0；
- 至少 1/3 B1 margin>=0.02，scene-balanced mean margin>0；
- mean Rigid coverage>=0.60；heldout `B1-B0>=-0.01`。

C 失败则 Stage B rejected、Stage C/Graph 保持锁定；C 通过也只把 Stage B 收口为 candidate，并单独请求 Stage C 授权。
B0/B1 delta 全量报告：如果 B1 与 B0 没有增益，不得声称 transpose 优于 current lift；但 plan 的 feature-graph gate
仍由 B1 自身能否形成稳定 separation 裁决。

## 9. 授权后的最小提交序列

1. `freeze-only`：显式 supersede P0 plan binding，确认本 proposal，冻结 240 image SHA；
2. `asset-only`：下载 4.55GB checkpoint，记录完整 SHA/license/source，不启动模型；
3. `resource-smoke`：单 H image、只测 faithful ViT-g/14 registers shape/VRAM；
4. `operator-only`：实现/测试 sparse B0/B1 与 lazy bilinear parity，不读 feature quality；
5. `H`→candidate freeze→`S one-shot`→family freeze→`C one-shot`；每个 terminal 同步 status/experiments/failure ledger。

validation、test、KITTI、Graph、SigLIP/CLIP/SAM backbone search 全程继续锁定。
