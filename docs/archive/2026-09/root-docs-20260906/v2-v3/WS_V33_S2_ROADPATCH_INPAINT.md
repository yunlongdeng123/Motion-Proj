# WorldSim V3.3 S2：RoadPatch-Lite 与 Inpaint360GS 预检

- Task：`WS-V33-S2-ROADPATCH-INPAINT-01`
- 状态：`done`
- Canonical B1：patch index r10 + RoadPatch r11
- Conditional B2：Inpaint360GS preflight r12=`blocked_single_3090`
- 硬件：单卡 NVIDIA GeForce RTX 3090 24 GiB
- 输入：immutable V3.1 D2 FP32 StreetGS checkpoint
- 方法边界：`GS-RoadPatching-inspired RoadPatch-Lite`，不是官方 GS-RoadPatching reproduction

## 1. 结论

S2 把“删除车辆后修背景”从 V3.2 的 Telea-generated full checkpoint，推进为可审计的：

```text
immutable D2 base
+ deterministic native-Gaussian patch delta
```

RoadPatch-Lite 不修改 base Background，不训练整场景，不使用 generated-background donor。它从同一场景的原生
Background 中建立确定性 1/2/4 m patch index，冻结真实 hole anchor 和 top-5 搜索，在 development view 选择最小
可见 delta，heldout 只做一次 confirmation。

canonical r11 只新增 `104` 个 Gaussian rows（high/boundary=`25/79`），heldout PSNR 退化
`0.084031 dB`，小于冻结的 `0.1 dB` 门；SSIM、LPIPS、static PSNR、static LiDAR MAE 也全部过门。

Inpaint360GS 上游 source、commit 和许可证可验证，但官方双环境、外部权重、StreetGS adapter 与官方 RTX 4090
硬件合同在当前主机不满足，因此 r12 如实封闭为 `blocked_single_3090`，没有伪造官方复现或 B2 质量指标。

## 2. 坐标与输入合同

DriveStudio scene-0230 的首个 CAM_FRONT 对齐坐标为：

```text
x = right
y = down / vertical
z = forward
BEV axes = (x, z)
```

这项核对修正了一个关键工程风险：V3.1 P3 package 使用 `(x,y)` 网格并绑定 V3.2 P2 FP16 mixed checkpoint，
不能直接当作 D2 FP32 道路 donor index。S2 只把它作为历史 schema 参考，重新从 D2 原生 Background 建索引。

D2 Background 有 `1,205,164` 行。V3.2 generated-background provenance 是从 `1,205,164` 开始连续追加的
`1,896` 行，所以以 D2 FP32 为 donor source 可在结构上排除全部 generated rows；runner 仍对 provenance 和
checkpoint SHA 做显式 fail-closed 验证。

## 3. Static patch index

每个 1/2/4 m patch 固定记录 28 维特征与 exact source provenance：

- geometry：plane normal/residual、density、scale、opacity、depth/vertical range；
- appearance：SH-DC/RGB mean/std、gradient proxy；
- semantics：actor mass 与静态背景证据；
- support：train-view observation、front-camera support、visibility mass；
- provenance：native flat indices、coarse chunk、source hashes。

### 3.1 道路层鲁棒化

whole-cell `max_scale/max_plane_residual` 会被同一平面格中的一个天空、立面或跨层 Gaussian 污染。r3 因此产生
`53,541` 个 patch、却没有一个 valid。最终合同是：

1. 逐行排除 actor 高语义质量、generated、低 visibility/mass 与 scale outlier；
2. 在 vertical axis `y` 上确定性选择宽度不超过 `0.75 m` 的 densest slab；
3. 仅对该主层计算 plane residual 与 vertical-normal gate；
4. sidecar `visible_view_count>=5`，并要求至少一个 front-camera frustum observation。

这不是放松 geometry 门，而是把道路层和无关空间层显式解耦。

### 3.2 Canonical r10

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/
20260810T193004Z__s2-patch-index-formal-s0-r10
```

| 项目 | 数值 |
|---|---:|
| native Background rows | 1,205,164 |
| eligible native rows | 702,506 |
| all patches | 15,591 |
| valid 1 m / 2 m / 4 m | 617 / 160 / 45 |
| valid total | 822 |
| generated donor rows | 0 |
| wall | 30.713 s |

| 产物 | bytes | SHA-256 |
|---|---:|---|
| `static_patch_index.npz` | 4,146,483 | `51561eecf66ac20f38d139abd9738c970cefe686f40ba9ae787ea62be74a1a4c` |
| `patch_index_manifest.json` | 13,130 | `565741c5b92c60a4a75552b71ff6c24758db605425618adefd0d0209f42d8845` |
| `summary.json` | 2,276 | `4216c65293680c71be1f042c3eb07ccf5e8ec619982dcbff17be71e56ba34f1b` |
| `status.json` | 321 | `754ab982a103f2993f42664ecbe1bdcb95a1d9f371c33d95ccd04b30edb98063` |

## 4. Hole anchor 与候选搜索

target anchor 只来自：

```text
S1 object-aware delete mask
∩ accepted SAM2 target mask
+ target-view first-hit depth
+ cross-view observed support
```

| target | delete pixels | valid depth | cross-view pixels | center (x,y,z) | patch |
|---|---:|---:|---:|---|---:|
| high-support | 15,178 | 6,139 | 6,994 | `(-15.8240, 2.1882, 46.0719)` | 4 m |
| boundary-support | 184 | 78 | 44 | `(-14.9516, 2.5408, 64.3057)` | 4 m |

top-K 固定为 `5`，搜索权重 geometry/appearance/semantic/visibility=`1/2/2/0.5`。donor 与 target 至少相隔
`5 m`，vertical offset 不超过 `0.5 m`。placement 只允许 `(x,z)` rigid translation、可选 yaw 和 `y` plane
offset；新增行采用 opacity feather、bounded RGB shift 与 scale clamp。

所有新增行保留 donor patch ID、native flat indices、source chunk、transform 和 source Gaussian hash。

## 5. 最小可见 delta

r8 暴露了 S2 的主要研究转折：使用几何合格但过密的 `2,150`-row delta，虽然 development target 可见，heldout
PSNR/SSIM 却退化 `-0.8553 dB/-0.00619`。该 run 保持 rejected，未从 heldout 反向选择 donor。

修复是在候选资格层冻结：

```text
maximum_rows_per_target = 512
```

超过上限的候选不参与 development ranking，而不是截断一个已选 dense patch。最终搜索自动选出：

| target | patch ID | rows |
|---|---|---:|
| high-support | `p4-x-000008-z+000009` | 25 |
| boundary-support | `p4-x-000009-z+000009` | 79 |
| total | — | 104 |

candidate 渲染时临时追加到 Background Parameters，完成后恢复同一对象 identity；不生成新 full checkpoint。

## 6. Canonical RoadPatch r11

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/
20260810T193140Z__s2-roadpatch-formal-s0-r11
```

### 6.1 Heldout confirmation

| 指标 | B0 | B1 | delta | 冻结门 | 结果 |
|---|---:|---:|---:|---:|---|
| PSNR | 28.157155 | 28.073124 | -0.084031 dB | >= -0.1 dB | pass |
| SSIM | 0.871450 | 0.870542 | -0.000908 | >= -0.005 | pass |
| LPIPS | 0.149666 | 0.151527 | +0.001861 | <= +0.01 | pass |
| static PSNR | 28.520749 | 28.523614 | +0.002865 dB | no regression | pass |
| static LiDAR MAE | 0.895636 m | 0.890384 m | -0.005252 m | no regression | pass |

heldout 只用于冻结候选后的确认；没有把 unsupported completion 报成 GT accuracy。

### 6.2 产物与资源

| 产物 | bytes | SHA-256 |
|---|---:|---|
| `roadpatch_delta.npz` | 24,557 | `a31053137e37bb36eb7f59d0250d525a9ebe274caf2903f5dd92a47063289014` |
| `selection.json` | 37,977 | `da54ad6d769507b6b47055686fabdcb67f88e221159b5c6e01e213b0ac5590f0` |
| `acceptance.json` | 44,786 | `9be398450e34a5b5a4f43dcfccd562b42439a4735a7efc9faaf97b59afa43cd0` |
| `summary.json` | 6,858 | `5de28a028b2780f80f4eb73e54bc7c7cd1ca32a9344953ad0f5b6f89940ad17d` |
| `status.json` | 454 | `f4780ce1fed07970f43906afc099ebac4a4694e6ebd52355f4f159601d030c87` |

- wall=`69.335 s`；
- peak CUDA allocated/reserved=`8,337,670,144 / 8,420,065,280 bytes`；
- D2 checkpoint before/after SHA-256 exact；
- index manifest 与 source snapshots exact；
- 无训练、无 optimizer step、无 base checkpoint mutation。
- RoadPatch 专项=`6 passed`；V3.3/V3.2 定向回归=`52 passed`；py_compile 通过；r10/r11/r12 的 8 个
  canonical source snapshots 与工作树 byte-exact。

## 7. Inpaint360GS 官方预检 r12

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/
20260810T193426Z__s2-inpaint360gs-preflight-s0-r12
```

固定事实：

- upstream commit=`d54c893285c6cb27788e05cce607e7d3cca6388a`，tree clean；
- Apache-2.0 license SHA=`41d805773f2aa0b36c2fb69491f64c3079fe3e0671c9848680645fc9e65d5a10`；
- 官方声明 RTX 4090/CUDA 11.8，main Python 3.10/torch 2.0+cu118；LaMa 使用独立 CUDA 10.2；
- 所需 CropFormer、Big-LaMa、SAM、DeAOT、GroundingDINO 权重均不在冻结位置；
- `/root/autodl-tmp/envs/inpaint360gs`、`/root/autodl-tmp/envs/lama` 不存在；
- 官方仓库没有 DriveStudio/StreetGS dynamic checkpoint adapter。

因此：

```text
status = blocked_single_3090
official_execution_attempted = false
```

| 产物 | bytes | SHA-256 |
|---|---:|---|
| `preflight.json` | 3,434 | `91b5c6a04cefc6086e4695584f57c0497bc9985ba36e874336b85cc4a11a830b` |
| `summary.json` | 545 | `263f336be0f9b5086ec0e743aa7d4f0238fa73b4b0daecbd7defa0e43793a8e3` |
| `status.json` | 151 | `292e90dfadf2f53bc5f39f63787e7d2cd3e1a7426c990bdac7f2047b02108ec9` |

该状态只说明当前官方执行前置条件不满足，不构成 Inpaint360GS 质量的正面或负面证据。

## 8. 失败账本

| run | 状态 | 发现与处置 |
|---|---|---|
| r0 | failed | 外层重定向先创建 run.log，non-empty run-dir fail-closed |
| r1 | failed | 旧 P3 绑定 P2 FP16 且使用 `(x,y)`，拒绝冒充 D2 native index |
| r2 | failed | intrinsics 实际 9 值，不是 4 值 |
| r3 | rejected diagnostic | whole-cell gate：53,541 patches、0 valid |
| r4–r6 | diagnostic | row filter/slab/support 合同逐步冻结；r6 得到 822 valid |
| r7 | diagnostic | 真实 GPU anchor/top-5 通过，未读 heldout |
| r8 | rejected | 2,150-row dense delta heldout 退化 |
| r9 | diagnostic done | 104-row delta 过门，但早于 formal index manifest |
| r10 | canonical done | static patch index |
| r11 | canonical done | B1 RoadPatch selection + heldout confirmation |
| r12 | blocked_single_3090 | B2 官方前置条件预检，未执行官方算法 |

## 9. 复现入口

在 `/root/autodl-tmp/motion_proj`：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj

/root/autodl-tmp/envs/drivestudio/bin/python \
  scripts/build_worldsim_v33_s2_patch_index.py \
  --config configs/worldsim_v33/s2_roadpatch_v1.yaml \
  --run-dir /root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/<new-index-run>

/root/autodl-tmp/envs/drivestudio/bin/python \
  scripts/run_worldsim_v33_s2_roadpatch.py \
  --config configs/worldsim_v33/s2_roadpatch_v1.yaml \
  --patch-index <new-index-run>/artifacts/static_patch_index.npz \
  --patch-index-sha256 <computed-sha256> \
  --patch-index-manifest <new-index-run>/artifacts/patch_index_manifest.json \
  --patch-index-manifest-sha256 <computed-sha256> \
  --run-dir /root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/<new-roadpatch-run> \
  --phase formal --device cuda:0

/root/autodl-tmp/envs/drivestudio/bin/python \
  scripts/preflight_worldsim_v33_s2_inpaint360gs.py \
  --config configs/worldsim_v33/s2_inpaint360gs_preflight_v1.yaml \
  --run-dir /root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/<new-preflight-run>
```

不得续写 canonical r10/r11/r12；复现必须使用新的 run ID，并继续验证所有 frozen input SHA。

## 10. Claim boundary 与下一步

允许声明：

- 同场景 native Gaussian donor 的确定性道路 patch reuse 已在单卡 3090 上落地；
- row-level layer isolation 解决了 whole-cell outlier poisoning；
- 104-row minimum-visible delta 在冻结 heldout 门内保持全局/静态质量；
- base + delta 的 authoring 语义已建立，完整 compose/rollback/package 由 S4 继续；
- Inpaint360GS 官方执行条件已审计并 fail-closed。

不允许声明：

- 已复现官方 GS-RoadPatching 或 Inpaint360GS；
- unsupported region 具有 GT correctness；
- 单 scene/双 target 结果已经证明跨场景普适性；
- S2 delta 已等价于 S4 的完整可组合 spatial-delta package。

S2 收口后唯一解锁任务为：

```text
WS-V33-S3-ASSET-VIEWSELECT-01
```
