# WorldSim 第三方依赖

- 更新时间：2026-08-11
- 当前计划：[`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md)

## V3.3 S5 审计

| 项目 | 路径 | 固定版本/许可/权重 | V3.3 裁决 |
|---|---|---|---|
| NVIDIA Harmonizer | `/root/autodl-tmp/third_party/worldsim_v32/harmonizer` | commit `dd5799e50855c5bcb1f6ef52a77b5b644b4798c0`；Apache license SHA `58d1e17f...d8bd`；JIT model SHA `ece8e2da...6e90` | 只作 semantic-gated insertion candidate；delete production 禁止写回 |
| R3D2 | `/root/autodl-tmp/third_party/worldsim_v33/r3d2` | commit `3fc6e317d9fea9d800d3f8706554ad6ac794d980`；tree `759dd48...79c7`；Apache license SHA `43070e2d...79c1` | `blocked_pretrained_model_unavailable`；无作者 exported pipeline；禁止从零训练 |
| SAM2.1 | `/root/autodl-tmp/third_party/worldsim_v32/sam2` | commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`；license SHA `c71d239d...ab4`；large checkpoint SHA `2647878d...d318` | 冻结 image prompt 回灌 detector；checkpoint before/after exact |

Harmonizer exported model 为 `1,448,843,112 bytes`，model card SHA=`f3fc6b26...4b23`。canonical S5
没有修改模型，也没有把其 unconstrained delete 输出发布为 production。R3D2 仓库提供训练/export/eval 代码，
但只引用基础 diffusion 组件，未发布作者训练并导出的推理 pipeline；该状态不是 R3D2 质量负结论。

SAM2 独立环境固定 Python `3.10.20`、torch `2.5.1+cu124`、torchvision `0.20.1+cu124`、numpy
`1.26.4`，conda/pip provenance SHA 分别为 `c9294494...0713 / aded7fb5...5d69`。r1 没有为无关
SciPy import 安装新包；实现改为仅在 gate builder 中 lazy import，保持该冻结环境不变。

## V3.3 S2 审计

| 项目 | 路径 | 固定版本/许可 | V3.3 裁决 |
|---|---|---|---|
| Inpaint360GS | `/root/autodl-tmp/third_party/worldsim_v33/inpaint360gs` | `d54c893285c6cb27788e05cce607e7d3cca6388a`，clean，Apache-2.0 | r12=`blocked_single_3090`；未执行官方训练/推理 |
| GS-RoadPatching | `/root/autodl-tmp/third_party/worldsim_v33/gs-roadpatching` | `468f812`；仅 project-page 静态文件，无算法源码与根 LICENSE | 只允许 `GS-RoadPatching-inspired RoadPatch-Lite`，不得写官方复现 |

Inpaint360GS 官方 README/环境文件声明：

- 验证硬件为 RTX 4090，CUDA 11.8；
- 主环境为 Python 3.10、torch 2.0+cu118；LaMa 使用独立 CUDA 10.2 环境；
- 需要 CropFormer、Big-LaMa、SAM、DeAOT/GroundingDINO 等外部代码或权重；
- 官方流程面向静态 COLMAP/object-aware Gaussian 初始化、分割、LaMa、PLY 与 5k finetune；
- 官方仓库没有 DriveStudio/StreetGS checkpoint adapter。

当前主机为 RTX 3090 24,576 MiB，`/root/autodl-tmp/envs/inpaint360gs` 与
`/root/autodl-tmp/envs/lama` 不存在，必需权重和 StreetGS adapter 也不存在。因此 canonical preflight：

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/
20260810T193426Z__s2-inpaint360gs-preflight-s0-r12
```

记录 `official_execution_attempted=false`、`blocked_single_3090`。这只表示当前官方执行合同不满足，不是对
Inpaint360GS 质量的负结论；不得下载替代权重、静默降低正式分辨率、改变 heldout split，或用 RoadPatch/Telea
输出冒充官方 B2。

## V2 历史依赖

- 历史计划：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)

## 当前驻留

| 项目 | 路径 | commit / 资产 | V2 用途 |
|---|---|---|---|
| AD-GS | `/root/autodl-tmp/third_party/AD-GS` | `9a208512e49c8ddbaa20387921d9648adcd21cb4` | 六场景冻结重建参考 |
| DGGT | `/root/autodl-tmp/third_party/dggt` | `a3276d2bbe4cbb03bcc117830b1836110a27adeb`，clean | M1 inference-only |
| DriveStudio | `/root/autodl-tmp/third_party/drivestudio` | `e59bda4fa681f829dbb1d65f0de582b0f633c450`，clean | M3 StreetGS/actor graph baseline |
| Grounded-SAM-2 | `/root/autodl-tmp/third_party/Grounded-SAM-2` | `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`，clean | M5 感知 evaluator |
| CoTracker3 | `/root/autodl-tmp/third_party/co-tracker` | V1 固定资产 | M5 可选 tracking evaluator |
| Depth Anything V2 | `/root/autodl-tmp/third_party/Depth-Anything-V2` | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` | source 留档；env/weight non-resident |

AD-GS worktree 中存在 V1 已登记的 compatibility 修改与编译产物；M0 必须读取正式 run 的
`source_snapshot/compatibility.patch`，不能把 live dirty 状态误报成 upstream clean。

## DGGT 权重

```text
model repo  xiaomi-research/dggt
revision    735ac9a6486057b1eb886c33a8c6dc79e0b43214
license     CC BY-NC 4.0（模型卡）；代码为 Apache-2.0
path        /root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt
bytes       5,411,266,466
sha256      fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9
```

该文件是本地完整候选，不等于 M1 已完成 provenance 验证。M1 必须核对固定 revision、license、远端元数据和
hash 后再复用；不得无理由重新下载。V1 的 5.39 GB `.partial` 不完整副本已列入清理。

## DriveStudio 当前缺口

- 源码与 `/root/autodl-tmp/envs/drivestudio` 驻留；环境为 Python 3.9 / torch 2.1.2+cu118；
- 历史数据/checkpoint 对应旧 mini/OccGS scenes `003/004/005`；
- 未发现 V2 `scene-0230/0242/0255` 的 DriveStudio processed data 或 actor-aware checkpoint；
- 因而 M3 默认从“官方 source/env 可用、V2 资产缺失”开始，不能走“已有 checkpoint”路径，除非 M3
  正式 audit 找到此前未索引且 hash/配置完整的资产。

## 下载与镜像

- Conda/PyPI：项目级 TUNA 镜像；
- Hugging Face：`HF_ENDPOINT=https://hf-mirror.com`，固定 revision 并校验 SHA-256；
- GitHub：先 `/etc/network_turbo`；用户允许学术加速作为传输 fallback，但结果必须核对官方 remote、固定
  commit、submodule 和 license；
- PyTorch/CUDA extension：版本与 wheel variant 以官方兼容矩阵为准，镜像只加速传输。

新增仓库必须登记 official URL、commit、submodule、license、local diff 和权重 SHA-256。不得用浮动
`main`、未固定 revision 或来源不明网盘进入正式 run。

## 历史依赖

OccGS/ReSim/SVD/cut-in 的依赖和研究结论已归档；它们不再授权执行。需要追溯时从
[`archive/2026-07/README.md`](archive/2026-07/README.md) 进入。
