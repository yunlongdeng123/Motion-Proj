# WS-V32 S0 路线与 SOTA 审计

- 日期：2026-08-10
- Task：`WS-V32-S0-ROUTE-AND-SOTA-AUDIT-01`
- 状态：`done`
- 项目基线：`d91e80eea33a1bf8b6596d2357ee0ccf357691cc`
- 当前分支：`research/worldsim-v3.2-semantic-repair`
- 事实配置：[`../configs/worldsim_v32/s0_sources_v1.yaml`](../configs/worldsim_v32/s0_sources_v1.yaml)
- 下一授权：仅 `WS-V32-S1-SEMANTIC-LIFT-01`

## 1. V3.1 冻结边界

- A2-D2 checkpoint SHA-256 固定为
  `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c`；
- A3 选择保持 `R0-off`，不恢复旧 R1；
- P2/P3 production assets 与 V3.1 terminal 保持只读；
- S0 未启动训练、推理或旧资产 mutation。

## 2. 官方源码审计

11 个公开仓库均固定到审计时的官方默认分支 HEAD，`git ls-remote` 与本地 commit 完全一致：

| 路线 | commit | license / gate | S0 结论 |
|---|---|---|---|
| SAM2 | `2b90b9f5ceec` | Apache-2.0 | S1 主路径 |
| SAGA | `2d4c5d77c857` | Apache-2.0 | 3D lift 参考，不直接采用其旧 3DGS runtime |
| Gaussian Grouping | `0ab60afed338` | Apache-2.0 | identity/grouping 参考 |
| SegmentAnythingin3D | `35739ac684c2` | Apache-2.0 | prior-guided lift 参考 |
| 3DGIC | `0fdbaed68026` | Inria 3DGS research license | S2 候选；商业使用需另审 |
| Inpaint360GS | `d54c893285c6` | Apache-2.0 | S2 候选 |
| VISTA | `95e47fd47255` | 无根许可证 | 阻塞执行，只保留方法参考 |
| Asset Harvester | `767b2439ce47` | Apache-2.0 | S3 主候选 |
| Harmonizer | `dd5799e50855` | Apache-2.0 | S4 候选；Cosmos base 为 gated |
| Omni-3DEdit | `13da5f2c9e87` | 无根许可证 | 阻塞执行，待许可证澄清 |
| CoIn | `a87cffde5b6b` | 无根许可证 | 仅 no-checkout 审计，阻塞执行 |

MV-SAM 在本次审计中未找到可固定的官方代码与 checkpoint，因此保持 conditional blocked；不从论文手写大型模型。

## 3. 权重与环境

S1 固定使用 SAM2.1 Hiera Large：

```text
repo      = facebook/sam2.1-hiera-large
revision  = 665f8e2ad61cf5f53d65644ff27c8ee525124610
file      = sam2.1_hiera_large.pt
bytes     = 898083611
sha256    = 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
mirror    = https://hf-mirror.com
```

文件已下载到：

```text
/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt
```

独立环境为 `/root/autodl-tmp/envs/worldsim-v32-sam`。无卡准备阶段设置
`SAM2_BUILD_CUDA=0`，避免编译可选 connected-components CUDA 扩展；这不阻止 SAM2 主推理，GPU 开机后可按需单独重建该可选扩展。

Asset Harvester、Harmonizer、Cosmos 与 Omni-3DEdit 只固定 HF revision，S0/S1 不下载、不安装、不执行。

## 4. 当前机器审计

当前是 AutoDL 无卡开机模式：

- cgroup memory max=`2,147,483,648 bytes`；
- `nvidia-smi` 不可用/无 GPU device；
- 数据盘审计时可用 `94,801,928,192 bytes`；
- Docker 不可用；
- GitHub 与 `hf-mirror.com` 可访问；
- `/proc/meminfo`/`free` 显示宿主机内存，不能替代 cgroup 合同。

因此源码固定、权重下载、环境安装和静态检查可在当前模式完成；任何 SAM2 mask 推理、3DGS render、semantic
lifting CUDA 路径必须在切回 GPU 实例后先过 preflight。

## 5. S1 解锁边界

S1 只允许：

1. scene-0230 train-view 的 SAM2 box-prompt temporal mask；
2. CAM_FRONT / FRONT_LEFT / FRONT_RIGHT；
3. heldout frame 完全排除；
4. actor registry 作为 hard identity prior；
5. semantic posterior 只写 sidecar，不写回 D2 checkpoint；
6. module-off 保持 D2 checkpoint bitwise exact；
7. original/delete/lateral smoke。

S2–S5 仍未授权。源码存在或权重可下载不等于方法可用、质量有效或许可证允许部署。
