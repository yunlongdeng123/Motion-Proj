# V2 第三方依赖

- 更新时间：2026-08-02
- 当前计划：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)

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
