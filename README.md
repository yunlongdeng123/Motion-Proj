# Motion-Proj

Motion-Proj 当前主线是**面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化 V3.1**。项目以
DriveStudio/StreetGS 和已经完成的动态 actor 编辑链为基础，研究校准、动态 Gaussian 资源分配、编辑后局部
3D 精修和部署优化，不以“大创新”或大型评测框架为前提。

## 当前入口

- 权威状态：[`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)
- 唯一当前计划：
  [`docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md`](docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md)
- 实验台账：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- 失败与防重复账本：[`docs/RESEARCH_FAILURES.md`](docs/RESEARCH_FAILURES.md)

V2 计划保留为历史执行合同，不再更新。V2 M0–M4 已完成；M5 的 0230/0242 checkpoint、scene-0255
CUDA 空 tensor 聚合诊断和未提交脚本作为部分证据冻结，不冒充完成的三场景压力测试。

## V3 模型链

```text
多相机日志 / 位姿 / LiDAR / 实例标注
                ↓
校准与 LiDAR 初始化
                ↓
静态背景 + 动态实例 + 天空 3DGS
                ↓
对象轨迹编辑
                ↓
编辑区域局部 Gaussian 精修
                ↓
剪枝、精度压缩、分块 / LOD 与实时渲染
```

主消融固定为：

| 实验 | 内容 |
|---|---|
| A0 | 原生 DriveStudio/StreetGS 三场景基线 |
| A1 | 成像与位姿校准增强 |
| A2 | actor-aware densification/pruning |
| A3 | 编辑后 local Gaussian refinement |
| A4 | pruning、FP16/量化、chunk/LOD 与资产注册 |

源码审计已经确认 StreetGS 原生包含 per-image affine、camera-pose residual 和 LiDAR 初始化，因此 A1 是
已有能力的严谨消融/增强；首要模型新增是 A2，而不是重新命名上游功能。三场景只用于模型消融与工程结论，
不包装成新的 benchmark 或大规模泛化结果。

A1 已完成 scene-0230 开发消融、scene-0242/0255 确认、冻结 E1/E2、LiDAR provenance、ISP/位姿诊断和
10 项逻辑矩阵 finalizer，正式终态为 `C*=C0-off / done_off`。原始端点方向存在场景依赖，但 C1 在两个确认
场景均未通过完整的 coverage + LPIPS 合同。A2 的 ancestry instrumentation、D1 quota-only 与 D2
boundary/residual smoke/formal 也已完成；fixed/matched 正式裁决都是 `tradeoff_non_dominated`。后续以 D2
作为边界优先研究资产、D1 作为低成本/全局质量回退，不宣称 D2 全面支配 D1。当前门禁是 A3 affected-set、
证据层级、深度语义、outside preservation 与局部质量端点：I0 semantic protocol 已冻结，首个工程门只实现
R0 exact alias 与 R1 affected Background opacity/scale。formal 训练仍未授权；D3/D4 未启动。

## 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
cd /root/autodl-tmp/motion_proj
```

项目脚本的镜像和缓存入口仍可复用：

```bash
source scripts/bootstrap_autodl_v2.sh
```

该脚本不会执行 `conda init` 或改写全局 pip/Conda 配置。大型环境、checkpoint、数据和正式 run 一律放在
`/root/autodl-tmp`。

## 代码与证据

- `motion_proj/resim/`：WorldState、actor registry、轨迹编辑和 typed render；
- `motion_proj/dynamic_editing_v2/`：可复用的 actor 真值、投影和局部评测设施；
- `motion_proj/worldsim_v3/`：A1 校准、端点、诊断、LiDAR provenance 与后续 WorldSim 模型组件；
- `/root/autodl-tmp/third_party/drivestudio/`：固定 DriveStudio/StreetGS 上游；
- `/root/autodl-tmp/runs/dynamic_editing_v2/`：V2 冻结证据；
- V3/V3.1 formal run 使用 `/root/autodl-tmp/runs/worldsim_v3/`；
- `docs/archive/`：历史计划与结论，不构成当前执行授权。

第三方版本、环境、迁移与资产保留规则见：

- [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
- [`docs/MACHINE_MIGRATION.md`](docs/MACHINE_MIGRATION.md)
- [`docs/ARTIFACT_RETENTION.md`](docs/ARTIFACT_RETENTION.md)
