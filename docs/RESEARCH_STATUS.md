# Research Status

- 更新时间：2026-08-05
- 当前路线：面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化 V3
- 当前任务：`WS-V3-P0-ROUTE-01`
- 状态：`running`
- 当前门禁：只完成 V3 文档切换与事实冻结；P0 提交后进入 A0
- 权威计划：[`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md)
- V3 启动 Git 基线：`research/dynamic-editing-v2@e691c1f`
- 当前分支：`research/worldsim-v3`

## 当前裁决

项目不再以“提出新的可编辑 3DGS”或 V2 M5/M6 大型失败评测为主线。V3 的交付目标是完整的 WorldSim
模型链和 A0–A4 消融：原生 StreetGS → 校准增强 → actor-aware 增密/剪枝 → 编辑后局部 Gaussian 精修
→ 部署优化。

核心模型问题固定为：

1. 动态 actor 是否应使用区别于静态背景的 Gaussian 增密与剪枝规则；
2. 对象移动/删除后，局部 3D Gaussian 短步精修是否能改善空洞、深度/透明度排序和时序闪烁。

三场景是模型消融场，不是新 benchmark。结果只支持当前数据、实现和资源合同下的模型/工程结论，不外推为
大规模泛化、物理真实性或闭环安全结论。

## V2 继承与冻结

### 已完成并继承

| Task | 终态 | V3 用法 |
|---|---|---|
| `DR-V2-M0-BOOTSTRAP-01` | done | 环境、资产、网络与 source provenance |
| `DR-V2-M1-DGGT-REPAIR-01` | done | 历史前馈范式对照；不再做非等价排行榜 |
| `DR-V2-M2-ACTOR-EVAL-01` | done | persistent actor、raw 轨迹、三相机投影和 frozen cohort |
| `DR-V2-M3-EDIT-BASELINE-01` | done | StreetGS checkpoint、actor registry、基础轨迹编辑 |
| `DR-V2-M4-EDIT-PILOT-01` | done | scene-0230 全序列编辑闭环和可复用指标设施 |

### M5 部分执行后冻结

`DR-V2-M5-STRESS-3SCENE-01` 没有完成，也没有产生 V2 预注册的 24 条序列、pseudo-hole/perception 全量结果
或三场景 final matrix。它不记为 `done` 或 `rejected`，只保留下列事实：

- scene-0230 held-out checkpoint：`398,652,534` bytes，SHA-256
  `24a39f27dfeed36bbdb01ee14211aec51b414e6ab0e61915b71c1dddcdf61e49`；high/boundary actor 分别
  `4,747/1,914` GS；
- scene-0242 checkpoint：`306,034,934` bytes，SHA-256
  `16179d8f99becb86b6893a18ff036af72d78c9897f7aa2b0e297b735dd6c5fda`；high actor `6,939` GS，
  boundary actor 为显式 `ABSTAIN`；
- scene-0255 数据准备和 sky 阶段已有产物，但原生训练阻塞于
  `datasets/driving_dataset.py` 的 CUDA `torch.cat(instance_dict[ins_id]["pts"], dim=0)`；
- r27 诊断输入为 166 个 CUDA float32 tensors，其中 152 个 `(0, 3)`，总计 177 scalars；无 OOM 证据；
- evaluation sequencer r16/r18 的 `running` terminal 属于容器中断遗留；现场无对应进程或 tmux，不得改写终态；
- M5 未提交的脚本、配置和测试保留在工作树中，P0 不清理、不覆盖、不混入 V3 文档提交。

V2 M6–M8 不再授权。V2 计划原文件保持不改，只作历史执行合同。

## V3 源码事实

DriveStudio 固定 commit `e59bda4fa681f829dbb1d65f0de582b0f633c450`。源码审计确认：

- 原生 `AffineTransform` 已提供 per-image RGB affine；
- 原生 `CameraOptModule` 已提供平移和旋转位姿残差；
- 原生数据链已用 LiDAR 初始化背景和动态实例；
- `RigidNodes` 仍对所有 actor Gaussian 使用统一的 gradient/scale/screen-size/opacity 阈值。

因此 A1 是已有校准能力的 off/native/enhanced 消融；A2 才是 V3 的首要模型新增。rolling shutter 只有在
processed data 存在真实 readout direction/time 后才可实现，否则必须报告 `not_supported`。

## V3 任务状态

| Task ID | 状态 | 当前结论/门禁 |
|---|---|---|
| `WS-V3-P0-ROUTE-01` | running | 文档编写中；尚未提交 |
| `WS-V3-A0-NATIVE-BASELINE-01` | pending | P0 提交后授权；先修 scene-0255 空 CUDA tensor 聚合 |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | pending | A0 后审计 Instant NuRec 官方代码与本地能力边界 |
| `WS-V3-A1-CALIBRATION-01` | pending | A0 三场景冻结后进入 |
| `WS-V3-A2-ACTOR-DENSIFY-01` | pending | A1 完成后按 D0–D3 小步消融 |
| `WS-V3-A3-LOCAL-REFINE-01` | pending | A2 后实施 affected-set 与短步局部精修 |
| `WS-V3-A4-DEPLOYMENT-01` | pending | A3 后做 pruning/precision/chunk/LOD |
| `WS-V3-R0-INTEGRATION-01` | pending | 汇总 A0–A4，不要求扩展到六场景 |

## 机器与工作树

- GPU：NVIDIA GeForce RTX 3090，24,576 MiB；driver `580.105.08`；最近审计 0 MiB；
- cgroup memory：90 GiB，`oom=0 / oom_kill=0`；
- 数据盘：约 67 GiB 可用；
- 无活跃研究 tmux/controller/GPU 进程；
- 当前非 V3 文档 dirty files 属于 V2 M5，必须保留。

## 下一步

完成 `WS-V3-P0-ROUTE-01` 的 Markdown/link/Git diff 校验并独立提交。提交后将当前任务切换为
`WS-V3-A0-NATIVE-BASELINE-01`，先把 r27 诊断固化为回归测试，再实施 scene-0255 最小兼容修复；不得
直接跳到 A2，也不得继续扩建 V2 M5 evaluator。
