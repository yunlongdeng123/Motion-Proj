# Research Status

- 更新时间：2026-08-06
- 当前路线：面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化 V3.1
- 当前任务：`WS-V3-A1-CALIBRATION-01`
- 状态：`running`
- 当前门禁：scene-0230 C0–C3 与 C* 冻结完成；顺序运行 scene-0242/0255 C0/C1 确认矩阵
- 权威计划：[`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md)
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

## A0 完成证据

- 实现提交：`436cfc1`（`fix(drivestudio): 过滤空 LiDAR 实例块`）；
- patch SHA-256：`54e7584b6d74431e58f626dfaadd69812d4058d54f82c7941e75aa11f5f94619`；
- frozen DriveStudio：`e59bda4`，实际训练使用独立 patched worktree
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-r2`，原始上游保持 clean；
- 定向测试：`16 passed`；patch apply/reverse-check 与 `git diff --check` 通过；
- scene-0255 canonical smoke：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/20260805T161656Z__scene0255-catfix-s0-r2`
  =`done`；原生 r27 mixed-empty CUDA cat 错误被复现，修复后为 `59×3 / 177 numel` 且点/颜色 exact pairing；
- 1-step 真实训练完成 dataset init、`966,259` background GS、`27,894` rigid GS、优化和 checkpoint 保存；
  controller duration `72.1 s`，peak GPU sample `8,388 MiB`，peak cgroup `5,971,820,544` bytes，
  `invalid_configuration=false`。

该 smoke 只解释兼容修复。A0 正式冻结还包括：

- scene-0255 新 30k run：`20260805T162355Z__scene0255-native30k-s0-r1`；
- scene-0230/0242 等价 checkpoint 复用 run：`20260805T171624Z__scene0230-reuse-eval-s0-r1` 与
  `20260805T171914Z__scene0242-reuse-eval-s0-r1`；
- 全图 PSNR（0230/0242/0255）：`24.934 / 29.107 / 25.230`；总 GS：
  `1,319,913 / 930,011 / 1,551,383`；训练时间：`3014.5 / 2006.2 / 2739.4 s`；
- high actor 区域 PSNR/SSIM/tight-crop LPIPS：`21.728/0.596/0.121`、
  `19.788/0.665/0.153`、`23.531/0.665/0.058`；scene-0242 boundary role 为预注册 `ABSTAIN`；
- actor mask 为 paired original/delete render 的模型 counterfactual diagnostic，不是真值分割；每场记录
  visible image 和 pixel coverage，checkpoint 评估前后哈希一致；
- 唯一汇总：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/20260805T175000Z__a0-three-scene-finalize-s0-r2`
  =`done`。r1 的训练资源 schema 字段差异已作为 `blocked` 保留，`00ba4e8` 修复后 r2 通过。

A0 的核心判断是：全图重建质量不能替代 actor/边界质量。scene-0242 全图 PSNR 最高，但 high actor PSNR
最低；scene-0255 boundary actor 区域 SSIM 仅 `0.526`。这为 A1/A2 提供目标，不构成跨场景因果结论。

## A1 开发场景完成证据

- 端点提交：`20c4276`；权威相机映射修复：`d85ef27`；LiDAR provenance：`14bc3c2`；
- 冻结 E1/E2 配置 SHA-256：
  `60c211625860c25edf92842b88bdb040ea8c180b12fe0fa78f2fc1c342bc4051`；
- C0/C1 有效正式端点 run：
  `20260806T141409Z__scene0230-c0-a1-e0-formal-full-camera-map-fix-s0-r2`、
  `20260806T141623Z__scene0230-c1-a1-e0-formal-full-camera-map-fix-s0-r1`，均为 `done` 且 checkpoint SHA 未变；
- E1 median/P90：C0 `0.05951/0.14719`，C1 `0.06289/0.15623`；coverage 为 `10.780%/10.614%`；
- E2 high actor mean/P90：C0 `0.004813/0.010895`，C1 `0.004751/0.010895`；boundary actor：
  C0 `0.003547/0.006353`，C1 `0.004450/0.007626`；
- 错误相机标签 run `20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 已显式
  `rejected / INVALID_CAMERA_ID_LABEL_MAPPING`，不得进入结果；
- 最小 LiDAR provenance 正式 run：
  `20260806T143644Z__scene0230-a1-lidar-provenance-formal-full-witness-s0-r1`=`done`；配置 SHA-256
  `f2fd1712cf4ddd75c1c4d1da4a426dcf7e1340a5fd943066401ba881f51c5639`；196 个 block、6,804,832 raw
  points、24 actor/75,002 actor points 均入账；
- 记录的 LiDAR/actor tensors exact match，但 CUDA visibility filter 使随机背景初始 GS 从源运行 946,484
  变为正式 witness 946,291。初始深度 median/P90=`7.679/35.958 m` 仅为
  `seed0_reconstructed_initialization_witness_not_exact_source_initialization`，不是源初始化 exact replay；
- A1 定向测试 `23 passed`；逐 Gaussian ancestry/parent-child/split-clone lineage 按 V3.1 后移至 A2。

scene-0230 四个配对 30k 训练均已完成；共同 initialization provenance SHA 为
`8951543c33f72f439068237f1a552fae660895f8906afbf4651f5f580981b898`：

| variant | global PSNR / LPIPS | boundary actor PSNR / LPIPS | high actor PSNR / LPIPS | total GS | train min |
|---|---:|---:|---:|---:|---:|
| C0-off | 27.746 / .1764 | 27.756 / .0687 | 25.358 / .0943 | 1,360,649 | 52.05 |
| C1-native | 24.979 / .1694 | 22.549 / .1033 | 21.696 / .1201 | 1,316,421 | 53.69 |
| C2-factorized-isp | 25.011 / .1677 | 22.583 / .1043 | 21.779 / .1174 | 1,322,979 | 52.26 |
| C3-bounded-pose | 28.109 / .1666 | 28.169 / .0657 | 25.137 / .0938 | 1,363,040 | 56.14 |

- C2/C3 训练 run：`20260806T144938Z__scene0230-c2-factorized-isp-formal30k-s0-r1`、
  `20260806T154834Z__scene0230-c3-bounded-pose-formal30k-s0-r1`；
- C2/C3 端点 run：`20260806T154541Z__scene0230-c2-a1-e0-formal-full-s0-r1`、
  `20260806T164852Z__scene0230-c3-a1-e0-formal-full-s0-r1`；均保持 checkpoint SHA 不变；
- 冻结 A1-D0 配置 SHA-256 为
  `a445078d3bea89a78a0c9e6544a94a2be4c9c2e71f45aec4a9d8878b4c6593c1`；正式诊断
  `20260806T170219Z__scene0230-a1-diagnostics-c0-c3-formal-s0-r1`=`done`；
- 输入速度层为 near-static/low/normal=`2/18/176` 帧；near-static 仅 2 帧，只作低支持描述；
- C3 学习位姿修正 translation median/P90=`1.703/2.338 mm`、rotation=`0.02553/0.03337°`，明显小于
  C1 的 `7.256/12.215 mm`、`0.1660/0.35465°`；这只是学习修正幅值，不是独立 pose GT；
- 选择实现提交 `60ef079`，无容差选择配置 SHA-256 为
  `a45699ebf696c875a18832f8db920a6106837a1e4f235dcd9036eff48dfbc609`；明确披露其在开发结果可见后、
  确认场景前操作化；
- 正式选择 run `20260806T171417Z__scene0230-a1-dev-selection-formal-s0-r1`=`done`，冻结
  `C*=C0-off / done_off`：C2 只改善 boundary role E2，high role 与 LPIPS 退化；C3 画质和位姿稳定性最好，
  但 E1/E2 均未严格改善。确认场景 C* 项登记为 C0 exact alias，10 个逻辑矩阵项对应 8 个唯一训练。

## V3 任务状态

| Task ID | 状态 | 当前结论/门禁 |
|---|---|---|
| `WS-V3-P0-ROUTE-01` | done | `076ebdc`；单一 V3 计划、V2 冻结边界、链接与 Git 校验通过 |
| `WS-V3-A0-NATIVE-BASELINE-01` | done | 3/3 30k/等价 checkpoint、held-out、registry、actor/boundary、GS 与资源矩阵完成 |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | pending | A0 后审计 Instant NuRec 官方代码与本地能力边界 |
| `WS-V3-A1-CALIBRATION-01` | running | 开发场景 4/4 与 C*=C0 冻结完成；下一门禁为 0242/0255 C0/C1 + C0 alias 确认 |
| `WS-V3-A2-ACTOR-DENSIFY-01` | pending | A1 完成后按 D0–D3 小步消融 |
| `WS-V3-A3-LOCAL-REFINE-01` | pending | A2 后实施 affected-set 与短步局部精修 |
| `WS-V3-A4-DEPLOYMENT-01` | pending | A3 后做 pruning/precision/chunk/LOD |
| `WS-V3-R0-INTEGRATION-01` | pending | 汇总 A0–A4，不要求扩展到六场景 |

## 机器与工作树

- GPU：NVIDIA GeForce RTX 3090，24,576 MiB；driver `580.105.08`；最近审计 0 MiB；
- cgroup memory：90 GiB，`oom=0 / oom_kill=0`；
- 数据盘：约 62 GiB 可用；
- 无活跃研究 tmux/controller/GPU 进程；
- 当前非 V3 文档 dirty files 属于 V2 M5，必须保留。

## 下一步

按冻结协议串行运行 scene-0242、scene-0255 的 C0/C1 30k，并分别用同一 E1/E2 配置回填；每个场景将 C* 项
登记为 C0 source run/checkpoint 的 exact alias。确认矩阵和 A1 finalizer 完成前不得进入 A2。
