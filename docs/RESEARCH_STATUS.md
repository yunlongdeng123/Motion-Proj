# Research Status

- 更新时间：2026-08-09
- 当前路线：面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化 V3.1
- 当前任务：`WS-V3-A2-ACTOR-DENSIFY-01`
- 状态：`running`
- 当前门禁：唯一 A2-D1 formal r1 已启动并正在执行 D0；下一门禁为 D0→D1 30k formal 完整终态
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

## A1 确认与正式终态

冻结确认配置 `configs/worldsim_v3/a1_confirmation_v1.yaml` SHA-256 为
`63a3cc607ccfddbb714cc81d0570da356263c01c5a68880345953023d2d6a8cd`，实现提交 `198a681`。四个确认训练和
端点 run 均 `done`、每场景 C0/C1 initialization SHA 相同、所有端点评估前后 checkpoint SHA 不变：

| scene / variant | global PSNR / LPIPS | E1 median / P90 / coverage | E2 high mean / P90 / coverage | E2 boundary mean / P90 / coverage |
|---|---:|---:|---:|---:|
| 0242 C0 | 30.064 / .1108 | .03147 / .08826 / 6.491% | .008264 / .020697 / 42.857% | `ABSTAIN` |
| 0242 C1 | 29.161 / .1122 | .03333 / .08971 / 6.423% | .008660 / .021708 / 42.857% | `ABSTAIN` |
| 0255 C0 | 27.255 / .2086 | .04348 / .14248 / 6.710% | .004772 / .009805 / 23.529% | .004032 / .009308 / 41.176% |
| 0255 C1 | 25.240 / .1921 | .04277 / .13626 / 6.751% | .003715 / .007704 / 21.569% | .003923 / .008784 / 41.176% |

- 0242 原始端点与全图指标偏向 C0；boundary role 按预注册继续 `ABSTAIN`；
- 0255 的 C1 E1/E2 error 较低，但 high-role coverage 降低，且 boundary/high actor LPIPS 均退化，未通过完整合同；
- exact alias run：`20260806T211000Z__scene0242-cstar-c0-exact-alias-s0-r1`、
  `20260806T211100Z__scene0255-cstar-c0-exact-alias-s0-r1`，均明确无新训练/评测；
- A1 finalizer `20260806T211248Z__a1-three-scene-finalize-s0-r1`=`done`：10/10 逻辑项、8/8 唯一训练，
  `C*=C0-off / done_off`。该结论是完整冻结合同下的 Pareto 选择，不是“所有场景每项指标 C0 都最好”。

## A2-I0 ancestry instrumentation 完成证据

- canonical r3 项目基线：`research/worldsim-v3@70cf2b2` + formal run 内不可变 source snapshot；当前实现提交：
  `271d876`；
- DriveStudio upstream：`e59bda4fa681f829dbb1d65f0de582b0f633c450`；patched worktree：
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-r5`；
- 配置 `configs/worldsim_v3/a2_instrumentation_v1.yaml` SHA-256：
  `bac1ec5b3642470a999e7f0cf8ddc9cf5b4d9a1445029c43ae92601929f4bfce`；
- instrumentation patch SHA-256：`87c084f77ed5d6395acce95abb992ca86004bdc47b68154878bf462a0fb345b0`；
- canonical formal run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T071500Z__a2-i0-ancestry-formal-s0-r3`=`done`；
- module-off/on 原生 checkpoint tensor 逐位一致、无 mismatch；off 不增加 ancestry key，on 增加且 round-trip；
- 8 个初始 Gaussian 经 split/clone/prune 后保留 10 个、累计分配 11 个 ID；来源计数 LiDAR/split/clone=`7/2/1`；
- actor/parent/lineage root、prune 后索引与 checkpoint 恢复通过；`nearest_lidar_distance` 对 actor 做 exact offline，
  background 因无有界参考集保持 deferred；
- boundary/photometric/depth/normal 在 I0 只冻结 attributed update API；无可靠 normal 时保持 schema-only；
- patched worktree verify、patch reverse-check、当前 working-tree `git diff --check` 和 WorldSim 定向测试
  `66 passed`。

该结果只关闭 deterministic synthetic `RigidNodes` instrumentation 门禁，不是 scene-0230 真实质量证据，
不授权直接启动 D1 formal。本次 source commit 只包含 A2-I0 代码、测试与直接相关文档，不混入保留的 V2 M5 文件。

## A2-D1 formal 协议冻结证据

- clean 协议/控制器/评测提交：`387dd501cd931b632ca4fd9950ee40b14bac6fce`；
- formal 配置：`configs/worldsim_v3/a2_d1_formal_v1.yaml`，SHA-256=
  `ad77db41d9d8c5172804a20b38a2dd92173c3639398d8abc24dc6f4799e8f8e7`；
- scene-0230 / seed 0 / D0→D1 / 每臂 30k；5k 保存只读 candidate checkpoint；
- matched-GS 只匹配干预域 `RigidNodes`：目标为 D0 30k 最终计数，D1 按绝对差最小、并列更早 step 选择；
  相对差 `<=2%` 才登记 done，否则 `ABSTAIN_BUDGET_NOT_MATCHED`；禁止事后 pruning、重训或 quota retune；
- held-out 端点为 global、high/boundary actor region 与 boundary band，以及两 actor 反事实 mask 并集之外的 non-target；
  counterfactual mask 明示不是 GT segmentation；
- `80 passed`；只读 preflight=`done`：GPU=`0 MiB`、free disk=`58.39 GiB`、memory.max=`90 GiB`、
  canonical r4 summary SHA 与三层 DriveStudio patch SHA 全部匹配；
- formal 尚未启动。该证据只解除启动门，不允许宣称 D1 质量改进，也不允许提前启动 D2。

## A2-D1 formal 当前运行

- canonical candidate run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T085400Z__a2-d1-paired-formal30k-s0-r1`；
- source commit=`f32f96b47619e05066d2ee11c899e38d07398e11`；tmux=`ws_a2_d1_f1`；terminal=`running`；
- manifest 已确认 D0/D1 物化配置除 variant/quota-enable 外匹配，formal config SHA=`ad77db41...f8e7`；
- 当前 stage=`train_d0_native_30000`，D1 尚未启动；D0 初始化 Background/RigidNodes=`946,484 / 75,002`；
- 启动后 GPU 约 `3.0 GiB`，cgroup 无 OOM。当前不得启动第二个 formal run 或 D2。

## A2-D1 quota-only 配对 smoke 完成证据

- 工程提交：`c9b2422af637370ca90f48b42a7d0131f458f96d`；配置 SHA-256：
  `6895370625080ccab327e731264e9ebb0f980499b8fec87d02d9efb2e56b14af`；
- DriveStudio upstream=`e59bda4`，canonical worktree=`/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d1-r5`，
  quota patch SHA-256=`c232af2c5fa532016943f399830c85ebba612078871b7c1a296bda816ae7bb1b`；
- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T081330Z__a2-d1-paired-smoke1k-s0-r4`，
  terminal=`done`，summary SHA-256=`ec219bb567799d4d84252e86bd4194620f6b5563d6032c43067ff8e155d3b8bd`；
- D0/D1 均为 scene-0230、seed 0、1000 step，顺序执行；配置除 quota enable/variant 外匹配，初始化 provenance 相同；
- actor threshold=`0.00025`，Background 保持原生 `0.0005`；初始/min/max actor 总量=`75,002 / 37,504 / 180,013`；
- D1 quota 5 次 event 接受 `93,057` children、拒绝 `30,171` parent；最终 `152,830` Rigid，24/24 actor
  不超过最大值；D0 最终 `125,915` Rigid；
- module-off tensor 逐位等价；D1 quota/ancestry checkpoint round-trip，D0/D1 原生 tensor finite；
- D0/D1 peak GPU=`12,807 / 12,795 MiB`，peak cgroup=`5,392,334,848 / 5,661,368,320 bytes`，
  duration=`110.91 / 110.97 s`，无 OOM；
- patch replay/reverse-check、synthetic integration 与 WorldSim 定向回归通过；当前回归为 `75 passed`；
- noncanonical r2 因前台 SSH 转 tmux 显式中止，r3 因 r2 遗留独立 session GPU 子进程被 idle preflight 拒绝；
  遵循 `PIVOT-F22` 精确回收后，r4 才作为 canonical，旧 terminal 不改写为 done。

该证据只授权冻结 D1 formal 协议；1000-step smoke 未执行冻结 held-out actor/boundary 质量合同，且 D1 Gaussian
更多，不能登记为方法改进或直接解锁 D2。

## V3 任务状态

| Task ID | 状态 | 当前结论/门禁 |
|---|---|---|
| `WS-V3-P0-ROUTE-01` | done | `076ebdc`；单一 V3 计划、V2 冻结边界、链接与 Git 校验通过 |
| `WS-V3-A0-NATIVE-BASELINE-01` | done | 3/3 30k/等价 checkpoint、held-out、registry、actor/boundary、GS 与资源矩阵完成 |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | pending | A0 后审计 Instant NuRec 官方代码与本地能力边界 |
| `WS-V3-A1-CALIBRATION-01` | done_off | 10/10 逻辑项、8/8 唯一训练；C*=C0；确认原始端点方向存在场景依赖 |
| `WS-V3-A2-ACTOR-DENSIFY-01` | running | I0、D1 smoke 与 formal 协议 done；唯一 r1 正在执行 D0→D1 30k formal |
| `WS-V3-A3-LOCAL-REFINE-01` | pending | A2 后实施 affected-set 与短步局部精修 |
| `WS-V3-A4-DEPLOYMENT-01` | pending | A3 后做 pruning/precision/chunk/LOD |
| `WS-V3-R0-INTEGRATION-01` | pending | 汇总 A0–A4，不要求扩展到六场景 |

## 机器与工作树

- GPU：NVIDIA GeForce RTX 3090，24,576 MiB；driver `580.105.08`；最近审计 0 MiB；
- cgroup memory：90 GiB，`oom=0 / oom_kill=0`；
- 数据盘：约 59 GiB 可用；
- 活跃 tmux=`ws_a2_d1_f1`；唯一 A2-D1 formal r1 正在使用 GPU；
- 当前非 V3 文档 dirty files 属于 V2 M5，必须保留。

## 下一步

A1 已收口，A2-I0、D1 quota-only paired smoke 与 formal 协议冻结均已通过。唯一 r1 已在 tmux 中顺序运行
scene-0230 D0→D1 30k；下一步是监控该实例并生成 fixed-step 与 matched-RigidNodes-budget 两个视图。formal 两臂
和 matched gate 完成前不启动 D2；更多 Gaussian 不自动解释为改进，也不混入 boundary/residual、scale cap、
LiDAR/visibility 或 D2–D4 因子。
