# 面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化计划 V3.1

- **版本**：V3.1
- **日期**：2026-08-09
- **项目根目录**：`/root/autodl-tmp/motion_proj`
- **执行环境**：AutoDL，单卡 NVIDIA GeForce RTX 3090 24 GiB，cgroup memory 90 GiB
- **当前分支**：`research/worldsim-v3`
- **A1 正式实现基线**：`198a681`（开发选择、确认矩阵、exact alias 与 finalizer）
- **A2-I0 实现基线**：`271d876`（ancestry instrumentation、module-off 等价、正式控制器与事实源）
- **A2-D1 工程基线**：`c9b2422`（quota-only policy、可重放 DriveStudio patch、配对 smoke controller）
- **A2-D1 formal 协议基线**：`387dd50`（30k 配对控制器、held-out/non-target 评测、matched-budget 裁决）
- **A2-D2 工程基线**：`1065264`（boundary/residual attribution、稳定排序、scale cap 与配对 smoke controller）
- **A2-D2 formal 协议基线**：`20b3f4d`（D1 exact alias、D2 单臂 30k、fixed/matched 与完整质量裁决）
- **A2-D2 formal 证据基线**：`482fba0`（唯一 30k 实例、5k grid、fixed/matched 与资源/不可变性证据）
- **A2 正式收口基线**：`2246693`（D2 formal 证据、非支配裁决、D2 research asset 与 D1 fallback）
- **A3-I0 协议 SHA-256**：`03fbf632645326692bbcf18ab18a08b5440c7733c709f925945c78018bb272d0`
- **A3 R0/R1 engineering guard 基线**：`9c639dd`（exact alias、row/Adam exact guard、DriveStudio patch 与 synthetic smoke）
- **A3 R1 真实 paired 工程基线**：`78741b3`（heldout-safe S-B/T0 sidecar、四单元 loss 注入、exact guard 与 checkpoint）
- **A3 R1 数值冻结基线**：`c02c8c7`；配置 SHA-256=`d9289df0b2ac7df7a7c408b5cb1601bc5f874e2922ebc9cb87961aacee43b3e3`
- **当前任务**：`WS-V3-A4-DEPLOYMENT-01`（`running`）
- **当前里程碑**：`P0 done / A0 done / A1 done_off / A2 done（tradeoff_non_dominated）/ A3 done（R1 rejected，A3*=R0-off）/ A4-P0 protocol frozen、profile next / D3-D4 not launched / F0 pending`
- **替代计划**：本文件替代 `DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md` 成为唯一当前计划
- **历史前序**：`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`、V3 原计划及其已完成事实

---

## 0. V3.1 修订目的

V3.1 创建时已完成路线切换和三场景原生基线，并进入 A1 正式实验；当前 A1 已收口、A2-I0 已完成，
A2-D1 quota-only 与 A2-D2 boundary/residual 的配对工程 smoke、formal 协议和唯一 30k formal run 均已收口。
D1→D2 的 fixed-step 与 matched-RigidNodes 两种视图仍为非支配 tradeoff；A2 以 D2 作为边界优先的后续研究资产，
保留 D1 作为低成本/全局质量回退。这是冻结完整 Pareto 后的工程分支选择，不构成“D2 全面更优”、统计显著性
或跨场景结论。
V3.1 不推翻 V3，而是在以下新事实基础上修正执行协议：

1. 6–7 月 WorldSim/GS 周报表明，工业链路的核心不是单个 3DGS 模型，而是：
   - 基础重建；
   - 动静态资产拆分；
   - 低质量资产路由与返修；
   - 生成式和谐化；
   - 多相机/多传感器渲染；
   - 可恢复的批量生产链。
2. 6 月离职交接资料进一步说明：
   - 3DGS 是 WorldSim 的显式场景资产与渲染层，不是完整世界模型；
   - 前馈模型更适合快速初始化，逐场景优化负责最终质量；
   - 编辑后必须区分“已有三维证据支持”与“完全未观测区域”；
   - 多相机 ISP、位姿抖动、动态资产、深度/法线、分块和 LOD 是相互独立的问题。
3. A1 已经暴露真实实验问题：
   - 原生 Affine 实际退化为场景级全局 RGB 仿射；
   - 不同变体曾因随机数消耗顺序造成初始化不配对；
   - C3 曾因零点导数错误被“零梯度锁死”；
   - 现有 C0/C1 结果只能说明画质差异，尚不能回答跨相机一致性与位姿稳定性。
4. 当前单卡预算不适合无差别跑完大矩阵。V3.1 改为：
   - `scene-0230` 作为开发场景；
   - `scene-0242/0255` 作为确认场景；
   - 开发场景完成子消融；
   - 确认场景只运行基线和冻结后的最优候选。

V3.1 的目标不是增加更多模块，而是让现有 A1–A4 更接近真实 WorldSim 模型生产链，并降低不可归因实验和无效算力消耗。

---

## 1. 权威边界

### 1.1 项目定位

> **面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化研究。**

本项目交付的是一条可复现、可编辑、可优化、可资产化的动态驾驶 3DGS 模型链：

```text
多相机日志 / 位姿 / LiDAR / 实例标注
        ↓
可审计预处理、初始化、相机校准
        ↓
静态背景 + 动态实例 + 天空 3DGS
        ↓
actor-aware Gaussian 资源分配
        ↓
对象轨迹编辑
        ↓
有三维证据支持区域的局部 Gaussian 精修
        ↓
剪枝、精度压缩、空间分块、资产注册与运行时加载
```

### 1.2 3DGS 与 WorldSim 的关系

本项目明确区分：

```text
3DGS：
显式三维资产、动态 actor、相机/深度渲染、编辑与实时显示

完整 WorldSim：
3DGS + 场景状态管理 + Agent/交通流 + 规划控制 + 传感器模型 + 闭环平台
```

因此，本项目可以主张“面向 WorldSim 的场景重建与资产链”，不能主张完成完整世界模型或自动驾驶安全闭环平台。

### 1.3 明确不做

- 不训练 Occupancy→多相机视频的十亿/百亿参数世界模型；
- 不恢复 V2 M5 的大型可信评测体系；
- 不重新实现基础 actor identity、scene graph 或轨迹编辑 API；
- 不把删除、平移、实例绑定包装成新贡献；
- 不接大型视频扩散作为 A3 主方法；
- 不实现完整 PBR、SSR、车辆动力学、交通流或强化学习；
- 不因周报中出现七相机、LiDAR、HIL 就在 V3.1 中扩大正式输入协议；
- 不把内部周报的“生产效率、膨胀比、复现度”等未定义口径直接作为本项目目标；
- 不把三场景实验外推为跨城市、夜间、长序列或量产泛化结论。

### 1.4 工业材料使用边界

6–7 月周报和离职交接资料只用于：

- 识别工业痛点；
- 调整模块优先级；
- 设计工程交付；
- 解释负结果和适用边界。

它们不作为代码、模型结构、指标数值或已完成能力的事实源。所有正式结论仍必须来自当前仓库、固定配置和可审计运行产物。

---

## 2. 已继承资产与当前现场

### 2.1 V2/V3 已完成资产

- DGGT：
  - 18/18 单视图；
  - 18/18 三视图；
  - 216/216 common target；
  - regional/dynamic/boundary 诊断完成。
- nuScenes：
  - `scene-0230/0242/0255`；
  - persistent `instance_token`；
  - raw 2 Hz 轨迹；
  - 三相机投影；
  - 4,356/4,356 exact-token mappings。
- StreetGS：
  - 三场景 30k checkpoint；
  - actor registry；
  - held-out 渲染；
  - per-actor Gaussian inventory；
  - `original/lateral/delete` 编辑接口。
- scene-0230：
  - 196 帧 × 3 相机 × 3 variant；
  - 1,764 张 RGB；
  - 9 个视频；
  - 非目标区域保持与编辑不变量检查。
- 工程基础：
  - WorldState；
  - actor registry；
  - typed render；
  - trajectory editor；
  - run manifest、terminal、资源守卫和 checkpoint hash。

### 2.2 A0 冻结结果

正式场景：

```text
scene-0230
scene-0242
scene-0255
```

A0 唯一汇总 run：

```text
/root/autodl-tmp/runs/worldsim_v3/
WS-V3-A0-NATIVE-BASELINE-01/
20260805T175000Z__a0-three-scene-finalize-s0-r2
```

关键结果：

| 场景 | held-out PSNR | background GS | rigid GS | high actor 状态 | boundary actor 状态 |
|---|---:|---:|---:|---|---|
| scene-0230 | 24.934 | 1,152,614 | 167,299 | available | available |
| scene-0242 | 29.107 | 843,756 | 86,255 | available | `ABSTAIN` |
| scene-0255 | 25.230 | 1,510,936 | 40,447 | available | available |

A0 已经说明：全图指标不能替代动态 actor 与边界质量。`scene-0242` 全图质量最高，但 high actor 区域质量并不最好；A2 有真实模型靶点。

### 2.3 A1 正式终态快照

- 分支：`research/worldsim-v3`
- A1 正式实现基线：`198a681`
- GPU：空闲
- 活跃训练/controller/tmux：无
- 数据盘剩余：约 62 GiB
- 当时任务：`WS-V3-A1-CALIBRATION-01`

已提交实现：

| commit | 内容 |
|---|---|
| `4958793` | C2 factorized ISP、C3 bounded pose、LiDAR 初始化审计 |
| `8debb9c` | A1 场景实验控制器 |
| `6895bd1` | 配对初始化 RNG/张量哈希控制 |
| `e4295fe` | 修复 C3 零点无梯度 |
| `20c4276` | 冻结并实现 A1 E1/E2 端点、正式控制器与定向测试 |
| `d85ef27` | 按 DriveStudio 权威定义修复 nuScenes 相机 ID 映射 |
| `14bc3c2` | 完成 A1 最小 LiDAR provenance 与初始深度 witness 审计 |
| `801db7a` | 同步 A1-E0、相机错误 run 与 LiDAR truth boundary |
| `95d0807` | 冻结并实现 ISP/位姿/速度分层 A1-D0 诊断 |
| `60ef079` | 冻结无容差 A1-S0 开发场景选择和 exact-alias 确认语义 |
| `198a681` | 冻结 A1 确认矩阵、exact-alias 登记器与三场景 finalizer |

已完成：

- C0/C1/C2/C3 配对 100-step smoke；
- C3 held-out 推理接口；
- A1-E0、LiDAR、A1-D0、开发选择与三场景 finalizer；定向 WorldSim 测试 `59 passed`；
- `scene-0230` C0/C1/C2/C3 四个配对 30k formal；
- `scene-0242/0255` C0/C1 四个确认 30k formal、冻结端点回填与两个 C0 exact alias。

`scene-0230` 完整开发结果：

| variant | global PSNR / SSIM / LPIPS | boundary actor PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | total GS | train min / peak MiB |
|---|---:|---:|---:|---:|---:|
| C0-off | 27.746 / .851 / .1764 | 27.756 / .892 / .0687 | 25.358 / .844 / .0943 | 1,360,649 | 52.05 / 24,077 |
| C1-native | 24.979 / .743 / .1694 | 22.549 / .700 / .1033 | 21.696 / .602 / .1201 | 1,316,421 | 53.69 / 24,035 |
| C2-factorized-isp | 25.011 / .743 / .1677 | 22.583 / .705 / .1043 | 21.779 / .608 / .1174 | 1,322,979 | 52.26 / 24,081 |
| C3-bounded-pose | 28.109 / .862 / .1666 | 28.169 / .897 / .0657 | 25.137 / .846 / .0938 | 1,363,040 | 56.14 / 23,779 |

当前只允许得出：

- C3 的全图与 boundary actor 画质最佳，学习位姿修正也最稳定；
- C0 的 high actor PSNR 略高于 C3，C3 的对应 SSIM/LPIPS 略好；
- C1/C2 使用较少 Gaussian，但 actor/boundary 质量明显低于 C0/C3；
- 这些画质和容量结果不能覆盖预注册 E1/E2 主端点。

开发场景选择和两个确认场景已经共同收口为 `C*=C0-off / done_off`。A1 已完成 `10/10` 个逻辑矩阵项；
C* 是 C0 exact alias，因此实际完成 `8/8` 个唯一训练。

#### A1-E0 正式回填结果

冻结配置：`configs/worldsim_v3/a1_endpoints_v1.yaml`，SHA-256
`60c211625860c25edf92842b88bdb040ea8c180b12fe0fa78f2fc1c342bc4051`。相机 ID 以 DriveStudio nuScenes
事实源为准：`0=CAM_FRONT / 1=CAM_FRONT_LEFT / 2=CAM_FRONT_RIGHT`。有效正式 run：

```text
C0  20260806T141409Z__scene0230-c0-a1-e0-formal-full-camera-map-fix-s0-r2
C1  20260806T141623Z__scene0230-c1-a1-e0-formal-full-camera-map-fix-s0-r1
C2  20260806T154541Z__scene0230-c2-a1-e0-formal-full-s0-r1
C3  20260806T164852Z__scene0230-c3-a1-e0-formal-full-s0-r1
```

| 端点（越低越好） | C0-off | C1-native | C2-factorized-isp | C3-bounded-pose |
|---|---:|---:|---:|---:|
| E1 valid / candidate / coverage | 28,744 / 266,631 / 10.780% | 29,151 / 274,658 / 10.614% | 31,299 / 275,877 / 11.345% | 29,846 / 268,826 / 11.102% |
| E1 median / P90 | .05951 / .14719 | .06289 / .15623 | .06544 / .16160 | .06309 / .15448 |
| E2 high mean / P90 / coverage | .004813 / .010895 / 26.316% | .004751 / .010895 / 28.070% | .004844 / .011734 / 28.070% | .004930 / .011734 / 26.316% |
| E2 boundary mean / P90 / coverage | .003547 / .006353 / 35.294% | .004450 / .007626 / 35.294% | .003346 / .005447 / 35.294% | .003592 / .006537 / 35.294% |

C2 只改善 boundary role E2，high role 和 actor/boundary LPIPS 退化；C3 的 E1 和两个 E2 role 均未严格优于
C0。四次评估前后 checkpoint SHA 一致；少量 E1/E2 panel 只做坐标/投影工程 sanity check，不作为人工质量结论。

首次 formal 端点 run `20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 使用了继承自旧脚本的错误
相机标签，导致本应相邻的相机对被错误解释；该 run 已保留并明确标记 `rejected / INVALID_CAMERA_ID_LABEL_MAPPING`，
不得进入汇总。

#### A1-D0 与开发场景选择

A1-D0 配置 `configs/worldsim_v3/a1_diagnostics_v1.yaml` SHA-256 为
`a445078d3bea89a78a0c9e6544a94a2be4c9c2e71f45aec4a9d8878b4c6593c1`。正式 run
`20260806T170219Z__scene0230-a1-diagnostics-c0-c3-formal-s0-r1`=`done`：输入速度层
near-static/low/normal=`2/18/176` 帧；C1 learned pose residual translation median/P90=`7.256/12.215 mm`、
rotation=`0.1660/0.35465°`，C3 bounded pose 为 `1.703/2.338 mm`、`0.02553/0.03337°`。near-static
只有 2 帧，且这些量是学习修正幅值，不是独立 pose GT。

选择协议实现提交为 `60ef079`；`configs/worldsim_v3/a1_dev_selection_v1.yaml` SHA-256 为
`a45699ebf696c875a18832f8db920a6106837a1e4f235dcd9036eff48dfbc609`。协议如实记录：V3.1 7.5 的文字要求
在开发结果已可见后、确认场景前被操作化为无容差严格 Pareto，没有新增事后数值阈值。正式 run
`20260806T171417Z__scene0230-a1-dev-selection-formal-s0-r1`=`done`，结论为
`C*=C0-off / done_off`。C2 仅单个 E2 role 改善；C3 画质和位姿稳定性最佳但没有主端点改善，均不满足候选合同。

#### A1 确认矩阵与正式终态

确认配置提交为 `198a681`，SHA-256 为
`63a3cc607ccfddbb714cc81d0570da356263c01c5a68880345953023d2d6a8cd`。`scene-0242/0255` 的 C0/C1
四个 30k 训练及冻结 E1/E2 回填均为 `done`，评估前后 checkpoint SHA 不变；两个 C* 项登记为指向对应 C0
source run/checkpoint 的 exact alias，没有新训练或评测。

| 场景 | C0 global PSNR / LPIPS | C1 global PSNR / LPIPS | 冻结合同结论 |
|---|---:|---:|---|
| scene-0242 | 30.064 / .1108 | 29.161 / .1122 | C1 不 eligible；boundary role 保持 `ABSTAIN` |
| scene-0255 | 27.255 / .2086 | 25.240 / .1921 | C1 原始 E1/E2 error 较低，但 coverage 与 actor/boundary 质量不满足合同 |

finalizer `20260806T211248Z__a1-three-scene-finalize-s0-r1`=`done`：`10/10` 逻辑项、`8/8` 唯一训练，
正式终态为 `C*=C0-off / done_off`。必须同时保留边界：原始端点方向存在场景依赖，不能表述成“C0 每场景每项指标都最好”。

### 2.4 当前 A2-I0 instrumentation 现场快照

- canonical r3 的项目提交基线：`research/worldsim-v3@70cf2b2` + 不可变 source snapshot；当前实现由
  `271d876` 收口；
- DriveStudio 上游：`e59bda4fa681f829dbb1d65f0de582b0f633c450`；patched worktree：
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-r5`；
- 冻结配置：`configs/worldsim_v3/a2_instrumentation_v1.yaml`，SHA-256
  `bac1ec5b3642470a999e7f0cf8ddc9cf5b4d9a1445029c43ae92601929f4bfce`；
- instrumentation patch SHA-256：`87c084f77ed5d6395acce95abb992ca86004bdc47b68154878bf462a0fb345b0`；
- 当前有效 formal run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T071500Z__a2-i0-ancestry-formal-s0-r3`=`done`；
- WorldSim 定向测试：`66 passed`；patched worktree verify、patch reverse-check 和当前 working-tree
  `git diff --check` 通过。

I0 已在真实 DriveStudio `RigidNodes` 类路径上的确定性合成 refinement contract 中验证：

- module-off 与 module-on 的全部原生 checkpoint tensor 逐位一致，无 mismatch；
- module-off 不增加 ancestry checkpoint key，module-on 增加并可 round-trip；
- 8 个初始 Gaussian 经 1 split、1 clone、1 prune 后保留 10 个，分配 11 个全局 ID；
- 最终来源计数为 LiDAR 7、split 2、clone 1；parent、lineage root、actor ID 与 prune 后索引一致；
- online 更新 `visibility_count/screen_grad`；boundary、photometric、depth、normal 只冻结归因 API；
- I0 r3 的 `nearest_lidar_distance` 只对 actor 做 exact offline materialization，background 因无有界参考集 deferred；
  后续 D2 final 虽记录 Background direct LiDAR roots/lineage，仍不等于 target ray 的 measured depth。

该 run 只关闭 ancestry instrumentation 工程门禁，不是 `scene-0230` 真实训练或质量证据；后续 D1 smoke
证据单独登记于 9.2.1，不能倒写或扩大 I0 结论。
本次 source commit 只允许包含 A2-I0 代码、测试与直接相关文档，不得混入保留的 V2 M5 工作。

---

## 3. V3.1 研究问题

V3.1 保留四个问题，不继续扩张：

### Q1：校准

> 受约束的多相机成像校准与位姿残差，能否改善跨相机一致性、动态边界和低速稳定性，而不是仅改变全图 PSNR？

### Q2：动态 actor 资源分配

> 原生统一梯度驱动增密/剪枝是否适合动态 actor？能否用 actor quota、边界尺度、残差和几何支持改善质量—Gaussian 数量 Pareto？

### Q3：编辑后局部三维精修

> 对已有多视图/LiDAR 支持的受影响区域，只优化局部 Gaussian，能否降低残影、空洞、深度排序错误和时序闪烁？

### Q4：工程生产链

> 能否把最终场景拆成可恢复、可加载、可回退的静态块与动态 actor 资产，并在质量可控的前提下降低大小、显存、加载时间和渲染开销？

F0/F1 前馈路线是辅助问题：

> 前馈重建是否更适合作为初始化，而不是与逐场景优化做非对等排行榜？

---

## 4. 两阶段实验设计

### 4.1 场景角色重新冻结

鉴于 `scene-0230` 的 A1 held-out 结果已经用于观察和调试，V3.1 不再假装三场景完全等价盲测。

```text
开发场景：
scene-0230

确认场景：
scene-0242
scene-0255
```

规则：

- 所有模块的候选、阈值和调度只在 `scene-0230` 冻结；
- `scene-0242/0255` 不再调参；
- 最终“支持”必须在两个确认场景方向一致，或至少 2/3 场景一致且确认场景无相反显著退化；
- `scene-0242` boundary actor 继续 `ABSTAIN`，不得换对象；
- 已经看过的 `scene-0230` 结果只能用于开发，不作为独立确认依据。

### 4.2 A1 新矩阵

开发阶段：

```text
scene-0230 × C0/C1/C2/C3 = 4 runs
```

确认阶段：

```text
scene-0242 × C0/C1/C* = 3 runs
scene-0255 × C0/C1/C* = 3 runs
```

其中 `C*` 是在 `scene-0230` 按预注册端点冻结的候选，可以是 C2、C3，也可以是 C0/C1。不得为了“必须有改进”
强制选择 C2/C3。当前已冻结 `C*=C0-off`；确认场景的 C* 项必须登记为同一 C0 source run/checkpoint 的
exact alias，不重复训练一个完全相同的变体。

总计：

```text
10 logical formal entries
当前完成 10/10
```

当 C*=C0/C1 时，10 个逻辑项对应 8 个唯一 30k 训练；当前唯一训练进度为 `8/8`。alias 必须保留独立矩阵项、
source run、checkpoint SHA 和 `alias_of`，但不得写成独立随机重复或独立模型证据。

### 4.3 A2/A3 新矩阵

A2：

```text
scene-0230：D0/D1/D2，D3/D4 条件式
scene-0242/0255：D0 + 冻结 D*
```

A3：

```text
scene-0230：R0/R1/R2/R3/R4 逐步消融
scene-0242/0255：R0 + 冻结 R*
```

这样保留因果可解释性，又避免所有子模块在三场景全排列。

### 4.4 公平性合同

- 训练图像、相机集合、held-out split、分辨率和 actor cohort 固定；
- 同一比较组必须显式重置 Python/NumPy/Torch RNG；
- 比较组必须保存初始化张量 SHA 和初始 Gaussian 数；
- 正式 held-out 不用于确认场景调参；
- fixed-step 与 matched-Gaussian-budget 两种视图同时报告；
- 不因某场景失败而换 actor、删场景或缩小分母；
- 每个 formal run 使用唯一 task ID、timestamp、source/config hash 和 terminal；
- 任何旧 `running` terminal 无进程时必须收口为 stale/interrupted，不得理解为后台运行。

---

## 5. 指标与端点合同

指标用于模型消融，不建设新的独立评测平台。

### 5.1 共用指标

| 维度 | 必报指标 |
|---|---|
| 全图 | held-out PSNR、SSIM、LPIPS |
| actor | actor PSNR、SSIM、tight-crop LPIPS、可见帧覆盖 |
| 边界 | 固定宽度 boundary band PSNR/SSIM/LPIPS |
| 规模 | total/background/per-actor GS |
| 资源 | train wall time、peak VRAM、peak RAM、checkpoint bytes |
| 不变量 | checkpoint hash、非目标参数 drift、registry/trajectory invariants |

### 5.2 A1 两个主端点

A1-E0 已在新 30k formal 前冻结并实现以下端点。唯一冻结配置为
`configs/worldsim_v3/a1_endpoints_v1.yaml`，SHA-256
`60c211625860c25edf92842b88bdb040ea8c180b12fe0fa78f2fc1c342bc4051`；后续 C2/C3 与确认场景不得改定义。

#### E1：跨相机光度残差

在相机重叠区选择满足以下条件的静态 3D 支持：

- 非天空；
- 非动态 actor；
- 不在明显深度不连续带；
- 有足够 opacity/visibility；
- 在两个相机均有效投影。

比较同一三维支持在相机对中的校正后 RGB，报告：

- 每相机对 median residual；
- P90 residual；
- coverage；
- 分静态背景、近场和远场。

禁止只报告拼接图视觉观感。

冻结实现细节：

- 相机对为 `CAM_FRONT_LEFT↔CAM_FRONT`、`CAM_FRONT↔CAM_FRONT_RIGHT`，双向投影；
- 以 8 px 网格采样，排除 GT sky/dynamic/egocar、相对深度不连续带、低 opacity、高 dynamic opacity 和
  目标相机遮挡不一致点；
- C1/C2/C3 对观测 RGB 应用对应 affine 的逆变换，兼容 `[H,W,3,4]` per-pixel 与 `[3,4]` global；
- residual 为 `[0,1]` RGB 三通道绝对差均值，近/远阈值 20 m；
- 聚合报告 pooled valid support 的 median/P90/candidate/valid/coverage；少于 256 个支持或 coverage `<1%`
  时显式 `ABSTAIN`。

#### E2：动态 actor 支持边界重投影误差

使用 A0/M2 已冻结的 actor 选择和三维投影事实源：

```text
冻结 actor 3D 支持投影
vs
original/delete 配对渲染产生的 actor effect-mask 边界
```

报告归一化双向边界距离、P90 和 coverage。该指标衡量的是 actor 三维支持与渲染边界一致性，不冒充精确语义分割 IoU。

实现前必须固定：

- 边界提取；
- dilation 宽度；
- 距离归一化；
- 无有效投影时的 `ABSTAIN`；
- 不得在看到 C0/C1 结果后切换定义。

已冻结为：paired original/delete 的 uint8 差分阈值 2、2 px dilation；actor 支持来自所选 RigidNodes 的
opacity render，阈值 0.05；边界为 1 px inner 8-connected boundary；距离为双向边界距离并按图像对角线归一化。
每个 role 至少 3 张有效图且 coverage 不低于 5%，否则显式 `ABSTAIN`。汇总报告 mean/median/P90、
candidate/valid image 与 coverage。

### 5.3 A1 诊断项

- 原始与优化后 pose residual；
- 相邻帧平移/旋转一阶差分；
- 二阶抖动；
- 按速度分层：
  - near-static；
  - low-speed；
  - normal；
- 按相机方位分层：
  - 前向高重叠；
  - 侧向低重叠；
  - 近场遮挡；
- ISP 参数幅值、时间平滑性和相机间分布；
- Gaussian 数量与增密曲线。

速度分层阈值必须在读取结果前由输入速度分布冻结，不允许事后选择。

### 5.4 阶段主端点

- A1：E1 跨相机光度残差 + E2 actor 边界重投影误差；
- A2：actor/boundary LPIPS 与 per-actor GS、wall time 的 Pareto；
- A3：支持区域 hole error、depth-order violations、temporal flicker；
- A4：checkpoint bytes、peak VRAM、cold-load time、P50/P95 FPS 与相对 A3 的质量损失。

---

## 6. 任务注册表

| Task ID | 状态 | 交付物 | 完成门禁 |
|---|---|---|---|
| `WS-V3-P0-ROUTE-01` | done | V3 路线与事实冻结 | 已提交并同步事实源 |
| `WS-V3-A0-NATIVE-BASELINE-01` | done | 三场景原生 checkpoint、registry、held-out/actor/资源基线 | 3/3 场景闭环 |
| `WS-V3-A1-CALIBRATION-01` | done_off | E1/E2、C0–C3 开发消融、两确认场景复核 | 10/10 逻辑项、8/8 唯一训练；C*=C0；finalizer done |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | pending | Instant NuRec 官方代码、输入输出、license、导出能力审计 | 形成可执行/不可执行事实结论 |
| `WS-V3-F1-FEEDFORWARD-INIT-01` | conditional | 前馈深度/高斯初始化 + StreetGS 短步精修 pilot | 只在 F0 输出可转换资产时启动；不阻塞 A2 |
| `WS-V3-A2-ACTOR-DENSIFY-01` | done | I0、D1/D2 smoke 与 formal、fixed/matched Pareto 和资产路由 | `tradeoff_non_dominated`；A2*=D2 boundary-priority，D1 fallback；D3/D4 未启动 |
| `WS-V3-A3-LOCAL-REFINE-01` | done | I0、R0 exact alias、R1 S-B 工程/replay 与 heldout 负结果 | R1 资源门失败且诊断 Pareto tradeoff；A3*=R0/D2 exact alias；formal、R2–R4 未授权 |
| `WS-V3-A4-DEPLOYMENT-01` | running | 端到端 profile、prune、FP16、chunk、registry、resume | P0 protocol 已冻结，当前执行 profile；P1/P2/P3/P5 未授权 |
| `WS-V3-R0-INTEGRATION-01` | pending | 最终模型链、负结果、复现包和工程说明 | 所有 terminal、配置和结论可追踪 |

---

## 7. A1：成像、位姿与初始化消融

### 7.1 固定变体

#### C0-off

- 删除 Affine/CamPose 模块；
- 作为诊断下界；
- 不能默认视为弱基线。

#### C1-native

- DriveStudio 原生 Affine + CamPose；
- 等同原生能力；
- 当前审计显示 Affine 实际退化为场景级全局 RGB 仿射。

#### C2-factorized-isp

输入只使用真实存在的元数据：

```text
camera_id embedding
+ normalized continuous timestamp basis
```

没有 exposure 元数据时，不允许声称使用 exposure embedding。

输出为受限 RGB 仿射：

- 参考相机锚定；
- 小幅增益和偏置；
- 防止自由成像模块吞噬几何误差；
- 保存逐相机、逐时间参数曲线。

#### C3-bounded-pose

在 C2 上增加：

- 有界平移残差；
- 有界轴角旋转残差；
- 逐相机时间平滑；
- pose warm-up；
- 校准模块分阶段启用。

建议固定调度：

```text
阶段 1：Gaussian warm-up，冻结 pose
阶段 2：开启 bounded pose + smoothness
阶段 3：固定或降低 pose 学习率，完成外观收敛
```

具体 step 在 `scene-0230` smoke/development 冻结后不再修改。

#### C4-rolling-shutter

当前固定为：

```text
not_supported
```

原因：

- 只有 frame timestamp；
- 无 exposure；
- 无 readout direction；
- 无 row timing。

不得从相机编号、采集顺序或视觉现象反推 timing。

### 7.2 A1-E0：端点冻结（done）

以下门禁已在新的 30k run 前完成：

1. E1/E2、coverage 与 `ABSTAIN` 已实现；
2. C0/C1 已用原 checkpoint 完成全 held-out 回填，无重训；
3. checkpoint 评估前后 SHA 一致；
4. QA panel 已做坐标和投影工程检查；
5. 公式、mask、相机对与配置 hash 已写入本计划和冻结 YAML；
6. 相机 ID 标签错误的旧 formal 已显式 `rejected`，修复后的 C0/C1 run 为唯一有效证据。

### 7.3 LiDAR provenance 边界（A1 最小集 done）

A1 只要求、且已经完成以下最小 provenance：

- 原始 LiDAR/随机 seed 输入来源；
- 每个输入块 SHA、点数和 actor 映射；
- 初始背景/actor Gaussian 数；
- actor 可见帧数；
- 初始 depth residual。

以下内容不作为 A1 done 门禁：

- 增密后逐 Gaussian 完整 ancestry；
- parent-child lineage；
- 所有复制/分裂的来源链。

这些在 A2 的 instrumentation stage 完成，避免 A1 无限扩张。

冻结配置：`configs/worldsim_v3/a1_lidar_provenance_v1.yaml`，SHA-256
`f2fd1712cf4ddd75c1c4d1da4a426dcf7e1340a5fd943066401ba881f51c5639`。有效正式 run：

```text
20260806T143644Z__scene0230-a1-lidar-provenance-formal-full-witness-s0-r1
```

正式事实：196 个 LiDAR/pose block、6,804,832 个 raw points；24 个 actor 输入共 75,002 点。记录的背景
LiDAR tensor、全部 actor point/color tensor、actor mapping 和 RigidNodes 初始计数均 exact match。初始 held-out
sparse depth 有效点 172,844，coverage=1.0；绝对 residual median/P90=`7.679/35.958 m`，相对 residual
median/P90=`0.6649/0.9077`。

该 depth 只能标为 `seed0_reconstructed_initialization_witness_not_exact_source_initialization`，不能冒充训练源初始化的
exact replay：DriveStudio 随机 near/far 点经过 CUDA visibility filter 后，即使 seed=0，重复初始化的背景 Gaussian
计数仍从源运行的 946,484 变化为 946,597、946,309、正式 witness 的 946,291。正式协议因此以已记录
LiDAR/actor tensor exact match 为门禁，把随机背景 exact reproduction 固定为 `report_not_gate`；没有事后设置计数
容差。逐 Gaussian ancestry、parent-child 和 split/clone 来源链按原计划后移至 A2 instrumentation。

### 7.4 A1 执行顺序

```text
A1-E0 端点冻结
→ 现有 C0/C1 补算 E1/E2
→ 完成 scene-0230 C2/C3 formal
→ 在 scene-0230 冻结 C*
→ scene-0242 C0/C1/C*
→ scene-0255 C0/C1/C*
→ A1 finalizer
```

### 7.5 A1 判定

增强候选必须：

- E1 或 E2 至少一个改善；
- 另一个不出现关键退化；
- actor/global LPIPS 不出现不可接受退化；
- GS 数和训练时间收益/代价透明；
- 两个确认场景方向一致，或准确记录场景依赖。

合法终态：

```text
done_enhanced
done_native
done_off
rejected
blocked
```

若 C0 最终 Pareto 最优，允许选择 C0；不能为了“模型必须更复杂”强行选择 C2/C3。

### 7.6 开发场景 C* 冻结（done）

A1-S0-v1 将 7.5 操作化为：E1 的 median/P90 必须同时严格下降且 coverage 不降低；E2 的两个冻结 role
必须全部 non-degraded，且至少一个 role 的 mean/P90 同时严格下降；另一主端点不得退化；global、两个 actor
region 和两个 boundary band 的 LPIPS 均不得高于 C0。比较不设容差。若多个候选通过，依次按改善主端点数、
global LPIPS、total GS、训练时间与固定候选顺序裁决；若无候选通过则回退 C0。

该操作化是在开发结果已可见后、确认场景前完成，因此只能称“确认前冻结的透明规则”，不能称盲测前数值阈值。
正式 finalizer 校验所有 source terminal、端点/诊断配置 SHA、配对初始化 SHA、source/checkpoint 绑定与评估前后
checkpoint 不变。结果为：

- C1：无主端点改善，actor/boundary LPIPS 退化；
- C2：boundary role E2 改善，但 high role E2 与 actor/boundary LPIPS 退化；
- C3：全部冻结 LPIPS 可接受，但 E1/E2 均无严格改善；
- 冻结 `C*=C0-off`，开发终态 `done_off`。

---

## 8. F0/F1：前馈范式与初始化

### 8.1 F0 官方能力审计

固定检查 Instant NuRec：

- 官方 revision；
- license；
- checkpoint provenance；
- 硬件要求；
- 相机模型；
- cadence；
- pose/LiDAR/实例输入；
- static/dynamic/sky/ISP 的实际 CLI 导出能力；
- 是否保留 actor registry；
- 一窗口 wall time、VRAM 和 schema。

网页演示能力不能写成本地 CLI 能力。

### 8.2 F1 条件式初始化 pilot

只有 F0 能输出可转换的 depth/point/Gaussian 资产时启动。否则可使用已跑通的 DGGT 作为工程 pilot，但必须明确模型身份。

固定 `scene-0230`，比较：

```text
I0：原生 LiDAR 初始化
I1：前馈 depth → 点云初始化
I2：前馈 Gaussian → StreetGS 初始化（仅在 schema 可严格转换时）
```

短步曲线：

```text
1k / 5k / 10k
```

报告：

- 达到同等质量的步数；
- 总 wall time；
- 前馈推理成本；
- actor/boundary 质量；
- registry 保留；
- 最终 Gaussian 数。

F1 不阻塞 A2，也不升级成新的完整训练主线。若转换成本和返修成本抵消提效，允许输出负结论。

---

## 9. A2：实例感知的动态 Gaussian 增密与剪枝

### 9.1 目标

不是简单“给车辆更多 Gaussian”，而是：

> 在动态 actor 内进行可解释的资源分配，抑制跨边界大尺度 Gaussian，并降低低质量 actor 进入后续生成式返修链的比例。

### 9.2 Instrumentation stage（I0 done）

I0 在 D1 前实现：

- `actor_id`；
- `init_source`：
  - LiDAR；
  - random-near；
  - random-far；
  - split；
  - clone；
- `parent_id` 或 lineage root；
- `visibility_count`；
- `screen_grad`；
- `boundary_contribution`；
- `photometric_residual`；
- `depth_residual`；
- `normal_residual`（已有可靠 normal 时）；
- `nearest_lidar_distance`。

模块关闭时必须逐位退化为原生 `RigidNodes`。

当前实现与边界：

- `gaussian_id/actor_id/init_source/parent_id/lineage_root_id/birth_step/generation` 随 split、clone、
  external actor replacement、prune 和 checkpoint round-trip 保持可审计；
- `visibility_count/screen_grad` 接入原生训练在线路径；
- boundary、photometric、depth、normal residual 当前只提供显式 update API，不在 I0 合成 run 中冒充已观测值；
- normal 没有可靠输入时保持 schema-only；I0 的 background bounded `nearest_lidar_distance` 保持 deferred，
  后续 direct LiDAR-root ancestry 仅按 provenance 使用；
- module-off 无额外 RNG draw、无额外 checkpoint key，原生 tensor 逐位相等；
- canonical r3 只覆盖 deterministic synthetic `RigidNodes` refinement，不替代 D1 的真实 `scene-0230` smoke。

### 9.2.1 D1 quota-only 工程门（smoke done）

- 实现提交：`c9b2422`；配置：`configs/worldsim_v3/a2_d1_v1.yaml`，SHA-256
  `6895370625080ccab327e731264e9ebb0f980499b8fec87d02d9efb2e56b14af`；
- DriveStudio upstream=`e59bda4`，canonical patched worktree=`/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d1-r5`，
  quota patch SHA-256=`c232af2c5fa532016943f399830c85ebba612078871b7c1a296bda816ae7bb1b`；
- D1 只改变 `RigidNodes` 的 actor threshold=`0.00025` 与 per-actor quota；Background 保持原生 threshold=`0.0005`
  且不启用 quota；native cull、boundary/residual、scale cap、LiDAR/visibility 与 D2–D4 均未混入；
- 初始 actor count 冻结为 24 项、合计 `75,002`；min/max 公式分别为
  `max(1, ceil(0.5 * initial))` 与 `max(min, min(12000, ceil(2.4 * initial)))`，总和为
  `37,504 / 180,013`；
- canonical paired smoke：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T081330Z__a2-d1-paired-smoke1k-s0-r4`，
  `terminal=done`，scene-0230 / seed 0 / D0→D1 各 1000 step 顺序执行；
- D0/D1 物化配置预算匹配，初始化 provenance 完全一致；module-off 原生 tensor 逐位相等，D1 quota 与 ancestry
  checkpoint 均可 round-trip，原生 tensor 无 NaN/Inf；
- D1 共执行 5 次 quota event，接受 `93,057` 个子点、拒绝 `30,171` 个超额 parent；最终 Rigid 总数
  `152,830`，24/24 actor 均不超过冻结最大值；D0 Rigid 总数为 `125,915`；
- D0/D1 duration=`110.91 / 110.97 s`，peak GPU sample=`12,807 / 12,795 MiB`，peak cgroup=
  `5,392,334,848 / 5,661,368,320 bytes`；无 OOM，结束后 GPU 回到 0 MiB。
- patch replay/reverse-check、synthetic integration 与 WorldSim 定向回归通过；当前回归为 `75 passed`。

该 smoke 只关闭真实训练路径、资源、quota 与 checkpoint 工程门；它没有执行冻结 held-out actor/boundary
质量合同，不能把更多 Gaussian 解释为方法改进。结论是“允许冻结 D1 formal 协议”，不是“D1 已通过方法门禁”。

### 9.2.2 D1 formal 结果（done）

- 唯一 canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T085400Z__a2-d1-paired-formal30k-s0-r1`，
  `terminal=done`；source commit=`f32f96b`；summary SHA-256=
  `e3b194c2ed0563385df70ca2043dbc791bedb21068d28dc9d75fb59984c166ac`；
- D0/D1 初始化 provenance SHA 均为 `8951543c33f72f439068237f1a552fae660895f8906afbf4651f5f580981b898`，
  初始 Background/RigidNodes 均为 `946,484 / 75,002`；两份物化配置仅 variant/quota-enable 不同；
- fixed 30k 的 D0/D1 Background/Rigid/total GS 分别为
  `1,182,619 / 177,628 / 1,360,247` 与 `1,201,057 / 105,412 / 1,306,469`；
- D1 5k 网格按冻结规则选中 15k checkpoint：Rigid=`176,741`，距 D0 目标 `887`（`0.499%`，小于 2%），
  Background/total=`2,432,701 / 2,609,442`；D0 matched 视图是 fixed D0 final 的 exact alias；
- fixed 质量轴 D1/D0 更优数=`12/7`，matched 质量轴=`9/10`；两种视图的质量与质量—成本裁决均为
  `tradeoff_non_dominated`。matched D1 改善 boundary-support 多数指标，但 global PSNR/SSIM/LPIPS 与 non-target
  均明显退化，因此不能压缩成单一“D1 更好”结论；
- D0/D1 30k duration=`2883.08 / 2099.33 s`，peak GPU sample=`23,867 / 23,989 MiB`，peak cgroup=
  `10,350,350,336 / 16,012,115,968 bytes`；matched 15k elapsed=`1127.66 s`，资源峰值按完整 D1 臂上界报告；
- 6×2 checkpoint 网格、quota/ancestry、原生 finite 与 24/24 actor 上限审计通过；三次 held-out/actor 评测前后
  checkpoint SHA 不变，high/boundary/non-target 均为 `done`，cgroup `oom=0 / oom_kill=0`。

冻结裁决为 `d2_unlocked=true`，下一动作只允许冻结 D2 boundary/residual ordering 与 boundary scale cap 协议。
该结果仅限 scene-0230 / seed 0；负向和 tradeoff 证据有效，更多 Gaussian 不是自动改进。

### 9.2.3 D2 协议与工程实现（paired smoke pending）

- 冻结配置：`configs/worldsim_v3/a2_d2_protocol_v1.yaml`，SHA-256=
  `acceb7f4ce0f8dc3745de2fcaca51659891cfd82e4175f5a0e5765d77a01e567`；依赖 D1 canonical summary
  SHA `e3b194c2...66ac` 与收口提交 `f380dd2`；
- D2 完整继承 D1 的 `0.00025` gradient eligibility、minimum recovery、per-actor maximum quota、split/clone
  cost、Background 原生策略与 native cull；唯一允许的新干预是 quota 内排序和 boundary scale cap；
- boundary 信号固定为训练帧 `image_infos.dynamic_masks` 的 3px 二值形态学轮廓带；photometric residual 固定为
  detached `mean(abs(outputs.rgb - image_infos.pixels), channel)`；二者在可见、有限且中心位于图像内的 gsplat
  `means2d` 像素坐标处做 nearest-center 采样，并按 Gaussian 记录算术运行均值与观察次数；
- actor 内稳定排序键固定为：boundary observed/mean 降序 → residual observed/mean 降序 → screen-grad 降序
  → Gaussian index 升序；gradient threshold 仍决定 eligibility，不把低残差 Gaussian 事后改成 eligible；
- boundary scale cap 只作用于 boundary mean>0 且有观察的 Gaussian；geometry 先读 pre-cap scale，然后在原生
  split/clone/cull 前把最大轴同比缩到原生 `densify_size_thresh × scene_scale`，保持各轴比例并清零被 cap 行的
  Adam 一、二阶矩；不新增 RNG draw；
- D3 depth/normal、D4 LiDAR/visibility/provenance pruning、非原生 cull 与 Background 干预全部显式禁止；
- 工程提交=`1065264762569c9832219936ddae6f063d6eaf07`；独立 DriveStudio worktree=
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d2-r8`；D2 patch SHA-256=
  `80fef55195906808d74394af0b997cfccbdb88fd7cb356b45240473e55f357cc`，四层 patch replay、reverse-check
  与六文件状态核验通过；
- D1/D2 materializer normalized-match 门禁、真实 `RigidNodes` synthetic integration 与联合回归=`29 passed`；
  synthetic 记录 boundary/residual 各 6 次、1 次排序/refinement、6 个 capped Gaussian，两 actor 均停在配额 10，
  optimizer moments 清零、checkpoint round-trip、module-off native state/RNG bitwise 全部通过；
- scene-0230 paired smoke r1 见下一节；工程实现通过本身不等于 D2 方法通过。

### 9.2.4 D2 配对工程 smoke 完成

- canonical run：`/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/`
  `20260809T111304Z__a2-d2-paired-smoke1k-s0-r1`，terminal=`done`；summary SHA-256=
  `749c7d15c27cc0798c267aa8af12857f3bea52a52ea9d00f7617a3b3edda3136`，manifest SHA-256=
  `5cb7879d898839b88a46c8ec7ec34141f3402245490416d589938658f33b4c8d`；source commit=`c594e0c`；
- D1/D2 物化配置 normalized match，初始化 provenance 与 50 actor 的冻结 initial quota 精确匹配；两臂均到 step 1000，
  D1=`Background 1,141,192 / Rigid 152,733`，D2=`Background 1,144,988 / Rigid 152,807`；
- D2 真实 attribution 共 1001 个 observation event，boundary/residual 各 `10,846,748` 个投影观测；5 个 native
  refinement/ordering event，累计 cap `365` 个 Gaussian，终态 boundary-observed live Gaussian=`56,732`；
  cap、optimizer/checkpoint round-trip、quota maximum 与 native finite 审计全部通过；
- D1/D2 stage duration=`142.17/141.99 s`，torch peak GPU=`9,615/9,620 MiB`，cgroup peak=
  `16,473,858,048/16,667,971,584 bytes`；`oom=0 / oom_kill=0`；
- 裁决：`d2_formal_unlocked=true`，只解锁 D2 formal 协议冻结。1k smoke 不是质量证据，D2 比 D1 多 74 个
  Rigid 与 3,796 个 Background Gaussian，不得据此宣称方法改进。

### 9.2.5 D2 formal 协议冻结

- 配置：`configs/worldsim_v3/a2_d2_formal_v1.yaml`，SHA-256=
  `b66cf795c55dfe65315ecf49c09951482d8d6809ce7d001b901942a6bd9a05bc`；实现提交=
  `20b3f4dc6bd09f371bb4cf1855370493b8abfc68`；39 项定向回归通过；
- D1 reference 不重训，固定为 D1 formal r1 的 immutable exact alias：summary SHA=`e3b194c2...66ac`，
  source=`f32f96b`，initialization provenance SHA=`8951543c...b898`，fixed checkpoint SHA=
  `c9d2a052...af52`，30k counts=`Rigid 105,412 / Background 1,201,057`；运行前后必须复核 checkpoint SHA；
- 唯一新训练臂为 D2 30k / seed 0，每 5k 保存一次；物化 D1/D2 formal configs 除 variant、actor ranking 与
  D2 enable 外 normalized match，D2 initialization provenance 必须与 D1 alias SHA 精确相同；
- fixed 视图比较 D1 alias 30k 与 D2 30k；matched 目标锁定 D1 fixed 的 `105,412` 个 Rigid Gaussian，从 D2
  5k–30k 网格按最小绝对 gap、再最早 step 选择，relative gap 必须≤2%；禁止 pruning/retrain/retune/mutation；
- held-out global、high-support、boundary-support、boundary band、non-target 与 exact Pareto 轴全部继承 D1；
  shared comparator 中旧 `d0/d1` 字段明确映射为 `D1 baseline / D2 candidate`，不得混淆结论标签；
- 只读 preflight=`done`，输出 SHA=`9cf49af0be9a2676c6c113bee963efb79704bb9434083857684f97bd19caaa28`；
  GPU used=`0 MiB`、free disk=`47.92 GiB`，合同、smoke、D1 alias、r8 patch 与 cgroup 门禁均通过。

### 9.2.6 D2 formal 结果与 A2 收口

- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T113230Z__a2-d2-formal30k-s0-r1`，
  terminal=`done`；source=`482fba0`，summary SHA-256=
  `9c41dfc83c9da0a14201e1c719fb3d0e2cf59dd1ad20cd279c6e1a9a1c97de7d`，manifest SHA-256=
  `260af5d99f3d3ece4f2c178f8c18385338432da9fbf94b7d8a4603163db20926`；
- D2 唯一 30k checkpoint SHA-256=
  `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c`，终态
  `Rigid 104,704 / Background 1,205,164 / total 1,309,868`；D1 reference 保持
  `Rigid 105,412 / Background 1,201,057`，checkpoint 运行前后 SHA 均为 `c9d2a052...af52`；
- 5k、10k、15k、20k、25k、30k 六个 checkpoint 均通过 finite、quota maximum 与 boundary scale-cap 审计。
  D2 共记录 `30,001` observation event，boundary/residual observations 各 `591,405,097`，`295` 个
  refinement event，累计 cap `70,764` 个 Gaussian，`158` 个 quota event 接受 `609,139` children；
- D1/D2 initialization provenance SHA 都是 `8951543c...b898`，物化配置符合配对合同。matched 目标为
  D1 fixed 的 `105,412` 个 Rigid；唯一合法候选为 D2 30k，gap=`708 / 0.67165%`，因此 matched D2 是 fixed
  D2 的 exact alias，不产生第二个训练、剪枝或调参分支；

| scene-0230 formal 指标 | D1 | D2 | D2-D1 方向 |
|---|---:|---:|---|
| global PSNR / SSIM / LPIPS | 27.770024 / .850915 / .177704 | 27.703188 / .850333 / .178344 | 三项轻微退化 |
| high-support actor PSNR / SSIM / LPIPS | 25.123809 / .840230 / .096602 | 25.071334 / .838744 / .095563 | PSNR/SSIM 退化，LPIPS 改善 |
| boundary-support actor PSNR / SSIM / LPIPS | 28.465795 / .899698 / .063419 | 28.678742 / .899129 / .064110 | PSNR 改善，SSIM/LPIPS 退化 |
| boundary-support boundary-band PSNR / SSIM / LPIPS | 25.770024 / .821572 / .048382 | 26.171399 / .828868 / .044568 | 三项改善 |
| non-target PSNR / SSIM / LPIPS | 26.890073 / .848493 / .058316 | 26.772341 / .847950 / .058242 | PSNR/SSIM 退化，LPIPS 轻微改善 |

- fixed 与 matched 的 strict-quality Pareto 都是 `tradeoff_non_dominated`：D1/D2/equal axes=`11/8/0`；
  quality-cost Pareto 也是 `tradeoff_non_dominated`：`14/9/1`。D2 训练 wall time=`2720.82 s`，高于 D1 的
  `2099.33 s`；两者 sampled peak GPU 都为 `23,989 MiB`，train cgroup peak 分别为
  `16,012,115,968 / 21,676,654,592 bytes`；
- 完整 run 的 `297` 条资源记录覆盖四个 stage，全部 `completed`，`oom=0 / oom_kill=0`；full-run peak GPU=
  `23,989 MiB`，peak cgroup=`25,837,490,176 bytes`（评测阶段），结束时 GPU=`0 MiB`；
- A2 正式状态为 `done`，裁决类为 `tradeoff_non_dominated`。后续 A3 冻结 `A2*=D2-boundary-residual`
  作为 boundary-priority research asset，并保留 D1 quota-only 作为低成本/全局质量回退。该选择基于完整 Pareto
  与 A2 的边界靶点，不设置结果后数值阈值，也不宣称 D2 支配 D1；
- `d3_unlocked=false`：当前没有可注册的可靠 depth/normal 输入；D4 也不因 ancestry instrumentation 已完成而
  自动启动。A2 不再追加 D3/D4 训练臂，条件项必须在后续独立协议中重新取得授权。

### 9.3 子消融

| 实验 | 改动 | 状态要求 |
|---|---|---|
| D0 | 原生统一规则 | baseline |
| D1 | actor/background 分离阈值 + 每 actor 最小/最大 quota | 必做 |
| D2 | D1 + boundary/residual 排序 + boundary scale cap | 必做 |
| D3 | D2 + depth/normal consistency | `not_unlocked`；有可靠、可注册 depth/normal 才做 |
| D4 | D2/D3 + LiDAR/visibility/provenance-aware pruning | `not_launched`；instrumentation 只是必要非充分条件 |

执行规则：

```text
D1 smoke → D1 formal
→ D2 smoke → D2 formal → A2 done / tradeoff_non_dominated
→ D3/D4 保持条件式、未启动
```

不得一次合并 D1–D4。

### 9.4 Actor 资产输出

每个 actor 输出：

```text
actor_quality.json
```

至少包括：

- instance token；
- Gaussian 数；
- 可见相机/帧数；
- boundary/residual；
- depth/LiDAR support；
- 新视角或 held-out 质量；
- 是否建议进入 A3 局部返修；
- 不可用原因。

它是模型资产路由信息，不是独立评测平台。

### 9.5 判定

同时报告：

- fixed-step；
- matched-Gaussian-budget；
- actor/boundary LPIPS；
- boundary error；
- per-actor GS；
- wall time；
- peak VRAM。

更多 Gaussian 不自动算改进。若 D1/D2 均不改善 Pareto，A2 可 `rejected`，A3 回退使用 A1 产物。当前正式结果
不是 dominance，而是 D1/D2 同处非支配前沿；因此 A3 使用 D2 boundary-priority asset，同时必须保留 D1 fallback
和上述完整成本/退化披露。

---

## 10. A3：证据分层的编辑后局部 Gaussian 精修

### 10.0 A3-I0 语义协议冻结

- 冻结配置：`configs/worldsim_v3/a3_local_refine_protocol_v1.yaml`，SHA-256=
  `03fbf632645326692bbcf18ab18a08b5440c7733c709f925945c78018bb272d0`；依赖 A2 closeout=`2246693`，
  D2 selected checkpoint SHA=`1a061247...e7c`、summary SHA=`9c41dfc8...de7d` 与 registry SHA=
  `ed57764e...0c68`；D1 checkpoint SHA=`c9d2a052...af52` 只作 fallback；
- 固定 `scene-0230 / scene_index 179 / seed 0 / cameras 0,1,2`，actor roles 为 high-support 与
  boundary-support，编辑只含已提交 M4 合同的 `lateral +1m` 和 `delete full trajectory`；19 个 stride-10
  held-out frames 仅评测，禁止进入优化或支持选择；
- source/edited footprint 继续使用已提交 paired RGB difference：uint8 threshold=`2`、counterfactual dilation=
  `2px`；affected union 再固定 dilation=`3px`，并纳入有注册支持的 vacated hole 与 first-hit depth-order conflict，
  tolerance=`0.05m`。mask 是 model-counterfactual diagnostic，不是真值分割；
- target actor 只作 context，确定性 edit 后 actor 参数、轨迹与 registry 全部冻结。R1 只允许更新 affected 且属于
  S-A/S-B 的 `Background._opacities/_scales`；position、color、seed、RigidNodes 与其他模型均冻结；
- S-A 必须使用排除 target view 的真实 alternate camera/time RGB 和 calibrated reprojection；S-B 只接受 T0
  LiDAR measured 或至少两视图 calibrated geometry，禁止 RGB loss；S-C 不更新、不 seed、不进 loss，只报告
  coverage/uncertainty/ABSTAIN。ancestry `nearest_lidar_distance` 只是 provenance，不是 T0 metric depth；
- expected/first-hit/measured depth 分别固定为 `diagnostic / T1 / T0`；无名 `depth` 产品禁止。R0 为 D2
  checkpoint immutable exact alias，R1→R4 必须一次只加一个因子；首个工程门只做 R0/R1；
- `formal_training_authorized=false`。R1 paired smoke 后冻结了 steps、逐字段 learning rate、affected/seed cap、
  first-hit alpha threshold 与资源上界；heldout 只读评测协议和负结果见 10.0.2，不能直接 formal；
- A3 独立实现不得依赖工作树中未提交的 V2 M5 config、`stress_metrics.py` 或 stress runner。协议 validator、
  support classifier、affected-mask 与逐 tensor outside audit 使用独立 V3 模块；新增 `12 passed`，联合 WorldSim
  V3/materializer 回归 `98 passed`。

### 10.0.1 A3 R1 真实 S-B/T0 paired smoke 与数值冻结

- heldout-safe sidecar run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T133911Z__a3-sb-sidecar-s0-r3`；
  manifest SHA-256=`42474f73fc563a2bba4c52cbec029bb4c28d33a21ca5f3d83ad4311bb7957273`，rows SHA-256=
  `c5756ecbc0eabee9a576a55297a1739aa20e2af578aa4a5a92e727701b5138fc`。选择 frame `0/31`、camera 0，
  与 19 个 heldout frames 交集为空；S-A=`ABSTAIN/0`；
- Background affected=`16,502` 行，其中 S-B mutable=`51`、S-C abstain=`16,451`；四个
  `high/boundary × lateral/delete` unit 共 8 个 T0 geometry loss 像素，S-B RGB loss 恒为 0；first-hit alpha
  `0.5` 在该门只作冻结 visibility diagnostic；
- canonical paired run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T135921Z__a3-r1-sb-paired4-s0-r2`；
  summary/manifest SHA-256=`ba4e2b853690f0b9c9bb7bfe039b4571db16c020ce726768a1ff884b09b3557d / de717ba0a5adb1afeb416a15a53ec55f471a8eb841882f784012b04ac86b596c`；
  step `30001–30004` 每个 unit 恰好一步，R1 checkpoint SHA-256=
  `e995e7c266d9fed4e64c86813718e46ab4576bbfdf60500a637bdaeaaba78cd1`；
- 两个授权字段每步均有 finite/nonzero gradient 与行内变化；行外参数和 Adam moments、其他字段、RigidNodes、
  actor trajectory/registry、tensor shape/dtype/order 全部 exact。wall=`50.16 s`、peak GPU=`8,286.86 MiB`、
  sampled cgroup peak=`22,076,936,192 bytes`、OOM=`0`；
- 数值冻结配置=`configs/worldsim_v3/a3_r1_numeric_freeze_v1.yaml`，SHA-256=
  `d9289df0b2ac7df7a7c408b5cb1601bc5f874e2922ebc9cb87961aacee43b3e3`：steps=`4`，opacity/scaling LR=
  `0.05/0.005`，affected/mutable cap=`16,502/51`，seed cap=`0`，alpha=`0.5`，资源 ceiling=
  `120 s / 12,288 MiB / 32 GiB / 650,000,000 run bytes`；
- frozen replay run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T140534Z__a3-r1-sb-frozen-replay4-s0-r1`；
  summary/manifest SHA-256=`7d820a53de21f505a5c56043d56556edb8d3a86510488ea3956b7cfa159187c6 / 393e65d5f91c0e2072eebd7c23a1161d46422502220ceeeaa18c04905fec646d`；
  四单元 loss 逐值相同，重现同一 checkpoint SHA，wall=`50.68 s`、peak GPU=`8,286.86 MiB`、
  sampled cgroup peak=`22,631,796,736 bytes`、OOM delta=`0`；
- 这些证据只关闭真实工程链与数值可重放门。S-A 未物化，S-B pixel quality claim 禁止，R2–R4 与 formal 仍未授权。

### 10.0.2 A3 R1 heldout 只读评测负结果与 A3 收口

- 结果前冻结协议=`configs/worldsim_v3/a3_r1_eval_protocol_v1.yaml`，SHA-256=
  `eb87a9f2ea7df9bdc050a8d4e4f3cdc7c6a1115ea6f4f69e2fd3c8011904b05a`；协议/评测器提交=
  `42508fb / c8fc560`，固定 19 heldout frames × 3 cameras、R0-only masks、T0/T1 geometry、non-target/global RGB
  safeguard、exact no-tolerance Pareto、checkpoint 不可变性与 `12,288 MiB` GPU ceiling；
- 首次运行后补充的资源审计、CPU checkpoint staging、Rigid quota 兼容与 per-view render-state 释放提交依次为
  `05cee1e / c9e3df4 / ef74622 / c2eb14f`；这些修复没有更改协议、阈值、mask、端点或 checkpoint。当前联合
  WorldSim V3 回归=`139 passed`；
- canonical negative run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T144037Z__a3-r1-heldout-eval-s0-r5`，
  exit=`1`，terminal=`blocked / A3 heldout resource ceiling failed: peak_gpu_memory_mib`；resource audit SHA-256=
  `d9536f4ec937bee0694a754038b22ab75a4b6b028f20e1e6f42e38e4db9a6280`，wall/GPU/cgroup/run bytes=
  `117.98 s / 14,241.40 MiB / 23,749,709,824 / 299,910`，OOM/OOM-kill delta=`0/0`。r2/r4 的完整指标路径也
  分别达到 `14,241.78 / 14,244.92 MiB`；r3 在指标前暴露并修复 Rigid quota device mismatch，不作结果；
- r5 原始 metric/global rows SHA-256=
  `04da7a2503460c075a3164c90d6c08436bbea9f4ec5560ea0417ee40e91aa939 / 04bf741e1da6cfe845b5ee6c9d4cccede54d79a1c8f7178e00abcf737ff7245e`；
  R0/R1 checkpoint 运行后 SHA 仍为 `1a061247...e7c / e995e7c2...8cd1`，run 内无 `.pth`；
- 因资源门失败，下列重算只作 diagnostic，不能升级为合格 heldout 证据：R0→R1 的 S-B first-hit coverage=
  `1.0→1.0`，depth-order violation=`0.915792→0.908173`，non-target RGB MSE=
  `0.002095031327→0.002095032019`，original-global RGB MSE=`0.002104032262→0.002104032654`；预冻结 exact
  comparator 为 `tradeoff_non_dominated`，不是 R1 pass；
- 正式裁决：单次 run 保留 `blocked`，R1 方法臂=`rejected_resource_gate_and_diagnostic_tradeoff`，A3 task=`done`。
  不事后上调显存阈值、不改成 `packed=true`/分块渲染挽回结果，不启动 formal、R2–R4 或独立 S-A 训练；
  `A3*=R0-off`，即 D2 checkpoint immutable exact alias。R1 checkpoint 只保留为负结果/工程复现资产。

### 10.1 核心边界

编辑后区域分为：

| 支持类型 | 含义 | 允许操作 |
|---|---|---|
| S-A observed | 其他时间/相机有真实观测 | 局部优化与补点 |
| S-B geometric | 有 LiDAR/至少两视图几何，无合法 RGB | 深度、opacity、scale；禁止 RGB loss |
| S-C unsupported | 完全未观测 | 不生成伪 3D 真值，不作为局部 3D 精修成功区域 |

A3 的主方法只解决 S-A/S-B。S-C 是生成式资产补全的边界，不通过增加高斯强行“恢复真实世界”。

### 10.2 affected set

对 lateral/delete：

- 目标 actor Gaussian 只作冻结的 context/mask 来源；
- source footprint；
- edited footprint；
- 后方静态 Gaussian；
- 局部深度排序冲突；
- 有 LiDAR/多视图支持的 hole；
- paired counterfactual `2px` mask dilation 后再做固定 `3px` affected dilation 对应的局部视锥。

affected set 外：

- Gaussian 参数冻结；
- optimizer state 冻结；
- checkpoint hash/drift 检查。

### 10.3 子消融

| 实验 | 内容 |
|---|---|
| R0 | 不精修 |
| R1 | opacity/scale 重激活与深度排序 |
| R2 | R1 + 有证据支持的 color/SH |
| R3 | R2 + LiDAR/多视图支持的局部 Gaussian seed |
| R4 | R3 + 时序一致性 |

优化顺序固定为：

```text
opacity/scale
→ S-A observed color/SH
→ evidence-backed seed
→ temporal consistency
```

V1 协议不允许 position update；若后续需要，必须作为 R1 与 R2 之间的独立新因子重新冻结。不得先用 RGB
幻觉生成三维点。

### 10.4 生成式上界

允许在 `scene-0230` 少量样本增加：

```text
H0：冻结的 2D Harmonizer / inpaint diagnostic
```

规则：

- 不训练大型扩散；
- 不成为 A3 主方法；
- 不写入三维 world state；
- 只用于说明局部 3D 精修与生成式补全的能力边界；
- 不因 H0 更好看而替换 R1–R4 的三维指标。

### 10.5 判定

主端点：

- S-A hole error；
- depth ordering violation；
- source residual；
- temporal flicker。

守卫：

- outside drift；
- non-target 参数变化；
- registry/trajectory invariant；
- affected Gaussian 数；
- local wall time。

若 A3 只在 S-C 依靠 2D 生成改善，则不能主张局部 3D 方法成立。

---

## 11. A4：端到端模型生产与部署优化

### 11.1 A4-P0：先做流水线 profile

#### 11.1.1 P0 结果前协议冻结

- 配置=`configs/worldsim_v3/a4_p0_profile_protocol_v1.yaml`，SHA-256=
  `8ba96278b7f65957480a343a21977e2e24a537462b7a0b042a3268684d27d9a4`；依赖 A3 closeout=
  `10eee3ad30c3729532afecdcc520c1ef542e0210`，production input 只允许 `A3*=R0-off` 的 D2 checkpoint
  `1a061247...e7c`、config `115deaf...5e68` 与 registry `ed57764e...0c68`；rejected R1 禁止进入 profile；
- train 与 render/eval 只读复用 A2-D2 formal 的不可变 stage JSON，不重跑 30k 或质量评测；prepare/load/runtime
  必须新进程测量，convert 在 P0 只做 checkpoint/registry inventory，不转换参数；缺失值必须显式 null+reason；
- 原生 `1600×900`，warm-up=`frame 0 / camera 0 ×2` 且 uint8 RGB hash 必须 exact；计时矩阵为
  frames `10/100/190 × cameras 0/1/2` 共 9 个 original view，每帧前后 CUDA synchronize，P50/P95 用
  nearest-rank；不保存 PNG/MP4；
- load 分为 process-cold 首次 load 与同进程 warm reload；filesystem cache 明示 uncontrolled，禁止驱逐 OS cache；
  资源 ceiling=`600 s / 16,384 MiB torch allocated / 24,576 MiB torch reserved / 24,000 MiB NVIDIA sampled /
  32 GiB cgroup / 50 MB run bytes`，OOM delta=`0/0`；
- recovery stage 固定为 `inventory→runtime_probe→aggregate→resume_audit`，completed stage 禁止覆盖；resume audit
  只做 read-only dry-run，不启动 GPU。P0 仍只是 profile，不登记质量改进或并发结论；P1/P2/P3/P5 未授权；
- validator preflight 已核对 10 个路径/hash/bytes，4 项协议测试与联合 WorldSim V3 回归=`143 passed`。协议提交时
  尚未读取任何新 A4 prepare/load/runtime 结果。

对最终 A3 产物拆分记录：

```text
prepare
train
render/eval
convert
load
runtime render
failure recovery
```

每段报告：

- wall time；
- peak VRAM/RAM；
- 输入/输出 bytes；
- cache 命中；
- terminal；
- 最小重跑单位。

工业周报说明，缓存、prepare、normal/depth、资源等待和失败重试可能比渲染器本身更耗时。V3.1 不再只报告训练时间和 FPS。

### 11.2 部署子实验

1. `P1-contribution-prune`
   - visibility；
   - opacity；
   - train/held-out contribution；
   - 可回退。
2. `P2-fp16`
   - 参数与渲染路径；
   - 逐字段误差审计。
3. `P3-chunk`
   - 静态空间块；
   - 动态 actor 独立资产；
   - chunk 边界检查。
4. `P4-lod`
   - 条件式；
   - 距离/投影大小；
   - 不作为最低完成集。
5. `P5-registry-resume`
   - converter；
   - asset manifest；
   - bytes/hash/schema；
   - stage checkpoint；
   - resume point；
   - reload smoke。

### 11.3 必报工程指标

| 维度 | 指标 |
|---|---|
| 资产 | checkpoint bytes、静态块数、actor 资产数 |
| 加载 | cold-load time、warm-load time、peak RAM/VRAM |
| 渲染 | P50/P95 frame time、FPS、相机数、分辨率 |
| 稳定性 | dropped frames、失败原因、重复加载一致性 |
| 恢复 | 最小重跑 stage、重跑 wall time |
| 并发 | 24 GiB 下可稳定运行的任务/场景数 |
| 质量 | 相对 A3 的 PSNR/LPIPS/actor/boundary 下降 |

每个性能数字必须同时记录：

- GPU；
- CUDA/driver；
- 分辨率；
- 相机数；
- warm-up；
- 同步方式；
- 统计窗口。

### 11.4 A4 最低完成集

```text
P0 profile
+ P1 contribution prune
+ P2 FP16
+ P3 spatial/actor chunk
+ P5 registry/resume
```

LOD、整数/低比特量化、完整 LiDAR raycast 和复杂 Web UI 均为可选项，不作为 R0 完成门禁。

### 11.5 最小可视化入口

R0 可提供简单 Scene Explorer：

- 加载 asset manifest；
- 切换相机；
- 播放时间序列；
- original/lateral/delete；
- 显示深度、actor ID、FPS、显存。

界面只作模型资产展示，不扩张为仿真平台项目。

---

## 12. R0：集成与最终交付

### 12.1 必交付

1. V3.1 最终计划与更新日志；
2. A0 三场景原生基线；
3. A1 两阶段校准结果；
4. F0 审计，F1 若启动则含初始化曲线；
5. A2 actor-aware 模块和 `actor_quality.json`；
6. A3 局部精修模型与支持类型输出；
7. A4 转换器、asset registry、resume 机制；
8. A0→A4 主表；
9. 质量—规模—时间—显存 Pareto；
10. 负结果和适用边界；
11. 可复现命令、配置 hash、checkpoint hash；
12. 最小可视化入口或完整离线渲染包。

### 12.2 最终结论允许形式

```text
calibration_enhanced
calibration_native_or_off_preferred
actor_aware_supported
actor_aware_rejected
local_refine_supported
local_refine_limited_to_observed_support
deployment_pareto_supported
engineering_blocked
```

不得把所有负结果统一写成“项目失败”。

---

## 13. 资源与运行合同

### 13.1 资源门禁

每个 GPU stage 前记录：

```bash
nvidia-smi
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
```

停止条件：

- cgroup memory 连续两次达到 90%；
- OOM/OOM-kill 增加；
- CUDA OOM；
- 磁盘安全余量低于 20 GiB；
- 必须静默降低分辨率、相机数、序列长度才能继续；
- 外部实例重建或文件系统异常。

A1 已经出现约 24.0 GiB 峰值，后续正式训练必须：

- 串行；
- 不与评测并行；
- stage 结束显式释放 GPU；
- 记录 densification 峰值；
- 不杀用户 Cursor/Jupyter/TensorBoard。

### 13.2 Run 目录

```text
/root/autodl-tmp/runs/worldsim_v3/<TASK_ID>/<instance_id>/
├── manifest.json
├── resolved.yaml
├── source_snapshot/
├── environment/
├── stages/
├── logs/
├── resource.jsonl
├── metrics.jsonl
├── artifacts.json
├── summary.md
└── terminal.json
```

终态：

```text
pending | running | blocked | done | rejected | interrupted
```

旧实例或旧 terminal 不得复用为新 run。

### 13.3 Pipeline resume

A2–A4 新增：

```text
stage_manifest.json
```

每个 stage 写：

- input hash；
- output hash；
- start/end；
- return code；
- resource peak；
- resume-safe；
- invalidation dependency。

失败后只重跑被失效的最小 stage。

---

## 14. 提交与文档合同

每个任务至少一个独立 Conventional Commit。

提交前：

1. `git status --short`；
2. 确认不混入其他 task dirty files；
3. 定向测试；
4. 配置解析；
5. `git diff --check`；
6. 核对 terminal、run、config/source hash；
7. 同步：
   - 本计划；
   - `RESEARCH_STATUS.md`；
   - `EXPERIMENTS.md`；
   - `RESEARCH_FAILURES.md`；
   - README；
8. commit 正文记录：
   - Task ID；
   - 测试；
   - 正式 run；
   - 指标；
   - 下一步。

A1 早期四个提交的正文不足不改写历史；`801db7a` 与本节开发场景台账已补齐任务、run、配置 SHA、测试和结论证据链。

---

## 15. 停止与扩展规则

- A1 E1/E2 未冻结前，不启动新的 30k；
- A1 `scene-0230` C2/C3 与 C* 未冻结前，不扩确认场景；该门禁现已通过；
- A2 D1 未通过 smoke 和同预算开发场景前，不做 D2；
- A2 LiDAR ancestry 不完整时，不做 D4；
- A3 不允许整场景重训冒充局部精修；
- A3 S-C unsupported 不回传伪真值；
- A4 不允许只报单帧峰值 FPS；
- 只有 A2 或 A3 在两个确认场景方向一致，才讨论扩到 6 场景；
- F1、H0、LOD、LiDAR raycast 均为条件项，不得阻塞主链；
- 不因提升不显著更换指标、actor、场景或确认集；
- 负结果是正式交付。

---

## 16. 当前唯一下一步

A1=`done_off`，A2=`done / tradeoff_non_dominated`。A3 的 R1 工程链和 bitwise replay 通过，但冻结 heldout
评测连续三条完整指标路径均越过 `12,288 MiB` GPU ceiling；r5 的资源无效 diagnostic 又是 geometry 改善与
两项 RGB safeguard 严格退化并存的 `tradeoff_non_dominated`。因此 R1 方法臂已 rejected，A3 task 以正式负结果
done，`A3*=R0-off`（D2 immutable exact alias）；formal、R2–R4 与独立 S-A 训练均未解锁。

当前唯一动作是按已冻结协议执行 `WS-V3-A4-DEPLOYMENT-01 / P0 profile`：

```text
1. 运行 protocol validator 和 GPU/disk/cgroup idle preflight，全部 exact 后才创建唯一 P0 run
2. 只读盘点 train/render-eval 历史 stage，执行 inventory、prepare/load/runtime probe、aggregate 与 dry-run resume audit
3. runtime 固定 2 个 warm-up + 9 个原生 1600×900 original views；同步计时且只存 hash/JSON，不存媒体
4. 报告 wall、torch/NVIDIA VRAM、cgroup RAM、bytes、cache 语义、terminal 与逐 stage 最小重跑单位
5. P0 收口前不启动 P1 prune、P2 FP16、P3 chunk 或 P5 registry/resume，不用部署因子倒改 A3 结论
```

D3/D4 继续未解锁；F0 独立非阻塞。负结果保留为最终交付，不因提升不显著更换端点、阈值或场景。

---

## 17. 更新日志

### 2026-08-09 — A4-P0 端到端 profile 协议冻结

- protocol SHA=`8ba96278...d9a4`，输入锁定 `A3*=R0/D2 exact alias`；rejected R1、训练、checkpoint mutation、
  prune/FP16/chunk/registry-resume 均未授权；
- 历史 train/render-eval stage 只读复用，新测固定 process-cold/warm load、2 warm-up、9 个原生分辨率同步 render、
  inventory 与 dry-run resume；filesystem cache 不冒充 cold；
- 资源 ceiling 和 recovery stage 在结果前冻结；validator 核对 10 个 immutable inputs，协议测试 4 passed，联合
  WorldSim V3 回归 143 passed；下一动作是唯一 P0 profile run。

### 2026-08-09 — A3 R1 heldout 资源门失败、诊断 tradeoff 与 A3 正式收口

- heldout 协议/评测器=`42508fb / c8fc560`，protocol SHA=`eb87a9f2...b05a`；后续只增加失败资源审计和
  等价内存诊断，当前联合回归 `139 passed`；
- r5 exit=1，peak GPU=`14,241.40 MiB > 12,288 MiB`，wall/cgroup/run bytes/OOM 其余门禁均通过；R0/R1
  checkpoint SHA 前后 exact，run 内无 checkpoint；
- 资源无效 diagnostic 为 depth violation 改善、coverage 不变、non-target/global RGB MSE 严格退化，exact Pareto=
  `tradeoff_non_dominated`；不得登记为合格 heldout 质量证据；
- R1=`rejected_resource_gate_and_diagnostic_tradeoff`，A3=`done`，`A3*=R0-off`。下一门禁转为 A4-P0 profile
  协议冻结；formal、R2–R4、S-A 训练以及通过上调阈值/更换 renderer 挽回均未授权。

### 2026-08-09 — A3 R1 真实 paired smoke、数值冻结与 bitwise replay 完成

- sidecar materializer/controller=`3b8526a / aac5213`；paired runner/guard=`d89e0ac`，CUDA preflight fix=`78741b3`，
  numeric freeze=`c02c8c7`；当前 A3 联合回归 `119 passed`；
- real sidecar 将 frame `0/31` 的四个 unit 冻结为 8 个 S-B/T0 loss pixels、51 mutable Background rows、
  16,451 S-C abstain rows；19 个 heldout frames 未进入 support/optimization，S-A 保持 0；
- canonical paired 与 frozen replay 的四单元 loss 完全相同，checkpoint SHA 均为 `e995e7c2...8cd1`；outside
  parameter/Adam、Rigid/trajectory/registry、shape/order exact，资源均低于冻结上界且无 OOM；
- `formal_training_authorized=false`、`quality_claim_authorized=false`；下一门禁是结果前冻结 heldout 只读评测协议。

### 2026-08-09 — A3 R0/R1 exactness guard 与 synthetic contract 完成

- implementation=`9c639dd5a0adcd1f8b5126f7f20d836815b127a6`；DriveStudio patch SHA-256=
  `155ec58fd2bfdc2e40357035dc20800bf2340b0c1c9ac5972c7c78efbd8cb69b`；独立工作树通过 apply/reverse、
  `py_compile` 与 import smoke；
- synthetic run=`20260809T132133Z__a3-r0-r1-synthetic-s0-r1`，summary SHA-256=
  `2ac123f0603120a103743e59680a31dd4cdf5b6d5fa45605d7c84d36ec337ada`，manifest SHA-256=
  `8ffa697e15d8a97108d8281a51313119c304fbf0f245d88bfbd127663fde27c4`；110 项联合回归通过；
- R0 重新计算 D2 checkpoint/config/protocol SHA 并只生成 immutable alias，optimizer steps=`0`、无新 checkpoint key；
- R1 synthetic 中只有 affected S-A/S-B Background opacity/scale 行变化；outside 参数和 Adam moments、
  Background position/color、RigidNodes/trajectory、tensor shape/order exact；
- 原 D2 与 A3 module-off 的 RGB/SSIM loss tensor 逐位相等；缺少 paired provenance/masks 时 fail closed；
- evidence tier 仍为 `synthetic_contract_only`，`paired_engineering_smoke_complete=false`、
  `formal_training_authorized=false`。下一门禁是真实 heldout-safe affected/support sidecar 与最小 paired smoke。

### 2026-08-09 — A3-I0 局部精修语义协议冻结

- config SHA-256=`03fbf632...72d0`；固定 D2 30k boundary-priority asset、D1 fallback、scene-0230/seed 0、
  两 actor roles、lateral/delete 与 held-out exclusion；
- 冻结 paired footprint morphology、affected union、S-A/S-B/S-C precedence、expected/first-hit/measured depth
  truth tiers、outside parameter/optimizer exact 与 R0→R4 单因子顺序；
- 首个工程门只允许 R0 exact alias 和 R1 Background opacity/scale；target actor、position/color/seed、S-C、
  D3/D4、whole-scene retraining 与大型 diffusion 禁止；
- D2 checkpoint 的 Background ancestry 含 `240,528` 个直接 LiDAR roots，但 ancestry 仅作 provenance，不能冒充
  measured depth；未提交 V2 M5 文件明确排除为依赖；
- 当前 `formal_training_authorized=false`；下一步实现 materializer/DriveStudio patch、module-off/outside exact 与
  synthetic paired smoke，smoke 后再冻结数值预算；新增 12 项、联合回归 98 项测试通过。

### 2026-08-09 — A2-D2 formal 与 A2 正式收口

- canonical run=`20260809T113230Z__a2-d2-formal30k-s0-r1`，terminal=`done`，summary SHA-256=
  `9c41dfc8...de7d`，D2 final checkpoint SHA-256=`1a061247...e7c`；D1 checkpoint 前后 SHA 不变；
- 六个 5k grid checkpoint、初始化同源、quota/cap/finite、297 条资源记录及四个 stage 全部通过，无 OOM；
- matched 选中 D2 30k，Rigid gap=`708 / 0.67165%`，因此 matched 是 fixed D2 exact alias；
- fixed/matched quality 与 quality-cost 都为 `tradeoff_non_dominated`；D2 boundary-support boundary-band
  PSNR/SSIM/LPIPS 三项改善，但 global、部分 actor/non-target 和 wall/cgroup 成本退化；
- A2 状态冻结为 `done`；A3 采用 `A2*=D2-boundary-residual` 作为 boundary-priority research asset，D1
  quota-only 保留为 fallback。此为完整 Pareto 后的工程分支选择，不是 dominance 或跨场景结论；
- `d3_unlocked=false`，D4 未启动；当前门禁转为 A3 affected-set、证据层级、深度语义和局部端点协议冻结。

### 2026-08-09 — A2-D2 boundary/residual 协议冻结

- 配置 SHA-256=`acceb7f4...e567`；D1 formal summary 与收口提交作为不可变前置；
- 冻结 3px dynamic-mask boundary band、投影中心 RGB L1 residual、稳定六键排序与复用 native size threshold 的 cap；
- 明确 D1 eligibility/quota/cull/Background 不变，D3/D4 因子禁止，module-off 必须与 D1 逐位等价；
- 新增纯函数与合同测试，并与 D1 quota、I0 ancestry 联合回归 `22 passed`；尚未实现 DriveStudio patch 或运行 smoke。

### 2026-08-09 — A2-D1 formal 正式收口

- canonical r1 terminal=`done`，summary SHA-256=`e3b194c2...66ac`，formal contract SHA-256=`ad77db41...f8e7`；
- D0/D1 各 30k、初始化 provenance 完全一致，6×2 checkpoint 网格与 24/24 actor quota 上限通过，无 OOM；
- fixed D0/D1 global PSNR/SSIM/LPIPS=`27.7481/.851207/.176319` 与 `27.7700/.850915/.177704`；
- matched 选中 D1 15k：Rigid=`176,741`，对 D0 `177,628` 的差=`887 / 0.499%`；D0 为 fixed final exact alias；
- matched D1 global=`25.9290/.825381/.217941`，boundary-support actor PSNR/SSIM/LPIPS=
  `29.2937/.902828/.061463`；表现为局部改善与 global/non-target 退化并存；
- fixed 与 matched 的质量裁决均为 `tradeoff_non_dominated`，故仅解锁 D2 协议冻结，不宣称 D1 全面改进。

### 2026-08-09 — A2-D1 formal 唯一实例启动

- run=`20260809T085400Z__a2-d1-paired-formal30k-s0-r1`；source commit=`f32f96b`；tmux=`ws_a2_d1_f1`；
- 启动登记时 terminal=`running`，stage=`train_d0_native_30000`，D1 未启动；该生命周期记录已由上方 done 终态取代；
- 两份物化配置除 variant/quota-enable 外一致；formal/base config SHA 与 canonical r4 summary SHA 已写入 manifest；
- D0 初始化计数 Background/RigidNodes=`946,484 / 75,002`，与冻结 scene-0230 起点一致；
- 启动后 GPU 约 `3.0 GiB` 且无 OOM；该条仅登记生命周期，不构成 checkpoint 或质量结果。

### 2026-08-09 — A2-D1 formal 协议冻结

- 协议/控制器/评测提交=`387dd50`；`configs/worldsim_v3/a2_d1_formal_v1.yaml` SHA-256=
  `ad77db41d9d8c5172804a20b38a2dd92173c3639398d8abc24dc6f4799e8f8e7`；
- fixed-step 冻结为 scene-0230 / seed 0 / D0→D1 / 每臂 30k；5k checkpoint 网格只用于只读 matched-budget 选择；
- matched view 以 D0 最终 RigidNodes 数为目标，D1 候选按绝对差、再按更早 step 选择，2% 之外明确 ABSTAIN；
- non-target 冻结为 high/boundary 两名 actor 的 model-counterfactual effect mask 并集之外，truth tier 不是 GT segmentation；
- `80 passed`；direct-script import 修复后只读 preflight=`done`：GPU=`0 MiB`、free disk=`58.39 GiB`、
  cgroup memory.max=`96,636,764,160` bytes，patch/SHA 与 canonical r4 summary SHA 全部匹配；
- 协议冻结提交时 formal 尚未启动；该提交本身只解除启动门，不构成 D1 质量结论。

### 2026-08-09 — A2-D1 quota-only 配对 smoke 收口

- 工程实现提交：`c9b2422`；DriveStudio quota patch SHA-256=`c232af2c...7bb1b`；
- 冻结 actor threshold=`0.00025`、Background native threshold=`0.0005` 与 24 actor min/max quota；
- canonical scene-0230 D0/D1 1000-step paired smoke r4=`done`，配置预算和初始化 provenance 匹配；
- module-off 逐位等价、quota/ancestry checkpoint round-trip、24/24 actor 上限与资源门禁通过；
- D1 最终 Rigid=`152,830`，高于 D0 的 `125,915`，明确不能据 Gaussian 数量宣称质量改进；
- r2 为前台 SSH 转 tmux 的显式中止，r3 被 GPU 非空预检拒绝；均非 canonical，不覆盖 r4；
- 当前门禁转为 D1 formal 协议、评测和 matched-Gaussian-budget 冻结，尚未启动 D1 formal。

### 2026-08-09 — A2-I0 ancestry instrumentation 收口

- 实现提交：`271d876`；
- DriveStudio `e59bda4` 的 RigidNodes split、clone、external replacement、prune 与 checkpoint 数据流已接入逐
  Gaussian ancestry；
- 冻结 I0 配置、init-source code、online/update API、module-off 等价合同和 legacy checkpoint 边界；
- canonical r3 formal smoke=`done`：原生 tensor 逐位一致、off 无额外 checkpoint key、on 可 round-trip；
- patch/worktree verify 与 66 项 WorldSim 定向测试通过；
- 明确该证据仅为 synthetic instrumentation contract，不是 scene-0230 质量或 D1 方法结论；
- 当前门禁转为 quota-only D1 配置/资源合同与配对 smoke。

### 2026-08-06 — V3.1 计划创建

- 保留 V3 的 A0–A4 主链和 V2 冻结边界；
- 将工业周报与离职交接资料转化为模型/工程优先级，不直接复制内部口径；
- 明确 3DGS 是 WorldSim 的场景资产和渲染层；
- 将实验设计改为 `scene-0230` 开发、`0242/0255` 确认；
- A1 正式矩阵从 12-run 全排列改为 10-run 两阶段协议；
- 修正 C2：无 exposure 元数据时只使用 camera + continuous time；
- 将 C4 rolling shutter 正式固定为 `not_supported`；
- A1 增加 E1/E2 端点、低速位姿与相机方位诊断；
- 将逐 Gaussian ancestry 从 A1 门禁移到 A2 instrumentation；
- F0 增加条件式 F1 前馈初始化 pilot；
- A2 增加 boundary scale cap、depth/normal、provenance 和 actor 资产路由；
- A3 明确 S-A/S-B/S-C 证据边界，并保留冻结生成式上界作为诊断；
- A4 扩展为端到端 profile、可恢复 stage、冷加载和并发指标；
- 计划创建时状态保持 `A1 running`，不提前进入 A2。

### 2026-08-06 — A1-E0 阶段快照

- C0/C1/C2/C3 配对 smoke 完成；
- C3 零梯度 bug 已修复；
- 配对初始化 SHA 已冻结；
- `scene-0230` C0/C1 30k formal 完成；
- C0 多数 PSNR/SSIM/actor/boundary 指标优于 C1，但 LPIPS、GS 数和端点尚未闭合；
- E1/E2 已冻结并用 C0/C1 全 held-out checkpoint 回填；
- 原生 C1 在开发场景的 E1 与 boundary actor E2 均劣于 C0，尚不构成最终 C* 裁决；
- 一次错误相机标签 formal 已显式 rejected，修复 run 为唯一有效端点证据；
- 最小 LiDAR provenance 已完成；随机背景 CUDA visibility filter 使 exact 初始化 replay 不成立，初始深度只作
  reconstructed witness；
- 当时门禁转为 scene-0230 C2/C3 30k formal，之后才能冻结 C*。

### 2026-08-07 — A1 开发场景与 C* 冻结

- C2/C3 配对 30k、actor/global 评估与冻结 E1/E2 回填完成；
- A1-D0 ISP/位姿/速度分层诊断完成；near-static 只有 2 帧，保留低支持边界；
- C3 的全图/boundary actor 质量与 learned pose correction 稳定性最佳，但 E1/E2 不优于 C0；
- A1-S0-v1 如实披露结果访问时点，以无容差严格 Pareto 操作化 7.5；
- 正式选择 `C*=C0-off / done_off`；确认矩阵使用 C0 exact alias，10 个逻辑项对应 8 个唯一训练；
- 当时门禁转为 scene-0242/0255 C0/C1 确认、端点回填、alias 登记和 A1 finalizer。

### 2026-08-07 — A1 确认矩阵正式收口

- scene-0242/0255 C0/C1 四个 30k、冻结端点回填与两个 C0 exact alias 全部完成；
- A1 finalizer 完成 `10/10` 逻辑项、`8/8` 唯一训练，正式终态 `C*=C0-off / done_off`；
- scene-0242 原始端点支持 C0，scene-0255 的部分原始 E1/E2 error 支持 C1，但 C1 未通过完整冻结合同；
- 当前门禁转为 A2 instrumentation，不再恢复 A1 为 running。

### 2026-08-06 — A0 done

- scene-0255 空 LiDAR tensor 兼容修复；
- 三场景 checkpoint、registry、held-out、actor/boundary 与资源矩阵闭环；
- 0242 boundary 继续 `ABSTAIN`；
- A0 终态不再恢复为 running。

---

## 18. Codex Agent 恢复提示词

```text
执行 docs/DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md。

WS-V3-A1-CALIBRATION-01 已固定为 done_off；不得恢复为 running。
WS-V3-A2-ACTOR-DENSIFY-01 已固定为 done / tradeoff_non_dominated；不得追加 D3/D4 或改写为 D2 dominance。
WS-V3-A3-LOCAL-REFINE-01 已 done；R1 方法臂因 frozen resource gate 与 diagnostic tradeoff 被 rejected，A3*=R0-off。
WS-V3-A4-DEPLOYMENT-01 当前为 running；P0 protocol 已冻结，只允许执行 P0 profile。

开始前：
1. 读取 AGENTS.md、RESEARCH_STATUS.md、RESEARCH_FAILURES.md、EXPERIMENTS.md 和 V3.1；
2. 检查 research/worldsim-v3 分支、HEAD、dirty files、run terminal、GPU/cgroup/磁盘；
3. 核对 A1 finalizer、10/10 logical、8/8 unique、C*=C0 exact alias 和 scene-dependent 原始端点边界；
4. 运行 WorldSim 定向测试、配置解析和 git diff 门禁；
5. 确认 GPU 空闲、cgroup 无 OOM、数据盘充足且无活动 controller；
6. 核对 A2-I0 r3、配置/patch SHA、module-off 逐位等价、lineage/prune/checkpoint round-trip 和 66 项测试；
7. 核对 clean A2-I0 source commit 只含 instrumentation、测试和直接相关文档，未混入 V2 M5；
8. 核对 A2-D1 `c9b2422`、配置/patch SHA、canonical r4、D0/D1 provenance 匹配和 24/24 quota 上限；
9. 核对 formal 协议提交 `387dd50`、配置 SHA `ad77db41...f8e7`、80 项测试和只读 preflight；
10. 核对 D1 formal r1 terminal/summary、初始化同源、6×2 grid、fixed/matched 两视图与 checkpoint 不可变性；
11. 核对 D2 paired smoke r1 terminal、summary SHA `749c7d15...3136`、provenance、真实 observation/order/cap、quota 与资源门禁；
12. 核对 D2 formal r1 terminal、summary SHA `9c41dfc8...de7d`、六个 grid、matched 0.67165%、D1 checkpoint 不变与资源终态；
13. 核对 A3 protocol SHA `03fbf632...72d0`、D2/D1 asset hashes、两 actor/两 edit、held-out exclusion、S-A/B/C 与三种 depth 语义；
14. 核对 A3 synthetic implementation `9c639dd`、summary/manifest、R0 alias 与 R1 outside/Adam exact；
15. 核对 sidecar manifest `42474f73...7273`、8 个 S-B/T0 pixels、51 mutable/16,451 S-C rows 与 heldout exclusion；
16. 核对 paired/frozen replay 两次 checkpoint SHA `e995e7c2...8cd1`、numeric freeze SHA `d9289df0...b3e3`、119 项测试和资源上界；
17. 核对 heldout protocol SHA `eb87a9f2...b05a`、r5 resource audit `d9536f4e...6280`、14,241.40 MiB
    超过 12,288 MiB、checkpoint exact、无新 `.pth` 与资源无效 diagnostic tradeoff；
18. 核对 A3 task=`done`、R1=`rejected_resource_gate_and_diagnostic_tradeoff`、A3*=R0/D2 exact alias；
19. 核对 A4-P0 protocol SHA `8ba96278...d9a4`、A3*=R0 输入、2+9 render matrix、资源/recovery contract 与
    143 项回归；当前只允许执行唯一 P0 profile；不得重跑 A3、上调阈值、切换 packed/分块 renderer 挽回结果，
    不得启动 formal、R2–R4 或把 S-B/S-C 登记为 RGB 质量成功。

不得恢复 A1/A2 或 V2 M5，不得依赖未提交 V2 M5 文件，不得把 ancestry 写成 measured depth，不得新增大型 diffusion。
```
