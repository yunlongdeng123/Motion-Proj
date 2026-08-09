# 面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化计划 V3.1

- **版本**：V3.1
- **日期**：2026-08-10
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
- **A4-P1 runner 基线**：`19cab2cf40b8ed8ef9a4ad1ba8cce4cc8cf67163`（train-only contribution、三臂物化、冻结质量门、runtime 与恢复审计）
- **A4-P2 协议基线**：`588e37e`；协议 SHA-256=`6558fb3f0864c7711add2bd8b61500670ddbf631be7e356f1eac77c57c136d4e`
- **A4-P2 runner / 账本修复基线**：`1cd9a6e` / `dcf2822`（10-field FP16、FP32 renderer adapter、19-audit finalizer）
- **A4-P3 协议 SHA-256**：`dfaaba79162961673b632271727c8a949c45519b1e75e5ed873badf999ad1b41`
- **A4-P3 runner 基线**：`aba55777f38a3d8e4363d2ff7d546d412214b481`（exact chunk package、内存重组、21-audit finalizer）
- **F0 Instant NuRec 审计协议 SHA-256**：`2004a0294cc4adb9750dd3bc78aac0b650c99338f761697c14afd8e71a6fd611`
- **当前任务**：`WS-V3-F0-FEEDFORWARD-AUDIT-01`（`running`）
- **当前里程碑**：`P0 done / A0 done / A1 done_off / A2 done（tradeoff_non_dominated）/ A3 done（R1 rejected，A3*=R0-off）/ A4 done（P0/P5/P1/P2/P3 complete；P1 rejected；P2 mixed checkpoint + P3 exact package selected）/ D3-D4 not launched / F0 running / F1 conditional / R0 pending`
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
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | running | Instant NuRec 官方代码、输入输出、license、导出能力审计 | 形成可执行/不可执行事实结论 |
| `WS-V3-F1-FEEDFORWARD-INIT-01` | conditional | 前馈深度/高斯初始化 + StreetGS 短步精修 pilot | 只在 F0 输出可转换资产时启动；不阻塞 A2 |
| `WS-V3-A2-ACTOR-DENSIFY-01` | done | I0、D1/D2 smoke 与 formal、fixed/matched Pareto 和资产路由 | `tradeoff_non_dominated`；A2*=D2 boundary-priority，D1 fallback；D3/D4 未启动 |
| `WS-V3-A3-LOCAL-REFINE-01` | done | I0、R0 exact alias、R1 S-B 工程/replay 与 heldout 负结果 | R1 资源门失败且诊断 Pareto tradeoff；A3*=R0/D2 exact alias；formal、R2–R4 未授权 |
| `WS-V3-A4-DEPLOYMENT-01` | done | 端到端 profile、prune、FP16、chunk、registry、resume | P0/P5/P1/P2/P3 全部闭环；P1 rejected；P2 mixed checkpoint + P3 exact package selected；21/21 P3 audits |
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

#### 8.1.1 F0 审计协议冻结

- 官方代码 checkout 固定为 NVIDIA `instant-nurec@1ce2288e646548e61fea6100bc58de3acd4bc8d0`，tree=
  `96e36fa4772f5ddada37dc3decb1be9d2e595dc0`；16 个关键文件逐 SHA-256 锁定且 checkout 必须 clean；
- source code、model weights、NCore dataset 的许可分别登记为 Apache-2.0、NVIDIA Open Model License 与
  NVIDIA Autonomous Vehicle Dataset License；dataset gated/terms acceptance 与 source code 许可分离；
- 固定三个当前 weights-only PTH 的 Hugging Face commit、bytes、SHA-256 与 Xet hash；旧 traced archive 不作为
  当前仓库支持的 checkpoint；
- 论文/模型卡完整模型的 static/dynamic/sky/ISP 能力与 standalone CLI 分表。当前 CLI 只读 NCore V4
  `.json/.lst`，接受 FTheta camera，可读 RGB/pose/intrinsics/mask/optional cuboids，不读 LiDAR；只导出
  static-layer PLY，不导出 dynamic、sky、ISP、actor registry/trajectory 或 depth/point map；
- 本机 smoke 为全前置条件合取：Python 3.11、`uv`、CC≥8.0、VRAM≥30,720 MiB、RAM≥32 GB、free disk≥100 GB、
  精确支持权重、合法 NCore 输入/terms 记录、exact clean checkout 与 CLI help 全部通过才授权。任一失败时不得构造
  inference command，也不得安装依赖、下载权重/gated 数据或启动 GPU；
- protocol=`configs/worldsim_v3/f0_instant_nurec_audit_v1.yaml`，SHA-256=
  `2004a0294cc4adb9750dd3bc78aac0b650c99338f761697c14afd8e71a6fd611`；runner SHA-256=
  `249f26d5cbff0687bfedf094c5386237365cdf24ddcf811d3c56b487e9868e4a`；协议/官方源码指纹测试=`8 passed`，
  WorldSim V3 联合回归=`241 passed`。
  本条写入时 formal 本机审计尚未运行，不构成 inference、wall/VRAM 或质量结果。

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

#### 11.1.1 P0 v1 阻塞与 v2 结果前重新冻结

- v1 配置=`configs/worldsim_v3/a4_p0_profile_protocol_v1.yaml`，SHA-256=`8ba96278...d9a4`，保持不可变；runner
  提交=`199abd99d642747241b79ce543c8eb9096553a1d`。formal r1=`20260809T151539Z__a4-p0-profile-s0-r1`
  完成 inventory/runtime/aggregate/no-torch resume audit 后，只因 `native_resolution_exact=false` 被 finalize 正确
  阻塞；terminal SHA=`9084e49c...e5ed`，不覆盖、不改写为 done；
- 根因是冻结的 A2 source config 已写明三路 `data.pixel_source.downscale_when_loading=[2,2,2]`。v1 将传感器
  `1600×900` 错当成当前 checkpoint 的模型原生分辨率；11 行 render 全部为 `800×450`，2 次 warm-up RGB hash
  exact。该观测只用于纠正 profile 输入合同，不登记为合格 P0 性能结果；
- v2 配置=`configs/worldsim_v3/a4_p0_profile_protocol_v2.yaml`，SHA-256=
  `43db718233589d847cb56e2497c4a75b20506e440e523177daaebfbc82c03f18`；production input、2+9 matrix、同步/
  nearest-rank、历史 stage 复用、资源 ceiling 和 recovery contract 均不变，只把模型原生尺寸纠正为
  `800×450` 并冻结 v1 protocol/manifest/runtime rows/resource audit/terminal 六项证据；
- production input 仍只允许 `A3*=R0-off` 的 D2 checkpoint `1a061247...e7c`、config `115deaf...5e68` 与
  registry `ed57764e...0c68`；rejected R1 禁止进入 profile。train 与 render/eval 只读复用，不重跑 30k 或质量
  评测；convert 只做 inventory；缺失值显式 null+reason；
- load 仍分 process-cold 首次 load 与同进程 warm reload；filesystem cache 明示 uncontrolled，禁止驱逐 OS cache；
  ceiling 仍为 `600 s / 16,384 MiB torch allocated / 24,576 MiB torch reserved / 24,000 MiB NVIDIA sampled /
  32 GiB cgroup / 50 MB run bytes`，OOM delta=`0/0`；
- r1 diagnostic 的 resource audit 自身通过：wall=`62.144 s`、allocated/reserved/NVIDIA=`7,913.31/8,232/
  8,574 MiB`、cgroup peak=`24,775,639,040 bytes`、run=`77,135 bytes`、OOM=`0/0`；由于 v1 分辨率合同错误，
  这些值不用于关闭 P0，只作为 v2 ceiling 风险审计；
- v2 validator 已 exact 核对 16 个路径/hash/bytes；协议测试=`7 passed`，联合 WorldSim V3=`152 passed`。P0 仍
  只是 profile，不登记质量改进或并发结论；P1/P2/P3/P5 未授权。

#### 11.1.2 P0 v2 正式结果与部署排序

- canonical run=`20260809T152923Z__a4-p0-profile-v2-s0-r2`，source commit=
  `b191afaaa88d5d356506fc29a36e6128959d8897`，exit=`0`，terminal=`done`；summary/manifest/resource/rows
  SHA=`0278a320...e92 / 12df93b3...a0f5 / b89c93bb...5fac / 4a94b1fb...a934`；13/13 required
  audits 全 true；checkpoint/registry 前后 SHA exact，run 内无 `.pth`、PNG、JPEG 或 MP4；
- 资产 inventory：checkpoint=`578,819,674 bytes`，registry=`3,721,428 bytes`，合计=`582,541,102 bytes`
  （`555.55 MiB`）；静态背景仍为 1 个 monolithic block；actor=`24`，其中 `23 available / 1 unavailable`；模型
  Gaussian=`1,205,164 Background + 104,704 RigidNodes`；P0 未做参数转换；
- 新测 wall=`60.784519 s`，prepare=`50.420569 s`（占 wall `82.95%`），trainer construction=`1.885644 s`，
  process-cold/warm load=`.391351/.397158 s`；filesystem cache 仍为 uncontrolled，不能宣称 filesystem-cold；
- 2 次 warm-up hash exact；9 个 `800×450` measured original views 覆盖 `frames 10/100/190 × cameras 0/1/2`，
  P50/P95=`.068017/.127388 s`，聚合 FPS=`16.377547`。这些数只适用于 scene-0230/seed-0/单进程/当前
  checkpoint 与同步口径，不构成并发或跨场景结论；
- resource audit=`passed`：torch allocated/reserved=`7,913.31/8,232 MiB`，NVIDIA sampled=`8,574 MiB`，
  cgroup peak=`24,474,128,384 bytes`（`22.79 GiB`），run bytes=`85,169`，disk free=`45,292,818,432 bytes`，
  OOM/kill delta=`0/0`；全部低于冻结 ceiling；
- no-torch resume dry-run=`.160304 s`，原子复用 inventory/runtime/aggregate 三个 completed stage，输入=
  `16,256 bytes`、输出=`919 bytes`，前后均无 GPU compute process；最小重跑单位为 none；
- P0 的主要瓶颈是 dataset prepare，而不是 checkpoint load；单进程 runtime 也没有触发显存门。没有证据支持先承担
  P1 prune、P2 FP16 或 P3 chunk 的质量/数值风险。下一门禁选择无模型变异的 P5 registry/resume 协议冻结：先把
  checkpoint、config、actor registry、stage/summary hashes 注册为不可变资产并做独立 reload/resume smoke；
  P1/P2/P3 继续未授权。

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

#### 11.2.1 P5 registry/resume 结果前协议冻结

- protocol=`configs/worldsim_v3/a4_p5_registry_resume_protocol_v1.yaml`，SHA-256=
  `51acb935f2b63b49992ac886c2756394f8bf362ccb2c7fcc5fe30d550a845874`，P0 closeout=
  `9811c03ce1d0c7f017f50d950fe55d5c2891e0b5`；9 项输入锁定为 P0 v2 protocol/manifest/summary/resource/
  rows/terminal 与 A3*=R0 checkpoint/config/actor registry 的 exact path/hash/bytes；
- 输出 schema=`worldsim-v3-deployment-registry-v1`，模式=`reference_only_immutable_manifest`。禁止复制或重写
  `578,819,674-byte` checkpoint；static asset 固定 1 个 `models.Background` reference、`1,205,164` GS，明确
  `independently_extractable=false` 与 `p3_chunk_not_authorized`；
- actor compact registry 固定 `24 total / 23 available / 1 unavailable / 104,704 GS`；每项保留 model index、token、
  class、availability、selector、count 与 flat-index hash，并引用 source registry SHA，不复制数千段 index ranges；
  unavailable actor 必须继续显式为空，不能在部署清单中静默修复或删除；
- reload smoke 只允许 fresh DriveStudio process 构造 trainer、只读加载 checkpoint 一次、核对 Background/RigidNodes
  总量以及全部 actor count/index hash；不 render、不构造 optimizer、不训练。filesystem cache 仍 explicit
  uncontrolled；
- stage=`input_audit→registry_materialize→reload_smoke→aggregate→resume_audit`，completed stage 不覆盖，失败只从
  首个无效 stage 向后重跑；最终 resume auditor 必须为 no-torch/no-GPU；
- ceiling=`180 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA sampled / 32 GiB cgroup /
  5 MB run bytes / 30 GB disk floor / OOM 0/0`；required audits=`14`，registry output ceiling=`2 MB`；
- validator 已 exact 核对 9 项输入；协议测试=`6 passed`，联合 WorldSim V3=`158 passed`。P5 是 packaging/recovery
  工程证据，不登记质量提升、独立 chunk、filesystem-cold 或 concurrency claim；P1/P2/P3 仍未授权。

#### 11.2.2 P5 formal 结果与后续部署裁决

- runner 实现提交=`4de2126e1fce3c8f0e3bed1d23b203daa57b8078`；formal r1=
  `20260809T155209Z__a4-p5-registry-resume-s0-r1` 在完成 input audit 和 registry materialize 后，于只读 checkpoint
  reload 后读取 `RigidNodes.points_ids` 时报 `AttributeError`，保留为 `blocked`。r1 terminal SHA=
  `61d30a112e00640ca4b6f3802f2184cb5dc03b00f6b2e291cbb54ef5d792773e`；已生成的 `14,729-byte` compact registry
  SHA=`e48bccdf...9039d`，不覆盖、不改写为 done；
- DriveStudio 源码核对表明 checkpoint state key 是 `points_ids`，但 `load_state_dict` 后运行时属性是 `point_ids`。
  修复提交=`0e899b2e6dcf7d5a091a0a4092ea99767c982357`，只更正运行时契约并增加成功/别名拒绝两条回归；协议、输入、
  ceiling 和测量口径均未改变。聚焦测试=`15 passed`，联合 WorldSim V3=`167 passed`；
- canonical r2=`20260809T155753Z__a4-p5-registry-resume-s0-r2`，source commit=`0e899b2...2357`，exit=`0`，
  terminal=`done`；summary/manifest/resource/registry SHA=`0c86ff68...8744 / 78830d74...58bd / f6c06df0...3ac4 /
  e48bccdf...9039d`，14/14 required audits 全 true；checkpoint 与 source actor registry 前后 SHA exact；
- deployment registry=`14,729 bytes`，canonical content SHA=`02467963...cb6`；资产总量仍为 `1 static /
  1,205,164 Background GS + 24 actors / 104,704 RigidNodes GS`，其中 `23 available / 1 unavailable`。全部 24 个
  actor 的 Gaussian count 与 flat-index hash 在 fresh reload 后逐项 exact；unavailable actor 保持显式空；
- fresh reload 只加载 checkpoint 一次，load=`.445515 s`、dataset prepare=`49.631015 s`、trainer construction=
  `1.987488 s`、reload total=`52.320687 s`，render=`0`，未构造 optimizer、未训练、未复制/重写 checkpoint，
  filesystem cache 仍 explicit uncontrolled；
- resource audit=`passed`：wall=`60.437454 s`，torch allocated/reserved=`7,188.73/7,226 MiB`，NVIDIA sampled=
  `7,564 MiB`，cgroup peak=`24,498,089,984 bytes`，finalize 前 run bytes=`102,229`，disk free=
  `45,291,683,840 bytes`，OOM/kill delta=`0/0`；全部低于冻结 ceiling；
- no-torch resume dry-run=`.127572 s`，未导入 torch、前后无 GPU compute process，按 hash 复用四个 completed
  stage，最小重跑单位=`none_all_completed_stages_reusable`；P5 因此登记 `done`，但只证明 reference-only packaging
  和单实例恢复，不证明独立 chunk、filesystem-cold、并发或质量提升；
- A4 最低完成集仍要求 P1/P2/P3。P0 表明 prepare 占主导且当前 load/runtime 未触发资源门，故不把 P5 成功解释为
  立即转换模型的理由；执行顺序冻结为下一步只制定 P1 contribution-prune 协议、validator 与测试，在新测量前固定
  contribution 语义、候选阈值、质量/资源 safeguards、回退资产与恢复合同。P2/P3 继续未授权，A4 保持 `running`。

#### 11.2.3 P1 contribution-prune 结果前协议冻结

- protocol=`configs/worldsim_v3/a4_p1_contribution_prune_protocol_v1.yaml`，SHA-256=
  `4f893c095989ed274ff431b9a19a8fd27ef5534e3a7df5a8112d07ed34ea429b`，P5 closeout=
  `4db43ddcc23d5c0fee5bf8fc0c254ad51301779b`；13 个文件加 1 个 mask 目录锁定 A3*=R0/D2 checkpoint/config/
  registry、D2 global/actor/boundary/non-target 基线、P0 profile 与 P5 registry/recovery canonical evidence；
- contribution 排名只使用 train frames `[5,45,85,125,165,195] × cameras [0,1,2]`。对 gsplat near→far
  intersections 计算遮挡感知 `T_before × alpha`，按 Gaussian 在 CPU float64 稳定累加并量化到 12 位小数；heldout
  frames `[10,50,90,130,170,190] × 3` 只报告 contribution audit，禁止参与排名或阈值选择；
- 候选预先固定为 source exact reference 与 `5%/10%/20%` 三个 arm；Background 和 23 个 available actor 分资产
  独立稳定排序，删除数=`floor(source_asset_count × fraction)`，不按 high/boundary role 特判；1 个 unavailable actor
  继续显式为空。所有 Gaussian 参数、`points_ids` 与 ancestry row 必须使用同一 keep mask，Sky/LPIPS/轨迹/step exact；
- candidate checkpoint 只允许一次原子写入，不新增 checkpoint key；每臂必须报告 checkpoint/count、逐资产 removed
  flat-index/gaussian-id hash、invariant before/after 与新 registry。source checkpoint 禁止复制、改写；不训练、不构造
  optimizer、不输出 raw render media；
- 质量裁决固定完整 `19 heldout frames × 3 cameras=57 views`，复用 source actor mask 的 33 个 PNG bytes exact，禁止
  candidate 重生成 mask。global PSNR/SSIM/LPIPS 最大退化=`0.10 dB / .002 / .002`；actor/boundary 最大退化=
  `0.20 dB / .005 / .005`，MAE 最大增加 `.002`；non-target 用更严格的 `0.10 dB / .002 / .002 / .001 MAE`；
  缺失或非有限 endpoint 直接 reject；
- 三个 candidate 全部按同一门禁执行，禁止看结果新增 arm/阈值。选择规则是通过 reload/count/invariant、全部质量、
  bytes/GS reduction 与资源门的最大 prune fraction；若无 candidate 通过，P1 method 登记 rejected，生产资产 exact
  fallback 到 source，P1 实验仍可 terminal done；bounded loss 不登记为质量提升；
- runtime 复用 P0 的 `frames 10/100/190 × cameras 0/1/2`、2 warm-up、`800×450`、CUDA sync 与 nearest-rank，
  对全部 arm 报 checkpoint/load/P50/P95/FPS/VRAM/RAM；filesystem cache 仍 uncontrolled，性能只报告、不反向选择
  质量阈值；
- recovery=`input_audit→contribution_scan→三臂 materialize/evaluate→runtime_profile_all_arms→aggregate→resume_audit`
  共 11 stage，completed stage 不覆盖，resume 必须 no-torch/no-GPU。ceiling=`1,800 s / 20,480 MiB allocated /
  24,576 MiB reserved / 24,000 MiB NVIDIA / 48 GiB cgroup / 2.5 GB run / 30 GB disk floor / OOM 0/0`；
  required audits=`21`；
- validator full preflight 已核对全部输入与 33-mask digest；协议测试=`11 passed`，联合 WorldSim V3=`178 passed`。
  本条写入时尚未执行任何 P1 新 measurement；下一动作只实现并提交 runner，P2/P3 仍未授权。

#### 11.2.4 P1 formal 结果与部署裁决

- runner 实现提交=`19cab2cf40b8ed8ef9a4ad1ba8cce4cc8cf67163`；提交前 23 项聚焦测试与 105 项
  WorldSim-prefix 回归通过，正式运行后完整 `tests/*worldsim_v3*.py` 回归=`190 passed`。canonical r1=
  `20260809T165058Z__a4-p1-contribution-prune-s0-r1`，exit=`0`、terminal=`done`、21/21 audits 全 true；
  summary/manifest/resource/terminal SHA=`7c5347e3...7119 / 486342ba...61ac / 8b6073ed...4b7c /
  80dd8178...c645`，source checkpoint/config/registry 与 33-mask 目录前后 exact；
- contribution scan 完整执行 18 train ranking views 与 18 heldout audit-only views，score NPZ=`30,376,517 bytes`、
  SHA=`0165401a...69a9`，15 个数组的 dtype/shape/content hash 全部复核；scan=`198.712 s`，峰值 torch
  allocated/reserved=`14,342.71/14,892 MiB`、NVIDIA sampled=`15,234 MiB`，未越过冻结资源门；
- b05/b10/b20 分别生成 `554,938,306 / 531,056,962 / 483,292,674-byte` checkpoint，相对 source 减少
  `23,881,368 / 47,762,712 / 95,527,000 bytes`；模型计数分别为 `1,144,906+99,480 /
  1,084,648+94,246 / 964,132+83,773`。每个 Background/available actor 的 floor removal、row alignment、
  ancestry、Sky/LPIPS/trajectory/step invariant、registry 与 fresh reload count 均 exact，unavailable actor 继续为空；
- source 质量 replay 对冻结历史端点逐项 exact。b05 已是最小预注册剪枝臂，但仍在 31 个 safeguard 中失败 3 项：
  global occupied PSNR `-0.117684 dB`、global PSNR `-0.110926 dB`、non-target PSNR `-0.125462 dB`，均超过
  `0.10 dB` 上限；b10/b20 分别失败 `12/15` 项。actor/boundary 的部分局部指标保持或改善，不能抵消全局与
  non-target 门禁失败；禁止事后新增更小 fraction 或放宽阈值；
- 全部 arm fresh reload 与 9-view runtime matrix 完成。source/b05/b10/b20 checkpoint load=`.365/.387/.365/.356 s`，
  render P50=`.0447/.0329/.0700/.0399 s`、P95=`.1402/.0785/.1538/.1008 s`、aggregate FPS=
  `18.11/23.64/14.54/22.73`；filesystem cache 未控制且性能只报告，非单调样本不用于质量选择；
- final resource audit=`passed`：wall=`605.281 s`，torch allocated/reserved=`14,342.71/14,892 MiB`，NVIDIA=
  `15,234 MiB`，cgroup=`26,264,842,240 bytes`，run=`1,610,165,885 bytes`，disk free=
  `43,679,989,760 bytes`，OOM/kill=`0/0`；no-torch resume=`2.316 s`，复用 10 个 completed stages，GPU launch=false；
- 三个 candidate 全部因质量门失败，故 `method_state=rejected_quality_or_integrity_gate`、selected arm=`p1-source`，
  生产资产保持 A3*=R0/D2 immutable exact alias；P1 实验以合规负结果 `done`。结论仅适用于 scene-0230/seed-0/
  冻结 36-view contribution 与 57-view quality 合同，不证明跨场景 pruning 失败，也不授权 post-hoc 新 arm。
  A4 最低完成集还缺 P2/P3；下一动作只冻结 P2 FP16 逐字段数值与质量协议，P3 继续未授权。

#### 11.2.5 P2 mixed-precision 结果前协议冻结

- protocol=`configs/worldsim_v3/a4_p2_mixed_precision_protocol_v1.yaml`，SHA-256=
  `6558fb3f0864c7711add2bd8b61500670ddbf631be7e356f1eac77c57c136d4e`，P1 closeout=
  `e733cbed8121643fca2223153c496a009b52283d`；exact 输入为 source checkpoint/config/registry、P1 canonical
  summary/manifest/resource/source-quality/resume/terminal 共 9 files，加 33-mask directory；P1 必须为 21/21 audits、
  selected=`p1-source`、method rejected、terminal done，禁止把 rejected prune arm 传给 P2；
- arm 只固定 `p2-source` 与 `p2-gs-param-fp16`。候选 checkpoint 仅把 Background/RigidNodes 的 `_scales`、
  `_quats`、`_features_dc`、`_features_rest`、`_opacities` 共 10 个 float32 tensor 以 IEEE binary16
  round-to-nearest-ties-even 转为 float16；candidate bitwise 必须等于 `source.to(float16)`，一次原子写入，schema/count/
  actor index 不变；
- source-only dtype audit 在任何 P2 render 前发现 Background `_means` 范围 `[-686.0377,2996.3384] m`，直接 FP16
  roundtrip 最大绝对误差=`0.999267578125 m`；Rigid means 也记录 `.0009765625 m`。因此两类 `_means`、Sky、LPIPS、
  trajectory、`points_ids`、ancestry/quota/boundary state 与 step 均预先冻结为 FP32/原 dtype exact，不允许看结果后改；
- runtime candidate 只让上述 10 个持久 Gaussian parameter 保持 FP16；`collect_gaussians` 后在进入 gsplat 前把
  means/scales/quats/RGB/opacities 显式转为 FP32，autocast=false。P2 研究的是 mixed-precision parameter storage，
  不授权或宣称 FP16 renderer kernel、Tensor Core speedup；checkpoint bytes 下降也不等于 peak VRAM 必然下降；
- 质量使用同一 57-view source replay 与 33 个 frozen masks。global PSNR/SSIM/LPIPS 最大退化=
  `.05 dB/.001/.001`；actor/boundary=`.10 dB/.0025/.0025`、MAE `+.001`；non-target=
  `.05 dB/.001/.001`、MAE `+.0005`。候选必须 31/31 全通过，缺失/nonfinite 直接 reject；
- runtime 固定 P0 的 `frames 10/100/190 × cameras 0/1/2`、2 warm-up、`800×450`、CUDA sync/nearest-rank；
  同时报告 checkpoint/load、persistent parameter bytes/dtype、renderer input dtype、P50/P95/FPS 与资源，filesystem
  cache 仍 uncontrolled，性能值不参与质量选择；
- selection 只有两种：候选 conversion/preservation/reload/dtype/quality/bytes/resource 全通过则选择
  `p2-gs-param-fp16`；否则 method rejected 并 exact fallback 到 `p2-source`。禁止事后增减字段或阈值；
- recovery=`input_audit→source_dtype_audit→materialize→evaluate→runtime→aggregate→resume_audit` 共 7 stages；
  ceiling=`900 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA / 48 GiB cgroup /
  1 GB run / 30 GB disk floor / OOM 0/0`，required audits=`19`。full validator exact 通过 10 项输入记录与 10 个
  converted source fields；协议测试=`9 passed`，联合 WorldSim V3=`199 passed`。本条无 P2 新 render/conversion
  measurement；下一动作只实现并提交 runner，P3 继续未授权。

#### 11.2.6 P2 mixed-precision formal 结果与部署裁决

- protocol/runner/fix=`588e37e / 1cd9a6e / dcf2822`。formal r1=
  `20260809T174337Z__a4-p2-mixed-precision-s0-r1` 已完成 conversion、两臂 57-view quality、runtime、aggregate 与
  no-torch resume，且 aggregate 选择 mixed arm；但参数账本只调用顶层 `trainer.named_parameters()`，遗漏普通
  `trainer.models` 映射内的 Gaussian Parameters，导致两臂都误记为 `9,883,392 bytes`、无 FP16 bucket，finalizer
  唯一失败 `checkpoint_reduction_and_runtime_matrix_exact`。r1 保留 `blocked`，terminal SHA=`5ef3dab6...74c0`；
- 修复 `dcf2822` 只显式遍历 mapped models、按 Parameter identity 去重并增加未注册映射回归；没有修改 protocol、
  转换字段、renderer、质量阈值或 selection。聚焦 P2=`20 passed`、完整 WorldSim V3=`210 passed`；
- canonical r2=`20260809T174850Z__a4-p2-mixed-precision-s0-r2`，source commit=`dcf2822...9860`，exit=`0`、
  terminal=`done`、19/19 audits 全 true。summary/manifest/resource/terminal SHA=`980f9b0f...1103 / bed45626...98cb /
  221d5e82...0df5 / 80dd8178...645`；source replay 31 endpoints 的最大绝对差=`0`；
- candidate checkpoint SHA=`7be87e8b...7448`，bytes=`432,111,754`，相对 source `578,819,674` 减少
  `146,707,920 bytes / 25.346049%`；candidate registry SHA=`69c4f38a...8a27`，`3,721,277 bytes`，actor counts/
  indices exact。10 个字段均 bitwise 等于 `source.to(float16)`，其余 `75` 个 tensor 与 schema exact；
- runtime persistent parameter bytes 从 source `394,641,424` 降到 `247,936,208`（`-146,705,216 / 37.174307%`），
  candidate=`146,705,216 float16 + 101,230,992 float32 bytes`。质量与 runtime 分别审计 `57/11` 次 renderer-input
  以及 `114/22` 次球谐输入，全部为 FP32 且 autocast=false；这不等于 FP16 renderer 或 Tensor Core claim；
- candidate 31/31 safeguards 全通过。最接近门限的是 high-support boundary LPIPS `+0.000036410`，只使用
  `.0025` 预算的 `1.4564%`；global human SSIM `-0.000009515`、boundary-support actor PSNR `-0.000755311 dB`，
  其余 endpoint 也均在冻结预算内。选择语义是 bounded quality loss，不是质量提升；
- 9-view runtime 仅报告：source/candidate load=`.33669/.47407 s`，P50=`.04583/.08721 s`，P95=
  `.13170/.09750 s`，FPS=`17.256/13.065`。cache 未控制且 P50/FPS 退化，禁止 speedup claim，也不参与质量选择；
- resource audit=`passed`：wall=`206.548 s`，allocated/reserved/NVIDIA=`7,754.05/8,072/8,426 MiB`，cgroup=
  `29,673,631,744 bytes`，run=`436,430,167 bytes`，disk free=`42,806,071,296 bytes`，OOM/kill=`0/0`；no-torch
  resume=`1.217 s`，6/6 completed stages 复用、GPU launch=false；
- `method_state=selected_mixed_precision_parameter_storage_fp32_render`，selected=`p2-gs-param-fp16`。P2 experiment=
  `done`；部署资产在 A3*=R0/D2 与 P1 source exact fallback 上增加冻结的 mixed storage 层。结论只覆盖
  scene-0230/seed-0/单卡/冻结矩阵；A4 最低完成集还缺 P3，下一动作只冻结 P3 chunk 协议，P4 继续条件式。

#### 11.2.7 P3 exact chunk package 结果前协议冻结

- protocol=`configs/worldsim_v3/a4_p3_chunk_protocol_v1.yaml`，SHA-256=
  `dfaaba79162961673b632271727c8a949c45519b1e75e5ed873badf999ad1b41`，P2 closeout=
  `e954e23cc4e81e4ba15ebcd9be4666cc3269ebe7`；exact 输入为 P2-selected mixed checkpoint/config/registry、P2
  canonical summary/manifest/resource/selected-quality/resume/terminal 共 9 files，加原 33-mask directory。P2
  必须为 terminal done、19/19 audits、31/31 safeguards 且 selected=`p2-gs-param-fp16`，禁止回退到原 FP32
  checkpoint 或接入 P1 rejected prune arm；
- static chunk 在任何 P3 materialization/render 前固定为原点 `[0,0] m`、`50 m` 固定 XY 半开网格，cell=
  `floor(float64(xy-origin)/50)`，按数值 `(ix,iy)` 排序。source-only audit 得到 `133` 个 occupied chunks、
  count=`1..330,169`，其中 `98` 个少于 100 rows、`7` 个不少于 10,000 rows，`.25 m` cell-edge band 含
  `69,393` rows；所有 sparse/outlier chunk 均保留，禁止 minimum-count、merge、cell-size search 或事后改网格；
- Background/RigidNodes 分别冻结 `25/26` 个 row tensors：mixed-precision render fields、全部 ancestry fields，
  以及 Rigid `points_ids`。每个 static/actor asset 携带严格升序 `int64 source_flat_indices` 与全部对应 row values；
  source static index inventory SHA=`d78fa6e...27cae`。24 个 actor 资产按模型索引 `0..23` 输出，23 个非空
  actor 的源行均为 interleaved、不得假设 contiguous slice；actor 14 必须输出含全部 zero-row fields 的显式空资产，
  actor inventory SHA=`384870e6...f23a`；
- package 固定为 manifest + shared skeleton + 133 static files + 24 actor files，共 `159` files。skeleton 逐路径
  保留所有非 row checkpoint state，并把每个 row tensor 替换为带 model/path/rows/shape/dtype 的唯一 sentinel；
  row values 只在所属 chunk/actor 存一次。manifest 必须绑定 protocol/source/grid/schema 及逐文件 relative path/
  SHA/bytes/count/bounds/index digest；禁止复制 source checkpoint 或落盘持久 reassembled checkpoint；
- candidate 只在内存中按源 flat index scatter，重建后 recursive container schema、每个 tensor shape/dtype/value
  SHA、模型/actor count、registry 与 P2 mixed-persistent/FP32-renderer adapter 必须 exact。质量仍用 57 views/33
  frozen masks：source 先以 `1e-6 PSNR / 1e-8 其余` 回放 P2 selected quality，chunk arm 再要求 57 个 RGB SHA
  逐 view 相等且 31 个 global/actor/boundary/non-target endpoints 在同一 exact tolerance 内；这不是质量提升门；
- runtime 固定两臂、`frames 10/100/190 × cameras 0/1/2`、2 warm-up、`800×450`、sync/nearest-rank。
  chunk load 必须读取并校验全部 133+24 assets 与 skeleton 后完整重组，不允许 selective loading/culling；cache
  uncontrolled，load/reassembly/P50/P95/FPS/VRAM/RAM/package bytes 只报告且不参与 selection；
- selection 只有两种：inventory/manifest/reassembly/reload/adapter/source replay/RGB+quality/resource 全通过则
  `selected_exact_chunk_package`；任一失败则 `rejected_chunk_integrity_quality_or_resource_gate` 并 exact fallback
  到 P2-selected checkpoint。recovery=`input→source_layout→materialize→reassemble→evaluate→runtime→aggregate→resume`
  共 8 stages；ceiling 延续 `900 s / 16 GiB allocated / 24 GiB reserved / 24,000 MiB NVIDIA / 48 GiB cgroup /
  1 GB run / 30 GB disk floor / OOM 0/0`，required audits=`21`；
- full validator exact 通过 10 项输入记录、133 static/24 actor inventory 与 25/26 row schema；协议测试=`12 passed`，
  联合 WorldSim V3=`222 passed`。本条没有创建 chunk package、render 或 formal run；下一动作只实现并提交 runner，
  不启动 P4、D3/D4 或 A3 R2–R4。

#### 11.2.8 P3 exact chunk package formal 结果与 A4 收口

- runner 提交=`aba55777f38a3d8e4363d2ff7d546d412214b481`；定向测试=`23 passed`，联合 WorldSim V3=
  `233 passed`。canonical r1=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A4-DEPLOYMENT-01/20260809T184240Z__a4-p3-chunk-s0-r1`，
  exit=`0`、terminal=`done`、21/21 audits passed；summary SHA-256=
  `f8e6e16639b685f4a9a80fd20a57d6553b9e09e7e5ddfc2b1a5a38604abca293`，manifest=
  `8b79d355...bd7af`，resource=`55ee6f0b...55e81`，terminal=`80dd8178...c645`；
- package manifest SHA=`35a3f1fe...64b8`，固定 `133 static + 24 actor + skeleton + manifest = 159 files`；
  package 总计 `444,177,055 bytes`，source checkpoint=`432,111,754 bytes`，开销=`12,065,301 bytes /
  +2.792171%`。158 个 payload=`444,033,142 bytes`，skeleton=`101,176,684 bytes / 51 sentinels`；没有复制
  source，也没有落盘 reconstructed checkpoint；
- 内存重组的 85 个 tensor path、recursive schema、shape/dtype/value SHA 与 non-tensor signature 全部 exact；
  Background `1,205,164` 行、RigidNodes `104,704` 行均 covered once，missing/duplicated=`0/0`。24 个 actor
  均有资产，actor 14 保留显式空资产；source checkpoint/registry 前后 SHA 保持
  `7be87e8b...7448 / 69c4f38a...48a27`；
- source replay 的 31 endpoints 最大绝对差=`0`；chunk 相对 source 的 57 个 RGB SHA、31 endpoints 与 masks
  全部 exact。quality 两臂分别记录 `57/114` renderer/SH observations，runtime 两臂记录 `11/22`，全部 renderer
  inputs 为 FP32 且 autocast=false；因此 P2 的 FP16-persistent/FP32-renderer 合同保持 exact；
- 9-view、2 warm-up、`800×450`、filesystem cache uncontrolled 的报告值为：source load=`.9071 s`、
  P50/P95=`.03013/.09446 s`、FPS=`21.2783`；chunk 全资产 load/reassembly=`4.1775 s`、P50/P95=
  `.03950/.10586 s`、FPS=`20.4471`。结果不支持 package size reduction、load speedup、render speedup、
  selective streaming 或 concurrency claim；chunk 的价值限于 exact spatial/actor asset separation；
- resource audit passed：wall=`221.786 s`，torch allocated/reserved=`7,614.99/8,066 MiB`，NVIDIA sampled=
  `8,420 MiB`，cgroup peak=`32,689,958,912 bytes`，run=`444,885,133 bytes`，disk free=
  `42,359,705,600 bytes`，OOM/kill=`0/0`；no-torch resume=`1.104 s`、7 actions、159 package artifacts verified、
  GPU launch=false；
- selection=`p3-chunk-package`，method=`selected_exact_chunk_package`，P3=`done`。P0/P5/P1/P2/P3 最低完成集
  已全部闭环，A4=`done`；P4 LOD 继续为可选项且不作为 R0 门禁。下一任务按依赖执行 F0 官方能力审计，之后再做 R0。

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

当前完成性：P0/P1/P2/P3/P5 均已形成 canonical terminal、资源/恢复审计和受限结论，A4 最低完成集满足。

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

P0 已由 v2 r2 正式关闭；P5 r1 的 runtime/state-key 混淆保留为 `blocked`，修复后 r2 已以 14/14 audits
正式关闭 P5。P1 r1 已完成 21/21 audits，但 b05/b10/b20 分别失败 3/12/15 个冻结质量 safeguard；方法
`rejected_quality_or_integrity_gate`，生产资产 exact fallback 到 source，P1 实验以负结果 `done`。P2 r1 的 mapped
parameter ledger 漏项保留为 `blocked`；只修账本后的 canonical r2 已 19/19 audits、31/31 safeguards 全通过，选择
`p2-gs-param-fp16`，P2=`done`。P3 canonical r1 已 21/21 audits、57 RGB/31 endpoints 和 checkpoint reassembly
exact，选择 `p3-chunk-package`；package 比 source 大 `2.792171%` 且 load/reassembly 更慢，性能只报告、不宣称收益。
P0/P5/P1/P2/P3 最低完成集已满足，A4=`done`。

R0 必交付仍要求 F0 审计，因此当前唯一动作切换为 `WS-V3-F0-FEEDFORWARD-AUDIT-01`：

```text
1. 使用已冻结 F0 协议运行 canonical 本机只读审计，核对 exact clean 官方 checkout 和 16 个文件指纹
2. 记录 CLI help、两组官方 focused tests、Python/uv/GPU/RAM/disk/token/weight/NCore/terms 前置条件
3. 只在全部前置条件通过时授权后续一窗口 inference；任一失败时 formal run 以 audit done、inference not-run 收口
4. 给出 F1 conditional_not_unlocked/unlocked 裁决；不启动 F1，不启动 P4、D3/D4 或 A3 R2–R4
```

D3/D4 继续未解锁；P4 继续条件式。负结果保留为最终交付，不因能力缺口改写官方/本地边界。

---

## 17. 更新日志

### 2026-08-10 — F0 Instant NuRec 官方/本机能力审计协议冻结

- protocol SHA=`2004a029...fd611`，官方 checkout 固定为 `1ce2288e...8d0` / tree `96e36fa4...5dc0`，16 个
  source files、三份权重的 commit/bytes/SHA/Xet、三层 license 与官方来源 URL 全部锁定；
- 明确论文完整模型与 standalone CLI 不等价：CLI 为 NCore V4/FTheta/CUDA、static PLY only，不读 LiDAR，
  不导出 dynamic/sky/ISP/actor registry/trajectory/depth；网页 demo 不登记为本地能力；
- 本机 smoke 采用 11 条全合取前置条件与 fail-closed 路由；失败时禁止构造 inference command、安装依赖、下载
  权重/gated 数据或启动 GPU；F1 预设为 `conditional_not_unlocked`，静态 PLY 不冒充 exact StreetGS checkpoint；
- runner SHA=`249f26d5...8e4a`，协议与 exact checkout 测试 8 passed、联合回归 241 passed。本条无 formal run 或 inference measurement；
  下一动作只运行已提交协议的 canonical F0 audit。

### 2026-08-10 — A4-P3 formal 与 A4 done

- runner `aba5577` 的 23 项定向测试与 233 项联合回归通过；canonical r1 terminal=`done`、21/21 audits，
  summary SHA=`f8e6e166...a293`；
- 133 static、24 actor、skeleton 与 manifest 共 159 files；85 tensors/non-tensor state exact reassembly，57 RGB 与
  31 endpoints exact，source/registry hash 不变，no-torch resume 通过；
- package=`444,177,055 bytes`，较 source checkpoint 大 `2.792171%`；全资产 load/reassembly=`4.1775 s`，未优于
  source `.9071 s`，render FPS=`20.447/21.278`。只登记 exact asset separation，不登记 size/load/render speedup；
- selected=`p3-chunk-package`，P3=`done`；A4 最低完成集全部满足，A4=`done`。下一任务为 R0 前置的 F0 官方能力审计。

### 2026-08-10 — A4-P3 exact chunk package 协议冻结

- protocol SHA=`dfaaba79...1b41`，输入 exact 锁定 P2-selected mixed checkpoint 与 P2 19/19 canonical evidence；
- static 固定 50 m XY 半开网格与 133-chunk source inventory；dynamic 固定 24 actor assets、显式 interleaved
  source indices 与 actor 14 zero-row asset；Background/Rigid row schema=`25/26`；
- package 固定 manifest+skeleton+157 data assets、内存 bitwise exact reassembly、57 RGB SHA/31 endpoints exact、
  两臂 9-view report-only runtime、8-stage recovery、21 audits 与 exact P2 fallback；
- full validator passed，协议测试 12 passed、联合 WorldSim V3 222 passed。本条没有创建 P3 package/render/formal
  run；下一动作只实现并提交 runner。

### 2026-08-10 — A4-P2 mixed-precision formal 完成并选择候选

- runner/fix=`1cd9a6e / dcf2822`；r1 因 mapped-model parameter ledger 漏项保留 blocked，r2=
  `20260809T174850Z__a4-p2-mixed-precision-s0-r2` exit=`0`、summary SHA=`980f9b0f...1103`、19/19 audits；
- mixed checkpoint=`432,111,754 bytes`，较 source 减少 `25.346049%`；persistent parameter bytes 减少
  `37.174307%`，10-field conversion、75 preserved tensors、schema、actor registry 和 FP32 renderer inputs exact；
- candidate 31/31 safeguards 通过；runtime P50/FPS 未改善且 cache 未控制，只报告、不作 speedup claim；资源门和
  no-torch resume 通过；selected=`p2-gs-param-fp16`、P2=`done`。A4 下一步只冻结 P3 chunk protocol。

### 2026-08-10 — A4-P2 mixed-precision 协议冻结

- protocol SHA=`6558fb3f...6d4e`，输入 exact 锁定 P1-selected source 与 P1 21/21 canonical evidence；只设
  source 和一个 mixed candidate，不继承 rejected prune arm；
- 候选只转换两模型的 scales/quats/features/opacities 共 10 tensors；Background means 的 source-only FP16
  roundtrip 最大误差近 `1 m`，故 means、Sky、trajectory、LPIPS 与 provenance 在结果前固定保留；
- persistent parameter FP16 与 renderer-input FP32 分离，禁止 FP16 kernel/Tensor Core claim；57-view 31 项质量门、
  9-view runtime、7-stage recovery、900 s/16 GiB torch/48 GiB cgroup/1 GB run 与 19 audits 已冻结；
- full validator passed，协议测试 9 passed、联合 WorldSim V3 199 passed。本条未创建 P2 formal measurement；
  下一动作只实现并提交 P2 runner，P3 未授权。

### 2026-08-10 — A4-P1 formal contribution-prune 完成并拒绝候选

- runner=`19cab2cf...7163`；canonical r1=`20260809T165058Z__a4-p1-contribution-prune-s0-r1`，exit=`0`，
  summary SHA=`7c5347e3...7119`，21/21 audits 全通过；36-view contribution score、三臂 checkpoint/registry、
  4-arm 完整质量与 runtime、no-torch resume 均完成，资源门通过；
- b05/b10/b20 checkpoint bytes 分别下降 `23.88/47.76/95.53 MB`，但最小 b05 已违反 global occupied PSNR、
  global PSNR 与 non-target PSNR 三个冻结门；b10/b20 分别失败 12/15 项，不新增 post-hoc arm；
- P1 method=`rejected_quality_or_integrity_gate`，selected=`p1-source immutable exact alias`，实验终态=`done`。
  A4 仍缺 P2/P3；下一步只冻结 P2 FP16 协议。

### 2026-08-10 — A4-P1 contribution-prune 协议冻结

- protocol SHA=`4f893c09...429b`，精确锁定 13 files + 1 frozen-mask directory；train-only 18-view contribution
  ranking、heldout audit-only 18 views 与完整 57-view quality split 明确分离；
- arm 固定 source/b05/b10/b20，逐 static/actor 资产稳定排序；checkpoint row alignment、invariant、registry、质量
  退化阈值、最大合格 fraction 与 exact source fallback 均在结果前冻结；
- 11-stage recovery、1,800 s/20 GiB torch/48 GiB cgroup/2.5 GB run ceiling 与 21 audits 已固定；validator full
  preflight、11 项协议测试、178 项联合回归通过。下一动作只实现并提交 P1 runner；P2/P3 仍未授权。

### 2026-08-10 — A4-P5 formal registry/resume 完成

- r1=`20260809T155209Z__a4-p5-registry-resume-s0-r1` 已生成 `14,729-byte` compact registry，但因混淆
  checkpoint key `points_ids` 与 runtime attribute `point_ids` 在 reload 后 blocked；旧 terminal 与 registry 保留；
- 修复=`0e899b2`，协议不变；canonical r2=`20260809T155753Z__a4-p5-registry-resume-s0-r2`，summary SHA=
  `0c86ff68...8744`，14/14 audits 全通过；1 static/24 actor、全部 count/index hash、source before/after 均 exact；
- reload=`52.321 s / one load / zero render`，resource audit passed，no-torch resume `.128 s` 复用四个 completed stage且
  无 GPU launch。P5=`done`；下一动作只冻结 P1 contribution-prune 协议，P2/P3 仍未授权。

### 2026-08-09 — A4-P5 registry/resume 协议冻结

- protocol SHA=`51acb935...5874`，输入锁定 9 项 P0/A3 immutable evidence；reference-only manifest 禁止复制/重写
  checkpoint，固定 `1 static / 24 actors（23 available）/ 1,309,868 total Gaussian`；
- reload 仅一次只读 checkpoint load，核对全部 actor index hash，不 render/训练；recovery 固定五 stage 与 no-torch
  resume；14 项 audits、资源 ceiling 和 2 MB registry ceiling 均在结果前冻结；
- validator full preflight passed；协议测试 6 passed，联合 WorldSim V3 158 passed。下一动作只实现并提交 P5 runner；
  正式运行及 P1/P2/P3 均未提前授权。

### 2026-08-09 — A4-P0 v2 formal profile 完成

- canonical r2=`20260809T152923Z__a4-p0-profile-v2-s0-r2`，exit=`0`，summary SHA=`0278a320...e92`，13/13
  audits 全通过；checkpoint/registry exact，无训练、checkpoint 或媒体输出，no-torch resume 复用通过；
- inventory=`582,541,102 bytes checkpoint+registry / 1 static block / 24 actors（23 available）`；prepare 占 wall
  `82.95%`，cold/warm load=`.391/.397 s`，render P50/P95=`.068/.127 s`，FPS=`16.378`；
- resources=`7,913 MiB allocated / 8,232 MiB reserved / 8,574 MiB NVIDIA / 22.79 GiB cgroup / OOM 0`，门禁通过；
  P0=`done`。下一门禁只冻结无模型变异的 P5 registry/resume 协议；P1/P2/P3 仍未授权。

### 2026-08-09 — A4-P0 v1 分辨率合同阻塞与 v2 重新冻结

- runner=`199abd9`；v1 formal r1 完成全部 probe 与 no-torch resume audit，但 11 行实际 render 均为 `800×450`，
  与 v1 的 `1600×900` 冻结值不符，因此 run=`blocked`，唯一失败 audit=`native_resolution_exact`；
- source config 的三路 `downscale_when_loading=[2,2,2]` 证明 `1600×900` 是传感器尺寸，当前 checkpoint 模型原生
  尺寸为 `800×450`；v1 结果只作合同纠错 diagnostic，不作为正式性能结论；
- v2 protocol SHA=`43db7182...3f18`，只纠正尺寸并冻结 6 项 v1 失败证据；其余输入、矩阵、资源与恢复门禁不变。
  该冻结点的下一动作是从新目录完整执行 v2，不复用 v1 measured runtime stage；P1/P2/P3/P5 当时均未授权。

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
WS-V3-A4-DEPLOYMENT-01 已 done；P0/P5/P1/P2/P3 全部闭环；P1 method rejected；P2 selected=p2-gs-param-fp16；P3 selected=p3-chunk-package。
WS-V3-F0-FEEDFORWARD-AUDIT-01 当前为 running；只审计 Instant NuRec 官方/本地能力，尚未授权 F1。

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
19. 核对 A4-P5 protocol SHA `51acb935...5874`、r1 blocked terminal SHA `61d30a11...773e`、runtime
    `point_ids` 修复 `0e899b2` 与 canonical r2 summary SHA `0c86ff68...8744`；确认 14/14 audits、24 actor index
    hashes、source before/after、resource 和 no-torch resume 全通过；
20. 核对 P1 protocol SHA `4f893c09...429b`、13 files + 1 mask directory、train-only 18-view ranking、b05/b10/b20、
    57-view safeguards、11-stage recovery、21 audits 与 178 项联合回归；
21. 核对 P1 runner `19cab2cf...7163`、canonical r1 summary SHA `7c5347e3...7119`、21/21 audits、source replay、
    b05/b10/b20 的 3/12/15 项质量失败、资源与 no-torch resume；保持 method rejected、source exact fallback；
22. 核对 P2 protocol SHA `6558fb3f...6d4e`、P1-selected source、10 个转换字段、Background means 近 1 m
    FP16 排除事实、31 项质量门、9-view runtime、7-stage recovery、19 audits 与 199 项联合回归；
23. 核对 P2 runner/fix `1cd9a6e/dcf2822`、r1 blocked terminal SHA `5ef3dab6...74c0`、canonical r2 summary
    SHA `980f9b0f...1103`、19/19 audits、31/31 safeguards、candidate checkpoint `7be87e8b...7448`、资源与 no-torch resume；
24. 核对 P3 protocol SHA `dfaaba79...1b41`、9 files + 1 mask directory、50 m/133 static chunks、24 actor
    assets、actor 14 empty、25/26 row tensors、159 package files、57 RGB/31 endpoints exact、8-stage recovery、
    21 audits 与 222 项联合回归；
25. 核对 P3 runner `aba5577`、canonical r1 summary SHA `f8e6e166...a293`、21/21 audits、159 files、85 tensor
    paths exact、57 RGB/31 endpoints exact、source/registry 不变、资源与 no-torch resume；保持 package +2.792171%、
    load/reassembly 更慢且无 streaming/load/render speedup claim；
26. 核对 F0 protocol SHA `2004a029...fd611`、官方 source revision/tree、16 个 source hashes、三份权重 provenance、
    static-Ply-only CLI boundary、11 项本机前置门与 8 项测试；只运行 canonical read-only audit。任一前置失败时不得
    构造 inference command。不得提前启动 F1、P4、A3 R2–R4 或 D3/D4。

不得恢复 A1/A2 或 V2 M5，不得依赖未提交 V2 M5 文件，不得把 ancestry 写成 measured depth，不得新增大型 diffusion。
```
