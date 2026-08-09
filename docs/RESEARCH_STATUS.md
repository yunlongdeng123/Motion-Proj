# Research Status

- 更新时间：2026-08-10
- 当前路线：面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化 V3.1
- 当前任务：`WS-V3-F0-FEEDFORWARD-AUDIT-01`
- 状态：`running`
- 当前门禁：A4-P0/P5/P1/P2/P3 已全部闭环，P3 canonical r1 选择 exact chunk package，A4=`done`；R0 前先完成 F0 官方能力审计，F1 尚未授权
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

A3 已给出受冻结合同约束的负答案：R1 S-B 四步工程链可重放，但 heldout evaluator 连续越过 GPU ceiling，且
资源无效 diagnostic 是 geometry 改善与 RGB safeguard 退化并存的 tradeoff。当前生产路由使用 R0/D2 exact alias，
不把 R1 checkpoint 升级为方法或部署基线。

A4-P0 v1 formal r1 已完成 probe 与无 torch resume audit，但 source config 的三路相机实际按 2 倍下采样加载，
模型原生 render 为 `800×450`，不是 v1 误写的传感器尺寸 `1600×900`。r1 因唯一 audit
`native_resolution_exact` 保留为 `blocked`；v2 只纠正该输入合同并冻结 r1 证据，不把 r1 性能登记为正式结果。
v2 validator 已核对 16 个 exact inputs，协议测试 7 passed，联合 WorldSim V3 回归 152 passed。

A4-P0 v2 formal r2 已以 `done` 关闭：13/13 audits 全 true，prepare 占 60.78 s wall 的 82.95%，cold/warm load
约 `.39/.40 s`，9 个模型原生 view 为 P50/P95 `.068/.127 s` 与 `16.38 FPS`；资源峰值为 `8,574 MiB`
NVIDIA sampled / `22.79 GiB` cgroup，OOM=0。P0 不证明并发或质量改进；它只支持先冻结无模型变异的 P5
registry/resume，而不先启动 prune、FP16 或 chunk。

P5 protocol SHA=`51acb935...5874` 已在新 P5 测量前冻结。r1 在成功生成 `14,729-byte` registry 后，因把 checkpoint
key `points_ids` 当作 runtime attribute 而 blocked；旧 terminal 保留。修复提交=`0e899b2`，未改变协议与测量合同。
canonical r2=`20260809T155753Z__a4-p5-registry-resume-s0-r2` 已 14/14 audits passed：reference-only registry
保持 `1 static / 24 actors（23 available / 1 unavailable）/ 1,309,868 total GS`，全部 actor count/index hash 与
source before/after SHA exact。reload=`52.321 s / one load / zero render`，资源门通过；no-torch resume=`.128 s`，
无 GPU launch并复用四个 completed stage。P5=`done`，不产生 chunk、filesystem-cold、concurrency 或质量 claim。

P1 完成时 A4 最低完成集仍要求 P2/P3，因此 task 当时保持 `running`。P1 protocol SHA=`4f893c09...429b` 在测量前冻结，runner=
`19cab2cf...7163`。canonical r1=`20260809T165058Z__a4-p1-contribution-prune-s0-r1` exit=`0`、21/21 audits
passed、summary SHA=`7c5347e3...7119`。36-view contribution score、b05/b10/b20 原子 checkpoint/registry、四臂
57-view global/actor/boundary/non-target 质量、9-view runtime 与 no-torch resume 均完成；source replay exact。
b05/b10/b20 分别减少 checkpoint `23,881,368 / 47,762,712 / 95,527,000 bytes`，但最小 b05 已使 global
occupied PSNR/global PSNR/non-target PSNR 退化 `0.117684/0.110926/0.125462 dB`，超过冻结 `0.10 dB` 门；
b10/b20 分别失败 12/15 项。全部候选因质量而 rejected，P1 method=`rejected_quality_or_integrity_gate`，生产路由
exact fallback 到 p1-source/A3*=R0-D2，实验终态=`done`。resource audit passed：wall=`605.281 s`、allocated/
reserved/NVIDIA=`14,342.71/14,892/15,234 MiB`、cgroup=`26,264,842,240 bytes`、run=`1,610,165,885 bytes`、
OOM=0；resume=`2.316 s`/10 stages/no torch/no GPU。

P2 protocol SHA=`6558fb3f...6d4e` 已在任何 P2 conversion/render 前冻结。输入 exact 指向 P1-selected source 与 P1
canonical evidence，不允许使用 rejected prune checkpoint。候选只转换 Background/RigidNodes 的 scales/quats/
features/opacities 共 10 tensors；source audit 显示 Background means 若 FP16 roundtrip 最大空间误差近 `1 m`，因此
means、Sky、LPIPS、trajectory 与 provenance 保留 FP32/原 dtype exact。runtime persistent parameters 为 FP16，
但进入 gsplat 前显式转 FP32、autocast=false，不宣称 FP16 renderer。57-view 31 项质量门、9-view runtime、
7-stage recovery、900 s/16 GiB torch/48 GiB cgroup/1 GB run ceiling 与 19 audits 已固定；full validator passed，
协议测试 9 passed、联合 WorldSim V3 199 passed。该冻结点的下一步只实现并提交 runner，P3 当时仍未授权。

P2 runner/fix=`1cd9a6e / dcf2822`。r1=`20260809T174337Z__a4-p2-mixed-precision-s0-r1` 的 conversion、quality、
runtime、aggregate 与 resume 均完成，但参数账本未遍历普通 `trainer.models` 映射，finalizer 唯一 audit 失败；r1
保持 `blocked`，terminal SHA=`5ef3dab6...74c0`。只修账本后的 canonical r2=
`20260809T174850Z__a4-p2-mixed-precision-s0-r2` exit=`0`、19/19 audits、31/31 safeguards、source replay exact，
summary SHA=`980f9b0f...1103`。candidate checkpoint=`7be87e8b...7448 / 432,111,754 bytes`，较 source 减少
`146,707,920 bytes / 25.346049%`；persistent parameters=`394,641,424→247,936,208 bytes / -37.174307%`。
runtime 只报告 source/candidate load=`.33669/.47407 s`、P50=`.04583/.08721 s`、P95=`.13170/.09750 s`、
FPS=`17.256/13.065`，不支持 speedup claim。resource passed：wall=`206.548 s`、allocated/reserved/NVIDIA=
`7,754.05/8,072/8,426 MiB`、cgroup=`29,673,631,744 bytes`、run=`436,430,167 bytes`、OOM=0；resume=
`1.217 s`/6 stages/no torch/no GPU。P2=`done`，selected=`p2-gs-param-fp16`；该时点 A4 仍缺 P3。

P3 protocol SHA=`dfaaba79...1b41` 已在任何 chunk materialization/render 前冻结，输入 exact 接 P2-selected mixed
checkpoint 与 P2 19/19 canonical evidence。static 使用原点 `[0,0] m` 的 50 m XY 半开网格，source-only audit
固定 `133` 个 occupied chunks（count `1..330,169`，98 个 `<100`，7 个 `>=10,000`），不允许稀疏/离群块丢弃、
merge 或 cell-size search。Background/Rigid row tensor schema=`25/26`；24 个 actor 均使用显式升序 source flat
indices，23 个非空 actor 全部 interleaved，actor 14 输出 zero-row asset。package 固定 manifest+skeleton+133 static+
24 actor=`159 files`，仅内存 scatter 重组，recursive tensor 必须 bitwise exact，禁止复制 source 或落盘重组
checkpoint。质量要求 source 回放 P2 exact、chunk 的 57 RGB SHA 与 31 endpoints exact；9-view runtime 读取全部
assets，只报告、不做 streaming/load/render speedup claim。8-stage recovery、900 s/16 GiB torch/48 GiB cgroup/
1 GB run ceiling、21 audits 与 P2 exact fallback 已固定；full validator passed，协议测试 12 passed、联合 WorldSim
V3 222 passed。本冻结点未创建 package/render/formal run；下一步只实现并提交 runner。

P3 runner=`aba55777...b481`。canonical r1=
`20260809T184240Z__a4-p3-chunk-s0-r1` exit=`0`、terminal=`done`、21/21 audits passed，summary SHA=
`f8e6e166...a293`。package manifest=`35a3f1fe...64b8`，`133 static + 24 actor + skeleton + manifest = 159 files`；
package=`444,177,055 bytes`，比 `432,111,754-byte` source checkpoint 大 `12,065,301 bytes / 2.792171%`。
85 个 tensor path 和 non-tensor state exact reassembly，Background/Rigid `1,205,164/104,704` rows covered once、
missing/duplicated=`0/0`，actor 14 显式为空；source checkpoint/registry SHA 前后不变。source replay 31 endpoints
max abs diff=`0`，chunk 的 57 RGB SHA、31 endpoints 与 masks 全 exact，P2 FP16-persistent/FP32-renderer adapter exact。
runtime 只报告：source/chunk load=`.9071/4.1775 s`、P50=`.03013/.03950 s`、P95=`.09446/.10586 s`、
FPS=`21.278/20.447`，filesystem cache uncontrolled；不支持 package size、load、render、streaming 或 concurrency
收益 claim。resource passed：wall=`221.786 s`，allocated/reserved/NVIDIA=`7,614.99/8,066/8,420 MiB`，cgroup=
`32,689,958,912 bytes`，run=`444,885,133 bytes`，disk free=`42,359,705,600 bytes`，OOM/kill=`0/0`；resume=
`1.104 s`/7 actions/159 artifacts/no torch/no GPU。selected=`p3-chunk-package`，method=
`selected_exact_chunk_package`，P3=`done`；A4 最低完成集满足，A4=`done`。

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
- 协议冻结提交时 formal 尚未启动。该证据本身只解除启动门，不构成 D1 质量结论。

## A2-D1 formal 完成证据

- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T085400Z__a2-d1-paired-formal30k-s0-r1`；
- source commit=`f32f96b47619e05066d2ee11c899e38d07398e11`；terminal=`done`；summary SHA-256=
  `e3b194c2ed0563385df70ca2043dbc791bedb21068d28dc9d75fb59984c166ac`；manifest SHA-256=
  `f10e6e654ab27289ccb1c995ebbe1ffde913009dbfb3eae0ab4c6414de18a560`；
- D0/D1 物化配置配对，初始化 provenance SHA 均为 `8951543c...b898`，初始 Background/RigidNodes=
  `946,484 / 75,002`；6×2 checkpoint 网格、quota/ancestry、native finite 与 24/24 actor 上限均通过；
- fixed 30k D0/D1：Background/Rigid/total GS=`1,182,619/177,628/1,360,247` 与
  `1,201,057/105,412/1,306,469`；global PSNR/SSIM/LPIPS=`27.7481/.851207/.176319` 与
  `27.7700/.850915/.177704`；质量轴更优数 D1/D0=`12/7`，裁决=`tradeoff_non_dominated`；
- matched 选中 D1 15k：Rigid=`176,741`，与 D0 target 差 `887 / 0.499%`；D0 视图为 fixed final exact alias；
  D1 Background/total=`2,432,701/2,609,442`，global=`25.9290/.825381/.217941`；质量轴更优数 D1/D0=
  `9/10`，裁决仍为 `tradeoff_non_dominated`；
- matched D1 boundary-support actor PSNR/SSIM/LPIPS=`29.2937/.902828/.061463`，优于 D0 的
  `27.1783/.882177/.068895`；但 non-target PSNR/SSIM/LPIPS=`24.3371/.822724/.090772`，劣于 D0 的
  `26.8707/.848887/.057715`。这是局部—全局 tradeoff，不是 D1 全面改进；
- D0/D1 train duration=`2883.08/2099.33 s`，peak GPU=`23,867/23,989 MiB`，peak cgroup=
  `10,350,350,336/16,012,115,968 bytes`；matched 15k elapsed=`1127.66 s`，资源按完整 D1 臂上界报告；
- fixed D0、fixed D1、matched D1 三次评测前后 checkpoint SHA 均不变，high/boundary/non-target 均 `done`，
  `oom=0 / oom_kill=0`。控制器登记 `d2_unlocked=true`，仅解锁 D2 协议冻结。

## A2-D2 协议冻结证据

- 配置：`configs/worldsim_v3/a2_d2_protocol_v1.yaml`，SHA-256=
  `acceb7f4ce0f8dc3745de2fcaca51659891cfd82e4175f5a0e5765d77a01e567`；
- immutable prerequisite：D1 canonical summary SHA=`e3b194c2...66ac`，D1 closeout commit=`f380dd2`；
- 真实信号只用训练帧 dynamic mask 的 3px 形态学轮廓带与 projected-center RGB channel-mean L1 residual；
  gsplat `means2d` 按像素坐标 nearest-center 采样，跳过不可见、非有限和中心出界项；
- per-actor quota 内排序为 boundary observed/mean → residual observed/mean → screen-grad → Gaussian index；
  D1 gradient eligibility、minimum recovery、maximum quota、split/clone cost、Background 与 native cull 全部不变；
- boundary scale cap 复用 native densify size threshold，pre-cap scale 先决定 split/clone geometry，再在原生 refinement
  前同比缩放三轴、保持 anisotropy，并清零 cap 行的 Adam moments；不新增 RNG draw；
- D3 depth/normal、D4 LiDAR/visibility/provenance pruning、非原生 cull 与 Background 干预明确禁止；
- 工程提交=`1065264762569c9832219936ddae6f063d6eaf07`；canonical worktree=
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d2-r8`；D2 patch SHA-256=
  `80fef55195906808d74394af0b997cfccbdb88fd7cb356b45240473e55f357cc`；replay/reverse-check 与六文件状态通过；
- D1/D2 materializer normalized-match、真实 `RigidNodes` synthetic integration 与联合回归=`29 passed`；boundary/residual
  各 6 次观测、1 次排序/refinement、6 个 cap、quota maximum、optimizer moments、checkpoint round-trip 和 module-off
  native state/RNG bitwise 均通过；
- paired smoke r1 见下一节；协议/工程门禁通过本身不等于 D2 方法通过。

## A2-D2 配对工程 smoke 完成证据

- canonical run=`20260809T111304Z__a2-d2-paired-smoke1k-s0-r1`，terminal=`done`；summary SHA=
  `749c7d15c27cc0798c267aa8af12857f3bea52a52ea9d00f7617a3b3edda3136`，manifest SHA=
  `5cb7879d898839b88a46c8ec7ec34141f3402245490416d589938658f33b4c8d`，source=`c594e0c`；
- D1/D2 configs normalized match，initialization provenance 与 frozen initial quota 精确匹配；两臂 step=1000，
  D1=`Background 1,141,192 / Rigid 152,733`，D2=`Background 1,144,988 / Rigid 152,807`；
- D2 observation event=`1001`，boundary/residual observations 各 `10,846,748`，refinement/ordering event=`5`，
  capped Gaussian=`365`，boundary-observed live=`56,732`；cap/quota/finite/checkpoint round-trip 全通过；
- D1/D2 duration=`142.17/141.99 s`，torch peak GPU=`9,615/9,620 MiB`，cgroup peak=
  `16,473,858,048/16,667,971,584 bytes`，`oom=0 / oom_kill=0`；
- 裁决=`d2_formal_unlocked=true`，仅解锁 formal 协议冻结；1k smoke 不登记质量改进。

## A2-D2 formal 协议冻结证据

- formal config=`configs/worldsim_v3/a2_d2_formal_v1.yaml`，SHA-256=
  `b66cf795c55dfe65315ecf49c09951482d8d6809ce7d001b901942a6bd9a05bc`；提交=`20b3f4d`；39 tests passed；
- D1 baseline 使用 formal r1 immutable exact alias，不重训：summary SHA=`e3b194c2...66ac`，provenance SHA=
  `8951543c...b898`，fixed checkpoint SHA=`c9d2a052...af52`，target Rigid=`105,412`；
- 唯一新训练为 D2 30k / seed 0 / 5k checkpoint grid；fixed 比较 D1 alias 与 D2 30k，matched 从 D2
  grid 匹配 D1 fixed Rigid target，最大 relative gap=2%，无 pruning/retrain/retune/mutation；
- held-out/high/boundary/non-target、checkpoint immutability、quality 与 quality-cost exact Pareto 完整继承 D1；
- read-only preflight=`done`，输出 SHA=`9cf49af0be9a2676c6c113bee963efb79704bb9434083857684f97bd19caaa28`；
  project=`20b3f4d`、GPU=`0 MiB`、free disk=`47.92 GiB`，所有依赖与资源门禁通过。

## A2-D2 formal 完成与 A2 裁决

- canonical run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T113230Z__a2-d2-formal30k-s0-r1`，
  terminal=`done`，source=`482fba0`，summary SHA-256=`9c41dfc83c9da0a14201e1c719fb3d0e2cf59dd1ad20cd279c6e1a9a1c97de7d`；
- D2 final checkpoint SHA-256=`1a061247...e7c`，counts=`Background 1,205,164 / Rigid 104,704`；D1 reference
  checkpoint 运行前后 SHA 都是 `c9d2a052...af52`，初始化 provenance SHA 精确匹配；
- 5k–30k 六个 checkpoint 全部通过 finite/quota/cap 审计；matched 选中 30k，Rigid gap=`708 / 0.67165%`，
  matched D2 因而是 fixed D2 exact alias；
- D1→D2 global PSNR/SSIM/LPIPS 从 `27.770024/.850915/.177704` 变为
  `27.703188/.850333/.178344`；boundary-support boundary-band 从 `25.770024/.821572/.048382` 变为
  `26.171399/.828868/.044568`。边界三项改善与 global/部分 actor/non-target 退化并存；
- fixed/matched strict-quality Pareto 都为 `tradeoff_non_dominated`（D1/D2/equal=`11/8/0`），quality-cost
  也为 `tradeoff_non_dominated`（`14/9/1`）；D2/D1 wall=`2720.82/2099.33 s`；
- 297 条资源记录、四个 stage 全部 completed，peak GPU=`23,989 MiB`，full-run peak cgroup=
  `25,837,490,176 bytes`，`oom=0 / oom_kill=0`，终态 GPU=`0 MiB`；
- A2 状态冻结为 `done`。A3 使用 D2 boundary-residual 作为 boundary-priority research asset，D1 quota-only
  作为低成本/全局质量 fallback；这不是 dominance 或跨场景结论。`d3_unlocked=false`，D4 未启动。

## A3-I0 语义协议冻结证据

- config=`configs/worldsim_v3/a3_local_refine_protocol_v1.yaml`，SHA-256=
  `03fbf632645326692bbcf18ab18a08b5440c7733c709f925945c78018bb272d0`；依赖 A2 closeout=`2246693`、
  D2 checkpoint SHA=`1a061247...e7c`、summary SHA=`9c41dfc8...de7d`、registry SHA=`ed57764e...0c68`；
- 固定 scene-0230 / seed 0 / cameras 0–2、high/boundary 两 actor、lateral/delete 与 19 个只读 held-out frames；
  D1 checkpoint `c9d2a052...af52` 只作 fallback；
- affected set 冻结为 paired source/edited footprint（threshold 2、2px dilation）、supported hole、first-hit conflict
  的并集，再做 3px dilation；target actor 只作冻结 context；
- S-A 要求排除 target view 的 alternate observed RGB + calibrated reprojection；S-B 只接受 T0 LiDAR measured 或
  至少两视图 geometry，禁止 RGB loss；S-C 不更新、不 seed、不进 loss；
- depth 产品继续分为 expected=`diagnostic`、first-hit=`T1`、measured LiDAR=`T0`；D2 Background ancestry 的
  `240,528` 个 direct LiDAR roots 只证明 provenance，不是 measured-depth GT；
- R0 为 D2 immutable exact alias；首个工程门 R1 仅允许 affected S-A/S-B Background opacity/scale，outside
  参数与 optimizer state、RigidNodes、trajectory、registry 全部 exact；
- `formal_training_authorized=false`。未提交 V2 M5 config/metrics/runner 明确排除为依赖；I0 当时要求 paired smoke
  后再冻结数值合同，该门已由下方 real paired/frozen replay 证据关闭；新增 `12 passed`，联合回归 `98 passed`。

## A3 R0/R1 engineering guard 与 synthetic closeout

- implementation=`9c639dd5a0adcd1f8b5126f7f20d836815b127a6`；DriveStudio patch SHA-256=
  `155ec58fd2bfdc2e40357035dc20800bf2340b0c1c9ac5972c7c78efbd8cb69b`；独立工作树=
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1`，apply/reverse、`py_compile`、import 均通过；
- canonical synthetic run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T132133Z__a3-r0-r1-synthetic-s0-r1`，
  terminal=`done`，summary SHA-256=`2ac123f0603120a103743e59680a31dd4cdf5b6d5fa45605d7c84d36ec337ada`，
  manifest SHA-256=`8ffa697e15d8a97108d8281a51313119c304fbf0f245d88bfbd127663fde27c4`；
- R0 materializer 重新命中 checkpoint/config/protocol SHA，只生成 immutable exact alias；optimizer steps=`0`，
  无新 checkpoint/key；
- R1 guard 在 Adam step 前只保留 affected S-A/S-B Background opacity/scale 行梯度，step 后逐位审计参数与 moments；
  synthetic 中授权行变化，outside、position/color、RigidNodes/trajectory、shape/order exact；
- 原 D2 与 A3 module-off 的 RGB/SSIM loss tensor 逐位相等；缺少 paired provenance/masks 会拒绝；
  checkpoint 实际布局为 Background=`1,205,164` 行、RigidNodes=`104,704` 行、trajectory=`196×24`；
- 联合 WorldSim V3/materializer 回归=`110 passed`。该 synthetic run 自身仍为 `synthetic_contract_only` 且记录
  `paired_engineering_smoke_complete=false`；后续 paired 门见下一节，`formal_training_authorized` 始终为 false。

## A3 R1 真实 paired smoke、数值冻结与 replay 证据

- heldout-safe sidecar run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T133911Z__a3-sb-sidecar-s0-r3`；
  manifest/rows SHA=`42474f73fc563a2bba4c52cbec029bb4c28d33a21ca5f3d83ad4311bb7957273 / c5756ecbc0eabee9a576a55297a1739aa20e2af578aa4a5a92e727701b5138fc`；
  frame `0/31` 与 heldout 交集为空；affected/S-B mutable/S-C=`16,502 / 51 / 16,451` rows，四 unit 共 8 个
  S-B/T0 geometry pixels，S-A/RGB=`0/ABSTAIN`；
- paired implementation=`d89e0ace37eda22434470849ec9940360c0e9251`，CUDA init fix=`78741b3abee07b2c39be6646c63928e8212b6a6b`；
  当前 DriveStudio patch SHA=`f1732f63ae38f9298cdbd45d38e91bbd9fb5d3dec46e4b96c647ef14db3c588a`，
  materializer 会移除 native regularizer，trainer 再次 fail-closed 校验；S-B occupancy 只认 T0 LiDAR；
- canonical paired run=`20260809T135921Z__a3-r1-sb-paired4-s0-r2`，summary/manifest SHA=
  `ba4e2b853690f0b9c9bb7bfe039b4571db16c020ce726768a1ff884b09b3557d / de717ba0a5adb1afeb416a15a53ec55f471a8eb841882f784012b04ac86b596c`；
  step `30001–30004` 的 opacity/scale 授权行均有 finite nonzero gradient/变化，outside parameter/Adam、
  Rigid/trajectory/registry、shape/order exact；checkpoint SHA=`e995e7c266d9fed4e64c86813718e46ab4576bbfdf60500a637bdaeaaba78cd1`；
- numeric freeze implementation=`c02c8c74c671362e86269bd7e00980bfa75ae1c9`；config SHA=
  `d9289df0b2ac7df7a7c408b5cb1601bc5f874e2922ebc9cb87961aacee43b3e3`，冻结 4 steps、LR `0.05/0.005`、
  affected/mutable cap `16,502/51`、seed cap 0、alpha 0.5 与资源 ceiling；联合回归=`119 passed`；
- frozen replay run=`20260809T140534Z__a3-r1-sb-frozen-replay4-s0-r1`，summary/manifest SHA=
  `7d820a53de21f505a5c56043d56556edb8d3a86510488ea3956b7cfa159187c6 / 393e65d5f91c0e2072eebd7c23a1161d46422502220ceeeaa18c04905fec646d`；
  四 unit loss 逐值一致并重现同一 checkpoint SHA；wall/GPU/cgroup=`50.68 s / 8,286.86 MiB / 22,631,796,736 bytes`，OOM delta 0；
- 结论仅为 `real_paired_engineering_and_bitwise_replay_done`。S-A 未物化，S-B pixel quality claim 禁止，
  `formal_training_authorized=false`，R2–R4 未授权。

## A3 R1 heldout 只读评测负结果与任务收口

- heldout protocol=`configs/worldsim_v3/a3_r1_eval_protocol_v1.yaml`，SHA-256=
  `eb87a9f2ea7df9bdc050a8d4e4f3cdc7c6a1115ea6f4f69e2fd3c8011904b05a`；冻结/评测器提交=
  `42508fb / c8fc560`；资源审计与内存诊断提交=`05cee1e / c9e3df4 / ef74622 / c2eb14f`；联合回归
  `139 passed`；
- closeout run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T144037Z__a3-r1-heldout-eval-s0-r5`，
  exit=`1`，terminal=`blocked / peak_gpu_memory_mib`；resource audit SHA=
  `d9536f4ec937bee0694a754038b22ab75a4b6b028f20e1e6f42e38e4db9a6280`；wall/GPU/cgroup/run bytes=
  `117.983 s / 14,241.399 MiB / 23,749,709,824 / 299,910`，冻结 GPU ceiling=`12,288 MiB`，OOM delta=`0/0`；
- r2/r4 的完整指标路径分别为 `14,241.777 / 14,244.924 MiB`，同样只失败 GPU ceiling；r3 在指标前
  失败于 Rigid quota CPU/CUDA validator，已修复且不作结果。未提高 ceiling，也未换 packed/分块 renderer；
- r5 metric/global rows SHA=
  `04da7a2503460c075a3164c90d6c08436bbea9f4ec5560ea0417ee40e91aa939 / 04bf741e1da6cfe845b5ee6c9d4cccede54d79a1c8f7178e00abcf737ff7245e`；
  R0/R1 checkpoint SHA 前后保持 `1a061247...e7c / e995e7c2...8cd1`，run 内无 `.pth`；
- 资源无效 diagnostic：coverage `1.0→1.0`，depth violation `0.915792→0.908173`，non-target RGB MSE
  `0.002095031327→0.002095032019`，original-global RGB MSE `0.002104032262→0.002104032654`；exact Pareto=
  `tradeoff_non_dominated`。该数值只刻画失败，不登记为合格 heldout 证据；
- 状态分层：r5 run=`blocked`，R1 arm=`rejected_resource_gate_and_diagnostic_tradeoff`，A3 task=`done`。
  `A3*=R0-off`，即 D2 checkpoint immutable exact alias；formal、R2–R4 与独立 S-A 训练未授权。

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
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | running | A4 已闭环；当前审计 Instant NuRec 官方代码、权重、输入输出、license 与本地能力边界 |
| `WS-V3-A1-CALIBRATION-01` | done_off | 10/10 逻辑项、8/8 唯一训练；C*=C0；确认原始端点方向存在场景依赖 |
| `WS-V3-A2-ACTOR-DENSIFY-01` | done | D1/D2 fixed/matched 均为 tradeoff；A2*=D2 boundary-priority，D1 fallback；D3/D4 未启动 |
| `WS-V3-A3-LOCAL-REFINE-01` | done | R1 resource gate failed，diagnostic tradeoff；R1 rejected，A3*=R0/D2 exact alias；formal、R2–R4 未授权 |
| `WS-V3-A4-DEPLOYMENT-01` | done | P0/P5/P1/P2/P3 complete；P1 rejected；P2 mixed checkpoint + P3 exact chunk package selected |
| `WS-V3-R0-INTEGRATION-01` | pending | 汇总 A0–A4，不要求扩展到六场景 |

## 机器与工作树

- GPU：NVIDIA GeForce RTX 3090，24,576 MiB；driver `580.105.08`；最近审计 0 MiB；
- cgroup memory：90 GiB，`oom=0 / oom_kill=0`；
- 数据盘：P3 r1 终态 free=`42,359,705,600 bytes`；
- A3 heldout r5、A4-P0 v1 r1、A4-P5 r1 与 A4-P2 r1 均保留 blocked；P0/P5/P2 canonical r2 与 P3 canonical r1 exit=`0`，GPU 无遗留进程；
- 当前非 V3 文档 dirty files 属于 V2 M5，必须保留。

## 下一步

执行 `WS-V3-F0-FEEDFORWARD-AUDIT-01`：冻结 NVIDIA Instant NuRec 官方 repository revision、license、paper 与
checkpoint provenance；静态审计相机/cadence/pose/LiDAR/instance 输入、standalone CLI 实际导出与 actor registry
保留能力；先做无权重/无 gated 数据的本机 CLI preflight，只有依赖和公开输入满足时才执行一窗口 inference smoke。
明确区分论文完整模型与 standalone CLI，不把网页演示写成本地能力。F1、P4、D3/D4 与 A3 formal/R2–R4 保持未解锁。
