# Experiments

- 更新时间：2026-08-06
- 活跃路线：面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化 V3.1
- 权威方案：[`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md)
- V2 历史方案：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V1 最终台账：
  [`archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md`](archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md)

本文件保留 V2 完整执行证据，并从 2026-08-05 起登记 V3。V2 M0–M4 已完成；M5 部分执行后停止扩张，
保持 `pending` 历史终态；M6–M8 不再授权。A0 三场景原生基线已完成，当前执行
`WS-V3-A1-CALIBRATION-01`。

## 1. 状态词

只使用：

```text
pending | running | blocked | done | rejected
```

`done` 表示预注册门禁满足；`blocked` 表示工程、资源或外部依赖阻塞；`rejected` 表示研究门禁失败。

## 2. V3 注册表

| Task ID | 状态 | 目标 | 完成门禁 |
|---|---|---|---|
| `WS-V3-P0-ROUTE-01` | done | 单一 V3 权威计划与 V2 事实冻结 | `076ebdc`；文档一致，链接与 Git diff 校验通过 |
| `WS-V3-A0-NATIVE-BASELINE-01` | done | 三场景原生 StreetGS 基线 | `20260805T175000Z__a0-three-scene-finalize-s0-r2`；3/3 完整矩阵 |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | pending | Instant NuRec 官方本地能力审计 | revision/license/input/output/asset-editability 审计和 1-window smoke |
| `WS-V3-A1-CALIBRATION-01` | done_off | 成像、位姿和 LiDAR 初始化消融 | 10/10 逻辑项、8/8 唯一训练；C*=C0；finalizer done |
| `WS-V3-A2-ACTOR-DENSIFY-01` | pending | actor-aware densification/pruning | 完成 D0–D3 小步消融；质量/GS 数/训练代价 Pareto |
| `WS-V3-A3-LOCAL-REFINE-01` | pending | 编辑区域局部 Gaussian 精修 | outside frozen；Tier-A/深度顺序/时序指标齐全 |
| `WS-V3-A4-DEPLOYMENT-01` | pending | pruning/precision/chunk/LOD 与资产注册 | pruning + 数值压缩 + chunk；不变量和质量-大小-速度 Pareto |
| `WS-V3-R0-INTEGRATION-01` | pending | 完整 A0–A4 结论与复现包 | 所有正式 terminal 可审计；结论不超出三场景证据 |

### `WS-V3-A0-NATIVE-BASELINE-01` 完成证据

- fix commit：`436cfc1`；DriveStudio upstream：`e59bda4`；compatibility patch SHA-256：
  `54e7584b6d74431e58f626dfaadd69812d4058d54f82c7941e75aa11f5f94619`；
- 完整 A0 定向测试：`16 passed`；
- canonical smoke：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/20260805T161656Z__scene0255-catfix-s0-r2`；
- terminal=`done`；torch=`2.1.2+cu118`；原生 mixed-empty cat 复现
  `invalid configuration argument`，patched output=`[59,3] / 177 numel / exact point-color pairing`；
- 真实 1-step DriveStudio 路径完成数据集、LiDAR 实例初始化、一次优化和 checkpoint；controller
  `72.1 s`，peak GPU sample `8,388 MiB`，peak cgroup `5,971,820,544` bytes，无 OOM；
- smoke checkpoint `320,832,362` bytes 只证明执行路径，不注册为 A0 最终模型。

正式三场景矩阵：

| scene | checkpoint / step | global PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | high boundary PSNR / SSIM | bg / rigid GS | train s / peak MiB |
|---|---|---:|---:|---:|---:|---:|
| 0230 | `24a39f…e49` / 30k | 24.934 / .740 / .169 | 21.728 / .596 / .121 | 20.165 / .603 | 1,152,614 / 167,299 | 3014.5 / 23,799 |
| 0242 | `16179d…fda` / 30k | 29.107 / .906 / .113 | 19.788 / .665 / .153 | 23.277 / .795 | 843,756 / 86,255 | 2006.2 / 12,783 |
| 0255 | `f8c81c…ef9` / 30k | 25.230 / .743 / .192 | 23.531 / .665 / .058 | 22.991 / .656 | 1,510,936 / 40,447 | 2739.4 / 24,057 |

- scene-0255 正式训练：`20260805T162355Z__scene0255-native30k-s0-r1`；0230/0242 通过 config normalized
  SHA、checkpoint bytes/SHA/step 和同实现合同注册复用；
- actor evaluator 提交 `01cd303`；counterfactual mask 明示不是真值分割，并记录 coverage。actor runs：
  `20260805T173900Z__scene0230-actor-metrics-s0-r1`、`20260805T174100Z__scene0242-actor-metrics-s0-r1`、
  `20260805T174300Z__scene0255-actor-metrics-s0-r1`；peak GPU `8,455 / 7,905 / 8,685 MiB`；
- 0242 boundary role 按注册表保留 `ABSTAIN`；其余 boundary actor 区域/边界带均有正式指标；
- finalizer r1 `20260805T174700Z__a0-three-scene-finalize-s0-r1` 因 native/reuse training resource 字段差异
  `blocked`；`00ba4e8` 归一化后 r2 `20260805T175000Z__a0-three-scene-finalize-s0-r2`=`done`，产出
  `a0_matrix.json/csv` 与 `a0_report.md`；
- A0 只支持固定三场景的描述性结论。跨场景 GS 数与质量不可作因果归因；A1/A2 必须做同场景受控消融。

工作树准备脚本首次创建旧候选 worktree 后，因 `git status --short` 的输出已被 `.strip()` 去除前导空格，
verification literal 写成带前导空格而失败；修正为 `M datasets/driving_dataset.py` 后 verify-only 通过。canonical
patch 为 r2 worktree 和上述 SHA；旧候选只解释首次 smoke，不进入 formal training。

### `WS-V3-A1-CALIBRATION-01` 当前证据

开发场景 `scene-0230` 已完成 C0–C3 30k formal；初始化 provenance SHA 均为
`8951543c33f72f439068237f1a552fae660895f8906afbf4651f5f580981b898`。固定 step 结果：

| variant | global PSNR / SSIM / LPIPS | boundary actor PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | total GS | train min |
|---|---:|---:|---:|---:|---:|
| C0-off | 27.746 / .851 / .176 | 27.756 / .892 / .069 | 25.358 / .844 / .094 | 1,360,649 | 52.05 |
| C1-native | 24.979 / .743 / .169 | 22.549 / .700 / .103 | 21.696 / .602 / .120 | 1,316,421 | 53.69 |
| C2-factorized-isp | 25.011 / .743 / .168 | 22.583 / .705 / .104 | 21.779 / .608 / .117 | 1,322,979 | 52.26 |
| C3-bounded-pose | 28.109 / .862 / .167 | 28.169 / .897 / .066 | 25.137 / .846 / .094 | 1,363,040 | 56.14 |

A1-E0 实现提交为 `20c4276`，相机映射修复为 `d85ef27`。冻结配置
`configs/worldsim_v3/a1_endpoints_v1.yaml` 的 SHA-256 为
`60c211625860c25edf92842b88bdb040ea8c180b12fe0fa78f2fc1c342bc4051`。相机对只使用 DriveStudio 权威映射
`0=FRONT / 1=FRONT_LEFT / 2=FRONT_RIGHT` 下的相邻对，支持双向投影、静态/可见/遮挡/深度边缘 mask、
near/far、coverage 与 `ABSTAIN`。

有效正式回填：

| variant / run | E1 valid/candidate/coverage | E1 median/P90 ↓ | E2 high mean/P90/coverage ↓ | E2 boundary mean/P90/coverage ↓ |
|---|---:|---:|---:|---:|
| C0 `20260806T141409Z__scene0230-c0-a1-e0-formal-full-camera-map-fix-s0-r2` | 28,744/266,631/10.780% | .05951/.14719 | .004813/.010895/26.316% | .003547/.006353/35.294% |
| C1 `20260806T141623Z__scene0230-c1-a1-e0-formal-full-camera-map-fix-s0-r1` | 29,151/274,658/10.614% | .06289/.15623 | .004751/.010895/28.070% | .004450/.007626/35.294% |
| C2 `20260806T154541Z__scene0230-c2-a1-e0-formal-full-s0-r1` | 31,299/275,877/11.345% | .06544/.16160 | .004844/.011734/28.070% | .003346/.005447/35.294% |
| C3 `20260806T164852Z__scene0230-c3-a1-e0-formal-full-s0-r1` | 29,846/268,826/11.102% | .06309/.15448 | .004930/.011734/26.316% | .003592/.006537/35.294% |

C2 只改善 boundary role E2，high role E2 与 actor/boundary LPIPS 退化；C3 全图、boundary actor 与位姿稳定性
最好，但 E1 和两个 E2 role 均未严格优于 C0。四次有效评估 checkpoint SHA 前后相同。QA panel 只确认投影落在
真实相邻视野和 actor 支持边界，不能替代人工质量裁决。

首次 formal `20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 继承了错误相机 ID 标签，实际把
非相邻画面当成预注册相机对，已保留为 `rejected / INVALID_CAMERA_ID_LABEL_MAPPING`；修复后 run 是唯一有效证据。

最小 LiDAR provenance 实现提交为 `14bc3c2`，冻结配置 SHA-256
`f2fd1712cf4ddd75c1c4d1da4a426dcf7e1340a5fd943066401ba881f51c5639`。正式 run
`20260806T143644Z__scene0230-a1-lidar-provenance-formal-full-witness-s0-r1`=`done`：196 个 LiDAR/pose block、
6,804,832 raw points、24 actor/75,002 actor points；记录的 LiDAR 与 actor tensors 全部 exact match，RigidNodes
初始计数 75,002 exact。held-out sparse depth 172,844/172,844，绝对 median/P90=`7.679/35.958 m`，相对
median/P90=`.6649/.9077`，checkpoint SHA 未变。

背景随机 near/far 点的 CUDA visibility filter 不提供跨初始化 exact replay：源背景初始计数 946,484，三次
replay 分别为 946,597、946,309、946,291。首次 strict smoke
`20260806T142900Z__scene0230-a1-lidar-provenance-smoke1-s0-r1` 因 exact SHA 门禁 `blocked`；协议在查看
正式 depth 结果前冻结为“LiDAR/actor tensor exact 是 gate，随机背景 exact 仅 report”，成功 smoke 和 formal 的
初始 depth 都明确标为 reconstructed witness。没有使用事后计数容差。逐 Gaussian ancestry 留到 A2 instrumentation。

A1-D0 配置 SHA-256 为 `a445078d3bea89a78a0c9e6544a94a2be4c9c2e71f45aec4a9d8878b4c6593c1`；
`20260806T170219Z__scene0230-a1-diagnostics-c0-c3-formal-s0-r1`=`done`。输入速度层为
near-static/low/normal=`2/18/176` 帧；near-static 只有 2 帧，不承担统计结论。C1 位姿修正 translation
median/P90=`7.256/12.215 mm`、rotation=`0.1660/0.35465°`；C3 为 `1.703/2.338 mm`、
`0.02553/0.03337°`。这些是学习修正幅值，不是独立 pose accuracy。

选择协议提交 `60ef079`；配置 `configs/worldsim_v3/a1_dev_selection_v1.yaml` SHA-256 为
`a45699ebf696c875a18832f8db920a6106837a1e4f235dcd9036eff48dfbc609`。协议如实披露结果访问，且不引入
数值容差。正式 run `20260806T171417Z__scene0230-a1-dev-selection-formal-s0-r1`=`done`，输出
`C*=C0-off / done_off`。若 C* 为 C0/C1，确认矩阵保留三个逻辑项但用 source run/checkpoint exact alias。
开发场景冻结时进度为 `4/10` 逻辑项、`4/8` 唯一训练；后续确认矩阵结果如下。

确认配置提交为 `198a681`，SHA-256 为
`63a3cc607ccfddbb714cc81d0570da356263c01c5a68880345953023d2d6a8cd`。正式结果：

| scene / variant | training run | endpoint run | global PSNR / LPIPS | E1 median / P90 | E2 high mean / P90 |
|---|---|---|---:|---:|---:|
| 0242 C0 | `20260806T172514Z__scene0242-c0-confirm-formal30k-s0-r1` | `20260806T181834Z__scene0242-c0-a1-e0-confirm-formal-full-s0-r1` | 30.064 / .1108 | .03147 / .08826 | .008264 / .020697 |
| 0242 C1 | `20260806T181957Z__scene0242-c1-confirm-formal30k-s0-r1` | `20260806T191202Z__scene0242-c1-a1-e0-confirm-formal-full-s0-r1` | 29.161 / .1122 | .03333 / .08971 | .008660 / .021708 |
| 0255 C0 | `20260806T191340Z__scene0255-c0-confirm-formal30k-s0-r1` | `20260806T200907Z__scene0255-c0-a1-e0-confirm-formal-full-s0-r1` | 27.255 / .2086 | .04348 / .14248 | .004772 / .009805 |
| 0255 C1 | `20260806T201041Z__scene0255-c1-confirm-formal30k-s0-r1` | `20260806T210645Z__scene0255-c1-a1-e0-confirm-formal-full-s0-r1` | 25.240 / .1921 | .04277 / .13626 | .003715 / .007704 |

0242 的 boundary role 继续 `ABSTAIN`。0255 C1 虽降低 E1/E2 error，但 high E2 coverage 从 `23.529%` 降至
`21.569%`，boundary/high actor LPIPS 也全部退化，故不通过冻结合同。两个 C* alias run 为
`20260806T211000Z__scene0242-cstar-c0-exact-alias-s0-r1`、
`20260806T211100Z__scene0255-cstar-c0-exact-alias-s0-r1`，不含新训练/评测。finalizer
`20260806T211248Z__a1-three-scene-finalize-s0-r1`=`done`，正式终态 `done_off`。A1 完成 `10/10` 逻辑项、
`8/8` 唯一训练；原始端点方向必须报告为 scene-dependent。

## 3. V2 冻结注册表

| Task ID | 状态 | 目标 | 当前输入事实 | 解锁条件 |
|---|---|---|---|---|
| `DR-V2-M0-BOOTSTRAP-01` | done | 事实源、分支、镜像与 bootstrap | 正式 run 完成；历史失败实例保留 | README/STATUS/PLAN 一致，bootstrap smoke 通过 |
| `DR-V2-M1-DGGT-REPAIR-01` | done | 修复 pointops2 并做 18-window inference | 1/3-view、common、regional 全部完成 | 18/18 + 216/216 + 完整运行合同 |
| `DR-V2-M2-ACTOR-EVAL-01` | done | nuScenes 真值 actor 评测适配器 | raw 2Hz 轨迹、4,356 exact mappings、6/6 cohort | 三 scene eligible 16/20/6，visual QA 通过 |
| `DR-V2-M3-EDIT-BASELINE-01` | done | DriveStudio/StreetGS 可编辑基线 | 30k checkpoint、registry、27-image smoke 完成 | scene-0230 remove/lateral/3-camera smoke |
| `DR-V2-M4-EDIT-PILOT-01` | done | scene-0230 真实编辑闭环 | 1,764 RGB、9 MP4、1,176 paired rows、16/16 checks | 两种编辑真实执行且证据可审计 |
| `DR-V2-M5-STRESS-3SCENE-01` | pending | 三场景编辑/去遮挡压力测试 | 0230/0242 checkpoint 与 0255 诊断已生成；任务未闭环并冻结 | 历史门禁未满足；V3 不再授权继续 |
| `DR-V2-M6-HYPOTHESIS-01` | pending | 基于真实失败做 novelty gate | 未生成 | V3 路线不再授权 |
| `DR-V2-M7-METHOD-01` | pending | 最小方法与 matched ablation | 未生成 | V3 路线不再授权 |
| `DR-V2-M8-HUMAN-01` | pending | 人工盲审与终局 | 未生成 | V3 路线不再授权 |

## 4. V2 启动前维护记录

2026-08-02 的文档归档和存储清理属于 maintenance，不冒充 `DR-V2-M0-BOOTSTRAP-01`：

- V1 当前态、实验台账、环境与报告已移入命名归档；
- V2 计划已按实际 checkpoint、DriveStudio 缺口和用户镜像偏好校准；
- 可再生中间产物的精确路径、字节数与恢复方式见
  [`archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md`](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)；
- AD-GS 六场景最终 60k checkpoint/render/metrics、processed 输入、raw subset、DGGT 完整预下载候选均受保护。

## 5. V1 冻结输入

| 资产 | 终态 | V2 用法 |
|---|---|---|
| AD-GS 六场景 exact reproduction | done，6/6 | 只读 checkpoint/render/metrics；不重复训练 |
| DGGT V1 run | blocked，未 inference | 只作为原始失败证据；V2 新环境重做 |
| V1 pseudo identity audit | 0/12 slots | 失败边界；不得当作真实编辑结果 |
| V1 候选 A novelty | rejected | 禁止复活“补身份 + 基础轨迹编辑”作为贡献 |

## 6. `DR-V2-M0-BOOTSTRAP-01`

### 工程失败实例

- run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T114342Z__bootstrap-s0`；
- terminal：`blocked / empty_shell_python_not_on_path`；
- 网络四源已可达，但非登录空 shell 中裸 `python` 不在 PATH，导致 `source_resolution.json` 未生成；
- 修复仅显式选择 `/root/miniconda3/bin/python`，没有安装依赖或改写全局环境。

### 正式完成实例

- 验证实例：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T114453Z__bootstrap-s0-r2`
  为 `done`；其后只为让 source snapshot 覆盖相对 `HEAD` 的 staged/unstaged 全部 M0 文件创建 r3，未改变
  bootstrap、资产或测试协议；
- run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T115419Z__bootstrap-s0-r3`；
- terminal：`done`；
- branch：`research/dynamic-editing-v2`；source commit：`09fbb55` + 本 M0 工作树快照；seed：`0`；
- empty shell/tmux shell：`PASS/PASS`；TUNA Conda/PyPI、HF mirror、GitHub：`4/4 HTTP 200`；
- AD-GS `model_60000`、42 test renders、138 train renders 与 processed 输入：`6/6`；
- DGGT preload：`5,411,266,466` bytes，SHA-256
  `fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9`；
- DriveStudio commit `e59bda4fa681f829dbb1d65f0de582b0f633c450` 与 env 可用，pilot assets `missing`；
- 测试：
  `python -m pytest -q tests/test_dr_pseudo_tracks.py tests/test_v71_actor_registry.py` → `7 passed`；
- `shellcheck` 未安装，按计划未为此污染环境；`bash -n` 通过。

## 7. `DR-V2-M1-DGGT-REPAIR-01`

### 冻结实现

- repo `a3276d2b`；model revision `735ac9a6`；checkpoint bytes/SHA-256 通过并 hardlink 复用；
- `/root/autodl-tmp/envs/dggt-v2`：Python 3.10 / torch 2.4.1+cu121 / 固定 NVIDIA CUDA 12.1
  compiler+runtime+headers；
- pointops2 upstream `python setup.py install`；CUDA forward/backward=`PASS`；
- compatibility patch 仅 `args.difix -> args.diffusion`，untouched 错误单独保留。

### 正式运行

| 证据 | 终态 | 覆盖/结果 |
|---|---|---|
| `20260802T125138Z__native-nusc-s0-r6` | native done；后续 common import blocked | 1-view 18/18；3-view 18/18；原生输出不受 common 失败影响 |
| `20260802T133151Z__common-retry-s0-r8` | done | AD-GS common target 216/216；GT 像素身份 216/216 |
| `20260802T133912Z__regional-s0-r9` | done | AD-GS 216 + DGGT 1-view 72 + DGGT 3-view 216 = 504 rows |

M1 均值：

| 协议 | PSNR | SSIM | LPIPS(Alex) | inference s |
|---|---:|---:|---:|---:|
| DGGT 1-view | 20.707359 | 0.856031 | 0.135780 | 1.785527 |
| DGGT 3-view | 21.165262 | 0.771051 | 0.165553 | 4.517659 |
| AD-GS same-target 1-view | 34.581860 | 0.951918 | 0.062490 | n/a |
| AD-GS same-target 3-view | 34.894344 | 0.951711 | 0.061447 | n/a |

区域诊断的动态区/边界带 PSNR 分别为：AD-GS `29.640118/29.480968`、DGGT 1-view
`22.999911/22.017347`、DGGT 3-view `22.902139/21.810579`。边界带固定为二值动态区
7x7 dilation XOR erosion。这些数值仅用于 failure characterization，不得解释为同预算排行。

### 失败实例

`r1–r5`分别固定了 pip backtracking、CUDA 11.8/cu121 compiler mismatch、缺 cusparse headers、
transformers 5.x/DTensor 和 diffusers 0.39/torch schema 不兼容；r6 common 固定 `flow_vis`
缺失；r7 固定重试封装字段错误。全部为独立 `blocked` run，没有覆盖原运行。

## 8. `DR-V2-M2-ACTOR-EVAL-01`

### 正式运行

`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M2-ACTOR-EVAL-01/20260802T140312Z__actor-eval-s0-r5`
=`done`。

| scene | raw actors | eligible | high-support | boundary-support |
|---|---:|---:|---|---|
| scene-0230 | 58 | 16 | `af663976db5e...` | `18c7f0c5fa6b...` |
| scene-0242 | 53 | 20 | `40f087d8d9d7...` | `2c820a798ad9...` |
| scene-0255 | 56 | 6 | `f4aa30b8d0b4...` | `80c08b992f1d...` |

- 预注册 support score 和字典序 tie-break 未调节，冻结时尚无 M3/M4 编辑输出；
- 4,356/4,356 observations 使用 timestamp+exact `sample_token`；无效投影不从分母中静默删除；
- raw 2 Hz 与 interpolated visualization 字段物理分离，本运行插值列表为空；
- 11 个输入 metadata 哈希、167 actor metrics、3 份 cohort CSV、6 组投影 panel、6 张 raw
  轨迹图与视觉 QA 齐全。

### 失败实例

- r1：错误假设磁盘 `sample.json` 含 devkit 运行时 `anns` 反向索引；
- r2：`ijson` Decimal 进入 JSON 运行合同；
- r3：近平面后的 invalid projection 没有统一零面积 schema；
- r4：只选最近 timestamp sweep 导致 raw sample token 不精确，protocol QA 失败；
- r5：改为 exact token 内再做 timestamp 最近选择，不改 actor 门槛，通过。

## 9. `DR-V2-M3-EDIT-BASELINE-01`

### 正式训练与恢复

- 原生训练 run `20260802T152252Z__native-train-s0-r8` 完成 100/1,000-step profile 和 30k
  训练，checkpoint=`386,398,646 bytes / step 30000 / SHA-256 8ed40576...a73f9e`；
- 训练后上游 full render 把帧累计在内存中，`577/588` 时 cgroup memory 连续两次超过 90%，
  守卫返回 `-15`；峰值 GPU `23,873 MiB`，峰值 cgroup `89,836,462,080 bytes`，
  `oom=0 / oom_kill=0`；r8 保持 `blocked`；
- r12 对 checkpoint step/bytes/hash、r8 formal stage 和 blocked terminal 做窄范围复核，未复制或修改
  checkpoint；训练语义完成与上游 post-render 未完成分开记录。

### 正式编辑 smoke

完成 run：
`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M3-EDIT-BASELINE-01/20260802T163930Z__formal-checkpoint-recovery-s0-r12`。

- registry 24 个 model：23 non-empty，1 个被原生训练裁剪为空并显式不可用；
- 冻结 actor `af663976... → true id 13 → column 13 → model 5 → 2,683 Gaussians`；
- `3 frames × 3 cameras × 3 variants = 27` PNG；original/remove/lateral 均非空且时间同步；
- lateral/remove mean absolute RGB diff 分别为 `0.0003448362 / 0.0002196175`；两者均在 2 个
  frame-camera 上非零；这只是 effect smoke，不是质量指标；
- checkpoint SHA、非目标参数、reload 后完整 RigidNodes state 均精确不变；
- post peak GPU `8,241 MiB`，peak cgroup `58,291,757,056 bytes`；readiness 11/11 available。

### 独立失败实例

- r4/r6/r7：旧 `gsplat/nvdiffrast` CUDA binary 不含 RTX 3090 SM 8.6；固定源码重建后通过；
- r5：tmux 非登录环境没有裸 `python`；改为前缀解释器；
- r8：训练完成后的累积 full render 触发内存守卫；
- r9：恢复 probe provenance 字段错误嵌套；
- r10：registry helper 在 DriveStudio 环境误依赖未声明 `ijson`；改为读取 16 MB 标准 JSON；
- r11：一个非目标 model 的 Gaussian slice 被训练裁剪为空；registry v2 显式标记 unavailable，
  仍要求所选 actor slice 非空。

## 10. `DR-V2-M4-EDIT-PILOT-01`

### 正式运行

`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M4-EDIT-PILOT-01/20260802T171000Z__scene0230-pilot-s0-r7`
=`done`。

- 固定 scene-0230 high-support actor `af663976...`，196 帧、三相机、original/lateral +1m/delete
  共 `1,764` 张 RGB，所有配套 depth/opacity/dynamic/target mask/footprint 与 9 个 MP4 完整；
- paired metrics=`1,176` rows；16/16 协议、不变量和产物检查通过；
- lateral/delete non-target PSNR=`93.394483/95.598042`，LPIPS(Alex, 256px)=
  `5.260851e-09/3.052960e-09`，source effect energy=`0.055526/0.031926`；
- actor-local 位移最大误差 `3.814697e-06 m`，rotation/size/canonical drift 和 multi-camera
  world mismatch 均为 `0`；
- 自动检查不冒充质量门禁；人工抽检只确认非黑、非重复 original、footprint 和目标差分可见。

### 独立失败实例

- `smoke_frame1_s0_r1`：float32 transform 往返最大误差高于不现实的 `1e-6 m`，其余检查通过；
  r2 将协议容差固定为 `1e-4 m` 后 16/16 通过，正式实测误差为 `3.814697e-06 m`；
- `debug_controller_s0_r5`：外层诊断 `timeout` 中断 controller，但 child 使用
  `start_new_session=True`，故需按精确 PGID 回收；未覆盖；
- `20260802T170600Z__...-r6`：调试 tmux 生命周期中断后保留 running terminal 证据；
  正式 r7 改用 nohup controller，资源守卫和 terminal 均闭环。

## 11. `DR-V2-M5-STRESS-3SCENE-01` 部分执行后冻结

该任务保持 `pending`，因为没有满足 V2 预注册完成门禁。2026-08-05 路线切换后不再继续扩建其大型评测链，
但已生成资产和失败诊断作为 V3 A0/A3 的输入保留。

### 已有证据

- scene-0230 held-out：checkpoint `398,652,534` bytes，SHA-256
  `24a39f27dfeed36bbdb01ee14211aec51b414e6ab0e61915b71c1dddcdf61e49`；
  high/boundary actor 均可用，分别 `4,747/1,914` GS；
- scene-0242 held-out：checkpoint `306,034,934` bytes，SHA-256
  `16179d8f99becb86b6893a18ff036af72d78c9897f7aa2b0e297b735dd6c5fda`；high actor `6,939` GS，
  boundary actor 显式不可用；
- scene-0255：数据准备与 sky 产物已落盘；r8 的 90% cgroup memory stop 与其后的 cache recovery 分开保留；
- r25–r27 把训练失败定位到 DriveStudio 实例点聚合的 CUDA `torch.cat`；r27 输入 166 个 CUDA float32
  tensor，其中 152 个为空 `(0, 3)`，总计 177 scalars，terminal=`done` 表示诊断完成，不表示训练完成；
- 当前无 M5 控制器、tmux 或 GPU 进程。r16/r18 的 `running` terminal 是容器生命周期中断证据，不改写。

### 未完成

- scene-0255 原生完整 checkpoint 与 actor registry；
- 三场景 × 两 actor × 四编辑的 24 条有效序列；
- pseudo-hole、perception 与跨场景 final matrix；
- M5 单独实现提交和 V2 M6 novelty gate。

未提交的 M5 config/scripts/tests 属于保留工作树，V3 P0 文档提交不得 stage 它们。scene-0255 修复在 V3 A0
以新 task、新 run 和最小 compatibility patch 执行，不能倒写旧 M5 terminal。

## 12. 当前唯一动作

`WS-V3-A1-CALIBRATION-01` 已 `done_off`。下一动作是 `WS-V3-A2-ACTOR-DENSIFY-01` instrumentation：先补齐
逐 Gaussian ancestry 与 split/clone lineage，再做只改变 actor/background quota 的 D1 smoke；不得一次混入
boundary、LiDAR、visibility 或 residual。
