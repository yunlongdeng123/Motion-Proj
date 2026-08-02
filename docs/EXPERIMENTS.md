# Experiments

- 更新时间：2026-08-02
- 活跃路线：动态驾驶场景可编辑重建与失败诊断 V2
- 权威方案：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V1 最终台账：
  [`archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md`](archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md)

本文件只登记 V2。M0–M2 已完成；当前只允许进入 M3。

## 1. 状态词

只使用：

```text
pending | running | blocked | done | rejected
```

`done` 表示预注册门禁满足；`blocked` 表示工程、资源或外部依赖阻塞；`rejected` 表示研究门禁失败。

## 2. V2 注册表

| Task ID | 状态 | 目标 | 当前输入事实 | 解锁条件 |
|---|---|---|---|---|
| `DR-V2-M0-BOOTSTRAP-01` | done | 事实源、分支、镜像与 bootstrap | 正式 run 完成；历史失败实例保留 | README/STATUS/PLAN 一致，bootstrap smoke 通过 |
| `DR-V2-M1-DGGT-REPAIR-01` | done | 修复 pointops2 并做 18-window inference | 1/3-view、common、regional 全部完成 | 18/18 + 216/216 + 完整运行合同 |
| `DR-V2-M2-ACTOR-EVAL-01` | done | nuScenes 真值 actor 评测适配器 | raw 2Hz 轨迹、4,356 exact mappings、6/6 cohort | 三 scene eligible 16/20/6，visual QA 通过 |
| `DR-V2-M3-EDIT-BASELINE-01` | pending | DriveStudio/StreetGS 可编辑基线 | source/env 存在；V2 scene 资产缺失 | scene-0230 remove/lateral/3-camera smoke |
| `DR-V2-M4-EDIT-PILOT-01` | pending | scene-0230 真实编辑闭环 | 未生成 | 两种编辑真实执行且证据可审计 |
| `DR-V2-M5-STRESS-3SCENE-01` | pending | 三场景编辑/去遮挡压力测试 | 未生成 | 3 scene、4 edits 有效 coverage |
| `DR-V2-M6-HYPOTHESIS-01` | pending | 基于真实失败做 novelty gate | 未生成 | 跨 3 scene 稳定失败且有独立 novelty delta |
| `DR-V2-M7-METHOD-01` | pending | 最小方法与 matched ablation | 未授权 | M6 done 且 endpoint/effect size 预注册 |
| `DR-V2-M8-HUMAN-01` | pending | 人工盲审与终局 | 未授权 | M7 有完整可审结果 |

## 3. V2 启动前维护记录

2026-08-02 的文档归档和存储清理属于 maintenance，不冒充 `DR-V2-M0-BOOTSTRAP-01`：

- V1 当前态、实验台账、环境与报告已移入命名归档；
- V2 计划已按实际 checkpoint、DriveStudio 缺口和用户镜像偏好校准；
- 可再生中间产物的精确路径、字节数与恢复方式见
  [`archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md`](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)；
- AD-GS 六场景最终 60k checkpoint/render/metrics、processed 输入、raw subset、DGGT 完整预下载候选均受保护。

## 4. V1 冻结输入

| 资产 | 终态 | V2 用法 |
|---|---|---|
| AD-GS 六场景 exact reproduction | done，6/6 | 只读 checkpoint/render/metrics；不重复训练 |
| DGGT V1 run | blocked，未 inference | 只作为原始失败证据；V2 新环境重做 |
| V1 pseudo identity audit | 0/12 slots | 失败边界；不得当作真实编辑结果 |
| V1 候选 A novelty | rejected | 禁止复活“补身份 + 基础轨迹编辑”作为贡献 |

## 5. `DR-V2-M0-BOOTSTRAP-01`

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

## 6. `DR-V2-M1-DGGT-REPAIR-01`

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

## 7. `DR-V2-M2-ACTOR-EVAL-01`

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

## 8. 当前唯一动作

执行 `DR-V2-M3-EDIT-BASELINE-01`。M3 提交前不进入 M4。
