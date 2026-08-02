# Experiments

- 更新时间：2026-08-02
- 活跃路线：动态驾驶场景可编辑重建与失败诊断 V2
- 权威方案：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V1 最终台账：
  [`archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md`](archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md)

本文件只登记 V2。M0 已完成；当前只允许进入 M1。

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
| `DR-V2-M1-DGGT-REPAIR-01` | pending | 修复 pointops2 并做 18-window inference | repo/full preload 存在；旧失败 env 已清理 | M0 done；1-view 18/18 或可信 upstream blocked |
| `DR-V2-M2-ACTOR-EVAL-01` | pending | nuScenes 真值 actor 评测适配器 | raw subset 与 metadata 驻留 | 三个 pilot scene 各至少 1 个合格车辆 |
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

## 6. 当前唯一动作

执行 `DR-V2-M1-DGGT-REPAIR-01`。M1 提交前不进入 M2，不创建 DriveStudio 正式训练 run。
