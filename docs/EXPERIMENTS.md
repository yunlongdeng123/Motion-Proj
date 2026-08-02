# Experiments

- 更新时间：2026-08-02
- 活跃路线：动态驾驶场景可编辑重建与失败诊断 V2
- 权威方案：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V1 最终台账：
  [`archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS_V1_FINAL.md`](archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS_V1_FINAL.md)

本文件只登记 V2。当前没有 V2 run，所有研究里程碑均未执行。

## 1. 状态词

只使用：

```text
pending | running | blocked | done | rejected
```

`done` 表示预注册门禁满足；`blocked` 表示工程、资源或外部依赖阻塞；`rejected` 表示研究门禁失败。

## 2. V2 注册表

| Task ID | 状态 | 目标 | 当前输入事实 | 解锁条件 |
|---|---|---|---|---|
| `DR-V2-M0-BOOTSTRAP-01` | pending | 事实源、分支、镜像与 bootstrap | 文档预整理完成；尚无 V2 run | README/STATUS/PLAN 一致，bootstrap smoke 通过 |
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

## 5. 当前唯一动作

执行 `DR-V2-M0-BOOTSTRAP-01`。M0 提交前不进入 M1，不创建正式 DGGT/DriveStudio run。
