# Motion-Proj V2 产物保留策略

- 更新时间：2026-08-02
- 当前路线：动态驾驶场景可编辑重建与失败诊断 V2
- 当前授权：[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
- 本次清理账本：
  [`archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md`](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)

## 1. V2 必须驻留

| 类别 | 路径 | 原因 |
|---|---|---|
| nuScenes 六场景 raw | `/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1/` | M1/M2 映射与真值 |
| 数据 manifest | `/root/autodl-tmp/data/dynamic_recon/manifests/` | provenance/hash |
| AD-GS processed | `/root/autodl-tmp/data/dynamic_recon/processed/adgs_nuscenes_v1/` | 当前 M1 adapter 直接读取 image/sky/semantic |
| AD-GS final | 六个正式 run 的 `model_60000/` | V2 冻结 checkpoint/render/metrics |
| V1 轻量证据 | `/root/autodl-tmp/runs/dynamic_recon/` 中 manifest/config/log/metrics/summary/terminal/source snapshot | 历史结论与失败边界 |
| DGGT full preload | `/root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt` | M1 固定 checkpoint 候选 |
| 感知 evaluator | Grounded-SAM、Grounding DINO HF cache、CoTracker3 | M5 固定 evaluator |
| V2 baseline | DriveStudio source/env 与历史最小实例资产 | M3 source/readiness audit |
| 文档 | 当前事实源、`docs/archive/`、人工 verdict/package | 研究连续性 |

原始数据、人工 verdict、唯一 provenance、正式终态文件和上述 V2 输入不得删除或原地改写。

## 2. 本次允许移除

- V1 失败或中断的 DGGT 环境和未完成 `.partial` checkpoint；
- 已关闭 ReSim V6 环境；
- AD-GS 已完成预处理后不再被 V2 使用的 DPT 环境/权重；
- 旧 pip/Conda 下载与包缓存；
- AD-GS 100/1,000-step profiling checkpoint/render；
- 成功 run 的临时 `work/` 和失败 run 的大型可再生 work payload；
- `processed_failed/` 副本。

删除这些载荷不删除对应 run 的 manifest、resolved config、stage JSON、日志、metrics、summary、terminal、
source snapshot，也不改变 V1 结论。

## 3. 后续清理规则

任何后续删除必须：

1. 以字面绝对路径列出目标并确认解析在 `/root/autodl-tmp/` 的预期子目录；
2. 删除前确认没有进程读取目标；
3. 在 Git 内记录大小、用途、恢复方法与受保护项；
4. 删除后复核最终 checkpoint/render/metrics、DGGT preload、raw/processed 数据和 Git 状态；
5. 新 run 使用新 instance ID，不因本机 non-resident 改写旧 terminal。

## 4. 存储门槛

- 新环境或训练启动前可用空间至少 60 GiB；
- run 过程中始终保留 20 GiB；
- 大权重优先复用同文件系统 hardlink，禁止无理由复制；
- 缓存统一写 `/root/autodl-tmp/cache/`，不要恢复旧 `/root/autodl-tmp/pip_cache` 或散落包缓存；
- 先清可重下载 cache/partial/中间 checkpoint，不从 raw、final checkpoint、正式证据或人工材料开始。
