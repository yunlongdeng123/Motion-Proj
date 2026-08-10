# Motion-Proj V3.3 启动前产物保留策略

- 更新时间：2026-08-11
- 当前研究授权：[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
- V3.3 计划：尚未执行；本文件只规定存储驻留边界
- 本次清理账本：
  [`archive/2026-08/pre-v3.3-cleanup/CLEANUP_MANIFEST.md`](archive/2026-08/pre-v3.3-cleanup/CLEANUP_MANIFEST.md)

## 1. V3.3 必须驻留

| 类别 | 路径 | 用途 |
|---|---|---|
| 三场景 processed 数据 | `/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval/{179,191,204}` | scene-0230/0242/0255 的图像、LiDAR、实例和相机事实源 |
| V3.1 D2 final | `.../20260809T113230Z__a2-d2-formal30k-s0-r1/.../checkpoint_final.pth` | V3.3 immutable FP32 base |
| V3.2 S1 canonical | `.../20260810T101739Z__s1-semantic-lift-s0-r6` | SAM2.1 fallback masks 与 semantic sidecars |
| V3.2 S2 canonical | `.../20260810T121829Z__s2-3dgic-adapted-s0-r3` | generated-background baseline 与 provenance |
| V3.2 S3 canonical | `.../20260810T112505Z__s3-asset-harvest-s0-r3` | manual 1/2-view actor baseline 与 selected actor asset |
| V3.2 S4 canonical | `.../20260810T131909Z__s4-harmonizer-nontemporal-s0-r3` | semantic reintroduction 负对照和 optional diagnostic |
| V3.2 R0 canonical | `.../20260810T134658Z__r0-final-integration-s0-r1` | mixed checkpoint、registry、semantic extension 与 exact chunk package |
| 运行环境 | `/root/autodl-tmp/envs/{motionproj,drivestudio,worldsim-v32-asset-harvester}` | V3.3 主代码、StreetGS、Asset Harvester/Harmonizer |
| 第三方与模型 | DriveStudio、Asset Harvester、Inpaint360GS、Harmonizer、必要 gsplat/nvdiffrast/PyTorch3D、Asset Harvester HF cache | V3.3 已知可复用依赖 |
| 文档与清理证据 | `docs/`、`/root/autodl-tmp/cleanup_manifests/20260811-pre-v33-space-reclaim/` | 研究连续性、删除清单和恢复边界 |

上述 canonical 路径、关键模型和三场景 processed 数据不得原地改写。V3.3 必须使用新的 run namespace；任何输入
变更都要通过新 task、冻结协议和 before/after SHA 审计。

## 2. 已转为 non-resident 的内容

- V3.2 及更早路线的非 canonical checkpoint、NPZ/NPY/PLY、图片、视频、panel 和生成候选；
- V3.1 D2 canonical 的 5k/10k/15k/20k/25k 中间 checkpoint；final 保留；
- DGGT、AD-GS、旧 SAM2 的 Conda 环境和 checkpoint；
- 完整 nuScenes 镜像、AD-GS processed、OccGS/ReSim 数据以及三个 scene-specific raw staging 副本；
- V1/V2/V3.1/V3.2 不再用于 V3.3 的第三方 checkout；
- pip/Conda/torch/Hugging Face 通用缓存；Asset Harvester 离线 HF cache 例外；
- Harmonizer 5,042,242,222-byte full `diffusion_harmonizer.pkl`；V3.3 使用的 non-temporal JIT 继续驻留。

旧 run 的 manifest、resolved config、JSON/Markdown summary、terminal、日志和 source snapshot 继续保留为轻量证据。
大型载荷 non-resident 不改变历史 `done/blocked/rejected` 结论，也不授权原 run ID 重跑或补写。

## 3. 恢复规则

1. canonical 资产从驻留路径直接读取，并先核对清理账本中的 SHA-256；
2. 被删除的包缓存、环境和公开 checkout 只在新任务明确需要时按固定 commit/revision 重建；
3. 被删除的旧 run 大载荷不从旧 terminal 恢复执行；若未来研究确需复现，使用新 task ID、协议和 run；
4. 被删除的 raw/历史数据只在 V3.3 之外出现新的数据需求时重新 staging，不得把 non-resident 写成数据损坏；
5. 详细删除目标以外部 `delete_dirs.tsv` / `delete_files.tsv` 为准，其 SHA-256 见清理账本。

## 4. 后续存储门槛

- 新环境或训练启动前可用空间至少 60 GiB；运行中始终保留 20 GiB；
- canonical checkpoint、registry、package、provenance 和正式 terminal 优先保留；
- 中间 checkpoint 只保留预注册要求的最小集合，阶段完成后优先清理；
- cache 统一写 `/root/autodl-tmp/cache/`，不得恢复散落的 `pip_cache` 或多份 Conda package cache；
- 大权重优先复用同文件系统驻留文件，禁止无理由复制；
- 文档备份按 `AGENTS.md` 集中进入对应 `codex-backups/<日期-任务>/`。
