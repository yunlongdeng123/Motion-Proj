# Motion-Proj WorldSim V4 终局产物保留策略

- 更新时间：2026-08-14
- 路线终态：`V4 closed / M1 rejected / M2 done with geometry caveat / M3 confirmed`
- 终局归档提交：`c7e4c969a95536d26d0a17a1c0d1d548f9a247dc`
- 当前研究授权：[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
- 终局证据入口：[`archive/2026-08/worldsim-v4-final/`](archive/2026-08/worldsim-v4-final/README.md)

## 0. V5 M2 active addendum（2026-08-14）

- `WS-V5-M2-GEOMETRY-FIRST-REPAIR-01` 的 r001–r009 全部保留在 `/root/autodl-tmp/runs/worldsim_v5/WS-V5-M2-GEOMETRY-FIRST-REPAIR-01/`；r001/r007 的 blocked terminal 不覆盖，r002/r003 的 union-mask artifact 保留为 request-unit 失败证据。
- r004 的逐 actor mask manifest 与 mask payload、r005–r009 的 request NPZ/diagnostics、summary/status/fingerprint/manifest/resolved/events/source snapshot 都是后续 Gaussianization forensic 与技术报告附录的只读输入，当前禁止清理。
- 轻量索引=`docs/archive/2026-08/worldsim-v5-m2/`；大型 checkpoint、request NPZ 与 render 仍驻留正式 run，不复制进 Git。
- 当前不存在 safe geometry candidate；在 Gaussianization forensic 和新的 development gate 完成前，不得把 G0/G1/G2/G3 任一 arm 标记为 selected，也不得清理 r005 的 staged candidate assets。

## 1. 必须驻留的轻量证据

| 类别 | 路径 | 用途 |
|---|---|---|
| V4 final archive | `docs/archive/2026-08/worldsim-v4-final/` | 计划/账本快照、附录索引、canonical 轻量证据与 SHA-256 清单 |
| M1 validation/rejection | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/{...r200,...r201}` | scene-disjoint rejection、calibration、逐 Gaussian/scene 诊断入口 |
| M2 development/validation | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M2-REPAIR-ROUTER-01/{...r212,...r222}` | frozen router、coverage、selective-risk 与 `+3.3908 m` geometry caveat |
| M3 validation/test/closeout | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/{...r238,...r335,...r336}` | frozen parameters、18-scene test、联合终局审计 |
| exact-once ledger | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T222011Z__m3-test-exact-once-ledger-s0` | 18/18 attempt/completion 与消费标记 |
| test freeze | `/root/autodl-tmp/motion_proj/V4_TEST_FREEZE.json` | source/freeze commit、split/config/asset SHA 和 test 协议 |
| 冻结源码与协议 | `configs/worldsim_v4/`、V4 相关 `scripts/`、`tests/`、`docs/WORLDSIM_V4_EVIDELTA_GS_PLAN.md` | 公式—配置—代码—测试—指标对应关系 |

上述 canonical run 不得原地续写、重命名、删除 terminal 或覆盖失败/abstain。任何复现必须使用新 task、新 run ID 和新输出目录。

## 2. 必须保留的外部资产引用

- V4 M1/M2/M3 canonical manifest 直接引用的 StreetGS/V3.3/AD-GS checkpoint、actor registry、sky mask、processed scene 和 validation/test staging；
- V3.3 immutable base 与 V3.3 R0 release，保留规则沿用其 final manifest 和 canonical run 证据；编辑过程恢复副本已按
  2026-08-17 文档清理策略移除；
- nuScenes 官方 metadata 与 V4 30-scene cohort 的 scene/token/split 指纹；
- `/root/autodl-tmp` 下 2026-08-13 到达的 KITTI tracking 压缩包在 V5 数据预检完成前不得删除或改名。

大型 checkpoint/processed scene 不复制进 Git。其驻留性由 canonical manifest、内容 SHA、路径和恢复来源共同确定；文档归档不是大型资产副本。

## 3. 可清理但本次未删除的内容

- V4 diagnostic/noncanonical 中间 checkpoint、重复渲染、panel、临时 cache 和已经被 canonical run 取代的候选；
- 失败 SSH launcher 的外部日志、重复 source checkout 和可从固定 commit 重建的包缓存；
- 不再被任何 canonical manifest、测试或 V5 forensic 引用的中间数组。

本次任务只做文档与轻量证据归档，没有删除任何 run、checkpoint、数据集、环境或第三方依赖。未来清理必须先生成独立 `CLEANUP_MANIFEST`，列出绝对目标、bytes、SHA/恢复来源和 canonical 引用检查。

## 4. KITTI 压缩包边界

当前发现的 V5 待审计输入包括：

- `/root/autodl-tmp/data_tracking_velodyne.zip`
- `/root/autodl-tmp/data_tracking_image_2.zip`
- `/root/autodl-tmp/data_tracking_image_3.zip`
- `/root/autodl-tmp/data_tracking_label_2.zip`
- `/root/autodl-tmp/data_tracking_oxts.zip`
- `/root/autodl-tmp/data_tracking_calib.zip`
- `/root/autodl-tmp/devkit_tracking.zip`

在完成 central-directory、成员计数、序列覆盖、可用空间和 SHA 审计前禁止直接解压。KITTI 只在 nuScenes V5 方法冻结后用于 frozen cross-domain；adapter smoke 不授权参数调优。

## 5. 恢复规则

1. 先读 `RESEARCH_STATUS.md`、`RESEARCH_FAILURES.md`、`EXPERIMENTS.md`；
2. 用 final archive 的 `SHA256SUMS` 校验轻量证据；
3. 按 canonical manifest 核对大型外部引用，缺失应写成 `blocked_asset_missing`，不得冒充方法失败；
4. V4 test attempt 已全部消费，禁止重跑或重新聚合 test source content；
5. V5 forensic 可以只读 V4 scene/run，但新 confirmatory split 必须与 V4 30 scenes scene-disjoint；
6. 新环境、数据 staging 和训练必须进入 V5 namespace，不得覆盖 V4 路径。

## 6. 存储门槛

- 新训练或大规模解压前可用空间至少 80 GiB，并预留 20 GiB 安全余量；
- canonical manifest/summary/status/fingerprint/source snapshot 优先于可重建 cache；
- cache 统一写 `/root/autodl-tmp/cache/`，大权重优先同文件系统复用；
- `docs/` 下不保留任何编辑过程备份或 `codex-backups/` 目录；已跟踪文档依靠 Git 历史恢复，短期恢复副本必须放在
  仓库外并在任务完成后删除。

## 7. 2026-08-17 清理登记

- V4 终局轻量包在清理前后均通过 `78/78` SHA-256 校验；canonical run、冻结配置、源码和测试均未删除。
- 两个 scratch `tmp` 清空但保留目录，旧 `/root/autodl-tmp/mnt` staging 产物删除；`docs/` 内编辑过程备份全部移除。
- 绝对路径、清理前大小、引用检查、不可恢复边界和保留项见
  [`archive/2026-08/worldsim-v4-cleanup-2026-08-17/CLEANUP_MANIFEST.md`](archive/2026-08/worldsim-v4-cleanup-2026-08-17/CLEANUP_MANIFEST.md)。
