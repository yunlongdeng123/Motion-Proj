# Motion-Proj WorldSim V4 终局产物保留策略

- 更新时间：2026-08-14
- 路线终态：`V4 closed / M1 rejected / M2 done with geometry caveat / M3 confirmed`
- 终局归档提交：`c7e4c969a95536d26d0a17a1c0d1d548f9a247dc`
- 当前研究授权：[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
- 终局证据入口：[`archive/2026-08/worldsim-v4-final/`](archive/2026-08/worldsim-v4-final/README.md)

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
- V3.3 immutable base 与 V3.3 R0 release，保留规则沿用其 final manifest 和 `docs/archive/2026-08/worldsim-v3.3/` 过程证据；
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
- 文档备份集中进入对应路线的 `codex-backups/<日期-任务>/`，不散落 `*.bak` 或 `*.codexbak.*`。
