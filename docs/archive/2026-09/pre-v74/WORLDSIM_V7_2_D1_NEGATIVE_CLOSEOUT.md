> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# WorldSim V7.2 D1 负向收口报告

> **2026-09-07 更正：此为旧 A1/D1 协议的历史报告，不是项目当前终态。** A1 对 G1 的负结果保留；B 缺少同协议候选比较不等于被证伪（`V71-F64`）。用户要求恢复 EAS-VGGT，见 [新计划](WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)；后续不再执行旧 A/B 队列（`V71-F65`），也不据本报告触发关机。

> 上一轮解释纠正（历史）：本报告保留 A1 负结果与当时交付记录。B 未完成方法候选比较，不能解释为科学失败；“V7.2 整体完成”由 `V71-F64` 撤回。当时恢复计划为 [research-first plan](WORLDSIM_V7_2_RESEARCH_FIRST_RECOVERY_PLAN.md)，现已被文首 EAS-VGGT 计划替代；旧数值和 canonical 不变。

## 结论

V7.2 解决了问题定义、数据隔离、对标基线和公平实验设计，但当前方法没有得到干净开发集支持。预注册 D1 判定为 `close_method_claim`：路线 A 的主候选没有越过 Actor-local TSDF；路线 B 只完成 LiDAR4D 公共协议能力运行，没有同协议方法增益。因此本版本交付可证伪诊断与基线报告，不进入 route-select、source-test、external-test、P3 或 P4，也不把负结果包装成投稿就绪的方法论文。

## 数据与 I/O

| 项目 | 结果 | 边界 |
|---|---:|---|
| 冻结 clean split | dev 4 logs；route-select 3；source candidates 3 | 仅 metadata 选取；按依赖链隔离 |
| 选择性 LiDAR 提取 | 3,882 keyframes；2,695,516,800 bytes | 只提取 dev+route；source candidates 未提取 |
| dev 物化 | 65 scenes；501 Actors；246 hazardous | ActorBundleV2 与 targets 分离 |
| dev held-out rays | 641,930 | surface forward 不读 target |
| build evidence | 2,059,365 points | 保留 origin、frame、ring、intensity 与 pose |

I/O run=`run://worldsim_v72/WS-V72-P2-CLEAN-LIDAR-IO-01/20260906T222500Z__clean-dev-route-lidar-io-r1`；dev data run=`run://worldsim_v72/WS-V72-P2-CLEAN-ACTOR-DATA-01/20260906T225000Z__clean-dev-actor-v2-s0-r2`。

## 方法与冻结规则

A1 从已适配的 AdaPoinTr checkpoint 开始，只增加由同一预测表面产生的 held-out-ray hit Huber 与 known-free early penalty。训练 100 epochs，seed=`7210`，没有 loss sweep。主候选在评测时固定为 75% measured TSDF + 25% A1 completion；纯 A1 只作诊断。D1 在读取 dev quality 前固定：512-point cap，相对最强基线 CD 至少降低 5%或 F-score 至少增加 2pp，同时 early 增量不超过 1pp、hit 降幅不超过 1pp。

训练 run=`run://worldsim_v72/WS-V72-P2-A1-OBSERVATION-CONSTRAINED-TRAIN-01/20260906T223000Z__a1-observation-train-s7210-r1`。冻结 checkpoint SHA-256=`d489e2bb5d056dc45907c46321c1a3d35fa6c1add8749945c00d4324a826e2d7`；final surface/hit/free loss=`.055098/.002022/.009084`。

## 干净 dev 对比

以下均为同一 501 Actors、同一 query rays、同一 surface-to-return 算子。`512` 是输出上限；原生点不足时不复制点，因此同时报告实际平均点数。

| 方法 | 实际点数 | CD-L1↓ (m) | F-score↑ | early↓ | hit↑ |
|---|---:|---:|---:|---:|---:|
| G0 raw fusion | 266.26 | .18566 | .74391 | .40910 | .52369 |
| **G1 Actor TSDF** | 401.49 | **.17860** | **.75032** | **.41146** | **.54972** |
| G3 AdaPoinTr | 512.00 | .22921 | .63442 | .43323 | .49104 |
| A1 observation constrained | 512.00 | .22980 | .63257 | .42897 | .49289 |
| A1 anchored surface | 449.20 | .19498 | .71454 | .43934 | .52215 |

纯 A1 相对 G3 将 early 降低 `.43pp`、hit 提高 `.18pp`，但 CD 增加 `.00060m`、F-score 降低 `.185pp`。主候选相对 G1 的 CD reduction=`-9.17%`、F-score gain=`-3.58pp`、early delta=`+2.79pp`、hit delta=`-2.76pp`。四项冻结判据全部不支持晋级。

Dev run=`run://worldsim_v72/WS-V72-P2-A1-OBSERVATION-CONSTRAINED-DEV-01/20260906T230000Z__a1-observation-dev-s7210-r1`；机械判定=`D1_DEV_GATE.json`；wall=`1498.27s`，RTX 3090 peak=`.828GiB`。

## 两条路线的最终状态

| 维度 | 路线 A：对象表面补全 | 路线 B：完整 Neural LiDAR |
|---|---|---|
| 任务输出 | target-free Actor surface | full range/return/intensity/point cloud |
| 强简单基线 | raw fusion、Actor TSDF | 官方 LiDAR4D pipeline |
| 强学习基线 | AdaPoinTr official/adapted/scratch | LiDAR4D 30k + ray-drop refinement |
| 当前增量 | A1 与 anchored A1 均未超过 TSDF | 只有 capability，没有 matched method candidate |
| 完整应用 | 未解锁 | KITTI-360 原生扫描导出仅证明能力 |
| D1 | fail | fail by missing required matched gain |
| 后续 | 不读 route/source final，不做 P3/P4 | 不扩 B1，不把跨任务绝对值当增益 |

LiDAR4D canonical=`run://worldsim_v72/WS-V72-B0-LIDAR4D-CAPABILITY-01/20260906T192557Z__lidar4d-kitti360-f4950-s0-r2`，ray-drop F1=`.95534`、depth RMSE=`2.94718m`、point CD/F=`.11733/.92081`，但数据、任务与分母不同，不能与对象补全表直接横比。

## 对 V7.1 原问题的回答

| 原疑问 | V7.2 处理 | 结果 |
|---|---|---|
| 问题定义不清 | 明确 build-only 输入、独立 target、输出 surface/full return 与 world-depth 合成 | 已解决为可执行合同 |
| 对标不足 | 引入 raw/TSDF/M8/AdaPoinTr 与完整 LiDAR4D | 简单 TSDF 比学习候选更强 |
| 实验缺乏说服力 | 日志级隔离、密度上限、同算子、预注册 effect+side-effect gate | 结果可证伪且没有事后改门 |
| 工作太窄 | 同时审计对象补全与完整 Neural LiDAR 两条路线 | 广度足以选路，但没有支持新的方法贡献 |

## 停止线与交付状态

`V71-F63` 记录科学拒绝。按冻结 both-fail rule，本版本不使用剩余三个 source candidates，不开启 route-select quality，不做 anchor/loss/threshold sweep，也不添加第三路线。V7.2 的正确终态是 `completed_negative_not_paper_ready`：任务、数据与基线基础设施可复用，当前方法主张关闭。若未来继续，应建立 V7.3 的新假设和新的独立证据，而不是在本 dev 集上救结果。

文稿已按负向技术报告收口并独立构建：`paper_v72/main.pdf=3 pages / 110,902 bytes`、`supplement.pdf=1 page / 49,107 bytes`、`arxiv.pdf=3 pages / 110,775 bytes`。构建日志无 overfull、undefined reference 或 undefined citation；main 与 supplement 全页渲染检查无截断和重叠。
