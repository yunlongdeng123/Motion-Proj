# Motion-Proj 产物保留策略

> **当前职责**：定义可删除与必须保留的边界，不记录某次实例的实时硬件状态。
> **历史清理账本**：[`archive/2026-07/v7-feasibility/ARTIFACT_RETENTION_20260719.md`](archive/2026-07/v7-feasibility/ARTIFACT_RETENTION_20260719.md)。
> **当前执行授权**：只看 [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)。

## 1. 必须保留

- 正式或 retrospective evidence 的 config、metrics、summary、result、fingerprint、终态标记与索引；
- 用户或指定评审者填写的人工 verdict、完整 review package 与对应提示词；
- V7 当前使用的三场景数据规格、occupancy、edit JSON、C0 legality screen 与轻量报告；
- 仍用于下一门禁的 checkpoint、baseline 输出和可重建所需的固定第三方 commit/license；
- `docs/archive/` 中的历史计划、终报、旧事实源与 RF 重开条件；
- raw nuScenes 及其许可约束范围内的必要本地数据。

必须保留不等于所有大型二进制永久保留。关闭路线的 checkpoint、candidate 视频、feature tensor、下载 cache 和
可重新下载权重可以按下述流程清理，但不得因此改写 run 结论。

## 2. 清理前置条件

任何删除必须同时满足：

1. 删除目标以字面绝对路径列出，并解析在明确的 `/root/autodl-tmp/...` 子目录内；
2. 先建立 Git 内清单或可恢复记录，说明大小、用途、上游 revision 和重建方式；
3. 核对没有正在读取目标的训练、推理、下载或评测进程；
4. 保护材料在清理前后以 hash、文件数和 JSON/JSONL/YAML 可解析性校验；
5. 不删除 raw data、人工 verdict、唯一 provenance、未关闭路线的必要 checkpoint 或当前 gate 输入；
6. 删除后使用新的 run ID 重算，禁止覆盖旧 run 目录或伪造原终态。

若目标、用途、恢复方式或保护范围不明确，停止并请求用户确认。

## 3. 历史 V7/V7.1 保护范围

| 类别 | 路径 | 处理规则 |
|---|---|---|
| B0 reconstruction | `/root/autodl-tmp/runs/occgs_resim/b0_recon/occgs_b0/` | 保留 `config.yaml`、eval JSON、`checkpoint_final.pth` 与关键渲染；3 个 `checkpoint_15000.pth` 已按正式清单删除 |
| O0 occupancy | `/root/autodl-tmp/data/occgs/occupancy/{003,004,005}/` | 保留为 WorldState/坐标与 UNKNOWN 失败证据，不是当前训练输入 |
| S0 edits | `/root/autodl-tmp/data/occgs/scene_specs/s0_edits/` | 保留逐 scene edit JSON；当前汇总覆盖缺口见 `EXPERIMENTS.md` |
| C0 counterfactual | `/root/autodl-tmp/runs/occgs_resim/c0_cf/` | 保留报告与历史 matched-ablation 证据；大图清理仍需先做证据索引 |
| C0 review material | `/root/autodl-tmp/data/occgs/reviews/c0_legality/` | 机器材料，不得改写成人工 verdict；人审前保留 |
| L0 / U0 | `/root/autodl-tmp/runs/occgs_resim/{l0_comp,u0_screen}/` | 保留轻量 JSON；图像只能在索引/hash 后按明确清单处理 |
| 历史正式 review | 历史 run 中的 `reviews.jsonl`、panel、prompt | 永久保护，除非用户明确授权迁移/删除 |

## 4. 可恢复但不默认驻留的资产

- SVD-XT：固定 Hugging Face revision 与 2026-07-19 清理细节见历史账本；
- ReSim CogVideoX checkpoint：V6 已关闭，历史重建边界见
  [`archive/2026-07/v6/C1_V6_FINAL_REPORT.md`](archive/2026-07/v6/C1_V6_FINAL_REPORT.md)；
- pip/conda/Hugging Face cache：仅在不影响当前环境复现且上游固定时清理；
- 历史已拒绝路线的 checkpoint/candidate：轻量证据和负结论必须继续驻留。

资产 non-resident 不等于历史实验“未测试”，也不自动授权重新下载或重跑。

## 5. 存储门槛

- 新路线安装/训练启动前要求可用空间至少 60 GiB，运行过程中始终保留 20 GiB 安全余量；
- raw nuScenes 优先 symlink，不复制公共盘全量；
- AD-GS exact reproduction 未通过前，不下载 Waymo/PandaSet、全量 Occ3D、全量 UniScene 权重或大型视频生成模型；
- 空间不足时先缩小未开始的协议，再评估可重建 cache；不得从受保护材料开始删。

## 6. 事实解释

- 清理只改变“本机是否驻留”，不改变 run ID、指标、人工结论、RF 状态或重开条件；
- V7 既有 run 缺 formal manifest 的问题由 `V7-EV-10` 处理，不能用清理账本补造 provenance；
- 历史清理批次的字节数、hash、测试结果和当时环境只在归档账本中解释，不回写为当前实例状态。

## 7. 2026-07-26 路线转向后的当前保护范围

| 类别 | 路径 | 处理规则 |
|---|---|---|
| cut-in 最终归档 | `docs/archive/2026-07/cutin-mining-closed/` | 永久保留报告、预注册、人工包、状态/实验快照与清理清单 |
| 失败总账 | `docs/RESEARCH_FAILURES.md` | 永久保护；路线切换只追加，不删除旧失败 |
| raw nuScenes | `/root/autodl-tmp/data/nuscenes/` | 不删除、不原地改写 |
| 本地官方 tar | `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/` | 只读；新路线只做选择性提取 |
| V7 evidence index | `/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json` | 永久保留，用于已清中间 checkpoint 的 hash/provenance |
| H1C final labels | `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/render_labels` | 保留；两个失败/修复前 dense 副本已按清单删除 |
| 新路线计划/台账 | `docs/DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`、`RESEARCH_STATUS.md`、`EXPERIMENTS.md` | 当前事实源 |
| 新路线 upstream | `/root/autodl-tmp/third_party/{AD-GS,dggt,...}` | 安装后固定 commit、license、diff；未经清单不得删除 |
| 新路线数据 manifest | `/root/autodl-tmp/data/dynamic_recon/manifests/` | 永久保护，派生 cache 可按 manifest 重建 |
| 新路线正式 runs | `/root/autodl-tmp/runs/dynamic_recon/` | config/metrics/summary/terminal/hash 与 final checkpoint 保护 |

本次已执行清理的精确路径、字节数、checkpoint 哈希、保留项和恢复路径只由
[`archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md`](archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md)
解释。后续清理不得把该批次扩展为未列出的路径。
