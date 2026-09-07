# WorldSim V7.2 task-first 历史执行计划

> **2026-09-07：本计划/交接的执行范围已被替代，status=`rejected`（目标不匹配，非全部方法被证伪）。** 当前使用 [EAS-VGGT Recovery Plan](WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md) 和 [最新状态](RESEARCH_STATUS.md)。下文 A/B、R1–R7、外部补全/神经 LiDAR 队列及 shutdown 判据属于历史，不据此启动任务。已完成的代码、I/O、run 和各自正/负结果保留；纠偏见 `V71-F64/F65`。

日期：2026-09-06

分支：`research/worldsim-v7.2-task-first-completion-lidar`

父提交：`79910be13e1f4740dbcd793fbfd45ee6d66020f8`

完整研究合同：`docs/WORLDSIM_V7_2_TASK_FIRST_OCCUPANCY_OR_NEURAL_LIDAR_PLAN.md`

## 当前问题

在公平输入、统一算子和相近成本下，现有对象补全或完整 LiDAR 重建方法是否仍存在几何完整性与观测一致性的缺口；若存在，V7.2 能否产生强于简单融合、标量权重和成熟外部方法的稳定增量。

路线 A 暂称 **对象表面补全**，只有存在可靠体积标签时才升级为 Occupancy。路线 B 必须输出完整扫描，包括返回存在性、测距、背景/Actor 深度排序和无回波协议；Actor 框内条件测距不能称为 Neural LiDAR Simulation。

## 当前阶段

| 任务 ID | 状态 | 交付或阻塞 |
|---|---|---|
| `WS-V72-P0-OPERATOR-ERRATUM-01` | done | M43 literal/categorical 字段隔离；真实 Actor state retention；M39 拒绝保留 |
| `WS-V72-P0-DATA-CONTRACT-01` | done | `ActorBundleV2`、`QueryRayBatch`、`RayTargets`、target-free 点表面 forward |
| `WS-V72-P0-DATA-ROLES-01` | done | nuScenes/AV2 metadata-only 暴露合并；最终 split 仍为空 |
| `WS-V72-P0-BASELINE-CAPABILITY-01` | done | 官方源码、checkpoint、依赖与数据适配已审计；GPU capability 为独立后续任务 |
| `WS-V72-D0-G0-RAW-FUSION-01` | done | 66 Actors / 34 logs 的 5 点密度曲线；只作 legacy diagnostic |
| `WS-V72-D0-G1-ACTOR-TSDF-01` | done | 同一 66 Actors / 34 logs 的 5 点密度曲线；44 raw + 22 processed provenance |
| `WS-V72-P0-G3-DATA-ADAPTER-01` | done | AdaPoinTr 593/66 Actor adapter 与官方 PCN checkpoint 已就绪 |
| `WS-V72-P1-D0-MATCHED-BASELINES-01` | done | G0--G3 与 W0--W4 完成；普遍三态缺陷主张关闭 |
| `WS-V72-B0-LIDAR4D-CAPABILITY-01` | done | 官方 KITTI-360 30k + refinement + final eval 完成 |
| `WS-V72-P2-CLEAN-SPLIT-FREEZE-01` | done | 4 dev / 3 route-select 日志冻结；3 source-test candidates 保持未读 |

状态只使用 `pending/running/blocked/done/rejected`。D1 前不得填充 `source_test` 或 `external_test`。

## 实际数据角色

| 数据 | metadata 规模 | 历史暴露 | 当前可用角色 | 结论 |
|---|---:|---:|---:|---|
| nuScenes trainval | 850 scenes / 68 logs | V7.1 corpus 1004 Actors 来自 164 scenes / 54 logs | train 54 logs；legacy diagnostic 2；exposure unknown 2；候选池 10 | 旧 120/20/20 scene split 有 10 个跨角色 log；不能声称日志级独立 |
| AV2 Sensor val | 150 logs | 四个历史 cohort 合计 80 logs | 磁盘上 80/80 均已消费；官方未消费 70 logs 尚未列出/下载 | M43 只作 legacy diagnostic；新 external split 尚未冻结 |

`configs/worldsim_v72/data_role_inventory.json` 保存原始 metadata 合并，`data_roles.json` 只登记互斥日志角色。候选池只是容量审计，不是测试集。

当前 nuScenes 只有 10 个全依赖链明确未暴露日志，达不到原目标规模。已按日志 token 字典序、完全不读质量地冻结 4 个 `dev` 与 3 个 `route_select`；余下 3 个只作为尚未打开的 `source_test` candidates。该缩小规模是明确证据限制，相邻 scenes 不扩大独立分母；冻结清单见 `configs/worldsim_v72/clean_route_split.yaml`。

## D0 最小比较

| 组别 | 几何/权重 | 当前状态 | 必须回答 |
|---|---|---|---|
| G0 | build-only 累积点/voxel surfel | CPU canonical complete | 点数与密度本身能解释多少结果 |
| G1 | Actor-local TSDF | CPU canonical complete | 与 G0 同 cohort；256/512 点形成非支配简单基线前沿 |
| G2 | M8 | legacy learned | 只作历史候选，不产出独立源域主张 |
| G3 | AdaPoinTr 适配版 | 源码、官方 PCN 权重和 adapter 就绪 | 从头训练与外部预训练分表 |
| W0–W4 | unit/support/density/scalar/F-O-U | 合同已就绪，等待 GPU | 三态证据是否优于同容量标量 |
| B0 | DyNFL；访问受阻时 LiDAR4D | 官方源码已固定 | 能否生成完整扫描并公平处理场景拟合 |

## 当前执行顺序（GPU 已恢复）

1. 物化冻结的 4 个 dev 与 3 个 route-select 日志，保持余下 3 个 source-test candidates 未读。
2. 在 clean dev 上运行 A1 观测约束 AdaPoinTr 与 G0/G1/G3 公平对照，并建立 B 的 clean development 证据。
3. 冻结 A/B 候选后只读 route-select 完成 D1；此时仍不读取 source/external test。

CPU 前置结果与开卡交接分别见 `docs/WORLDSIM_V7_2_D0_CPU_PREFLIGHT_REPORT.md` 和
`docs/WORLDSIM_V7_2_GPU_HANDOFF.md`。
