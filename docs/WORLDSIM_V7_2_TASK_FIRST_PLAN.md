# WorldSim V7.2 当前执行计划

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
| `WS-V72-P0-BASELINE-CAPABILITY-01` | running | 官方代码与依赖已审计；CUDA 环境和数据能力运行等待 GPU/数据 |
| `WS-V72-P1-D0-MATCHED-BASELINES-01` | blocked | 需要至少 1×RTX 3090 24 GiB 运行 AdaPoinTr 与正式 matched evaluation |

状态只使用 `pending/running/blocked/done/rejected`。D1 前不得填充 `source_test` 或 `external_test`。

## 实际数据角色

| 数据 | metadata 规模 | 历史暴露 | 当前可用角色 | 结论 |
|---|---:|---:|---:|---|
| nuScenes trainval | 850 scenes / 68 logs | V7.1 corpus 1004 Actors 来自 164 scenes / 54 logs | train 54 logs；legacy diagnostic 2；exposure unknown 2；候选池 10 | 旧 120/20/20 scene split 有 10 个跨角色 log；不能声称日志级独立 |
| AV2 Sensor val | 150 logs | 四个历史 cohort 合计 80 logs | 磁盘上 80/80 均已消费；官方未消费 70 logs 尚未列出/下载 | M43 只作 legacy diagnostic；新 external split 尚未冻结 |

`configs/worldsim_v72/data_role_inventory.json` 保存原始 metadata 合并，`data_roles.json` 只登记互斥日志角色。候选池只是容量审计，不是测试集。

当前 nuScenes 只有 10 个明确未分配日志，达不到计划中的 `dev 20 + route_select 12 + source_test 30`。D1 前必须在以下边界内重新分配：缩小并公开实际独立样本规模，或接入具有公开协议的第二数据集；不能把相邻 scenes 拆开扩大分母。

## D0 最小比较

| 组别 | 几何/权重 | 当前状态 | 必须回答 |
|---|---|---|---|
| G0 | build-only 累积点/voxel surfel | CPU 实现中 | 点数与密度本身能解释多少结果 |
| G1 | Actor-local TSDF | 旧实现可复用 | 与 G0/G2 在同一 Actor、同一算子上比较 |
| G2 | M8 | legacy learned | 只作历史候选，不产出独立源域主张 |
| G3 | AdaPoinTr 适配版 | 官方源码已固定 | 从头训练与外部预训练分表 |
| W0–W4 | unit/support/density/scalar/F-O-U | 等待 v2 cache 与 GPU | 三态证据是否优于同容量标量 |
| B0 | DyNFL；访问受阻时 LiDAR4D | 官方源码已固定 | 能否生成完整扫描并公平处理场景拟合 |

## GPU 恢复后的第一顺序

1. 构建一小批 `ActorBundleV2`，实测峰值内存并完成 AdaPoinTr CUDA 扩展 capability。
2. 在 legacy development 上跑 G0/G1/G2/G3 同对象、同密度、同算子主表。
3. DyNFL 若 Waymo 授权和数据可用，先跑官方单场景；否则用完整 KITTI-360 数据跑 LiDAR4D 官方 sequence capability。
4. 只有两路均有公平强对照后才做 D1；此时仍不读取最终 source/external test。
