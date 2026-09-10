# WorldSim 当前实验台账

## P0 存储里程碑已完成（2026-09-10）

WS-V74-P0-STORAGE-01 done；执行基线 d6bea861。删除 4105 个已列明目标，实际释放 131.802 GiB，完成时可用 194.487 GiB。
包括 5 处冻结视觉前缀和退役 V6 传感器重渲染帧/旧 dense logits；按 run/组保留 340 个代表文件。原始数据、V73 checkpoint/表面/射线/指标、环境和模型保留。
恢复旧逐帧分析可能需要重新推理或恢复历史依赖；不把缓存清理解释成无成本完整复现。精确路径/大小/恢复方式：docs/autoresearch/worldsim_v74/p0/storage_plan.json；实际删除和空间差：storage_result.json。
failure_ledger_delta=none（未新增科学失败）；V74-F01/F02 继续。CPU 数据预处理 running，尚未关机。


日期：2026-09-10；V74 从 `01af4739` 建分支。下表只登记真实执行；P0 不属于方法效果实验。

| task | 状态 | 数据/seed | 证据、结果与下一步 |
|---|---|---|---|
| WS-V74-P0-DOCS-01 | done | 无训练；seed 不适用 | 归档旧执行文档，统一 V74 合同和双数据集域内验证；本提交文档 diff |
| WS-V74-P0-STORAGE-01 | done | 不读模型质量 | 清理记录将写 autoresearch/worldsim_v74/p0/storage_result.json |
| WS-V74-P0-DATA-01 | running | nuScenes/AV2；确定性元数据排序 | 每对象 BUILD/QUERY 分离、probe 分层、FIT/DEV/FINAL 身份；实际计数待导出完成 |
| WS-V74-METHOD-TOURNAMENT-01 | pending | 各数据集独立 FIT；主 seed=7401、确认 seed=7402 | WEX/RIF/DCS 无训练、无评分、无裁决 |

failure_ledger_refs=[V73-F02,V73-F03,V73-F04,V73-F05,V73-F09,V74-F01,V74-F02]。
本里程碑 failure_ledger_delta=新增 V74-F01/V74-F02；文档归档本身无新科学失败。
[V73 及以前完整实验台账](archive/2026-09/pre-v74/V73_EXPERIMENTS.md) 保留全部历史记录；V73 最终论文和保存的指标不重跑。
