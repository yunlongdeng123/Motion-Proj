# 当前状态：V74 P0 完成，等待有卡开机

日期：2026-09-10。分支 `research/worldsim-v7.4-method-tournament` 从 V73 最终 `01af4739` 建立；文档里程碑 d6bea861、存储里程碑 463ef199 均已 push；数据执行基线 463ef199。

| task | 状态 | 真实结果 |
|---|---|---|
| WS-V74-P0-DOCS-01 | done | 76 份非主线文档归档，AGENTS/导航/主计划统一；保留 V73 历史与论文 |
| WS-V74-P0-STORAGE-01 | done | 清理 4105 项可重建缓存/旧逐帧副本，实际释放 131.802 GiB；最终准备时可用 194.126 GiB |
| WS-V74-P0-DATA-01 | done | 1425 对象 BUILD/QUERY 分离、1071391 点几何缓存；43 对象/8 日志 probe，另保留 23 缺输入合同对象；AV2 新 FINAL 10 日志原始窗口已下载 |
| WS-V74-METHOD-TOURNAMENT-01 | pending | 等用户有卡开机；WEX/RIF/DCS 尚未实现训练或取得质量结果 |

| 数据集/角色 | 日志 | 对象 | 有 BUILD | 缺 BUILD | 无自有 QUERY 返回 |
|---|---:|---:|---:|---:|---:|
| nuscenes FIT | 20 | 414 | 371 | 43 | 83 |
| nuscenes DEV | 5 | 75 | 67 | 8 | 23 |
| av2 FIT | 17 | 807 | 764 | 43 | 96 |
| av2 DEV | 3 | 129 | 114 | 15 | 23 |

完整细节与命令：[P0 交接报告](WORLDSIM_V74_P0_HANDOFF.md)；机器可读：`autoresearch/worldsim_v74/p0/handoff_summary.json`。
两数据集分别 FIT 与评价；AV2 新 FINAL 在 `configs/worldsim_v74/av2_final.json`，只有原始文件准备完成，规范坐标转换/测试未执行。
nuScenes 暂无可证明全新未曝光的最终日志；保留现有域内开发，不能宣称双数据集独立最终确认已经齐备。43/48 的 probe 差额来自部分预定日志可用对象不足，不按留出质量换对象；11 个 probe 对象没有自有 QUERY 返回。

本实例 cgroup 0.5 CPU/2 GiB、GPU 不可访问；CPU 分离导出峰值 0.375 GiB，局部几何峰值 0.097 GiB，无 OOM，无训练/模型评价。
下一步：完成本里程碑 push，确认所有任务退出后 shutdown；实际回执保存于本次任务本地 outputs。用户有卡开机后按 V74 主计划推进三独立候选与强控制，不恢复 V73 launcher 或旧调度。
failure_ledger_delta=V74-F02 的 CPU 规避已完成、GPU 资源需求仍 active；V74-F01 仍 active，无新增科学失败/候选裁决。历史：[V73 状态快照](archive/2026-09/pre-v74/V73_RESEARCH_STATUS.md)。
