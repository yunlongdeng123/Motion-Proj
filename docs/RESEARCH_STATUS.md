# V74 方法竞争：有卡恢复，首次实现进行中

2026-09-10，执行基线 e782b2b4。用户已解除 P0 阶段终点。RTX 3090 24GiB、14 CPU、90GiB 内存；当前没有资源阻断，V74-F02 GPU 状态已解决。P0 结果保持，原始数据不重跑。

| task | 当前状态 | 实际证据 |
|---|---|---|
| WS-V74-RESUME-01 | done | start/resource_resume.json，分支干净、GPU 空闲 |
| WS-V74-FIT-CALIBRATION-01 | done | nuScenes 129812 / AV2 672669 跨 BUILD 平面比较，epsilon=.174520/.200000m；DEV 质量未读 |
| WS-V74-METHOD-TOURNAMENT-01 | implementing | WEX/RIF/DCS 独立实现及强控制；43 ready/8 日志 + 23 缺 BUILD 合同对象 |

主 seed=7401，确认=7402；输出≤4096三角形；各自 FIT/DEV。尚无 DEV 模型质量或候选裁决；V74-F03 初始冲突图遗漏已修复，V74-F04 登记 WEX 的标准求解器替代/支撑缺失风险。FIT 定标是各自数据内完成，局部平面残差含曲率/轨迹/归属代理误差，不是噪声真值；容差以上仍记冲突。原 nuScenes FINAL 数据缺口 V74-F01 保留。
OSQP 1.0.4 已安装，SciPy 1.15.3/HiGHS 可用。WEX 域子系统实现及其固定/贪心/MILP/软域控制已经写入；80 个解析约束算例 r1/r2 和两数据集各2对象 FIT pilot 已完成，不能代替完整三维机制与真实验证。公共硬求交/评价接入已有 V73 定义；保存机制算例、初始/事件/最终资产及全部对象分母。详细失败见 [V74 失败过程](WORLDSIM_V7_4_FAILURES.md)。failure_ledger_delta=resolve V74-F02 GPU resource; no science verdict。

WEX 实际证据：解析 multiple_front_constraints 20例中，r2 WEX20/20、贪心13/20、MILP20/20；MILP累计 .143s，WEX .230s。其余有正确候选的解析组贪心/MILP同样成功，不能声明机制不可替代。
FIT4对象中 WEX/MILP BUILD正确命中分别均为40/41、908/3195、74/74、1145/4208；两个较密对象缺正确候选2242/3009束，域选择无法凭空补齐。WEX计算 .575/45.30/.076/126.40s，MILP .060/.884/.089/2.471s；首个资产与V73首交点读出数值差0。无DEV/FINAL质量读取。
证据 `autoresearch/worldsim_v74/a_wex/{domain_r1_summary,domain_r2_summary,fit_pilot_results,fit_pilot_resources}.json`。failure_ledger_delta=V74-F03 resolved initial implementation; V74-F04 active hypothesis risk，尚无主方法晋级/淘汰裁决。

---

# P0 完成时的历史状态

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
