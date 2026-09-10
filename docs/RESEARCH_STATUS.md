# 当前状态：WorldSim V7.4 P0

日期：2026-09-10；分支：`research/worldsim-v7.4-method-tournament`；基线提交：`01af4739`（V73 最终收口）。
当前用户授权：先完成文档清理归档、数据盘清理与 CPU 数据预处理，汇报后由用户有卡开机。

| 任务 | 状态 | 实际证据与下一步 |
|---|---|---|
| WS-V74-P0-DOCS-01 | done | 活跃 AGENTS/README/计划已统一；旧计划移入 archive/2026-09/pre-v74；V73 证据保留 |
| WS-V74-P0-STORAGE-01 | running | 初始数据盘 638/700 GiB；分类清理可重建缓存，保留关键证据 |
| WS-V74-P0-DATA-01 | running | 复用两套真实对象观测；准备隔离的 BUILD/QUERY 与固定 probe；预留 AV2 新测试身份 |
| WS-V74-METHOD-TOURNAMENT-01 | pending | P0 交接后推进独立 A/B/C 与强控制；当前无 V74 方法分数 |

当前资源：cgroup 0.5 CPU、2 GiB 内存，GPU 访问被拒绝；宿主 112 核/755 GiB 不是实例预算。逐对象流式 CPU 处理可行，不加载数 GB RGB 拼包。
nuScenes 现有历史角色覆盖全部 trainval 日志，不能直接宣称找到十个全新独立日志。现有域内 FIT/DEV 可准备，严格最终确认限制记录为 V74-F01。
两数据集分别训练/评价；旧 AV2 20 日志已曝光，可拆作 V74 FIT/DEV，不能继续当 FINAL。失败记录沿用 V73-F02/F03/F04/F05/F09；三候选均未开始，不写 SURVIVORS 或 NO_SURVIVOR。

证据与交接：[P0 报告](WORLDSIM_V74_P0_HANDOFF.md)；[V73 完整历史状态](archive/2026-09/pre-v74/V73_RESEARCH_STATUS.md)。
failure_ledger_delta：新增 V74-F01（独立 nuScenes 身份不足）、V74-F02（当前无 GPU 与 CPU 配额）；文档冲突已通过归档解决，不是算法失败。
