# 当前：V8.1 GPU_PILOT_PASSED / 双3090并行（2026-09-13）

修正记录：[V81-F03](research_failures/entries/V81-F03.md)。首轮DVGT导出遗漏官方RDF→FLU与gt_scale_factor=0.1的还原，并误用LiDAR时间戳ego pose；已按官方合同修复，39项原生forward无需重跑，错误导出另存。新增坐标/尺度/时间戳测试通过；此为实现错误，不能作为模型badcase。两主模型队列继续独立运行。

用户已开启2×RTX3090并授权继续，DVGT固定GPU0、VGGT固定GPU1独立推理。task=WS-V81-GPU-P2-01；run=20260913-dual3090-r1；seed=8101。首例full6已完成：DVGT 10.638GiB / 0.865s，VGGT 7.535GiB / 1.011s（前向时间，不含模型加载）。28核/180GiB cgroup；保留官方完整FP32权重，BF16 autocast。没有训练或V8.2方法开发。

冻结r2的408任务主队列接续；full6 DISCOVERY，稀疏/重排 VIEW_DIAGNOSTIC，合成纹理 SYNTHETIC_DIAGNOSTIC。完整自然四格匹配组仍为0。VGGT首例camera-baseline scale CV=0.691，必须分开审计全局尺度不稳定与局部形状误差。人工verdict=null；尚无可推广failure结论。资源不需扩容；不沿用V74关机指令。

failure_ledger_refs=V81-F01/F02、V74-H2-F09/F10/F11；failure_ledger_delta=V81-F03（导出合同修正）。证据：`/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01/registration.json`及两模型首例result.json。以下为冻结历史。

# 当前：V8.1 CPU_COMPLETE_WAIT_GPU（2026-09-13）

WS-V81-CPU-01 / 20260913-cpu-r2；稀疏视角×低纹理failure discovery。27日志、34场景、68窗口、6120 ROI、1527几何候选；8案例卡、49视觉复核，36混杂排除。同日志/语义/距离完整四格匹配组=0，不能计算自然H1交互效应。全部DISCOVERY，AV2 reserve质量保持封存。

无卡可执行工作完成：DVGT/VGGT/DGGT官方权重落盘、3模型strict meta加载和真实输入预处理通过、7项检查通过、408任务队列与诊断输入冻结。模型推理=0，H1–H5未检验，V8.2 NO_GO_PENDING_EVIDENCE。CPU阶段已停止，无自动续跑；请求2×48GB GPU或1×80GB，主机≥64GB RAM/8 vCPU。GPU不会自动补足自然对照数据缺口。

failure_ledger_refs=V74-H2-F11/F09/F10、V74-F01；failure_ledger_delta=V81-F01/F02。人工verdict=null。入口：[CPU报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md)、[图谱](WORLDSIM_V8_1_FAILURE_ATLAS.md)、[GPU交接](WORLDSIM_V8_1_CPU_HANDOFF.md)。以下为冻结历史。
# Research Failures：渐进读取入口

最终精简：GitHub 实际下载 ZIP 约 **22.2 MB**；同口径本地打包105.25→21.45 MB，减少约80%，跟踪文件展开约53.6 MB（原375.9 MB）；两批共307个资产、原文件322.49 MB已归档。全部代码/配置/Markdown、失败查询索引和V7.4报告图保留。旧下载包不会自动变小，需重新下载当前分支；完整 Git 历史保持，未强推改写。

**当前：V7.4 已收尾，未得到经过验证的论文主方法。** H1 保持 NO_SURVIVOR；H2 当前 ordered 实现关闭，整体 witness-native 假说未被全面证伪。用户已结束本轮，不自动延长诊断或启动新方法。

| 要解决的问题 | 从哪里读 |
|---|---|
| V7.4 最终结论与研究过程教训 | [F11 收尾短卡](research_failures/entries/V74-H2-F11.md) → [收尾报告](WORLDSIM_V7_4_CLOSEOUT.md) |
| 哪些动机已排除，什么仍开放 | [累计研究边界](research_failures/BOUNDARIES.md) |
| GPU 训练实现为何停止 | [F08](research_failures/entries/V74-H2-F08.md) |
| 薄结构为何退化、普通控制解释了多少 | [F09](research_failures/entries/V74-H2-F09.md) → [F10](research_failures/entries/V74-H2-F10.md) |
| 上半场三个候选为何失败 | [H1 最终失败](WORLDSIM_V7_4_FAILURES.md)；关键 V74-F04/F09/F10 |
| 按版本或 ID 渐进检索 | [版本目录](research_failures/VERSIONS.md)、[新记录目录](research_failures/ENTRIES.md)、下方查询命令 |
| 如何新增可复用资产 | [维护约定与模板](research_failures/README.md) |

```bash
python scripts/query_research_failures.py --id V74-H2-F11 --detail --limit 1
python scripts/query_research_failures.py --id V74-H2-F10 --detail --limit 1
python scripts/query_research_failures.py --query 后继 --limit 5
python scripts/query_research_failures.py --topic first_return --version V73 --limit 10
python scripts/query_research_failures.py --record RF0013 --detail --max-lines 80
```

无 `--detail` 时只输出目录；多页用 `--offset`，长正文用 `--line-offset`。历史 14485 行已迁移为 80 个分片、1084 条历史记录，正文保留；新卡另由查询器读取。见 [迁移清单](research_failures/migration.json)。不默认全文读取。

主题：`first_return`、`coverage`、`novelty`、`data_evidence`、`engineering`、`resources`、`optimization`、`legacy_policy`。旧阶段的 RUNNING、WAIT_GPU、未裁决及历史权限仅代表当时状态，不能恢复为当前执行授权。旧哈希/门控要求不适用；当前用户要求与 [scaling law](../auto-research_scaling_law.md) 优先。

failure_ledger_delta：新增 V74-H2-F11（收尾与规划纠偏），F08–F10 及 V73/H1 历史证据保持原结论。

大资产已外移、原值保留：[归档入口](archives/worldsim_v74_closeout_20260913/README.md)。逐文件查询与恢复不需要重新读完整历史账本。

追加精简完成：两批合计移出 307 个批量/生成资产，原始文件共 322.49 MB；两个完整包均已保存远端和本地副本。V7.4 核心图保留，旧论文与 V7.3 非架构图改为按需恢复。 [追加归档](archives/worldsim_v74_closeout_20260913/LEGACY_MEDIA.md)。
