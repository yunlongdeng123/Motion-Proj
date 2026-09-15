# 当前里程碑：三模型36项完成，Omega512按用户指定来源加入（2026-09-15）

`WS-V74-MAINFIG-01 / 20260915-first-return-r1`：官方VGGT、DVGT-1、Pi3X完成6场景×6/12图推理；单RTX3090。原生输出保存，普通尺度/标定/置信度和共同6输出图控制完成或汇总中。用户提供Omega512公开镜像并授权下载使用，替代本轮未获访问的416重现版，版本与来源必须明示。75对象全分母、52有QUERY、23未定义、11886射线。无训练、无封存集、无闭环/false-safe结论、无关机或调度。详见主图实验报告；人工verdict=null；failure_ledger_delta=pending。以下为历史。

# 当前：V7.4 官方模型首回波与主图实验（2026-09-15）

用户授权单张RTX3090。task `WS-V74-MAINFIG-01 / 20260915-first-return-r1`；首批仅官方VGGT、VGGT-Ω 416 reproduction、DVGT-1、Pi3X。固定旧DEV 6场景/5日志/75对象，BUILD sample3六环视与sample3+4十二视图配对，QUERY sample2/5独立评价。官方Omega权重401，等待用户配置授权；其余继续。无训练、无封存集、无关机或自动调度。原始模型输出、已知标定/BUILD定尺度额外信息诊断与网格首交读出分开记录。Early不等于false-safe；闭环未测。旧白车仅代表微调VGGT DPT＋LiDAR融合适配系统。人工verdict=null。failure_ledger_refs=V73-F03/F09、V74-F06、V74-H2-F12；delta=pending。以下为历史。

# 当前：V7.4-H2 首次回波重新取证完成（2026-09-15）

已回到 `research/worldsim-v7.4-h2-generative-surface`，V8.1 分支保留。task `WS-V74-H2-REDISCOVERY-01 / 20260915-cpu-r1`，状态 **CPU_DONE_WAIT_GPU**。恢复7个旧案例、8364束双精度回放、3对象×7系统配对、3个RGB上下文和15张图；0新模型推理、0训练。白色轿车两个留出时刻Early59/557、18/195，0.5m仍32/752；旧VGGT-native实际为微调DPT＋LiDAR融合，不能代表官方VGGT。旧20日志r7−native融合15/20同增Hit/Early，12/20四项同增；强r6对照完整保留。

官方VGGT/Ω、Pi3X、MapAnything、DVGT、DGGT的共同失效仍待原生输出和尺度/读出控制；Early不等于false-safe。只准备有限GPU验证，不恢复已关闭H2训练或旧调度。旧20日志已曝光，10日志reserve质量未读；人工verdict=null。failure_ledger_refs=V73-F02/F03/F04/F09、V74-F06、V74-H2-F11；delta=V74-H2-F12。当前不继承历史shutdown授权，未关机、无自动续跑。

[本轮报告与组件图](WORLDSIM_V7_4_H2_FIRST_RETURN_REDISCOVERY.md)；[GPU接续](WORLDSIM_V7_4_H2_REDISCOVERY_GPU_HANDOFF.md)；[F12](research_failures/entries/V74-H2-F12.md)。以下为历史。

# 当前：V7.4-H2 存量 badcase 重新取证（2026-09-15）

用户已授权回到 `research/worldsim-v7.4-h2-generative-surface`，当前无卡。task `WS-V74-H2-REDISCOVERY-01` / `20260915-cpu-r1`：追查覆盖与首次回波冲突的旧真实案例，审计 VGGT 与后继方法的实测边界。只执行存量 CPU 分析、可视化与后续实验准备，不自动恢复已关闭 H2 学习器或旧调度，不继承历史关机授权。V8.1 独立分支保留。旧20日志确认现为已曝光资料，10日志新 reserve 继续封存。人工 verdict=null。以下是历史。

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
