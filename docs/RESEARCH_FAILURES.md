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
