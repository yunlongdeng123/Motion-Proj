# Research Failures：渐进读取入口

当前路线：V74-H2；[状态](RESEARCH_STATUS.md) → [研究边界](research_failures/BOUNDARIES.md) → 相关 failure ID → 证据。不要默认读完历史账本。

| 要解决的问题 | 从哪里读 |
|---|---|
| 哪些方法族已关闭，什么仍开放 | [累计研究边界](research_failures/BOUNDARIES.md) |
| 上半场三个候选为何失败 | [V74-H1 最终失败](WORLDSIM_V7_4_FAILURES.md)；关键 V74-F04/F09/F10 |
| 按版本查找 | [分页版本目录](research_failures/VERSIONS.md) |
| 按 ID、主题、关键词查找 | 下方查询命令；历史 1084 条目录 |
| 下半场新失败 | [新记录目录](research_failures/ENTRIES.md) |
| 如何新增可复用资产 | [维护约定与模板](research_failures/README.md) |

```bash
python scripts/query_research_failures.py --id V74-F10 --detail --limit 1
python scripts/query_research_failures.py --query 后继 --limit 5
python scripts/query_research_failures.py --topic first_return --version V73 --limit 10
python scripts/query_research_failures.py --record RF0013 --detail --max-lines 80
```

主题：`first_return`、`coverage`、`novelty`、`data_evidence`、`engineering`、`resources`、`optimization`、`legacy_policy`。
无 `--detail` 时仅输出目录；多页用 `--offset`，长正文用 `--line-offset`。同一 ID 的不同阶段记录分别保留，不合并成互相矛盾的“当前结论”。

迁移：原 14485 行 → 80 个历史分片、1084 条可查询记录；正文顺序完整保留，详见 [迁移清单](research_failures/migration.json)。旧哈希/门控要求是历史文本，**不适用于当前工作**；当前用户要求和 [scaling law](../auto-research_scaling_law.md) 优先。

下半场CPU已收口：V74-H2-F01（数据）、F02（迁移/路径修复）、F03（数值）、F04（预算规范化修复）、F05（远层排序/特征截断修复）；尚无A的科学或新颖性裁决。人工verdict=null。H1最终短卡优先返回，旧阶段记录继续可查。
