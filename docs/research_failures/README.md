# 失败资产维护

入口 [RESEARCH_FAILURES](../RESEARCH_FAILURES.md) → [版本](VERSIONS.md) / [ID](IDS.md) → 单条证据。当前状态只写 [RESEARCH_STATUS](../RESEARCH_STATUS.md)，不写进本目录首页。

## 存储与维护

- `history/part-*.md` 与 `history_index.json`：冻结的原始正文、原行号及阶段顺序。不要改写正文或重排记录；已有事实更正通过新卡链接原证据。
- `entries/ID.md`：可维护的独立失败卡。相同问题更新原卡，保留原始结果、正例、控制及结论变化。
- `VERSIONS.md`、`versions/`、`IDS.md`、`ids/`、`ENTRIES.md`：由 `python scripts/build_research_failure_index.py` 生成；新增或修改卡后再生成，避免人工漏项。
- `defined_ids` 是明确标题定义，`referenced_ids` 是引用；只有引用时不虚构独立失败。主题用于检索，不自动裁决失败层级。
- 没有新增失败时不修改总入口。任务运行失败写实验/运行证据；只有可复用的工程教训才需要单独 failure 卡。

```markdown
# V74-H2-Fxx：具体失败边界
<!-- metadata: {"title":"具体失败边界","defined_ids":["V74-H2-Fxx"],"referenced_ids":[],"versions":["V74-H2"],"topics":["engineering"]} -->

类型：工程 / 数值 / 优化 / 表示 / 科学 / 新颖性 / 证据不足。
命题、触发条件及输入信息范围：
最强控制与实际结果（含正例、负例、完整分母）：
排除的解释、仍开放的问题与适用范围：
证据：task/run、配置、输入角色、原始与关键资产、成本及报告链接。
failure_ledger_refs / failure_ledger_delta：
人工 verdict：null
```

失败卡记录某次实验的结论及边界，不写当前任务队列或新的执行授权。技术报告按协作规则配组件图。大型证据只链接保存位置和恢复入口，轻量索引不冒充资产备份。
