# 跨版本失败资产

本页只提供失败证据导航。当前执行状态见 [RESEARCH_STATUS](RESEARCH_STATUS.md)，实验进度见 [EXPERIMENTS](EXPERIMENTS.md)。

| 查什么 | 入口 |
|---|---|
| **以前各版本的 failure** | [全部版本](research_failures/VERSIONS.md)；包含早期 N1 / H1 / PIVOT 至 V7.7 |
| **已知 failure ID** | [全量 ID 目录](research_failures/IDS.md)，区分独立定义与仅有引用 |
| V7.4-H1 和 H2 F01–F22 | [独立卡目录](research_failures/ENTRIES.md) |
| V7.5 工程与能力边界 | [V75-F01 输入绑定](research_failures/entries/V75-F01.md)、[V75-F02 DriveEditor 分母与接口](research_failures/entries/V75-F02.md)；[r9 收尾证据](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md) |
| V7.6 对象资产质量、复现与数据适配 | [V76-F03 当前HUGSIM/VAD-GS资产不足以支撑高保真编辑](research_failures/entries/V76-F03.md)、[V76-F01 COLMAP track视图错绑](research_failures/entries/V76-F01.md)、[V76-F02 SAM遮挡身份错绑](research_failures/entries/V76-F02.md)；[收口证据](v76/CLOSEOUT.md) |
| V7.7 对象资产适配 | [V77-F01 框内点集不等于完整可编辑对象](research_failures/entries/V77-F01.md)、[V77-F02 成熟补景与单图GLB拼接未达两场景高保真](research_failures/entries/V77-F02.md)；[24对象与三时刻控制](v77/P0_RESULTS.md)、[显式资产POC](v77/EXPLICIT_POC.md) |
| 已排除的解释与仍开放的问题 | [累计研究边界](research_failures/BOUNDARIES.md) |
| 原始过程和原文完整性 | [历史分片说明](research_failures/migration.json)、[本次恢复核对](archive/2026-09/v74-0920/FAILURE_AUDIT.json) |
| 怎么新增或修订 | [维护规范](research_failures/README.md) |

旧正文完整保留：迁移前 **14,485 行**对应 **80 个分片、1084 条阶段记录**，本次与 Git 原文逐行比对相同。阶段记录数不等于独立科学失败数；无 ID 的版本也可按版本/关键词查询。

```bash
python scripts/query_research_failures.py --version V73 --limit 10
python scripts/query_research_failures.py --id V73-F09 --detail --limit 2
python scripts/query_research_failures.py --id V74-H2-F22 --detail --limit 1
python scripts/query_research_failures.py --id V75-F02 --detail --limit 1
python scripts/query_research_failures.py --query coverage --limit 5
```

`--offset` 翻页，`--max-lines` / `--line-offset` 控制长正文。无 `--detail` 只返回目录。历史中的 RUNNING、WAIT_GPU、待办与电源权限都是历史信息；原文路径按当时 `docs/` 基准解释，迁移见[V7.4 路径表](archive/2026-09/v74-0920/PATH_MAP.json)及[归档总目录](archive/README.md)。
