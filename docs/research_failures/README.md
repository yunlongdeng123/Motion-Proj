# 失败资产维护

读取顺序：总入口（约一页）→ BOUNDARIES → 版本/关键词/ID → 一条详情 → 所需 NPZ/指标。历史目录按原出现顺序保留；最新裁决优先于旧阶段状态。

历史小节是证据原文，不是全部重新审核过的分类。`defined_ids` 是标题中明示 ID；`referenced_ids` 是正文引用。主题是检索标签，不是失败层级的自动裁决。无 ID 的运行阶段具有顺序记录号 RFxxxx。

新失败在 `entries/ID.md` 新建一个文件，首个注释采用下面 metadata。查询器会自动读取新文件，无需重写 14k 行历史。阶段运行状态进入 EXPERIMENTS；同一个失败的修复/收口更新同一资产，保留原证据链接，不再复制整篇日志。

```markdown
# V74-H2-Fxx：具体失败边界
<!-- metadata: {"title":"具体失败边界","defined_ids":["V74-H2-Fxx"],"referenced_ids":[],"versions":["V74-H2"],"topics":["engineering"]} -->

状态/分类：工程 / 数值 / 优化 / 表示 / 科学 / 新颖性 / 证据不足；填写其一并说明证据。
命题与触发条件：
最强控制与同信息预算：
实际结果（含正例与负例）：
排除的假设/不再重试的方法族：
仍开放的假设与适用范围：
卡点一手来源 → 迁移位置 → 一次有针对性的修正：
证据：task/run、输入角色、配置、初始化/首次退化/最终资产、责任射线/片、状态和资源成本。
failure_ledger_refs / failure_ledger_delta：
人工 verdict：null
```

每个里程碑同步 RESEARCH_STATUS、EXPERIMENTS 和总入口/相关 ID。不要把数据不足写成科学失败，不把修 bug 写成机制成立。历史相对链接沿原 `docs/` 基准解释；可用原文件路径或 Git 历史定位。大型外部证据只记录保存位置与恢复入口，不声称 Git 轻量索引等于资产备份。
