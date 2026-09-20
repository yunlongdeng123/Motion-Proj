# Motion-Proj 协作规则

## 文档职责

- 当前阶段、执行范围、阻塞和下一步只写 [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md)，更新同一份快照，不在文件顶部堆叠历次状态。
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 是按 task/run 查实验的索引；配置、资源、完整结果进入对应报告和 `docs/autoresearch/`。
- [docs/RESEARCH_FAILURES.md](docs/RESEARCH_FAILURES.md) 只做跨版本失败入口。新证据写 `docs/research_failures/entries/ID.md`，同一失败更新原卡；没有新增失败就不改入口或硬造 ID。
- 本文件只放长期协作规则，不放进度、结果、累计次数、任务终态或关机安排。README 只做导航，不复制研究状态。
- 文档迁移保持用户归档目录，修复链接并记录路径映射；历史正文、原始结果和正反案例均保留。历史文档不是当前授权。
- 里程碑按职责更新相关文件，禁止把相同摘要同步粘贴到多本账。修改 failure 卡后运行 `python scripts/build_research_failure_index.py` 更新目录。

## 研究与沟通

- 遵循用户最新要求和 [auto-research_scaling_law.md](auto-research_scaling_law.md)。先确认问题真实且值得研究，再用简单强控制决定方法；不预设机制或指定失败结论。
- 开始前渐进读取当前状态及相关 failure ID，不默认读完整历史。明确工程错误、数据缺口、资源限制与科学否定的区别。
- 实验登记唯一 task/run ID、输入角色、配置、seed、证据路径、资源、结果与 `failure_ledger_refs`；收口记 `failure_ledger_delta`，没有新增写 `none`。
- BUILD 与 QUERY 分开；方法只接合法输入。额外真值/度量锚点、训练重叠、缺输入及未知分母明示。原始与适配系统、诊断与下游结果分开表述。
- 技术报告和 paper 必须有简单 architecture components 图，用模块、箭头和少量标签说明输入、组件、数据流与输出。
- 默认简体中文文档和代码注释。人工 verdict 只由用户或指定评审填写，不代填。
- 实质卡点先查相关一手论文、官方实现或文档，记录实际迁移点与证据边界；同一问题不机械重复检索。

## 代码、Git 与资源

- 远端修改先检查用户工作，局部暂存与备份后上传；保留用户已有改动，不改写共享历史。小步提交并 push 当前分支，使用 Conventional Commits。
- 提交前查看暂存 diff 并运行 `git diff --cached --check`。验证与风险相称，不添加哈希、校验和、指纹或大量无关测试。
- Python 优先 `/root/autodl-tmp/envs/motionproj/bin/python`；CPU 工作按当前 cgroup 配额控制线程，必要时设置 `CUDA_VISIBLE_DEVICES=`。
- 不从旧报告自动恢复任务。电源操作只依据当前会话授权及状态中的适用范围；关机前确认没有训练、评价、数据、渲染或会启动任务的控制器，并保存、推送结果。
- 删除或外移资产先确认当前授权与恢复入口；保留原始输入、关键 checkpoint、固定表面、指标和失败证据。轻量索引不等于完整备份。
