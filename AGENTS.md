# Motion-Proj 执行约定

## 当前授权：WorldSim V7.4 / 三独立候选研究（2026-09-10）

唯一当前状态是 `docs/RESEARCH_STATUS.md`，主计划是 `docs/WorldSim_V74_Method_Tournament_Plan.md`。
从 V73 最终分支建立 `research/worldsim-v7.4-method-tournament`；V73 科研保持冻结。
P0 文档归档、数据盘清理和现有 FIT/DEV 的 CPU 数据准备已完成；详见 P0 交接报告。用户已于 2026-09-10 有卡开机并授权继续，现进入三独立候选方法实验。
后续按 WEX/RIF/DCS 三候选计划推进，不自动增加第四候选或拼接失败方法。
nuScenes 与 AV2 优先各自 FIT/train、DEV 与 FINAL/test；跨数据集零样本不是要求。
旧 5 个 nuScenes DEV 和 20 个 AV2 确认日志均已曝光，不冒充新 FINAL。

## 工作与资源

- 每次实质卡点先检索相关顶会一手论文、优秀开源官方实现或文档，结合当前接口迁移，记录来源、实际接入点和未证实边界；同一问题不机械重复检索。
- 不加哈希、校验和、指纹；不增加过度校验或门控，不运行大量 smoke/回归测试。Git 自身版本标识用于普通提交追溯。
- 小步提交并及时 push 当前远端分支，不改写 V73 或已共享历史。每个里程碑同步三本总账。
- 资源不足时先完成仍可执行的 CPU/整理工作、保存结果并 push，确认没有训练、评价、数据、渲染或会启动任务的控制器，再执行 shutdown 并提示加卡。不能只看宿主 CPU/内存，读取当前 cgroup 配额。
- 计划的科学淘汰规则仍然有效；研究失败、数据缺失与资源不足分别记录。用户已解除 P0 阶段终点，持续推进至计划裁决/扩展结束或实际资源不可抗力。
- 已授权清理与后续实验无关的大文件。优先删除可重建缓存和退役中间数据；保留原始输入、最终/关键 checkpoint、固定表面、指标、失败证据和重建入口。按路径、类别、大小、原因及恢复方式记录，不能把归档文档当作大文件备份。

## 研究事实与文档

- 先渐进式读取三本总账当前节和相关 failure ID；`docs/archive/`、旧论文和旧配置只作历史依据，不包含当前执行授权。
- 按用户最新要求，V74 详细过程写 `docs/WORLDSIM_V7_4_FAILURES.md`；`docs/RESEARCH_FAILURES.md` 维护 ID 索引与结论并链接详情，保留 V73 历史。两处不能相互矛盾。
- 正式实验记录唯一 task/run ID、输入角色、配置、seed、证据路径、资源和结果。启动登记 `failure_ledger_refs`，收口登记 `failure_ledger_delta`（无新增写 none）。
- BUILD 与 QUERY 分开存储/加载；对象求解器只接 BUILD。FIT 可用本数据集训练监督；DEV/FINAL 真值由评价器持有。保留缺输入、无自有返回与未知分母。
- 技术报告和 paper 必须包含模块、箭头和少量标签构成的简单 architecture components 图。
- 默认简体中文文档和代码注释；只解释必要的意图。文档不留编辑备份；已跟踪文件通过 Git 历史恢复。
- 人工 verdict 只能由用户或指定评审填写，不代填。

## Git 与环境

- 使用 Conventional Commits，一次一个逻辑主题；正文写问题、改动、必要验证、task/run 与证据路径。
- 提交前查看暂存 diff 并执行 `git diff --cached --check`；仅做风险相称的验证。
- Python 优先 `/root/autodl-tmp/envs/motionproj/bin/python`；P0 设置 `CUDA_VISIBLE_DEVICES=`，逐对象处理，线程按 cgroup 配额控制。
