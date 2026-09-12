# Motion-Proj 执行约定

## 当前授权：WorldSim V7.4 下半场 / GPU P1（2026-09-12）

用户已开启GPU并修订主计划为 `docs/WorldSim_V74_Second_Half_Generative_Surface_Plan.md` v1.1。当前执行该计划，尽可能把有益的几何求交、事件教师和训练批量放到GPU。`docs/RESEARCH_STATUS.md` 为当前状态；历史WAIT_GPU已被本次授权替代。
方案制定与实质卡点重规划遵循 `auto-research_scaling_law.md`。唯一主命题是 observation witness 决定共享表面存在/支撑域/位置，真实首面归属由硬几何产生；不能靠重命名二维片或加损失宣称创新。
分支 `research/worldsim-v7.4-h2-generative-surface`；V73和H1 NO_SURVIVOR冻结。P1完整机制/同信息强控制通过后才进入P2真实训练；B有条件备用，C暂缓。两域各自训练/测试，不要求跨域零样本，旧DEV不是独立FINAL。

## 工作与资源

- 每次实质卡点先检索相关顶会一手论文、优秀开源官方实现或文档，结合当前接口迁移，记录来源、实际接入点和未证实边界；同一问题不机械重复检索。
- 不加哈希、校验和、指纹；不增加过度校验或门控，不运行大量 smoke/回归测试。Git 自身版本标识用于普通提交追溯。
- 小步提交并及时 push 当前远端分支，不改写 V73 或已共享历史。每个里程碑同步三本总账。
- 资源不足时先完成仍可执行的 CPU/整理工作、保存结果并 push，确认没有训练、评价、数据、渲染或会启动任务的控制器，再执行 shutdown 并提示加卡。不能只看宿主 CPU/内存，读取当前 cgroup 配额。
- 计划的科学淘汰规则仍然有效；研究失败、数据缺失与资源不足分别记录。本次用户授权继续GPU研究，按计划的证据和停止规则推进。H1 科学冻结不阻止用户授权的 H2 新抽象。
- 已授权清理与后续实验无关的大文件。优先删除可重建缓存和退役中间数据；保留原始输入、最终/关键 checkpoint、固定表面、指标、失败证据和重建入口。按路径、类别、大小、原因及恢复方式记录，不能把归档文档当作大文件备份。

## 研究事实与文档

- 先渐进式读取三本总账当前节和相关 failure ID；`docs/archive/`、旧论文和旧配置只作历史依据，不包含当前执行授权。
- `docs/RESEARCH_FAILURES.md` 只维护短入口；先读 `docs/research_failures/BOUNDARIES.md`，再用 `scripts/query_research_failures.py --id ID --detail --limit 1` 查阅。新失败写 `docs/research_failures/entries/ID.md`，H1 细节仍在 `docs/WORLDSIM_V7_4_FAILURES.md`，不把运行日志继续堆进总入口。
- 正式实验记录唯一 task/run ID、输入角色、配置、seed、证据路径、资源和结果。启动登记 `failure_ledger_refs`，收口登记 `failure_ledger_delta`（无新增写 none）。
- BUILD 与 QUERY 分开存储/加载；对象求解器只接 BUILD。FIT 可用本数据集训练监督；DEV/FINAL 真值由评价器持有。保留缺输入、无自有返回与未知分母。
- 技术报告和 paper 必须包含模块、箭头和少量标签构成的简单 architecture components 图。
- 默认简体中文文档和代码注释；只解释必要的意图。文档不留编辑备份；已跟踪文件通过 Git 历史恢复。
- 人工 verdict 只能由用户或指定评审填写，不代填。

## Git 与环境

- 使用 Conventional Commits，一次一个逻辑主题；正文写问题、改动、必要验证、task/run 与证据路径。
- 提交前查看暂存 diff 并执行 `git diff --cached --check`；仅做风险相称的验证。
- Python 优先 `/root/autodl-tmp/envs/motionproj/bin/python`；P0 设置 `CUDA_VISIBLE_DEVICES=`，逐对象处理，线程按 cgroup 配额控制。
