# 最新用户授权：V8.1 GPU阶段（2026-09-13）

2×RTX3090已就绪；DVGT/GPU0、VGGT/GPU1独立并行推理，继续科学发现与可视化。无卡阶段停止要求已由本次开卡授权接续。V8.2不开发；不继承V74关机指令。当前task WS-V81-GPU-P2-01。

# 当前：V8.1 稀疏视角 × 低纹理失效发现（2026-09-13）

用户新授权执行 docs/WorldSim_V81_SparseView_LowTexture_Failure_Discovery_Plan.md。只做科学发现、badcase/goodcase与可视化对比；先完成所有可行的无卡工作，停止并提示用户开GPU。V8.2才研究方法；不重开V7.4 family。当前分支 research/worldsim-v8.1-sparseview-lowtexture。

首批task WS-V81-CPU-01；所有自然数据为DISCOVERY。raw VGGT不代表VGGD，代码未开放方法只记假说。实际GPU输出之前不填写模型failure、rendering指标或human verdict。GPU接续见 docs/WORLDSIM_V8_1_CPU_HANDOFF.md；没有自动恢复器。V7.4收尾关机记录只作为历史，不代表本轮新的电源指令。

以下为继承的工作规则与V7.4历史；与本轮用户指令冲突时按当前V8.1授权执行。
# Motion-Proj 执行约定

## 当前状态：V7.4 已收尾（2026-09-13）

用户已结束 V7.4；H1 NO_SURVIVOR，H2 当前 ordered 实现关闭，未形成经过验证的论文主方法。P1.5/P1.6 完成，整体 witness-native 假说未被全面证伪。本轮只更新失败资产、规划规则与清理当前分支大文件，不启新研究。最终状态见 docs/WORLDSIM_V7_4_CLOSEOUT.md 与 V74-H2-F11。

后续方案与卡点重规划遵循 auto-research_scaling_law.md，尤其第33条：先 failure discovery 与简单强控制，再找有效办法；禁止先有 fancy mechanism 再不断补必要性理由。有效主方法出现后立即整理 paper story，之后才研究有增量的次要方法。不预设下一主方法。

分支 research/worldsim-v7.4-h2-generative-surface；只清理此分支当前树，V73/H1 与已共享历史保持。外移大资产先保留仓库外完整归档和恢复索引，核心代码、文档、失败证据可追溯。推送并确认没有活动任务后沿用 shutdown 授权。

## 工作与资源

- 每次实质卡点先检索相关顶会一手论文、优秀开源官方实现或文档，结合当前接口迁移，记录来源、实际接入点和未证实边界；同一问题不机械重复检索。
- 不加哈希、校验和、指纹；不增加过度校验或门控，不运行大量 smoke/回归测试。Git 自身版本标识用于普通提交追溯。
- 小步提交并及时 push 当前远端分支，不改写 V73 或已共享历史。每个里程碑同步三本总账。
- 资源不足时先完成仍可执行的 CPU/整理工作、保存结果并 push，确认没有训练、评价、数据、渲染或会启动任务的控制器，再执行 shutdown 并提示加卡。不能只看宿主 CPU/内存，读取当前 cgroup 配额。
- 计划的科学淘汰规则仍然有效；研究失败、数据缺失与资源不足分别记录。V7.4 已结束；旧计划中的待办不代表继续研究授权。
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

## 收口后电源授权（2026-09-12）

用户明确要求V74收口后shutdown AutoDL，条件为没有正在运行的任务、代码push且文档更新完毕。当前研究已按优化限制收口，条件核实后执行关机；后续需新的开机/研究授权，不设自动恢复器。
