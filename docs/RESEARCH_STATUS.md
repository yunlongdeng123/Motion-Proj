# 当前：V8.1 CPU准备中（2026-09-13）

WS-V81-CPU-01；主方向稀疏视角×低纹理 failure discovery。r1完成12日志、14场景、28窗口、2520 ROI、589几何候选与可视化；14个高重叠候选复核暴露栅栏/多层前景混杂，因此按原metadata顺序补齐27个落盘日志的r2，不放宽筛选标准。

SOTA推理=0，H1–H5未检验，V8.2暂NO_GO（未有证据，不是科学否定）。官方代码/权重和隔离环境准备中。failure_ledger_refs=V74-H2-F11/F09/F10、V74-F01；failure_ledger_delta=V81-F01/F02。人工verdict=null。用户要求CPU完成后停下提示开GPU。

入口：[V8.1执行](WORLDSIM_V8_1_PLAN.md)、[方法审计](WORLDSIM_V8_1_METHOD_AUDIT.md)、[CPU报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md)。以下为冻结历史。
# V7.4 已收尾：CLOSED_WITHOUT_VALIDATED_MAIN_METHOD（2026-09-13）

最终精简：GitHub 实际下载 ZIP 约 **22.2 MB**；同口径本地打包105.25→21.45 MB，减少约80%，跟踪文件展开约53.6 MB（原375.9 MB）；两批共307个资产、原文件322.49 MB已归档。全部代码/配置/Markdown、失败查询索引和V7.4报告图保留。旧下载包不会自动变小，需重新下载当前分支；完整 Git 历史保持，未强推改写。

仓库资产清理完成：241 个大表/渲染场景/编译包已从当前分支移出，原值完整存于仓库外归档并有本地副本；核心代码、文档、报告内嵌图和失败查询索引保留，原始 runs 不删除。见 [归档清单与恢复方法](archives/worldsim_v74_closeout_20260913/README.md)。本次没有新实验；failure_ledger_delta 仍为 F11（追加资产位置），F08–F10 数值不变。

用户明确结束 V7.4。H1 NO_SURVIVOR 保持；H2 当前 ordered 实现关闭，GPU P1/P1.5/P1.6 完成但没有建立经验证的论文主方法。没有新训练、新方法或新科学否定实验。整体 witness-native 假说仍开放，开放问题不代表继续执行授权。

[收尾报告与流程图](WORLDSIM_V7_4_CLOSEOUT.md)；[最终短卡 F11](research_failures/entries/V74-H2-F11.md)；[更新后的 scaling law](../auto-research_scaling_law.md)。failure_ledger_refs=V74-H2-F08/F09/F10；failure_ledger_delta=V74-H2-F11。规则转为 failure discovery → 简单强控制 → 有效主方法 → paper story → 有增量的次要方法。仓库资产清理已完成；既有 runs 不删除。以下全部是历史阶段记录。

# 当前：P1.6 已完成，普通控制仅部分解释（2026-09-13）

P1.6 诊断、普通BUILD物理控制、同父残差诊断及架构/三张科学图已完成，无训练。旧70条失败在控制闭环对应步骤全部HIT，但A自由空间变差、C2出现3条新共享支撑HIT→EARLY；结论PARTIAL_EXPLANATION_NOT_JOINT_SUCCESS。关闭薄结构法向漂移要求新机制的依据，整体witness假说开放；不新立项。主计算23.43s/峰值分配0.0581GiB，资源充足。

[最终报告](WORLDSIM_V7_4_H2_P16_DIAGNOSTIC_REPORT.md)；[短卡F10](research_failures/entries/V74-H2-F10.md)；[资产索引](autoresearch/worldsim_v74_h2/p16/README.md)。failure_ledger_refs=F08/F09；failure_ledger_delta=F10；用户评分仍3/6 WeakReject。代码和文档push、证据备份且无活动任务后依既有授权shutdown。以下为历史记录。

本轮对照工程修正：r1前向batch12→r2原设置8+4，首次中心输出差6.59微米→0；主要结论不变，A free_m主结果以r2 0.03010m为准。r1资产与历史提交保留。

# V74-H2 P1.5 已完成：当前 ordered 实现关闭

2026-09-13：P1.5 存量失败审计完成，未训练新模型。A 薄结构 52/52 首次 HIT→EARLY 保持同一首面；连续位置漂移经斜入射放大已解释退化。C2 有 12 条短暂同类退化后恢复，不能称全程稳定。完整解释见 [P1.5 报告与架构](WORLDSIM_V7_4_H2_P15_FORENSICS_REPORT.md)，渐进入口 [F09](research_failures/entries/V74-H2-F09.md)。

当前 A/ordered 实现关闭，witness-native reconstruction operator 假设开放；不立项 ownership-stable 方法，不启 B/C 或真实训练。人工评分沿用用户复审约 3/6 Weak Reject；没有新的科学/新颖性裁决。failure_ledger_delta=F08 人工复审 + F09 几何失效边界。代码、文档推送且无活动任务后按用户授权关机；以下为历史阶段记录。

# P1.5 分析完成，报告整理中

2026-09-13：存量 216 状态查询比较、70 条首次退化、同父状态教师插值完成；无训练。A 薄结构 52/52 是同片连续前移，C2 12 条短暂同类退化后恢复。当前 ordered 实现关闭；未满足新方法立项条件。failure_ledger_delta=V74-H2-F09；[证据卡](research_failures/entries/V74-H2-F09.md)。以下是历史里程碑。

# V74-H2 P1.5：存量失败轨迹审计中（2026-09-13）

用户已授权在当前开机环境继续 P1.5；仅分析已保存 A/C2/教师轨迹，必要时重算同状态的冻结教师动作，不训练模型。当前 A 实现仍关闭；人工复审约 3/6 Weak Reject。witness-native operator 假设开放，ordered-chain 和 novel representation 均无必要性证据。

[审计定义](WORLDSIM_V7_4_H2_P15_FORENSICS_PLAN.md)；[人工复审 F08](research_failures/entries/V74-H2-F08.md)。failure_ledger_delta=F08 人工复审与 claim 边界。P2/B/C 均未进入。以下为历史状态。

# V74-H2 当前状态：STOP_IMPLEMENTATION_OPTIMIZATION

2026-09-12，执行计划1.1后停止当前A训练实现。资源充足；12任务自身八步几何闭环未满足最小能力，一次固定DAgger修正未改善A最佳闭环目标。按计划6.4/16停止该实现，不进入P2。

- [GPU执行报告与架构](WORLDSIM_V7_4_H2_GPU_P1_REPORT.md)；[机器判定](autoresearch/worldsim_v74_h2/gpu_p1/decision.json)；[全部run目录](autoresearch/worldsim_v74_h2/gpu_p1/run_registry.json)。
- 完成：80+20合成硬教师、80例C1普通块搜索、80例C3同池束搜索、80例共同池贪心、A/C2八步训练、12任务过拟合和一次固定修正。
- 尚未完成：完整A/C2的80例必要性比较、C4/C5、真实两域训练/DEV、独立FINAL、场景应用。因学习能力不足未进入，不伪造结果。
- A科学/新颖性verdict=null，人工verdict=null；不宣布witness-native假设无效。B/C未启用，H1 NO_SURVIVOR冻结。
- [CPU数据与归档准备](WORLDSIM_V7_4_H2_CPU_HANDOFF.md)继续可用。width32之外能力未测，当前结论只覆盖本实现。

failure_ledger_delta: F06合成角色区间修复，F07初始GPU实现修复，F08一次优化修正未解除最小闭环能力限制。遵循[scaling law](../auto-research_scaling_law.md)。

资源收口：[实际进程记录](autoresearch/worldsim_v74_h2/gpu_p1/resource_closeout.json)。无研究worker，未配置自动恢复；该记录是关机前的资源快照。数据盘剩余约183.75GiB。

关机安排（2026-09-12用户明确授权）：研究已按上述边界收口，CPU/GPU无研究任务、无研究cron或tmux。当前文档提交推送后立即执行 `shutdown -h now`，不自动重启研究。见[关机前记录](autoresearch/worldsim_v74_h2/gpu_p1/shutdown_preparation.json)，命令结果保存在本地任务对话。

追加精简完成：两批合计移出 307 个批量/生成资产，原始文件共 322.49 MB；两个完整包均已保存远端和本地副本。V7.4 核心图保留，旧论文与 V7.3 非架构图改为按需恢复。 [追加归档](archives/worldsim_v74_closeout_20260913/LEGACY_MEDIA.md)。
