# 当前：六日志有限补证完成，普遍phantom严重危害主张降级（2026-09-15）

WS-SIM-INTERACTION-01/20260915-r1及COVERAGE-01：48前向、96扫描、390+59组官方策略/PDM执行。唯一新增名义接触在0061且对象为路锥，真实LiDAR基线仅1.6cm余量。删除FOV外缺失束8/8交叠、FOV内0/8；只注入路锥错误0/8，修复路锥仍7/8。主要差距为观测范围；恢复全部缺失后的2个Pi3X残余保留，但仅一个域适配/近边界来源。降级普遍phantom→严重事故，不以更多极端诊断或训练维持。累计18次HUGSIM闭环、72几何前向、644策略/车辆执行；PDM不是新位姿传感器闭环。整体目标active，人工verdict=null，failure_ledger_delta=V74-H2-F16。见docs/WORLDSIM_SIMULATION_INTERACTION_FINDINGS.md。以下为历史。

# 当前：LiDAR策略和官方车辆模型实测完成，近车交互批次推进中（2026-09-15）

WS-SIM-LIDAR-01：195组官方TransFuser及195组PDM车辆执行，nuScenes适配、非反应式4秒跟踪，不混称传感器闭环。三场景无新增演员交叠；全局BUILD尺度后24个完整扫描执行ADE变化−0.0738至+0.0949m，末端最大变化1.1398m。公交车82/84早交案例未新增碰撞且ADE改善，保留反例。GPU校准发现HUGSIM RGB+ED+S深度未除alpha，旧raw深度诊断需更正，LTF闭环不受该通道影响。6不同日志近车窗口已按真实状态先冻结后运行，不按模型误差挑选；目标active，人工verdict=null，failure_ledger_delta=V74-H2-F15。见docs/WORLDSIM_SIMULATION_LIDAR_IMPACT.md。以下为历史。

# 当前：原生闭环与碰撞采样影响已实测，研究继续（2026-09-15）

WS-SIM-IMPACT-01/20260915-r1：18次HUGSIM/LTF闭环、24次四官方模型推理。LTF无LiDAR输入依赖实测为0；固定控制器迭代预算的同轨迹采样对照中，0013可由碰撞终止变为完成，0041无变化。证明碰撞表示采样影响，不等于自然前馈phantom危害。48组替换对终止点覆盖为0，阴性不可解释；帧对应/标定待解。详见docs/WORLDSIM_SIMULATION_IMPACT.md与V74-H2-F14。目标active，不以工程接通收口；人工verdict=null，failure_ledger_delta=V74-H2-F14。以下为历史。

# 当前：实际端到端仿真影响研究已启动（2026-09-15）

用户授权自主接通官方端到端仿真，判断重建误差是否实质影响传感器、轨迹和碰撞；若首回波无影响，转查其他前馈几何/物理缺陷。WS-SIM-IMPACT-01/20260915-r1：已冻结HUGSIM官方三个普通nuScenes场景0013/0038/0041及官方LTF seed0，正在安装独立运行环境、下载公开资产。代码审查确认HUGSIM-LTF无LiDAR首回波输入，静态碰撞依赖语义/透明度筛选后的GS中心点数；WorldEngine当前MTGS也未实现LiDAR渲染。尚无闭环结果，不将接口审查算作完成。F13反例保持，旧H2不重启，封存集不打开；人工verdict=null，failure_ledger_delta=pending。详见 docs/WORLDSIM_SIMULATION_IMPACT.md。以下为历史。

# 当前：官方模型首回波主图实验完成（2026-09-15）

WS-V74-MAINFIG-01/20260915-first-return-r1 与 SECONDARY-01/20260915-secondary-r1：首批 VGGT、用户指定 Ω 原始512镜像、DVGT-1、Pi3X共48前向完成；DVGT-2/DGGT共24前向完成，DGGT108张原生输入视图Gaussian渲染；NKSR/NoKSR各34原生网格、33可评价对象，41输入不足保持未定义。6场景/5已曝光DEV日志/75对象，52有QUERY，11886束；无新独立日志、无训练、无封存集、无闭环/false-safe结论、无关机或自动调度。

共同低位置回波在四模型主协议下早交0.413–0.666m；它位于框底附近，不能标成真实车身。旧0.323m车门射线已被四个首批官方模型修复。共同6输出图控制中DVGT-1/Pi3X可同时改善Hit/Early；更多输出面带来的集合效应不证明内在必然trade-off。DGGT全部12窗口深度/姿态与VGGT完全相同，不重复计为独立基础几何失败。NKSR白车111/752 Early，其中61有后方正确交点；NoKSR80/752，其中38有后方正确交点。结论收窄为表面重建与物理首回波一致性缺口，尚未证明SOTA普遍凭空生成假几何。10张图各PNG/PDF/SVG、图册、中文报告与英文图注完成。

人工verdict=null；failure_ledger_delta=V74-H2-F13。详见主图报告与F13。以下为历史记录。

# 当前里程碑：三模型36项完成，Omega512按用户指定来源加入（2026-09-15）

`WS-V74-MAINFIG-01 / 20260915-first-return-r1`：官方VGGT、DVGT-1、Pi3X完成6场景×6/12图推理；单RTX3090。原生输出保存，普通尺度/标定/置信度和共同6输出图控制完成或汇总中。用户提供Omega512公开镜像并授权下载使用，替代本轮未获访问的416重现版，版本与来源必须明示。75对象全分母、52有QUERY、23未定义、11886射线。无训练、无封存集、无闭环/false-safe结论、无关机或调度。详见主图实验报告；人工verdict=null；failure_ledger_delta=pending。以下为历史。

# 当前：V7.4 官方模型首回波与主图实验（2026-09-15）

用户授权单张RTX3090。task `WS-V74-MAINFIG-01 / 20260915-first-return-r1`；首批仅官方VGGT、VGGT-Ω 416 reproduction、DVGT-1、Pi3X。固定旧DEV 6场景/5日志/75对象，BUILD sample3六环视与sample3+4十二视图配对，QUERY sample2/5独立评价。官方Omega权重401，等待用户配置授权；其余继续。无训练、无封存集、无关机或自动调度。原始模型输出、已知标定/BUILD定尺度额外信息诊断与网格首交读出分开记录。Early不等于false-safe；闭环未测。旧白车仅代表微调VGGT DPT＋LiDAR融合适配系统。人工verdict=null。failure_ledger_refs=V73-F03/F09、V74-F06、V74-H2-F12；delta=pending。以下为历史。

# 当前：V7.4-H2 首次回波重新取证完成（2026-09-15）

已回到 `research/worldsim-v7.4-h2-generative-surface`，V8.1 分支保留。task `WS-V74-H2-REDISCOVERY-01 / 20260915-cpu-r1`，状态 **CPU_DONE_WAIT_GPU**。恢复7个旧案例、8364束双精度回放、3对象×7系统配对、3个RGB上下文和15张图；0新模型推理、0训练。白色轿车两个留出时刻Early59/557、18/195，0.5m仍32/752；旧VGGT-native实际为微调DPT＋LiDAR融合，不能代表官方VGGT。旧20日志r7−native融合15/20同增Hit/Early，12/20四项同增；强r6对照完整保留。

官方VGGT/Ω、Pi3X、MapAnything、DVGT、DGGT的共同失效仍待原生输出和尺度/读出控制；Early不等于false-safe。只准备有限GPU验证，不恢复已关闭H2训练或旧调度。旧20日志已曝光，10日志reserve质量未读；人工verdict=null。failure_ledger_refs=V73-F02/F03/F04/F09、V74-F06、V74-H2-F11；delta=V74-H2-F12。当前不继承历史shutdown授权，未关机、无自动续跑。

[本轮报告与组件图](WORLDSIM_V7_4_H2_FIRST_RETURN_REDISCOVERY.md)；[GPU接续](WORLDSIM_V7_4_H2_REDISCOVERY_GPU_HANDOFF.md)；[F12](research_failures/entries/V74-H2-F12.md)。以下为历史。

# 当前：V7.4-H2 存量 badcase 重新取证（2026-09-15）

用户已授权回到 `research/worldsim-v7.4-h2-generative-surface`，当前无卡。task `WS-V74-H2-REDISCOVERY-01` / `20260915-cpu-r1`：追查覆盖与首次回波冲突的旧真实案例，审计 VGGT 与后继方法的实测边界。只执行存量 CPU 分析、可视化与后续实验准备，不自动恢复已关闭 H2 学习器或旧调度，不继承历史关机授权。V8.1 独立分支保留。旧20日志确认现为已曝光资料，10日志新 reserve 继续封存。人工 verdict=null。以下是历史。

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
