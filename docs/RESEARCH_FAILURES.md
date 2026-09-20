# SparseDrive 固定队列终态（2026-09-20）

状态：failed；实际已保存前向 0/40。环境/模型错误不计作科学失败；未完成任务不补计。

六相机输入 → 官方 SparseDrive → 保存原生三秒轨迹 → 固定参考评价 → 保存后关机。

结果为开环回放，非闭环事故或几何因果证明；真实路线提示、训练集重叠、重建监督与共同裁剪控制保持明示。既有六日志结果与 Ω 局部几何因果结论不变。人工 verdict=null；failure_ledger_delta=none。

证据：docs/autoresearch/worldsim_simimpact/closeout_20260920/。完整日志与原始结果保留 runs/ 下。

# 当前：用户要求收口与完成后关机（2026-09-20）

原生SparseDrive环境于9月15日准备完成，模型前向尚未开始；scene0004既有模型的60幅六相机配对渲染已完成。9月20日核对远端仍在线、GPU空闲，上次自动关机链未部署。本次按用户明确授权补充一次性固定40前向+评价链，失败不重试、依赖任务跳过，保存终态并shutdown；不扩实验或训练。预计前向不得提前记入统计。详细范围、计算上限和终态入口见docs/WORLDSIM_SIMULATION_CLOSEOUT_20260920.md。

科学结论保持：六日志四模型均有框定位/IoU退化，48预选前车条件中心匹配全部保留；Ω局部几何校正恢复定位，但删除控制也有效。没有普遍phantom造成严重仿真危害的主论文证据。累计72几何、222检测、818策略/PDM、18HUGSIM+4SplatAD反馈闭环不变。人工verdict=null；failure_ledger_delta=none（沿用F20–F22），执行遗漏不伪装为科学失败。以下为历史。

# 当前：新来源真实基线与一次路线控制完成，停止该场景拟合（2026-09-15）

WS-SIM-FRESH-NATIVE-01 / 20260915-r1：有限新日志窗口仅scene-0002达元数据条件，1770原始文件完整。八真实起点＋八路线提示对照共16策略/PDM配对：ADE/FDE 1.667/4.642m→1.422/3.727m，框交叠1/8→0/8；均未通过固定ADE≤1m、FDE≤2m准入，不启动新SplatAD拟合。路线提示是额外GT空间路线信息，真实基线残余不能归因重建。scene-0002是本轮新来源但属于模型训练集，非封存确认。

BridgeSim代码核查为PDM-lite及二维runtime ray LiDAR，另有独立可选3D PointCloudLidar；尚未运行。准备官方原生nuScenes SparseDrive-S stage2，两预热＋八起点已冻结、权重/环境准备中，尚未前向；不把接口准备计为科学结果。累计818策略/PDM、222 CenterPoint、72几何前向、18HUGSIM＋4SplatAD反馈闭环。goal active；人工verdict=null；failure_ledger_delta=V74-H2-F22。见docs/WORLDSIM_SIMULATION_FRESH_BASELINE.md。以下为历史。

# 当前：六日志原生感知矩阵完成，预选前车47/48保留完整匹配（2026-09-15）

WS-SIM-FF-COHORT-PERCEPTION-01 / 20260915-r1：82新增检测＋20复用完成102项。52目标，真实中心52/52、IoU40/52；四模型六/十二图已齐。补回原缺失后，48个元数据前车条件全部中心匹配、47个IoU匹配；没有原可靠目标在两固定置信度和两预算下持续丢失中心匹配。两新候选是0028右侧约15m行人，置信度0.74–0.81、中心误差0.34–0.48m，只有IoU/定位下降，未证明驾驶危害。本轮不做其局部网格扫参/训练。Ω十二图明显改善，正例和新增匹配完整保留。

累计222 CenterPoint；72几何前向、802策略/PDM、18 HUGSIM＋4 SplatAD闭环不变。六日志全部已曝光，每日志一时刻，当前扫描＋真实强度/九帧历史的额外信息控制，非完整传感器或独立确认。有限筛查收口；整体goal active、人工verdict=null；failure_ledger_delta=V74-H2-F21。下一次有意义推进须补独立来源与实际规划接口，不重跑这批或把低杠杆框误差继续细调成Hero。见docs/WORLDSIM_SIMULATION_SIX_LOG_PERCEPTION.md。以下为历史。

# 当前：局部网格修改获得有限感知因果证据，普通删除同样有效（2026-09-15）

WS-SIM-FF-LOCAL-ASSET-01 / r2：两个冻结下游候选、两输入预算，10次CenterPoint。Ω车辆局部网格校正后重新投射完整扫描，中心误差2.073/1.954m→0.086/0.030m，IoU0.347/0.370→0.836/0.804；同面删除也恢复（0.089/0.086m）。支持当前固定真实历史适配器内的局部几何→传感器→感知作用，未证明生成式修复必要性。Pi3X行人距离改善但检测不稳定恢复，关闭本轮径向修复子命题。均未通过预注册超出删除控制的Hero准入。不能把2m中心匹配失配称作车辆完全消失，亦无独立日志/规划/安全证明。

全部CenterPoint累计140次；72几何前向、802策略/PDM、18 HUGSIM＋4 SplatAD闭环不变。见docs/WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md和F20。两候选不继续扫参；下一步若推进应冻结新的时刻/日志及完整传感器信息范围，而非重复当前几何诊断。goal active；人工verdict=null；failure_ledger_delta=V74-H2-F20。以下为历史。

# 当前：完整预算闭环与原生检测完成，局部强度恢复解释主要漏检（2026-09-15）

SplatAD 30001步完成；194 LiDAR／678相机留出、64策略/PDM与2次反馈闭环完成。LiDAR-only平均ADE差−0.014/+0.026m，两闭环无框交叠。冻结nuScenes CenterPoint真实机动车32/32，原生raw/median16/20；工程车真实8/8、原生0/1，局部真实扫描恢复8/8，而median仅恢复强度即7/8、位置恢复仅2/3。保留传感器感知坏例，不晋级纯几何或phantom Hero。四前馈模型两日志18次检测包含DVGT正例，不能主张普遍同样失效。全部CenterPoint130次，累计18 HUGSIM＋4 SplatAD反馈闭环、72几何前向、802策略/PDM。见docs/WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md及F19。

本轮有限工程车诊断收口，不细切ROI或追末帧残余；不启动第二个SplatAD拟合。后续仅复核已有Ω车辆／Pi3X行人检测候选的真实支持，再判断实际局部资产修复是否必要。尚无前馈几何资产修复后的独立下游恢复。goal active；人工verdict=null；failure_ledger_delta=V74-H2-F19。以下为历史。

# 当前：原生RGB＋LiDAR反馈已实际运行，8k试跑差距主要经RGB通道（2026-09-15）

完整预算接续已准备：`run_native_full_evaluation.sh` 仅在拟合成功且第30000步checkpoint存在时运行全留出传感器评价、日志时刻匹配的模态对照及两次反馈闭环；当前尚未启动，不增加执行次数。已有TensorBoard快照读至第24000步，成本/质量复核来自既有日志。0073横穿拖车仅列为后续候选，第二个场景未启动；无新增failure（沿用F18），goal active。

WS-SIM-NATIVE-CLOSEDLOOP-01 / scene0004-step008000-pilot2：8日志时刻×6模态对照，64策略/PDM预测，raw/median各一次4秒实际新位姿传感器闭环。仅LiDAR的平均ADE增量−0.017/+0.015m，RGB-only为+0.288m；RGB通道归因未区分视觉几何、外观或采样时刻。两闭环无标注框交叠，最小间距+0.566/+0.572m；5.17/5.54m终点误差不能全部归为重建，真实输入初始预测本身已有3.07m。当前是8000步接入试跑，不是完整30001步SOTA失效结论；完整拟合仍运行。累计18次HUGSIM＋2次本轮反馈闭环、72几何前向、738策略/PDM预测。自然几何恢复与独立严重危害确认仍缺。goal active；人工verdict=null；failure_ledger_delta=V74-H2-F18。见docs/WORLDSIM_SIMULATION_NATIVE_CLOSED_LOOP_PILOT.md。以下为历史。

# 当前：下游优先判据生效，Pi3X有限因果审计完成，原生传感器拟合中（2026-09-15）

原生留出接口补充：固定第8000步checkpoint的4帧LiDAR和6相机验证已全部完成，raw/median点云并列保存；距离绝对误差中位数0.11–0.22m且仍有长尾。首场景拟合快照10000/30001步；尚无新位姿RGB+LiDAR闭环结果。该接口补证不增加策略/PDM次数，不晋级几何坏例。failure_ledger_delta=none（沿用F17研究结论），人工verdict=null。

用户明确改为先发现仿真退化、再做局部几何恢复；Early/Chamfer仅作诊断。23项空间传感器定位＋7项真实网格平面/删除对照完成。Pi3X六图右区传感器恢复有信号，十二图不重复；有2734点支持的局部路面网格修复，两者都未消除接触，不晋级Hero动机，不继续细切/扫参。既有48组下游重排：7个新增名义接触均在0061，2个终点变化>1m，无新增停止或超过2m/s²的额外制动。累计18次HUGSIM闭环、72几何前向、674策略/PDM执行。官方SplatAD完整0004/0061数据3449文件已提取，CUDA已编译；0004已稳定拟合、阶段checkpoint落盘，完整新位姿RGB+LiDAR闭环尚未完成。原始与median两条官方点云输出均保留。整体goal active；人工verdict=null；failure_ledger_delta=V74-H2-F17。见docs/WORLDSIM_SIMULATION_DOWNSTREAM_FIRST.md及WORLDSIM_SIMULATION_PI3X_CAUSAL_AUDIT.md。以下为历史。

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
