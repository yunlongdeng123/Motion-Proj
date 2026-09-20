# 当前研究状态

更新：2026-09-21。分支：`research/worldsim-v7.4-h2-generative-surface`。本文件是唯一当前快照；历史见[V7.4归档](archive/2026-09/v74-0920/STATUS_PRE_V75.md)，实验按[索引](EXPERIMENTS.md)查阅。

## 当前方向与授权

按用户最新要求，V7.5先定义“WorldSim做得好”，主问题调整为 **What reconstruction state is needed to make counterfactual generative simulation faithful?** 评价干预遵循、不受影响的事实、世界/可见性一致，以及相对于可信参考的策略后果保真度。重放、几何、视频观感和策略表现均不能单独代替该目标。完整定义和架构见[PROBLEM](v75/PROBLEM.md)，参考边界与有限实验见[评价协议](v75/COUNTERFACTUAL_EVALUATION.md)。

单张RTX3090正式推理既有授权保留；任何OOM立即停止，不自动降配置、重试或调研多卡。本轮只完成定义、接口查证及已结束控制的归档，未启动新反事实推理。无训练、旧队列恢复或关机任务。模型与样例资源就绪，下载heartbeat已暂停。

## 已有证据及保留边界

- [单卡基线](v75/BASELINE.md)、[原生开发样例](v75/NATIVE_COHORT.md)和[AV2桥接](v75/AV2_BRIDGE.md)证明公开单视图链路可运行；工程失败[V75-F01](research_failures/entries/V75-F01.md)已绕过，第三方源码未修改。
- [定位](v75/LOCALIZATION.md)、[自然读出](v75/NATURAL_STATE.md)和[可见性窗口](v75/VISIBLE_COHORT.md)保留全部好坏表现、参考排除和普通尺度控制。未建立稳定额外放大、普遍SOTA失效或驾驶危害，不再追加长时搜索。
- 两个[接近任务](v75/APPROACH_CLOSED_LOOP.md)已接通生成RGB→策略→动作→ego/相机→下一段生成。真实输入与GT生成分别门控；中心/尺度/目标LiDAR没有同时恢复动作与执行。固定6日志×3起点的运动跟车窗口18→0，平移、尺度、关联、运动和时长扩展均关闭。
- 形状主候选02678d04在唯一额外seed43中，配对行进差由+2.757m反为−0.324m，平均误差和欠制动差也反向，已关闭，不追加seed44。24642607保留两seed同向的小效应，普通类别先验已改善；承认普通解，不升级为严重危害或复杂主方法必要性。
- [原生DVGT契约与三时刻控制](v75/NATURAL_STATE.md#原生点图契约与有限时间上下文控制)：两次固定官方前向，14视角中13个方向误差改善，严重单帧错位大部分由普通历史观测解释。两例仍未通过事前原生投影筛查，不能把残余直接当生成badcase；不继续长度/坐标搜索。
- 已完成[速度先验控制](autoresearch/worldsim_v75/velocity_prior_control/README.md)：20条既有流CPU回放，0新模型调用。主项平均误差下降但额外制动恶化，同场景两个GT seed也未通过，未准入新生成；全部正反结果保留，不扫描速度/噪声。

这些实验支持接口能力与有限误差边界，**不证明重建普遍成熟，也不证明反事实正确**。DVGT距离读出使用GT尺寸/朝向、LiDAR诊断有额外信息、交通非反应式，原有信息边界继续有效。当前只有单一IDM，没有成熟驾驶多策略排名证据。

## 本轮交付与下一步

`WS-V75-CF-DEFINITION-01 / 20260921-r1`完成质量定义、独立参考层级、实际输入接口核对和单项有限实验设计，0模型调用。旧问题协议迁入[v75历史协议](v75/protocols/PROBLEM_20260920_CLOSED_LOOP.md)，没有改写旧结果。可视化审阅页为本地`outputs/V75_WorldSim_Definition/index.html`，图和页面轻量归档在[定义证据](autoresearch/worldsim_v75/counterfactual_definition/)。

下一项是**有参考支持的对象移除资格检查**，先只审阅已有两个开发任务，最多一个合格目标。必须有支持被显露内容的独立观测、与移除后一致的初帧/条件，以及确实进入模型的重建差异；未满足就记数据/接口缺口。资格满足后才冻结至多六段固定相机的配对生成；有可恢复状态错误后才决定至多三段反馈。具体target、参考初帧和阈值尚未冻结，当前没有待启动队列。不同时铺横移、新视角幅度、多模型或多策略实验。

保留V7.4 [F20](research_failures/entries/V74-H2-F20.md)、[F21](research_failures/entries/V74-H2-F21.md)、[F22](research_failures/entries/V74-H2-F22.md)边界，不恢复失败的TransFuser域基线。人工verdict均null；本轮`failure_ledger_delta:none`，不新增失败卡，不在AGENTS或FAILURES重复状态。
