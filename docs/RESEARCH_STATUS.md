# 当前研究状态

更新：2026-10-17T09:10:39.043881+08:00。最新用户要求修正后自查并重启训练。当前run `backend_restore_r15`，代码 `implementation_r15_train_v3`；controller.py PID8244，train PID变化，须核对cmdline。旧B/C、R14不恢复。禁止subagent/新chat。[R15组件、继承差异与证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/README.md)，[实验索引](EXPERIMENTS.md)。

真实训练最近记录150/45000，G30000/M10000/J5000；每1000步释放GPU锁，他人优先，共享环境不改。原始静态608434+Actor163387，不读GN11学习权重；恢复SH3、1024立方环境图、原hybrid全图/静态/动态/天空RGB、100倍可见mask、OBB软边界、原优化调度/拓扑与guarded SGCM。道路无影参考59423/184019=32.29%，其余有RGB合成监督，阴影由实际编辑后高斯几何/位姿/光照计算。

22 CPU和真实三阶段GPU自查通过；原EnvLight输出maxerror0，抽检124596无干净参考道路有颜色梯度。step1及调度修复后step2的非空Adam/模型/重渲染实际恢复通过。首段提前增密错误已保留200..249实际更新证据，从尚未增密的step1恢复，额外199..248已耗另记；历史成本25049..25196。核心模型前向未改；新调度首600末14900，与SGCM offset50不重叠。

初始前后向约25h，增密/teacher/Adam/审计/IO/49编辑另计，真实ETA须按metrics更新。心跳b-c-15每10min，稳定再30/60；只阶段/失败/异常需操作通知。1k预览；3k/9k/30k/40k根代理必须看预览/证据并放行，不能仅平均误差或alpha下降称⑥成功。3k是opacity reset节点，需结合4k。完整49x65动态/7车delete及旧三连续编辑/阴影反事实尚未完成，当前质量未验收。

正式delete仅5/6/12/18/25/26/37，旧0/1/2->5/26/18；另外4无旧独立资产，仅历史场景外观。静态无时间自由度、待定不是背景，GT固定、QUERY为已看过开发集。完整训练/49复核/报告源码交付且所有本任务、其他GPU及锁队列空闲后才按授权AutoDL关机；暂时空闲不关。

human_verdict=null，problem_solved=null，failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。原始输入、旧模型、历史失败和用户dirty工作保留。
