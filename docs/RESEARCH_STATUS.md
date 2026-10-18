# 当前研究状态

更新：2026-10-18。当前 task=WS-V82-R15-REPAIR-4K-20261018，保留 WS-V82-R15-IDENTITY-COMPARISON-20261018 后续；failure_ledger_refs=[V82-F06]，failure_ledger_delta=none。用户最新授权：修复 R15 并从相同原始测量重新进行有界 4K。无 subagent，无 R16/R17；原暂停 45K 不恢复。[组件、修复与实验合同](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/index.html)。

修复入口已实现：Actor 不可重置来源锚点、累计0.25m位移、真实框+0.10m容许域（保留初始异常种子的包络）、相对初始σ上限1.5倍；每次Actor增密2%、总增密容量4倍；当前全场景正/已知负/实测道路贡献审计，撤销被反证的确定状态，充分重复实测3D才改owner；旧source tag不再拦截候选，真实道路和过大支撑否决整点转车；审计在1500/2500/3500，避开3000 opacity reset。已验证FP64因子阴影接入模型和恢复入口。

26项针对性CPU测试通过，真实49帧/43Actor/771821种子输入预检通过。旧CPU reference集成套件12/16通过，其余4项在未改动R15上相同复现；真实GPU自检尚未运行，不冒称全套通过。新远端队列PID20154等待旧5训练、旧5渲染和冻结三值投影完成，再按GPU自检→从头4000步→49帧五身份评估顺序运行。原30K学习率时间表、原三值mask与loss保留；旧459点池补入不重放，协议差异已登记。旧5最近读取14400/30000，仍非最终基线。

待验证：新旧同一a5/a6/a18/a26/a37的框外率、源点漂移、点数、三值状态、owner实际转移、车体保留、独立/删除RGB与alpha。a6原始碎片及a37静态外观残影未获得质量通过结论；a37道路上车形不授权把道路点全转Actor。组合修复不能归因到单项。所有49帧属于同一重建窗口，无独立编辑GT，human_verdict=null，problem_solved=null。

历史旧3和R15冻结4K身份渲染已完成；旧5完整30K、冻结三值投影后续仍在同一工作范围。[既有三值交互与谱系](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/tristate/index.html)。原GN11 30K、历史旧3和R15 4K检查点全部保留，用户其他dirty文件不变。

GPU保持运行以完成已授权的新旧任务。必须新修复4K及评估、旧对照、冻结投影、报告保存和v82推送均完成，并确认所有worker和会启动任务的controller结束，才按本次授权关闭AutoDL。旧队列完成不是关机条件。
