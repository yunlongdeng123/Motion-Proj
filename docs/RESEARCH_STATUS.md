# 当前研究状态

更新：2026-10-17T20:14:36+08:00。用户最新要求暂停R15训练、保留checkpoint，查明慢速与a6/a37分层问题并推送v82。**训练保持暂停，禁止自动续训或关机。** 无subagent/新chat。[诊断与严重性表](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/paused_diagnostic_20261017/README.md)、[HTML及分层图](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/paused_diagnostic_20261017/report.html)、[实验索引](EXPERIMENTS.md)。

run `backend_restore_r15`：controller8244、flock49808、train51627已核实身份后退出。最后日志4100，完整可恢复点4000；独立保留在服务器 `backend_restore_r15/paused_checkpoint_004000_20261017/checkpoint.pt`，模型/13组非空Adam状态/RNG已读回。4000后未落盘部分不冒称已保存。main/latest、pre_owner3000及历史输入/失败保留。相关心跳b-c-15原本PAUSED；D/PAUSE_REQUESTED.json和execution_policy禁止续训，controller增加启动暂停保护并实际验证未启动子worker。旧controller仍有初始step2断言，恢复前需要另行处理入口，不可盲重开。

生产后端 `implementation_r15_train_v3` 的表示、loss、配置和CUDA ABI未改。新增诊断副本参数更新0，计时用lr=0 Adam。同步单帧4k前后向19.92s，关阴影1.68s；Python/Torch阴影循环有大量内核提交/同步，分块、重计算、5方向与Actor增密放大开销。FP32协方差/求逆失稳且阴影明显依赖chunk，FP64参考复现错误；a6/a37关阴影删除后静态车形残影仍在，3k审计split/owner转移均0，同表面深度竞争不覆盖大部分锚帧。完整49分层/46训练贡献/6锚帧反事实已完成，质量未通过。详细数值仅进入专门报告，不重复堆叠账本。

45k及最终49×65编辑未完成。本次明确暂停取代此前连续训完与交付后关机安排。下一步先修阴影数值及积分调度，再分别验证资产碎片/静态外观/审计时机；用户未授权恢复训练。报告里的提速候选没有应用，不能宣称修复完成。

human_verdict=null，problem_solved=null，failure_ledger_refs=[V82-F01,V82-F05,V82-F06]，failure_ledger_delta=[V82-F05,V82-F06]。原始BUILD、固定GT、冻结mask、已见开发QUERY边界、旧模型及原用户dirty工作保留。未关机。
