# 当前研究状态

更新：2026-11-26（Asia/Singapore）。

当前范围 `SCENE065-POSTPROCESSING-COMPARE-20261126`：[实际运行、资源与结果](autoresearch/scene065_postprocessing_20261126/results_summary.json)，[技术判断](autoresearch/scene065_postprocessing_20261126/review.json)。冻结高斯和模型权重；同一原始输入、同一mask、同一置灰输入，078—098与132—152各21帧，480×320。已完成修正版ProPainter无调色/调色控制，以及VACE1.3B每段25步冻结推理；新增训练与高斯更新均0。保留修正版 ProPainter；本次 VACE 1.3B 配置不晋升 正式对照使用共同置灰输入和同一补全范围。ProPainter减少车形残留与硬矩形接缝，但道路仍有涂抹和大面积暗区；VACE在087段生成带红色图案的新物体，在141段重新生成一辆黑色轿车，违背删除后空路面的目标。本轮停止追加VACE适配，不将两种后端标为最终画质通过。完整视频、关键帧、原始PNG及as-run源码已取回，更新原实验报告；失败接线与原始RGB泄漏试跑单列。本轮不新增控制通道、训练或蒸馏。完成依赖核验后按长期规则释放GPU。failure_ledger_refs=[V82-F06]，failure_ledger_delta=none。

三维重建历史：[R7—R9资产与证据](autoresearch/scene065_teacher_gaussian_rebuild_20261125/README.md)保留；其2800步场与本次11月16日handoff输入分别记账，历史human verdict不变。

二维教师独立范围 `ROAD-2D-TEACHER-PROMPT-20261120`：[五seed提示词对照与组件](autoresearch/road_2d_teacher_prompt_20261120/publication.html)完成。用户已采纳整帧1152×768＋注意力LoRA T0为后续默认；本轮5seed×2prompt×6任务正式60次，另4次真实GPU smoke，4进程并行，训练与三维修改0。合成两底图等权ΔE76 2.239597→2.242254、共同频谱偏差1.201894→1.141373，但改善配对仅7/20与8/20，seed波动未降低；新prompt不晋升，不再追加调词。12次旧1152三图重放和实际W/K门通过，60原始PNG及全部对照资产已取回核验；真实隐藏道路无GT。保持历史活动提示词，原CPU/GPU报告及其他三维任务保留。method human_verdict=用户采纳当前审阅样例；prompt human_verdict/problem_solved=null。实例按用户明确要求保持开机。

[上一共享分支状态快照](autoresearch/scene065_teacher_gaussian_rebuild_20261125/prior_upstream_status_20261125.md)保留完整历史；历史阶段计划不替代本段当前范围。
