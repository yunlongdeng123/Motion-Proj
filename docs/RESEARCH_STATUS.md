# 当前研究状态

更新：2026-09-27，v77。`WS-V77-DELETE-REPAIR-20260927/r1`出现明确退化，已按用户“差则记录failure并回退”的要求停止，恢复上一轮DriveEditor完整DELETE方案作为默认。没有正在运行的生成任务；不自动重启该精确轮廓输入实验。

## 当前默认

`configs/worldsim_v77/delete_pipeline_current.json`指向`WS-V77-DELETE-FULL-20260927/r1`：scene_0230/actor22、scene_0255/actor25，原DriveEditor mask配置、冻结Ω背景、原GLB。旧BUILD/QUERY入口与1a2d5189相同，原资产未替换。回退保留旧版再生车辆、模糊与Ω错误遮挡的已知缺陷，不表示高保真通过。V7.6关闭状态不变。

## 本轮范围与结果入口

两旧scene A/B各30帧；新增official_000/actor12 A30帧、B10帧后停止，共17个完成窗口，0230零证据所以B复用A。三例新补景均出现白车或灰白车形，检测器能抓清晰车但漏检残影；全部阻断进入Ω。第三目标没有旧DriveEditor/GLB基线，保留为失败诊断，未纳入默认世界。零新Ω前向、零训练，旧资产与全部对照保留。

审核页：`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-delete-repair/index.html`。上方为恢复后的旧版原视频/factual/DELETE，下方为三案例失败对照、精确mask、真实证据、原生输出和guard。详情见[最终报告](v77/DELETE_REPAIR.md)；唯一run证据在`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1`。

## 下一步与边界

等待用户看报告；不继续相同失败配置、不扩训练集、不恢复ProPainter、MOVE、定时任务或历史关机安排。若继续，本轮只留下“生成条件范围与最终精确写回范围可单独控制”的未验证假设，不预填改善。GT/LiDAR辅助、开发scene、隐藏背景无GT、官方训练重叠未知等边界保持。

`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
