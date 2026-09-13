# 当前：V8.1 有限补证收口（2026-09-13）

WS-V81-CLOSE-01完成：10旧候选内部复核；8新AV2日志840格模型前筛查，3日志18项新推理。旧440项未重跑、无训练。木板墙最强法向候选参考不稳；灰墙297点/0.103m好例保留。VGGT两个新日志有0.429/0.290m可恢复差距，单图度量锚点后MAE0.018/0.472/0.337m；DVGT两个后向单相机坐标/覆盖合同未建立，不计科学failure或稳健负例。

四环证据不足，V8.2 NO_GO；当前共同失效子命题降低优先级，停止本批次。其他8方法已纳入方法族badcase report，DGGT等方法级实证仍缺，条件下游扩展未触发；不能宣称整个V8.1已测完。无自动续跑、确认reserve封存、未关机。人工verdict=null；failure_ledger_refs=V81-F01/F02/F03/F04；failure_ledger_delta=V81-F05。

[有限收口报告](WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md)；[方法族badcase report](WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)；[推进判定](WORLDSIM_V8_1_TO_V8_2_DECISION.md)。以下为历史。

## 接续边界

本批任务已结束，2×3090充足，不自动重跑。当前入口为有限收口报告及方法族badcase report。后续只有明确新任务才扩展；其他方法的机制假说不能填写成实测failure。

原始GPU P2：`/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01`，440项；本次`/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01`，18项新forward。复核：`scripts/audit_worldsim_v81_residual.py`与`scripts/evaluate_worldsim_v81_bounded_intervention.py`，均为CPU读取现有输出。获取、筛查、freeze脚本用于记录本次构建；不可盲目重跑freeze覆盖原登记时间。

新单图relative导出只在显式`--allow-relative-single-view`下使用`*_depth_z_native.npy`，`metric_scale=null`，不声称米制。新队列结果已改写为VIEW_DIAGNOSTIC与实际rich/removed/restored；stdout中底层通用入口的临时full6标签不作为最终登记。旧自然full6结果未修改。

DVGT新rear单相机的point-frame/coverage状态保持未解决，不按当前camera-z评价器推进该科学主张。确认reserve未消耗。不存在继承V74关机的新授权。
