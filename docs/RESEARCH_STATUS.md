# 当前研究状态

2026-10-01，v77；默认wm-3090-1001。`WS-V77-TARGET-PROTECTED-20260929/r9`过程盘点/有界工厂收口（覆盖缺额、训练0）；`r10`有限过程同80张量/160步训练、四臂76窗、19例模型与22候选HTML交付完成。不重复启动，final未用，无电源/定时操作。

r10未形成稳定收益，不替换原/r7/r8：旧GT保护车MAE对r8下降5.18%，但新过程GT洞内/保护车分别变差16.41%/11.90%；扫过验证P006在真实GT道路上补出额外车辆，真实DEV关键失败也仍在。

实际r7/r8均无达到阈值的B扫过；r8已有12个静止A＋运动ego。r10训练50/25world，实际11静止A＋ego、4扫过、12显露（可重叠）；新扫过val仅1world、dense train仅1world、一秒窗口。其他帧证据仅GT框表面几何代理，真实隐藏GT未知，不把统计等同因果或视频通过率。

下一有界控制：先更换过程来源与窗口抽取，达到原r9的8扫过train／3扫过val／2隔离val-world覆盖目标；同时匹配实际投影速度、尺度变化与合法截边过程。数据准入和冻结完成后才做同模块/同预算控制，不重复训练当前四个扫过例。 尚未启动；保留全部输入/拒绝/旧基线/新权重。参见[报告与组件图](v77/TARGET_PROTECTED_TEMPORAL_CONTROL_R10.md)、[指标](autoresearch/worldsim_v77/target_protected_20260929/r10/results_summary.json)、[审核](autoresearch/worldsim_v77/target_protected_20260929/r10/review_link.md)。failure_ledger_refs: [V77-F02]。
