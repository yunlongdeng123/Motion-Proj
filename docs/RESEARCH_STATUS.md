# 当前研究状态

2026-10-02，wm-3090-1001，v77。Temporal reveal＋合法projected actor-state，先条件后surfel。

r37–r40已修曝光匹配、多地图/空LiDAR和SAM实际提示帧/JPEG溯源；15新来源尚无新增训练准入。S016独立QA0；r40提示帧改善身份疑点但未新增可用样本，控制关闭。旧Q060输入/条件2保留，Q046条件仍1，r36过滤未推广。

取消每scene前三截断后440→612窗口；原440及r41过滤逐条重现，但固定过程筛选仍0来源；停止将截断作为主因。r43用实际空间检查替代固定B距离预筛选：6scene来源已冻结，阶段为有界来源对照已完成；4候选独立QA为0/0/1/0，均未准入。控制器状态`CPU_complete_pending_geometry_result`；运行目录`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r43`。空间统计：{"sources": 6, "spatial_candidates": 4, "candidate_scenes": 3, "attempts": 13, "rejects": {"ground_support_gap": 5, "collision_clearance": 2, "ground_fit_unavailable": 2, "size_border_or_ego": 2}, "mask_QA_and_condition_not_done": true, "training_steps": 0, "training_admission": 0}。实例统计：{"protected_depth_order": 2, "insufficient_other_frame_evidence": 1, "instance_identity_or_visibility_uncertain": 1}。没有重复作业；按真实进程/状态断点续跑。

当前新训练准入0，50条完整配方未齐，正式A/B/C训练0步；身份分支与完整surfel未启动。GT camera/track/LiDAR明示POC辅助，完整Y仅监督/离线QA，条件仍必须从最终H遮后RGB构造。模型入口保留r21 sam_full_v2，r26裁洞失败不推广。人工verdict空。

下一步先修生成入口中camera→A→指定B及显露关系的前置检查，不再扩本次六来源的位置／速度网格。四例独立QA为0/0/1/0，全部拒绝。报告：[r42–r43与组件图](v77/TARGET_PROTECTED_SPACE_FIRST_R42_R43.md)；当前审核页本地`outputs/v77-target-protected-r43/index.html`，旧r38页面保留。

数据盘约44–45GiB可用，用户允许必要时清不重要旧物，本轮未删除。没有新增定时任务。真实DELETE保护车与无车背景两侧跨scene收益未达到，未满足关机条件。继续更新同一[V77-F02](research_failures/entries/V77-F02.md)。
