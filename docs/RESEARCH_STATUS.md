# 当前研究状态

更新：2026-09-27（Asia/Singapore），分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。用户授权开始完整v77 DELETE工程。只运行DriveEditor已训练好的deletion、冻结Ω及原GLB；自动合法性模块暂缓，无新MOVE、定时任务或ProPainter实验。

## 当前工程

`WS-V77-DELETE-FULL-20260927/r1`已登记并运行。scene_0230/actor22，50帧/5秒；scene_0255/actor25，100帧/10秒；六相机10Hz。DriveEditor仅处理有目标mask的相机窗口，共24窗，每窗10帧、stride9、同时间重叠首帧条件，seed42/25步。空mask视图复制源RGB。0230保留已验证的收紧矩形，CAM5的两个SAM离散污染帧已保存原件并只留最大连通块；0255保留官方扩大mask，固定并保存随机结果。

背景按每时刻B_t存储，不称持久静态或4D世界。六相机RGB重跑Ω；GT相机、框外LiDAR单尺度参与米制控制。原位GLB的RGBA和相机Z层在本地Blender5.2.2渲染，保留0230车头180度和主点修复。已知z=5平面确认Z pass直接为camera z，不能额外除射线模长。QUERY只切visible=False；纯几何和RGB补足分开导出。

## 证据与边界

远端run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-FULL-20260927/r1`，以drive_state.json/omega_state.json和实际进程判断进度，不重复启动。完整工程尚未完成，人工verdict保持null。

此前1秒DriveEditor正反例已完成，见[实测报告](v77/DRIVEEDITOR_RESULTS.md)及本地`outputs/v77-driveeditor-deletion/index.html`。0230扩大mask残影、收紧mask改善，0255去车后路面清晰；栏杆细节仍有形变。新长窗口结果不能继承短窗通过结论。隐藏背景无真实GT、两个已曝光开发scene、训练重叠未核对、SAM小目标/近裁面门控保留。旧MOVE拒绝保持。

## 下一步

完成24窗→150时刻Ω→显式DELETE指令和遮挡合成→原视频/factual/DELETE连续审核页；检查窗口接缝、跨相机深度兼容性、残留目标与邻车保持。不得把工程打通等同高保真通过。无训练、无新生成器；无电源操作授权。

V7.6仍按[V76-F03](research_failures/entries/V76-F03.md)关闭。`failure_ledger_refs: [V77-F02]`；`human_verdict: null`。
