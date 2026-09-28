# 当前研究状态

更新：2026-09-28，v77。用户已给九个旧例的原位重建/DELETE评分，并把下一阶段收敛到**已训练 DriveEditor 的 DELETE 补景能力审计**。旧九例属于已调试 DEV，评分原文登记于 [新审计协议](v77/DELETE_AUDIT_PROTOCOL.md)，旧版视频与全部原始产物保留；不按这九例做最终测试。

当前唯一任务 `WS-V77-DELETE-AUDIT-20260928/r1`。从官方 nuScenes val 150 scene 的 GT 元数据与固定种子预选 45 scene/70 个单车辆 clip，另隔离 25 个未看的 final scene。大小、相机、可见度、车辆投影与昼夜/密度覆盖均按预定规则达到；每 clip 26 个实际相机时间帧，共 1820 时刻、1802 个不同原始 JPEG。四个最近相机曝光偏差 70–85ms，另有少量 10Hz 位置复用同一 12Hz 曝光，按输入时间误差记录。全部目标身份、26 帧 GT 投影和公共盘精确文件清单已登记。

1802张原始JPEG已全部提取、逐张解码，缺失/损坏/重复均0。用户已开启GPU，明确要求推理、每半小时检查、完成后关闭AutoDL。RTX3090 24GB已确认。SAM2完成70/70例，中位4.135秒；21例有GT投影存在但core为空的帧，须结合可见像素判断输入资格。主环境缺iopath，已用现有worldsim-v77-sam2环境完成跟踪，没有安装新依赖。后台控制器已启动DriveEditor，随后顺序编码四列视频与HTML。远端run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1`。状态以controller_state.json、drive_state.json和真实进程为准。

输入预检初版黄框坐标比例错误已修复并留证。A025提示由第2帧换第12帧，其他GT占框14.6%→3.3%；统一p2规则覆盖70例并保留旧提示，20例仍占框≥20%，须复核实例mask。投影包络交叠不能直接等同实际邻车像素泄漏。A012的一帧裁剪看不清目标，保留疑点。数据盘已扩至600GB，启动前剩154GB；本轮RGB约266MB，空间足够。完整数据集/训练集不在本轮磁盘预算内。

下一步：同一规则完成70例、保存原生与最终输出、每例在冻结prompt抽一帧做助手粗分类/输入资格说明。导出原视频目标框／模型洞／原生生成／最终DELETE四列同步视频，核验解码与本地HTML链接，更新报告并提交推送。`DriveEditor factual reconstruction`保持未运行/不可判；用户0/1/2 verdict留空。预计总耗时5–7小时，按实际生成吞吐更新。

本次半小时监控ID为`v77-delete`。全部产物保存、HTML同步和提交推送完成，且确认无其他训练、评价、数据、渲染或启动控制器后，按用户本次授权关闭AutoDL；不因推理报错直接关机。关机后删除本次监控。无Ω、GLB、训练或逐场景调参。

见[GPU执行记录与组件图](v77/DELETE_AUDIT_GPU_RUN.md)、[CPU预检报告](v77/DELETE_AUDIT_CPU_PREFLIGHT.md)、[审计协议](v77/DELETE_AUDIT_PROTOCOL.md)。失败卡继续用V77-F02；启动阶段failure_ledger_delta为none，尚无新的模型质量结论。
