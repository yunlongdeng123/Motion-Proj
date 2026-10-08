# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 `wm-vgpu-1008`。

唯一run `WS-V77-TARGET-PROTECTED-20260929/r50` 已完成固定真实DELETE扩大排查。独立输入检查87例，36个0/1分剔除，20个官方train开发场景42个输入2分目标。42例SAM无空mask；独立实例检查40例通过，R013/R066因吞入邻车拒绝，不算补景模型失败。40个DELETE全部完成、20景仍在，其中两景各1例，其余2–3例。

固定r46官方DriveEditor原权重+r21完整SAM，seed42/25steps/10帧/1024×576；无Adapter、训练、参数扫描或时间模块修改。400帧写回验证通过，原生与最终视频均保留。独立subagent每例只抽f05粗分类，标签可重叠：{'film_or_ghost': 14, 'car_body_extends_onto_road': 4, 'none_visible': 21, 'actor_regeneration': 4, 'protected_actor_deformation': 1}。它不是人工0/1/2、隐藏区域真值或视频通过率。后车可见证据充分9例、薄弱8例、无候选21例、不确定2例，不能当完整隐藏车身已知。

同一页面 `outputs/v77-real-delete-r50/index.html` 展示40例四列原视频/mask/原生DELETE/固定写回；两例mask拒绝及全部旧低质量输入另存归档。主干没有更新，因此本轮是问题排查，不能宣称模型收益。后续按有真实证据的结构失败匹配遮挡数据，Y必须是真实视频，再单独验证内部空间层；不自动训练、不改时间层。5个隔离train景与旧val隔离不动，本轮缓存来源不是总体/最终评测。

GPU计算已结束、进程退出，用户可切CPU；本轮无自动关机或定时任务。GPU报告32GB RTX 4080 SUPER，CPU16核，推理累计36.5分钟、PyTorch峰值已分配显存21.86GiB；数据盘余量见run资源记录，无清理。[报告与组件图](v77/REAL_DELETE_STRUCTURE_R50.md)，[同一失败卡V77-F02](research_failures/entries/V77-F02.md)。
