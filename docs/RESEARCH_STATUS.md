# 当前研究状态

更新：2026-09-27（Asia/Singapore）。远端分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。用户最新要求：继续使用DriveEditor已训练好的deletion权重做补景。自动指令合法性模型/规则暂缓，由助手结合图像、视频和几何证据检查；不追加ProPainter、不恢复定时任务。

## 当前执行

沿用唯一实验 `WS-V77-DRIVEEDITOR-COMPARE-20260927/r1`，两scene此前尚无DriveEditor生成。3090已恢复且无GPU作业。上次下载在约10.7GB时遇到HTTP200代替Range206而退出，关机作业记录`failed_without_shutdown`；本轮恢复剩余分片，校验后串行生成两scene。旧关机任务已结束，本轮继续授权替代旧关机安排，不启动电源操作。原hold及状态已备份至run的`resume_20260927_backup`。

输入：scene_0230/actor22/CAM2/f0–9、scene_0255/actor25/CAM3/f15–24，各10帧、1024×576、10Hz。官方训练后DriveEditor deletion，seed42、25步、单帧解码、单3090串行CFG；20帧条件已与官方删除分支一致。逐场景保存原生整帧、mask内合成及原RGB对照，实际生成后导出视频，不把旧结果当新结果。

权重下载状态 `/root/autodl-tmp/work/v77_driveeditor_restore/state.json`。本轮只接DriveEditor背景补全部件；deletion的valid_mask为0，跳过SV3D去噪主干。完整文件属于现有加载方式，不替代VGGT/Hunyuan/显式编辑器。零训练；当前不裁剪权重、不造训练集。

## 判断与下一步

先看目标是否消失、道路/围栏是否清楚且时序稳定、邻车是否保持。当前官方扩大mask有邻车投影交叠，不能把输入范围风险当生成损伤；必须看实际输出。原生整帧与局部合成并列，避免合成隐藏原生改动。短窗和单相机通过才考虑长窗/跨相机/冻结Ω重建，不自动升级成可靠背景世界。

原位factual用户反馈“问题不大”，主要视觉问题是DELETE背景糊。保留原Hunyuan GLB、0230车头180°修正、相机投影修复及全部旧证据。旧MOVE0230碰框、0255近距且静态占据拒绝保持；本轮只DELETE，无新MOVE。R3联合背景参考候选为0，不把伪观测地面当真实GT。见[V77-F02](research_failures/entries/V77-F02.md)、[执行登记](v77/DRIVEEDITOR_RUN.md)。

VGGT系列仍为重建基座；V7.6按[V76-F03](research_failures/entries/V76-F03.md)关闭。`failure_ledger_refs: [V77-F02]`；本轮结果待生成；`human_verdict: null`。
