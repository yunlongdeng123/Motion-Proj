# 当前研究状态

更新：2026-09-27（Asia/Singapore），分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。最新授权：主要推进DriveEditor已训练好的deletion补景；自动结构化指令合法性模块暂缓，由助手检查图像/视频和现有几何证据。不恢复定时任务、不追加ProPainter。

## 当前结果

`WS-V77-DRIVEEDITOR-COMPARE-20260927/r1,r2`已完成：两scene，3次DriveEditor推理，30生成帧；每段10帧/10Hz、1024×576、seed42、25步、单3090串行CFG/单帧解码，零训练。

0230/actor22/CAM2/f0–9：r1官方扩大mask仍有明显黑色车形；唯一r2控制只把mask改为SAM外包矩形+固定边距，主要车形残影消失、路面路缘可辨，栏杆局部仍变形。0255/actor25/CAM3/f15–24：r1官方扩大mask已去掉SUV，背景路面清晰度较旧糊影改善，围栏/车列细节仍需复核。助手已查看全部30生成帧；人工verdict保持null，不认定全流程高保真通过。

官方权重已完整下载12,059,467,678字节、4077张量，三次加载均无缺失/多余key。旧下载关机作业曾因Range错误退出，本轮继续授权替代旧关机安排；没有关机、重复控制器或定时任务。全部GPU生成已结束。

## 审核与边界

本地审核：`outputs/v77-driveeditor-deletion/index.html`，每scene同时间窗原视频/原生/DELETE合成，另有0230扩大mask失败与收紧mask改善对照、明确目标、mask视频、详细注释。远端run保留全部原始产物；[报告与architecture](v77/DRIVEEDITOR_RESULTS.md)、[汇总](autoresearch/worldsim_v77/driveeditor_run_20260927/summary.json)、[V77-F02](research_failures/entries/V77-F02.md)。11视频实解码通过，30帧PNG合成逐像素一致。

两个已曝光开发scene，1秒单相机，隐藏背景无真实GT，官方训练重叠未核对。r2同时改变范围和随机中心策略，未分离因果。原GLB、0230车头180°与相机修复保留；本轮未运行Ω。只DELETE，无新MOVE；0230旧碰框、0255旧近距和静态占据拒绝保持。

## 下一步

固定当前mask方案先扩展连续时间窗，检查目标再出现、栏杆漂移、邻车保持，再做跨相机检查，随后才接冻结VGGT-Ω背景重建。现有权重已展示改善，暂不造训练集、不训练Ω、不增加第二补景模型、不做自动合法性网络、语言或RL。

V7.6仍按[V76-F03](research_failures/entries/V76-F03.md)关闭，VGGT系列为重建基座。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
