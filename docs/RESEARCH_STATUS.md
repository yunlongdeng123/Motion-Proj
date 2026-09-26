# 当前研究状态

更新：2026-09-27（Asia/Singapore）。远端分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。**用户最新指令：DriveEditor权重下载完成后关闭远端AutoDL，用户休息。只完成下载、校验、保存推送和关机，不接着运行推理。**

## 当前执行与关机授权

关机作业 `WS-V77-DOWNLOAD-POWEROFF-20260927/r1`，原话：“你下好之后帮我把远端autodl关机，我先睡了”。这是本次下载结束后的单次操作，不创建Codex定时任务、cron或循环研究任务；先前`v77-pipeline`自动化保持已删除。

权重状态：正在断点续传；校验完成后关机；本次不启动推理。

01:10左右实例重启为无GPU模式，原下载及其后接推理的shell已退出；已保留约7.75GB分片，恢复同一官方12,059,467,678字节文件。CPU模式仅0.5核/2GiB，下载及检查限制CPU线程为1。下载器使用本轮LocalTUN重新确认的代理；断点续传，不丢弃分片。下载依赖本地代理保持联网。

新作业依次：恢复下载→safetensors结构与完整字节数检查→同步落盘→更新本状态并提交推送→检查其他实际作业→调用AutoDL平台关机流程。下载失败、存在未提交的其他改动、推送失败或其他作业仍运行时，不提前关机，明确记录失败状态。没有新的模型训练、生成或MOVE。

下载状态：`/root/autodl-tmp/work/v77_driveeditor_restore/state.json`。关机状态与日志：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DOWNLOAD-POWEROFF-20260927/r1/`；`state.json`只有写成`shutdown_requested`才说明已准备发送平台关机信号，不应把正在下载的记录当已关机。脚本为 `scripts/worldsim_v77/v77_download_then_shutdown.py`，持有单作业锁。

## 下次继续前

DriveEditor两scene尚未生成。`WS-V77-DRIVEEDITOR-COMPARE-20260927/r1`下已写入`hold_inference_for_shutdown.json`，controller会拒绝自动接回推理。下次用户要求继续且恢复GPU后，再显式解除此hold并执行；不恢复定时任务、不追加ProPainter实验。

已准备 scene_0230/actor22/CAM2/f0–9、scene_0255/actor25/CAM3/f15–24，各10帧、1024×576、10Hz。官方训练后DriveEditor deletion，seed42、25步、单帧解码、单3090串行CFG；20帧条件与官方删除分支一致。推理与视频导出代码已保存到v77；[执行说明](v77/DRIVEEDITOR_RUN.md)是前一轮登记，当前授权以本快照为准。

只复用DriveEditor的删除/背景补全能力。官方deletion的valid_mask为0，跳过SV3D去噪主干；完整checkpoint下载属于现有加载方式，不代表替代VGGT/Hunyuan/显式编辑器。模型索引中SV3D主干约3.05GB，可否裁剪需以后独立验证，本次不展开。

## 保留的研究边界

原位factual用户反馈“问题不大”，主要视觉问题是DELETE背景糊。保留原Hunyuan GLB、0230车头180°修正、已修复的相机投影和全部旧证据；旧MOVE碰撞/静态占据拒绝保持，没有新MOVE准入。R3联合背景参考候选为0，不复制伪观测地面。见 [V77-F02](research_failures/entries/V77-F02.md)、[DriveEditor评估](v77/DRIVEEDITOR_ASSESSMENT.md)。ProPainter短窗结果仅作为已完成历史保留，不据此预判DriveEditor质量。

VGGT系列仍为重建基座；V7.6按 [V76-F03](research_failures/entries/V76-F03.md) 关闭。当前没有新的科学否定，不改失败卡；`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: none`；`human_verdict: null`。
