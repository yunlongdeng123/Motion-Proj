# 当前研究状态

更新：2026-09-27（Asia/Singapore）。远端分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。**按用户最新纠正：删除任务 `v77-pipeline` 的定时自动化，直接继续 DriveEditor 补景。禁止恢复定时任务，不追加 ProPainter 实验。**

## 当前执行

`WS-V77-DRIVEEDITOR-COMPARE-20260927/r1` 已固定 scene_0230/actor22/CAM2/f0–9 和 scene_0255/actor25/CAM3/f15–24，各10帧、1024×576、10Hz。接下来只执行 C=官方训练后 DriveEditor deletion，seed42、25步、单帧解码、既有单3090串行CFG适配，零训练。冻结mask直接来自前轮官方函数预检；两scene共20帧的12项条件返回值与官方删除分支逐张量一致。不是另做补景网络。

**尚无这两个scene的DriveEditor生成结果。** 约12.06GB官方checkpoint恢复中，已启动一次性“断点下载→结构检查→两scene串行推理”命令；下载未完成时GPU空闲是预期状态。禁止重复启动。下载状态 `/root/autodl-tmp/work/v77_driveeditor_restore/state.json`，推理状态在run根目录 `drive_state.json`；实际进程以 `pgrep -af 'v77_restore_driveeditor|v77_driveeditor_controller|v77_driveeditor_compare'` 核实。已有下载分片保留。该命令是当前实验的一次性作业，不是定时任务。

用户纠正前已完成A/B共4个ProPainter短窗对照，原始证据保留，不继续扩大这条支线；不得拿其结果代替DriveEditor结论。官方大mask在0230覆盖28.5%–73.3%画面、0255约15%–16%，有邻车范围风险；DriveEditor实际生成尚未运行，不能据此预判其补景质量。结果必须保留原生整帧与原图mask外合成两版，不把合成保护当成模型原生保持。

## 执行后的工作

直接检查DriveEditor生成日志和20张原生输出：目标是否消失、邻车/围栏是否改写、局部结构与时序是否稳定；制作原RGB／原生生成／局部合成的视频审核页。若出现资源或工程错误，定位该错误；不据此否定方法。人工verdict保持null。只有视频达到继续检验的质量，再接冻结Ω检查重建与跨视角一致性；当前不造数据、不训练。

原位factual的用户反馈为“问题不大”，当前主要问题为DELETE背景糊；原Hunyuan GLB、放置修正和全部旧证据保留。0230车头180°校正；旧MOVE碰撞限制、0255静态占据拒绝均保留，没有新的MOVE准入。R3两例联合背景参考候选为0，不复制伪观测地面。见 [V77-F02](research_failures/entries/V77-F02.md)、[R3](v77/PIPELINE_R3.md)、[DriveEditor评估](v77/DRIVEEDITOR_ASSESSMENT.md)。

## 资源与交付

一张RTX3090 24GiB；DriveEditor源码 `/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor`，环境 `/root/autodl-tmp/envs/driveeditor/bin/python`。保留此前单卡适配源码，不覆盖。没有Ω/Hunyuan前向、没有训练、没有新MOVE。

本次登记与执行说明见 [DRIVEEDITOR_RUN](v77/DRIVEEDITOR_RUN.md)，run `/root/autodl-tmp/runs/worldsim_v77/WS-V77-DRIVEEDITOR-COMPARE-20260927/r1`。下载完成后需要核对实际完成状态，不能把本快照里的计划当结果。现有本地审核页 `outputs/v77-pipeline-r3/index.html` 仍是旧结果；暂不以它冒充DriveEditor输出。

VGGT系列仍为重建基座；V7.6按 [V76-F03](research_failures/entries/V76-F03.md) 关闭。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02（执行范围纠正）`；`human_verdict: null`。
