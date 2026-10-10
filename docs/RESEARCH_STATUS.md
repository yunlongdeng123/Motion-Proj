# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户授权自主完成P1，无human in loop；hold后自主有界排查，不随便关AutoDL。当前保持开机，旧关机记录仅作历史。

P0工程闭环完成；P1尚未通过，P2/P3未开始。原paper-bidirectional-m4正式1000→2000完成，新增1000更新、758个训练视频、平均6.01秒/步、峰18.40GiB，梯度全部有限、冻结梯度0。相同三valid×两倍率六窗全部25帧，两名gpt-6-sol/xhigh/no-fast助手独立审完150帧，门禁hold。海豚窄遮挡窗较500/1000出现可见原生轮廓与摆尾进步，仍不稳定；滑板两窗仍无视角转换/跳跃，白鲸未随近景放大/姿态，宽海豚仍变形。真实中心硬写回不计模型收益，2%论文预算阴性不能否定整方法。[实测与组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

零训练fps单因素配对已完成：正式2000滑板.33窗/seed2026/25步，两CFG支ID6→7之外实际输入逐值相同；ID6全部25原始PNG重现正式2000。59.65秒、峰9.47GiB，原生像素响应0.00417，独立全25帧审核no_clear_gain。正确单位visible VAE与实际融合condition解码均保留中心人物/板/跳跃，但原生生成没有跟随；这不唯一定位具体模块，不改正式协议。

当前继续独立单训练片段全噪声分布容量诊断：正式2000源/原Adam，固定0fc958cde2/start2，每更新重新抽σ/ε，最多512更新；前后同seed2036/25步的纯噪声QUERY及完整teacher分别保存、独立审核。teacher含带噪GT，不计QUERY；诊断只保存可训练权重、无Adam、不能续训或进入正式恢复。实际PID/命令见同run `single_clip_capacity_step2000_512_state.json`，不直接追加正式长训、不混诊断权重。启动前修正脚本对FCNet既有8个未参与edge分支的误报，只按精确名称记录无梯度/无Adam state；其余514个参数要求逐个增步，正式损失未改。

此前500/1000六窗、四组64步控制、非法完整GT latent oracle、带GT单sigma teacher均保存。后者可恢复粗动作但x_t含GT、非法条件不计QUERY；未唯一定位传播或否定U-Net。启动本次fps诊断前遇Windows默认GBK元数据写出，远端UTF-8读取失败，发生在模型加载前；保存栈/状态，改显式UTF-8后只重启同一两支诊断，模型/协议未改。

1951有效YouTube训练视频与六数据包在，DAVIS90/附录YT60固定IDs不改、valid排除正式60。四指标尚未计算，protocol_verified=false；GT训练flow/首帧CLIP与可见QUERY差异、single3090 bf16、YT60 valid_all_frames与论文test名称、FVD倍率汇总仍明示。[正式协议](v81/YOUTUBE_VOS_P1_R1.md)。关键100/500/1000/2000与原输入/失败媒体保留；仅清理过可重构内层zip，外层7z及RGB未动。容量诊断启动前数据盘free约4.05GB、系统6.50GB；最终仅保存约1.61GB模型，不保存额外4.83GB完整Adam断点，不盲删模型/数据。

本地 `outputs/v81-paper-p1/condition_gap_review.html` 展示500/1000/2000六列同步视频、fps配对、全帧图板和独立审核；新增24条2000与8条fps视频实际完整解码，旧媒体不重复重编码。Git仅代码/轻量实测，源码ZIP<100MB；human_verdict=null，failure_ledger_refs=[V77-F02]仅历史证据边界，failure_ledger_delta=none。最近周额度余9%、2张重置卡，未触发3%阈值、未消费。15分钟监控持续，正常无变化静默，P1目标未完成。[轻量记录](v81/P1_CONDITION_GAP_GPU_R1.json)。
