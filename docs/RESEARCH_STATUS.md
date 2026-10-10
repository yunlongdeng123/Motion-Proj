# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户授权自主P1，无human in loop；质量hold后继续有界排查，AutoDL保持开机，旧关机记录不适用。

P0工程闭环完成，P1尚未通过，P2/P3未开始。正式2000六窗/150帧独立审核hold；海豚.125较500/1000有局部轮廓与摆尾进步，滑板与白鲸仍不随输入动作/尺度。四指标尚未计算、protocol_verified=false，不宣称论文复现或DELETE收益。[实测与组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

单训练片段512诊断已完成：514个有Adam历史的参数均2000→2512，数值/梯度有限、冻结梯度0；GT-x_t单步teacher误差下降15.9%，原生Gaussian QUERY仍无猫头鹰脸/双翼及翼姿跟随，独立25帧hold。诊断末态只保存1.61GB模型、无Adam/RNG，不能恢复正式训练。随后零更新CFG配对59.10秒、峰9.49GiB；普通1→3支全25PNG重放、15项实际输入逐值相同。恒1使后半段主体更模糊并右移，全25帧独立判regression，仅限此片段；不采用该采样改动，不重做同项或扫CFG/σ/seed。

当前已启动正式2000→5000原协议有界学习点，最多新增3000更新；从原2000完整Adam/RNG恢复，不混512诊断权重。父18955、子19021已核对实际命令，快照时间2026-10-10T01:57:29.527801+00:00记录2011步；这是快照，不是实时进度。按此前6.01秒/步预计训练约5小时，随后同三valid×两倍率六窗/25帧全部独立审核，不自动续100K。新证据未见确定接线bug；2%预算和单片段阴性不足否定正式学习曲线。[协议](v81/YOUTUBE_VOS_P1_R1.md)。

数据盘约2.2GB、系统盘约6.5GB；只在5000保存一份约4.83GB完整断点至 `/root/v81_checkpoint_retained/paper_bidirectional_m4_step005000/`，phase/train建软链接入口，不是外部耐久备份。原100/500/1000/2000断点、真实RGB、6包与失败媒体保留，本轮无清理。系统盘保存后余量有限，下一阶段需重新核对空间，不盲删资产。

本地 `outputs/v81-paper-p1/condition_gap_review.html` 保留正式500/1000/2000对照，新增512 before/after和CFG双支完整视频、10张全帧图板与独立QA；新增16视频均实际完整解码25帧。1951训练视频及固定DAVIS90/YT60名单不改；train GT flow/首帧CLIP与visible QUERY差异、bf16/单卡、YT60来源及FVD汇总仍明示。Git仅代码/轻量实测，源码ZIP<100MB。human_verdict=null，failure_ledger_refs=[V77-F02]仅历史证据边界、failure_ledger_delta=none。[记录](v81/P1_CONDITION_GAP_GPU_R1.json)。最近实际周额度余7%、2张重置卡未用；15分钟监控继续，正常进度静默，剩余<3%才使用已授权卡。
