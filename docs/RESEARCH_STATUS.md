# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户授权自主P1，无human in loop；质量hold后继续有界排查，AutoDL保持开机，旧关机记录不适用。

P0工程闭环完成，P1尚未通过，P2/P3未开始。正式2000六窗/150帧独立审核hold；海豚.125较500/1000有局部轮廓与摆尾进步，滑板与白鲸仍不随输入动作/尺度。四指标尚未计算、protocol_verified=false，不宣称论文复现或DELETE收益。[实测与组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

单训练片段512诊断已完成：514个有Adam历史的参数均2000→2512，数值/梯度有限、冻结梯度0；GT-x_t单步teacher误差下降15.9%，原生Gaussian QUERY仍无猫头鹰脸/双翼及翼姿跟随，独立25帧hold。诊断末态只保存1.61GB模型、无Adam/RNG，不能恢复正式训练。随后零更新CFG配对59.10秒、峰9.49GiB；普通1→3支全25PNG重放、15项实际输入逐值相同。恒1使后半段主体更模糊并右移，全25帧独立判regression，仅限此片段；不采用该采样改动，不重做同项或扫CFG/σ/seed。

正式2000→5000新增3000更新、六窗推理已完成；完整5000断点与514个step5000 Adam状态保留，24新视频全部解码。两名独立助手全150帧审核整体hold：海豚.125躯体/尾鳍有明确局部进步，其余五窗仍动作/尺度/构图脱节，滑板尤为明显。单滑板.125的FCNet边序零更新对照已完成，普通25帧精确重放/19项模块配对通过；独立全25帧no_clear_gain。末尾JSON保存错误已CPU恢复，不重跑GPU，不认定排序是bug。已启动原协议5000→7500有界学习点，不改正式协议、不自动续100K。[实测与组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

用户授权的数据盘清理已释放105.95GB，可用2.21→108.16GB；5000保存后系统盘余1.66GB。删除已完整解压的6个重复归档、5个非在用旧环境、旧v77优化器与未入选step32，并精简9个已结束P1诊断的优化器；最终模型逐张量核对，正式100/500/1000/2000完整断点、512末态、真实RGB/标注/评分与failure媒体保留。退役优化器不可精确续训，旧环境需按配方重装；不要重下已解压数据。[清单与恢复边界](v81/STORAGE_CLEANUP_20261010.md)。已按原配置在5000保存约4.83GB完整断点至 `/root/v81_checkpoint_retained/paper_bidirectional_m4_step005000/`，phase/train软链接不是外部耐久备份。

本地 `outputs/v81-paper-p1/condition_gap_review.html` 保留正式500/1000/2000对照、新增5000六窗和边序两支原生/硬写回对照；本轮32新视频均实际完整解码25帧，独立审核150+25帧。1951训练视频及固定DAVIS90/YT60名单不改；train GT flow/首帧CLIP与visible QUERY差异、bf16/单卡、YT60来源及FVD汇总仍明示。Git仅代码/轻量实测，源码ZIP<100MB。human_verdict=null，failure_ledger_refs=[V77-F02]仅历史证据边界、failure_ledger_delta=none。[记录](v81/P1_CONDITION_GAP_GPU_R1.json)。最近实际周额度余3%、2张重置卡未用；15分钟监控继续，正常进度静默，剩余<3%才使用已授权卡。
